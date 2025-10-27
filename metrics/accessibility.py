import cv2
import pathlib
import trimesh
import numpy as np
from warnings import warn
from dataclasses import dataclass, field
from pydantic import BaseModel
from matplotlib import pyplot as plt
from scenes import Scene
from vlm import BaseVLM
from .base import BaseMetric, MetricResult
from .registry import register_vlm_metric

# ----------------------------------------------------------------------------------------

class ObjFunctionalSidesAssessment(BaseModel):
    obj_id: str
    obj_description: str
    functional_sides: list[str]
    reason: str

class ObjFunctionalSidesResponseFormat(BaseModel):
    assessments: list[ObjFunctionalSidesAssessment]

@dataclass
class AccessibilityMetricConfig:
    """
    Configuration for the accessibility metric.

    Attributes:
        image_resolution: the resolution of the image
        scale_margin: the margin to add to the scale
        obj_height_threshold: the height threshold for objects
        access_area_width: the width of the access area
        access_area_offset: the offset of the access area from the object
        floor_color: the color of the floor
        obj_color: the color of the objects
        access_area_color: the color of the access area
    """

    image_resolution: int
    scale_margin: float
    obj_height_threshold: float
    access_area_width: float
    access_area_offset: float
    floor_color: list[int] = field(default_factory=lambda: [255, 0, 0])
    obj_color: list[int] = field(default_factory=lambda: [0, 255, 0])
    access_area_color: list[int] = field(default_factory=lambda: [0, 0, 255])

# ----------------------------------------------------------------------------------------

@register_vlm_metric(config_class=AccessibilityMetricConfig)
class AccessibilityMetric(BaseMetric):
    """
    Metric to evaluate whether objects are blocked by other objects.
    """

    def __init__(self,
                 scene: Scene,
                 vlm: BaseVLM,
                 output_dir: pathlib.Path,
                 cfg: AccessibilityMetricConfig,
                 **kwargs) -> None:
        """
        Initialize the metric.

        Args:
            scene: the scene to evaluate
            vlm: the VLM to use for the metric
            cfg: the accessibility metric configuration
            output_dir: the output directory
        """

        self.scene = scene
        self.vlm = vlm
        self.output_dir = output_dir / "accessibility"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg

        self.vlm.reset()

        # Prepare scene information
        self.half_image_resolution = self.cfg.image_resolution // 2
        self.obj_info = [f"obj_id: '{obj_id}', obj_description: '{self.scene.obj_descriptions[obj_id]}'" for obj_id in self.scene.get_obj_ids()]
        
        t_floors = [t_arch for arch_id, t_arch in scene.t_architecture.items() if arch_id.startswith("floor")]
        self.t_floor = trimesh.util.concatenate(t_floors)
        self.t_floor_center = self.t_floor.bounds[0] + self.t_floor.extents / 2
        self.scale = self._get_scale(self.t_floor.vertices, self.t_floor_center)
    
    def _get_scale(self, floor_vertices: np.ndarray, floor_center: np.ndarray) -> float:
        """
        Get the scale for mapping scene to image coordinates.

        Args:
            floor_vertices: the vertices of the floor
            floor_center: the center of the floor
        
        Returns:
            scale: the scale for mapping scene to image coordinates
        """

        denormed_floor_vertices = floor_vertices - floor_center
        denormed_floor_vertices = denormed_floor_vertices[:, :2]
        scale = np.max(np.abs(denormed_floor_vertices)) + self.cfg.scale_margin
        return scale
    
    def _scene_to_image_coordinates(self, scene_x: float, scene_y: float, scale: float) -> tuple:
        """
        Convert scene coordinates to image coordinates.

        Args:
            scene_x: the x coordinate in the scene
            scene_y: the y coordinate in the scene
            scale: the scale of the scene
        
        Returns:
            x_image: the x coordinate in the image
            y_image: the y coordinate in the image
        """

        scene_y = -scene_y
        x_image = int(scene_x / scale * self.half_image_resolution) + self.half_image_resolution
        y_image = int(scene_y / scale * self.half_image_resolution) + self.half_image_resolution
        return x_image, y_image
    
    def _get_floor_mask(self, verbose: bool = False) -> np.ndarray:
        """
        Get the floor mask.

        Args:
            verbose: whether to visualize during the run
        
        Returns:
            mask: the floor mask
        """

        # Initialize the floor mask
        mask = np.zeros((self.cfg.image_resolution, self.cfg.image_resolution, 3), dtype=np.uint8)

        # Center the floor vertices at the origin
        floor_vertices = self.t_floor.vertices - self.t_floor_center
        floor_vertices = floor_vertices[:, :2] # Ignore z coordinate (height)

        # Draw floor
        for face in self.t_floor.faces:
            face_vertices = floor_vertices[face]
            face_vertices_image = [self._scene_to_image_coordinates(x, y, self.scale) for (x, y) in face_vertices]

            pts = np.array(face_vertices_image, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], self.cfg.floor_color)

        # Prepare object bounding box info
        # Schema: [center_x, center_y, extent_x, extent_y, angle_around_z]
        obj_bboxes = np.empty((0, 5))
        for obj_id in self.scene.get_obj_ids():
            
            # Get the object center and extents in the default pose
            obj_bbox_center = self.scene.get_obj_bbox_center(obj_id) - self.t_floor_center
            obj_bbox_extents = self.scene.get_default_pose_obj_bbox_extents(obj_id)

            # Ignore objects above the height threshold
            if obj_bbox_center[2] > self.cfg.obj_height_threshold:
                continue
            
            # Get the object angle around the z-axis
            rotation_angle = self.scene.get_obj_z_rotation(obj_id)
            
            # Store the object bounding box
            bbox_info = np.asarray([*obj_bbox_center[:2], *obj_bbox_extents[:2], rotation_angle])
            obj_bboxes = np.vstack([obj_bboxes, bbox_info])

        # Draw object bounding boxes
        for bbox_info in obj_bboxes:
            center_x, center_y, extent_x, extent_y, rotation_angle = bbox_info

            # Get where the object bbox is on the image
            on_image_center = self._scene_to_image_coordinates(center_x, center_y, self.scale)
            on_image_size = (int(extent_x / self.scale * self.half_image_resolution), int(extent_y / self.scale * self.half_image_resolution))

            # Calculate 2D bbox vertices
            box_points = cv2.boxPoints(((on_image_center[0], on_image_center[1]), on_image_size, -np.rad2deg(rotation_angle)))
            box_points = box_points.astype(int)
            
            # Draw the object bbox
            cv2.fillPoly(mask, [box_points], self.cfg.obj_color)
        
        plt.title("Accessibility - Mask")
        plt.imshow(mask[:, :, ::-1])
        plt.savefig(self.output_dir / "a0_mask.png")
        if verbose:
            plt.show(block=True)
        plt.close()

        return mask
    
    def _get_accessibility_score(self, floor_mask: np.ndarray, obj_id: str, side: str, verbose: bool = False) -> float:
        """
        Get the accessibility score of the object from the given side.

        Args:
            floor_mask: the floor mask
            obj_id: the object ID
            side: the side of the object
            verbose: whether to visualize during the run
        
        Returns:
            score: the accessibility score
        """

        # Initialize the accessibility map with the floor mask
        accessibility_map = floor_mask.copy()

        # Compute the access area direction, distance, and extents based on the side
        match side:
            case "front":
                access_area_direction = np.array([0, -1, 0])
            case "back":
                access_area_direction = np.array([0, 1, 0])
            case "left":
                access_area_direction = np.array([-1, 0, 0])
            case "right":
                access_area_direction = np.array([1, 0, 0])
        
        obj_default_pose_bbox_extents = self.scene.get_default_pose_obj_bbox_extents(obj_id)
        match side:
            case "front" | "back":
                access_area_distance = obj_default_pose_bbox_extents[1] / 2
                access_area_extents = [obj_default_pose_bbox_extents[0], self.cfg.access_area_width]
            case "left" | "right":
                access_area_distance = obj_default_pose_bbox_extents[0] / 2
                access_area_extents = [self.cfg.access_area_width, obj_default_pose_bbox_extents[1]]
        
        # Compute the center of the access area
        obj_default_pose_center = self.scene.get_default_pose_obj_bbox_center(obj_id)
        distance_from_center = access_area_distance + self.cfg.access_area_width / 2 + self.cfg.access_area_offset
        access_area_default_pose_center = obj_default_pose_center + access_area_direction * distance_from_center
        access_area_center = (np.asarray(self.scene.get_obj_matrix(obj_id)) @ np.append(access_area_default_pose_center, 1))[:3]
        access_area_center = (access_area_center - self.t_floor_center)[:2]

        # Compute where the access area is on the image
        on_image_center = self._scene_to_image_coordinates(access_area_center[0], access_area_center[1], self.scale)
        on_image_size = (
            int(access_area_extents[0] / self.scale * self.half_image_resolution),
            int(access_area_extents[1] / self.scale * self.half_image_resolution)
        )

        # Get the object rotation angle and calculate the 2D access area vertices
        rotation_angle = self.scene.get_obj_z_rotation(obj_id)
        box_points = cv2.boxPoints(((on_image_center[0], on_image_center[1]), on_image_size, -np.rad2deg(rotation_angle)))
        box_points = box_points.astype(int)
        
        # Draw the access area
        access_area_mask = np.zeros((self.cfg.image_resolution, self.cfg.image_resolution), dtype=np.uint8)
        cv2.fillPoly(access_area_mask, [box_points], 255)
        accessibility_map[access_area_mask == 255] = accessibility_map[access_area_mask == 255] + self.cfg.access_area_color

        # Draw the access area again at the center of another canvas to compute the mask area
        access_area_size_canvas = np.zeros((self.cfg.image_resolution, self.cfg.image_resolution), dtype=np.uint8)
        mean_x, mean_y = np.mean(box_points, axis=0).astype(int)
        cv2.fillPoly(access_area_size_canvas, [box_points - (mean_x, mean_y) + self.half_image_resolution], 255)

        figure_short_obj_name = f"{self.scene.obj_descriptions[obj_id].split(' ')[0]}"
        if "instance" in self.scene.obj_descriptions[obj_id]:
            figure_short_obj_name += f"-{self.scene.obj_descriptions[obj_id].split('- ')[-1]}"
        
        plt.title(f"Accessibility - {figure_short_obj_name} - {side} area")
        plt.imshow(accessibility_map[:, :, ::-1])
        plt.savefig(self.output_dir / f"a1_{obj_id}-{side}.png")
        if verbose:
            plt.show(block=True)
        plt.close()
        
        # Compute the accessibility score as the ratio of the access area on the floor that is not blocked by other objects
        area_mask_size = np.sum(access_area_size_canvas == 255)
        accessible_color = np.asarray(self.cfg.floor_color) + np.asarray(self.cfg.access_area_color)
        mask_on_floor_size = np.sum(np.all((accessibility_map[access_area_mask == 255] == accessible_color), axis=-1))
        score = mask_on_floor_size / area_mask_size

        print(f"Accessibility score of {obj_id} ({self.scene.obj_descriptions[obj_id]}) - {side}: {score}")
        
        return score

    def run(self, verbose: bool = False) -> MetricResult:
        """
        Run the metric.

        Args:
            verbose: whether to visualize during the run
        
        Returns:
            result: the result of running the metric
        """

        # Get the functional sides of the objects
        prompt_info = {
            "obj_info": "\n".join(self.obj_info)
        }
        response: ObjFunctionalSidesResponseFormat | str = self.vlm.send("obj_functional_sides",
                                                                         prompt_info=prompt_info,
                                                                         response_format=ObjFunctionalSidesResponseFormat)

        if type(response) is str:
            warn("The response is not in the expected format.", RuntimeWarning)

        # Get the floor mask
        floor_mask = self._get_floor_mask(verbose=verbose)

        # Compute the accessibility score for each object for each functional side
        accessibility_scores = {}
        for assessment in response.assessments:
            obj_id = assessment.obj_id
            accessibility_scores[obj_id] = {}

            for side in assessment.functional_sides:
                accessibility_scores[obj_id][side] = self._get_accessibility_score(floor_mask, obj_id, side, verbose=verbose)
        
        # Compute the maximum accessibility score for each object
        for obj_id, scores in accessibility_scores.items():
            if len(scores.keys()) == 0:
                accessibility_scores[obj_id]["max"] = -1 # No applicable functional sides
            else:
                accessibility_scores[obj_id]["max"] = np.max(list(scores.values()))
        
        result = MetricResult(
            message=f"Accessibility score for all objects: {[scores['max'] for scores in accessibility_scores.values()]}",
            data=accessibility_scores
        )

        print(f"\n{result.message}\n")

        return result
    