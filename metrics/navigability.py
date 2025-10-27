import cv2
import pathlib
import trimesh
import numpy as np
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
from scenes import Scene
from .base import BaseMetric, MetricResult
from .registry import register_non_vlm_metric

# ----------------------------------------------------------------------------------------

@dataclass
class NavigabilityMetricConfig:
    """
    Configuration for the navigability metric.

    Attributes:
        calculate_object_area: whether to calculate the object area ratio
        image_resolution: the resolution of the image
        robot_width: the width of the robot
        scale_margin: the added margin for scaling the scene
        obj_height_threshold: the height threshold for objects
        floor_color: the color of the floor in the image
        obj_color: the color of the objects in the image
    """

    calculate_object_area: bool = False
    image_resolution: int = 256
    robot_width: float = 0.2
    scale_margin: float = 0.2
    obj_height_threshold: float = 2.0
    floor_color: list[float] = field(default_factory=lambda: [255, 0, 0])
    obj_color: list[float] = field(default_factory=lambda: [0, 255, 0])

# ----------------------------------------------------------------------------------------

@register_non_vlm_metric(config_class=NavigabilityMetricConfig)
class NavigabilityMetric(BaseMetric):
    """
    Metric to evaluate scene navigability.

    Adapted from:
    https://github.com/PhyScene/PhyScene/blob/main/scripts/eval/walkable_metric.py
    """

    def __init__(self,
                 scene: Scene,
                 output_dir: pathlib.Path,
                 cfg: NavigabilityMetricConfig,
                 **kwargs) -> None:
        """
        Initialize the metric.

        Args:
            scene: the scene object
            cfg: the navigability metric configuration
            output_dir: the output directory for saving images
        """

        self.scene = scene
        self.output_dir = output_dir / "navigability"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg

        # Prepare scene information
        self.half_image_resolution = self.cfg.image_resolution // 2
        
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
    
    def run(self, verbose: bool = False) -> MetricResult:
        """
        Run the metric.

        Args:
            verbose: whether to visualize during the run
        
        Returns:
            result: the result of running the metric
        """

        # Create empty image
        image = np.zeros((self.cfg.image_resolution, self.cfg.image_resolution, 3), dtype=np.uint8)

        # Center the floor vertices at the origin
        floor_vertices = self.t_floor.vertices - self.t_floor_center
        floor_vertices = floor_vertices[:, :2] # Ignore z coordinate (height)

        # Draw floor
        for face in self.t_floor.faces:
            face_vertices = floor_vertices[face]
            face_vertices_image = [self._scene_to_image_coordinates(x, y, self.scale) for (x, y) in face_vertices]

            pts = np.array(face_vertices_image, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(image, [pts], self.cfg.floor_color)

        plt.title("Navigability - Floor Plan")
        plt.imshow(image[:, :, ::-1])
        plt.savefig(self.output_dir / "n0_floor_plan.png")
        if verbose:
            plt.show(block=True)
        plt.close()

        # Erode the floor with robot width to simulate robot size
        robot_width = int(self.cfg.robot_width / self.scale * self.half_image_resolution)
        kernel = np.ones((robot_width, robot_width))
        image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)
        
        plt.title("Navigability - Eroded Floor Plan")
        plt.imshow(image[:, :, ::-1])
        plt.savefig(self.output_dir / "n1_eroded_floor_plan.png")
        if verbose:
            plt.show(block=True)
        plt.close()

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
            cv2.drawContours(image, [box_points], 0, self.cfg.obj_color, robot_width)
            cv2.fillPoly(image, [box_points], self.cfg.obj_color)
        
        plt.title("Navigability - Eroded Floor Plan with Bounding Boxes")
        plt.imshow(image[:, :, ::-1])
        plt.savefig(self.output_dir / "n2_eroded_floor_plan_with_bboxes.png")
        if verbose:
            plt.show(block=True)
        plt.close()

        # Calculate object to floor plan area ratio
        if self.cfg.calculate_object_area:
            obj_pixel_count = 0
            floor_pixel_count = 0
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if tuple(image[i][j]) == self.cfg.obj_color:
                        obj_pixel_count += 1
                    elif tuple(image[i][j]) == self.cfg.floor_color:
                        floor_pixel_count += 1
            object_area_ratio = obj_pixel_count / (floor_pixel_count + obj_pixel_count)
        
        # Get floor-colored pixels as binary walkable map and do connected component analysis
        walkable_map = image[:, :, 0].copy()
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(walkable_map, connectivity=8)

        plt.title("Navigability - Walkable Map")
        plt.imshow(walkable_map)
        plt.savefig(self.output_dir / "n3_walkable_map.png")
        if verbose:
            plt.show(block=True)
        plt.close()

        # TODO: Support for doors

        # Calculate walkable rate
        walkable_map_max = np.zeros_like(walkable_map)
        if num_labels > 1:
            # Find the connected component with the largest area
            for label in range(1, num_labels):  # Skip background
                mask = np.zeros_like(walkable_map)
                mask[labels == label] = 255

                plt.title(f"Navigability - Connected Component {label}")
                plt.imshow(mask)
                plt.savefig(self.output_dir / f"n4_connected_component_{label}.png")
                if verbose:
                    plt.show(block=True)
                plt.close()

                if mask.sum() > walkable_map_max.sum():
                    walkable_map_max = mask.copy()
            
            # Calculate walkable rate from the largest connected component
            rate = walkable_map_max.sum() / walkable_map.sum()

            plt.title("Navigability - Largest Connected Component")
            plt.imshow(walkable_map_max)
            plt.savefig(self.output_dir / "n5_largest_connected_component.png")
            if verbose:
                plt.show(block=True)
            plt.close()
        else:
            rate = 0.

        result = MetricResult(
            message=f"Navigability is {rate:.2f} with {num_labels - 1} connected components.",
            data={
                "navigability": rate,
                "connected_components": num_labels - 1,
            }
        )
        if self.cfg.calculate_object_area:
            result.data["object_area_ratio"] = object_area_ratio

        print(f"\n{result.message}\n")
        
        return result
