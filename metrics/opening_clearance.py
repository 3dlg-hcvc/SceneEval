import cv2
import pathlib
import trimesh
import numpy as np
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
from scenes import Scene
from spatial import BoundingBox, BoundingBoxConfig
from .base import BaseMetric, MetricResult
from .registry import register_non_vlm_metric

# ----------------------------------------------------------------------------------------

@dataclass
class OpeningClearanceMetricConfig:
    """
    Configuration for the opening clearance metric.

    Attributes:
        bounding_box: configuration for bounding boxes used in the metric
        within_room_point_offset: the offset distance from the opening for placing test points to check which side is within the room
        map_pixel_per_meter: the pixel density (pixels per meter) of the 2D occupancy map
        door_check: whether to check doors for clearance
        door_room_side_only: whether to check only the side of the door that faces into the room
        door_use_width_as_extrude: whether to use the door width as the extrude distance for the clearance box
        door_extrude_distance: the distance in front of doors to extrude for the clearance box (if not using door width)
        window_check: whether to check windows for clearance
        window_room_side_only: whether to check only the side of the window that faces into the room
        window_front_extrude_distance: the distance in front of windows to extrude for the clearance box
    """

    bounding_box: BoundingBoxConfig = field(default_factory=lambda: BoundingBoxConfig())
    within_room_point_offset: float = 0.1
    map_pixel_per_meter: int = 100
    door_check: bool = True
    door_room_side_only: bool = True
    door_use_width_as_extrude: bool = True
    door_extrude_distance: float = 1.0
    window_check: bool = True
    window_room_side_only: bool = True
    window_front_extrude_distance: float = 0.5

# ----------------------------------------------------------------------------------------

@register_non_vlm_metric(config_class=OpeningClearanceMetricConfig)
class OpeningClearanceMetric(BaseMetric):
    """
    Metric to evaluate whether openings (doors and windows) are clear of obstructions.
    """

    def __init__(self, scene: Scene, output_dir: pathlib.Path, cfg: OpeningClearanceMetricConfig, **kwargs) -> None:
        """
        Initialize the metric.

        Args:
            scene: the scene to evaluate
            output_dir: the output directory for saving results
            cfg: the configuration for the metric
        """

        self.scene = scene
        self.output_dir = output_dir / "opening_clearance"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        
        self.gravity_direction = np.array([0, 0, -1])
        self.front_vector = np.array([0, -1, 0])
            
    def _get_opening_directions(self, arch_id: str, t_floor: trimesh.Trimesh, into_room_only: bool = True) -> np.ndarray:
        """
        Get the normal vectors of an opening (door or window).

        Args:
            arch_id: the architecture element ID
            t_floor: the floor mesh defining the room
            into_room_only: whether to return only directions that point into the room

        Returns:
            direction_vectors: a 2D array with vectors as rows
        """
        
        # Get front and back directions
        arch_matrix = self.scene.get_arch_matrix(arch_id).to_3x3()
        front_direction = np.asarray(arch_matrix) @ self.front_vector
        back_direction = -front_direction
        front_back_directions = np.stack([front_direction, back_direction])
        
        # If not testing within room only, return both directions
        if not into_room_only:
            return front_back_directions
        
        # Test which direction(s) is within the room
        arch_bbox_center = self.scene.get_arch_bbox_center(arch_id)
        ray_origins = arch_bbox_center + front_back_directions * self.cfg.within_room_point_offset
        ray_directions = np.tile(self.gravity_direction, (ray_origins.shape[0], 1))
        _, ray_indices, _ = t_floor.ray.intersects_location(ray_origins, ray_directions, multiple_hits=False)
        
        # Only keep directions that point into the room
        into_room_directions = np.atleast_2d(front_back_directions[ray_indices])

        return into_room_directions

    def _get_opening_extrude_distance(self, opening_id: str, with_thickness: bool) -> float:
        """
        Get the extrude distance in front of an opening (door or window) for the clearance box.

        Args:
            opening_id: the architecture element ID
            with_thickness: whether to include half the opening thickness in the extrude distance

        Returns:
            extrude_distance: the extrude distance in front of the opening
        """

        # Get opening type
        opening_type = "door" if opening_id.startswith("door") else "window" if opening_id.startswith("window") else None
        if opening_type not in ["door", "window"]:
            raise ValueError(f"Unsupported opening type: {opening_type}")

        # Determine extrude distance based on opening type
        match opening_type:
            case "door":
                if self.cfg.door_use_width_as_extrude:
                    extrude_distance = self.scene.get_default_pose_arch_bbox_extents(opening_id)[0]
                else:
                    extrude_distance = self.cfg.door_extrude_distance
            case "window":
                extrude_distance = self.cfg.window_front_extrude_distance

        if with_thickness:
            half_opening_thickness = self.scene.get_default_pose_arch_bbox_extents(opening_id)[1] / 2
            extrude_distance += half_opening_thickness

        return extrude_distance
    
    def _get_objs_interfering_with_opening_one_side(self,
                                                    opening_id: str,
                                                    side_direction: np.ndarray,
                                                    obj_bboxes: dict[str, BoundingBox]) -> list[str]:
        """
        Get the list of object IDs that interfere with one side of an opening (door or window).

        Args:
            opening_id: the architecture element ID
            side_direction: which side of the opening to consider
            obj_bboxes: a dictionary mapping object IDs to their bounding boxes
        
        Returns:
            interfering_obj_ids: a list of object IDs that interfere with the specified side of the opening
        """
        
        # Get opening information
        opening_bbox_center = self.scene.get_arch_bbox_center(opening_id)
        opening_width, opening_depth, opening_height = self.scene.get_default_pose_arch_bbox_extents(opening_id)
        opening_coord_axes = self.scene.get_arch_matrix(opening_id).to_3x3()

        # Get extrude distance for the clearance box, including half the opening thickness as the origin of the opening is at its 3D center
        extrude_distance = self._get_opening_extrude_distance(opening_id, with_thickness=True)
                
        # Create the clearance bounding box
        clearance_bbox_center = opening_bbox_center + side_direction * (extrude_distance / 2)
        clearance_bbox_half_size = np.array([opening_width / 2, extrude_distance / 2, opening_height / 2])
        clearance_bbox = BoundingBox(clearance_bbox_center, clearance_bbox_half_size, opening_coord_axes, cfg=self.cfg.bounding_box)
        
        # Check which objects interfere with the clearance box
        interfering_obj_ids = []
        for obj_id, obj_bbox in obj_bboxes.items():
            bbox_points = obj_bbox.sample_points()
            contains = clearance_bbox.contains(bbox_points)
            if np.any(contains):
                interfering_obj_ids.append(obj_id)
            
                # Debug visualization
                # t_box = trimesh.creation.box(clearance_bbox.half_size * 2, transform=clearance_bbox.no_scale_matrix)
                # t_box.visual.face_colors = [255, 0, 0, 100]
                # pc_colors = np.array([[0, 255, 0, 100] if c else [0, 0, 255, 100] for c in contains])
                # pc = trimesh.points.PointCloud(bbox_points, colors=pc_colors)
                # scene = self.scene.trimesh_scene.t_scene.copy()
                # scene.add_geometry(t_box)
                # scene.add_geometry(pc)
                # scene.show()
        
        return interfering_obj_ids

    def _get_world_to_opening_matrix(self, opening_id: str) -> np.ndarray:
        """
        Get the transformation matrix from world coordinates to the local frame of an opening (door or window).

        Args:
            opening_id: the architecture element ID
        
        Returns:
            world_to_opening_matrix: the transformation matrix from world to opening local frame
        """
        
        opening_transform = np.asarray(self.scene.get_arch_matrix(opening_id))
        world_to_opening_matrix = np.linalg.inv(opening_transform)
        
        return world_to_opening_matrix

    def _get_vertices_in_frame(self, vertices: np.ndarray, world_to_opening_matrix: np.ndarray) -> np.ndarray:
        """
        Transform vertices in world coordinates to the local frame (of an opening).

        Args:
            vertices: a 2D array with (x, y, z) coordinates in the world frame as rows
            world_to_opening_matrix: the transformation matrix from world to opening local frame
        
        Returns:
            vertices_local: a 2D array with the vertices in the local frame as rows
        """
        
        vertices_homogeneous = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
        vertices_local = ((world_to_opening_matrix @ vertices_homogeneous.T).T)[:, :3]  # Drop the homogeneous coordinate

        return vertices_local

    def _scene_to_image_coordinates(self, uw_scene_points: np.ndarray, opening_uvw_min: np.ndarray, opening_uvw_max: np.ndarray) -> np.ndarray:
        """
        Convert 3D scene points to 2D image coordinates.

        Args:
            uw_scene_points: a 2D array with (u, w) coordinates in the scene as rows in the opening frame
            opening_uvw_min: the minimum (u, v, w) coordinates of the opening in its local frame
            opening_uvw_max: the maximum (u, v, w) coordinates of the opening in its local frame

        Returns:
            image_points: a 2D array with 2D image points as rows
        """

        image_x = (uw_scene_points[:, 0] - opening_uvw_min[0]) * self.cfg.map_pixel_per_meter
        image_y = (opening_uvw_max[2] - uw_scene_points[:, 1]) * self.cfg.map_pixel_per_meter  # Reverse y for image coordinates
        image_points = np.stack([image_x, image_y], axis=1).astype(np.int32)
        
        return image_points

    def _project_interfering_objs_to_opening_2d(self,
                                                opening_id: str,
                                                interfering_obj_ids: list[str],
                                                side_direction: np.ndarray,
                                                figure_save_dir: pathlib.Path,
                                                verbose: bool = False) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Project interfering objects onto the 2D plane of an opening (door or window).
        
        Args:
            opening_id: the architecture element ID
            interfering_obj_ids: a list of object IDs that interfere with the opening
            side_normal: the normal vector of the side of the opening being considered
            figure_save_dir: the directory to save figures
            verbose: whether to visualize during the run
        
        Returns:
            opening_map: the 2D occupancy map of the opening
            occupancy_maps: a dictionary mapping object IDs to their 2D occupancy maps on the opening
        """
        
        # Get the transformation matrix from world to opening local frame
        world_to_opening_matrix = self._get_world_to_opening_matrix(opening_id)
        
        # Denote u, v, w, where u is width, v is depth, and w is height in the opening's local frame
        
        # Get opening dimensions in its local frame
        opening_vertices = np.asarray(self.scene.t_architecture[opening_id].vertices)
        opening_vertices_uvw = self._get_vertices_in_frame(opening_vertices, world_to_opening_matrix)
        opening_uvw_min = np.min(opening_vertices_uvw, axis=0)
        opening_uvw_max = np.max(opening_vertices_uvw, axis=0)

        # Image dimensions are based on opening dimensions
        image_width = int(np.ceil((opening_uvw_max[0] - opening_uvw_min[0]) * self.cfg.map_pixel_per_meter))
        image_height = int(np.ceil((opening_uvw_max[2] - opening_uvw_min[2]) * self.cfg.map_pixel_per_meter))
        opening_map = np.zeros((image_height, image_width), dtype=np.uint8)

        plt.title(f"Opening Map")
        plt.imshow(opening_map)
        plt.savefig(figure_save_dir / f"m1_opening_map.png")
        if verbose:
            plt.show(block=True)
        plt.close()
        
        # Get the side of the opening that we are considering
        side_direction_uvw = world_to_opening_matrix[:3, :3] @ side_direction
        extrude_distance = self._get_opening_extrude_distance(opening_id, with_thickness=True)

        # Project each interfering object to the 2D plane of the opening
        occupancy_maps = {}
        for obj_id in interfering_obj_ids:
            
            # Get the object's vertices in the opening frame
            obj_vertices = np.asarray(self.scene.t_objs[obj_id].vertices)
            obj_vertices_uvw = self._get_vertices_in_frame(obj_vertices, world_to_opening_matrix)

            # Only consider vertices within the extrusion distance in front of the opening
            obj_faces = self.scene.t_objs[obj_id].faces
            obj_triangles_uvw = obj_vertices_uvw[obj_faces]
            obj_triangles_min_v = np.min(obj_triangles_uvw[:, :, 1], axis=1)
            obj_triangles_max_v = np.max(obj_triangles_uvw[:, :, 1], axis=1)
            
            # Depend on which side of the opening we are considering, check if triangles are within the extrusion distance
            if side_direction_uvw[1] < 0: # Front side (negative v) in local frame
                interfering_obj_triangle_mask = (obj_triangles_min_v <= 0) & (obj_triangles_max_v >= -extrude_distance)
            else: # Back side (positive v) in local frame
                interfering_obj_triangle_mask = (obj_triangles_min_v <= extrude_distance) & (obj_triangles_max_v >= 0)

            # Keep only faces that are within the extrusion distance
            interfering_faces = obj_faces[interfering_obj_triangle_mask]

            # If there are any interfering faces, project them onto the opening plane as an occupancy map
            if len(interfering_faces) > 0:
                occupancy_map = np.zeros_like(opening_map)
                
                # Get the (u, w) coordinates of the interfering triangles's 2D projection
                vertices_uw = obj_vertices_uvw[:, [0, 2]]
                interfering_triangles_uw = vertices_uw[interfering_faces.reshape(-1)].reshape(-1, 3, 2) # (num_triangles, 3, 2) 3 vertices for each 2D triangle, each with (u, w)

                # Draw triangles on the occupancy map
                for triangle_uw in interfering_triangles_uw:
                    tri_img = self._scene_to_image_coordinates(triangle_uw, opening_uvw_min, opening_uvw_max)
                    cv2.fillConvexPoly(occupancy_map, tri_img, 255)
                
                # Store the occupancy map for the object
                occupancy_maps[obj_id] = occupancy_map
                
                plt.title(f"Object {obj_id} Projection")
                plt.imshow(occupancy_map)
                plt.savefig(figure_save_dir / f"m2_obj_{obj_id.split('_')[0]}_occupancy_map.png")
                if verbose:
                    plt.show(block=True)
                plt.close()
                
                # Debug visualization
                # plt.title(f"Object {obj_id} projection on opening {opening_id}")
                # plt.imshow(occupancy_map)
                # plt.show(block=True)
                # plt.close()

            # Debug visualization
            # scene = self.scene.trimesh_scene.t_scene.copy()
            # scene.apply_transform(world_to_opening_matrix)
            # pc = trimesh.points.PointCloud(obj_vertices_uvw, colors=[0, 255, 0, 100])
            # scene.add_geometry(pc)
            # scene.show()

        return opening_map, occupancy_maps

    def _summarize_occlusion_info(self,
                                  occupancy_maps: dict[str, np.ndarray],
                                  figure_save_dir: pathlib.Path,
                                  verbose: bool = False) -> dict[str, float]:
        """
        Summarize occlusion information from occupancy maps.

        Args:
            opening_id: the architecture element ID
            occupancy_maps: a dictionary mapping object IDs to their 2D occupancy maps on the opening
            figure_save_dir: the directory to save figures
            verbose: whether to visualize during the run
            
        Returns:
            occlusion_info: a dictionary mapping object IDs to their occlusion percentages
        """
                
        occlusion_info = {
            "opening_occluded": len(occupancy_maps) > 0,
            "total_occlusion": 0.0,
            "object_occlusions": {}
        }
        
        if len(occupancy_maps) == 0:
            return occlusion_info
        
        opening_size = list(occupancy_maps.values())[0].size
        combined_occupancy_map = np.zeros_like(next(iter(occupancy_maps.values())))
        
        # Calculate per object occlusion percentages
        for obj_id, occupancy_map in occupancy_maps.items():
            occluded_area = np.count_nonzero(occupancy_map)
            occlusion_percentage = occluded_area / opening_size
            occlusion_info["object_occlusions"][obj_id] = occlusion_percentage

            # Combine occupancy maps
            combined_occupancy_map |= occupancy_map

        plt.title("Combined Occupancy Map")
        plt.imshow(combined_occupancy_map)
        plt.savefig(figure_save_dir / f"m3_combined_occupancy_map.png")
        if verbose:
            plt.show(block=True)
        plt.close()

        # Calculate total occlusion percentage
        total_occluded_area = np.count_nonzero(combined_occupancy_map)
        total_occlusion_percentage = total_occluded_area / opening_size

        occlusion_info["total_occlusion"] = total_occlusion_percentage
        return occlusion_info        

    def _process_openings(self,
                          opening_type: str,
                          opening_ids: list[str],
                          t_floor: trimesh.Trimesh,
                          room_side_only: bool,
                          obj_bboxes: dict[str, BoundingBox],
                          result: MetricResult,
                          verbose: bool = False) -> None:
        """
        Process a group of openings.
        
        Args:
            opening_type: the type of opening ("door" or "window")
            opening_ids: list of opening IDs to process
            t_floor: the floor mesh for inside-room checks
            room_side_only: whether to check only the room side
            obj_bboxes: dictionary mapping object IDs to their bounding boxes
            result: the metric result to store data in
            verbose: whether to visualize during the run
        """
        print(f"Checking {opening_type} clearance...")
        
        for opening_id in opening_ids:
            pathlib.Path.mkdir(self.output_dir / f"{opening_id}", exist_ok=True)
            
            # Get directions to check for the opening
            directions_to_check = self._get_opening_directions(opening_id, t_floor, into_room_only=room_side_only)
            
            # Check each direction
            for direction in directions_to_check:
                
                # Set up figure save directory
                rounded_dir_str = direction.round(4).tolist()
                print(f"\n--- Checking {opening_type} '{opening_id}', side: {rounded_dir_str}...")
                fig_save_dir = self.output_dir / f"{opening_id}/side_{rounded_dir_str}"
                pathlib.Path.mkdir(fig_save_dir, exist_ok=True)

                # Find objects in the interference region
                interfering_obj_ids = self._get_objs_interfering_with_opening_one_side(opening_id, direction, obj_bboxes)
                print_str = '\n'.join(interfering_obj_ids) if len(interfering_obj_ids) > 0 else " None"
                print(f"> Found {len(interfering_obj_ids)} interfering objects:{print_str}")

                # Project interfering objects to the 2D plane of the opening and summarize occlusion information
                _, occupancy_maps = self._project_interfering_objs_to_opening_2d(opening_id, interfering_obj_ids, direction, fig_save_dir, verbose=verbose)
                occlusion_info = self._summarize_occlusion_info(occupancy_maps, fig_save_dir, verbose=verbose)
                print(f"> Occcluded: {occlusion_info['opening_occluded']}, Total occlusion: {occlusion_info['total_occlusion']}")

                # Store the results
                clearance_key = f"{opening_type}_clearance"
                result.data[clearance_key].setdefault(f"{opening_id}", []).append({
                    "direction": direction.tolist(),
                    "interfering_obj_ids": interfering_obj_ids,
                    "occlusion_info": occlusion_info
                })

    def run(self, verbose: bool = False) -> MetricResult:
        """
        Run the metric.

        Args:
            verbose: whether to visualize during the run
        
        Returns:
            result: the result of running the metric
        """
        
        # Prepare the floor mesh for inside-room checks
        t_floors = [t_arch for arch_id, t_arch in self.scene.trimesh_scene.t_architecture.items() if arch_id.startswith("floor")]
        t_floor = trimesh.util.concatenate(t_floors)
        
        # Prepare bounding boxes for all objects
        obj_bboxes = {}
        for obj_id in self.scene.get_obj_ids():
            obj_bbox_center = self.scene.get_obj_bbox_center(obj_id)
            obj_bbox_half_size = self.scene.get_default_pose_obj_bbox_extents(obj_id) / 2
            obj_coord_axes = self.scene.get_obj_matrix(obj_id).to_3x3()
            obj_bboxes[obj_id] = BoundingBox(obj_bbox_center, obj_bbox_half_size, obj_coord_axes, cfg=self.cfg.bounding_box)
        
        result = MetricResult(
            message="",
            data={
                "check_doors": self.cfg.door_check,
                "door_room_side_only": self.cfg.door_room_side_only,
                "check_windows": self.cfg.window_check,
                "window_room_side_only": self.cfg.window_room_side_only,
                "door_clearance": {},
                "window_clearance": {}
            }
        )

        # Collect opening IDs based on configuration
        opening_types_to_process = []
        if self.cfg.door_check:
            door_ids = [door_id for door_id in self.scene.t_architecture.keys() if door_id.startswith("door")]
            opening_types_to_process.append(("door", door_ids, self.cfg.door_room_side_only))
        if self.cfg.window_check:
            window_ids = [window_id for window_id in self.scene.t_architecture.keys() if window_id.startswith("window")]
            opening_types_to_process.append(("window", window_ids, self.cfg.window_room_side_only))
        
        # Process each opening type
        for opening_type, opening_ids, room_side_only in opening_types_to_process:
            self._process_openings(opening_type, opening_ids, t_floor, room_side_only, obj_bboxes, result, verbose)
        
        # Generate summary messages
        for opening_type, _, room_side_only in opening_types_to_process:
            clearance_data = result.data[f"{opening_type}_clearance"]
            num_openings = len(clearance_data)
            num_sides_checked = sum([len(opening_info) for opening_info in clearance_data.values()])
            num_sides_occluded = sum([1 for opening_info in clearance_data.values() for side_info in opening_info if side_info["occlusion_info"]["opening_occluded"]])
            result.message += f"Num {opening_type}s: {num_openings}, only room side: {room_side_only}, {num_sides_checked} sides checked, {num_sides_occluded} sides occluded\n"

        print(f"\n{result.message}\n")

        return result
