import json
import trimesh
import numpy as np
from dataclasses import dataclass
from warnings import warn
from pydantic import BaseModel
from scenes import Scene
from vlm import BaseVLM
from .base import BaseMetric, MetricResult
from .registry import register_vlm_metric

# ----------------------------------------------------------------------------------------

@dataclass
class SupportMetricConfig:
    """
    Configuration for the support metric.

    Attributes:
        num_samples_per_square_meter: the number of samples per square meter
        min_num_samples: the minimum number of samples
        support_distance_threshold: the distance threshold to consider as support
        epsilon: the epsilon value to move the object slightly in reverse of the gravity direction
        normal_facing_threshold: threshold for normals facing gravity direction
        use_existing_support_type_assessment: whether to use existing support type assessment if available
    """

    num_samples_per_square_meter: int = 256
    min_num_samples: int = 32
    support_distance_threshold: float = 0.01
    epsilon: float = 0.005
    normal_facing_threshold: float = 0.9
    use_existing_support_type_assessment: bool = False

# ----------------------------------------------------------------------------------------

class SupportTypeAssessmentResponseFormat(BaseModel):
    support_type: str
    reason: str

# ----------------------------------------------------------------------------------------

@register_vlm_metric(config_class=SupportMetricConfig)
class SupportMetric(BaseMetric):
    """
    Metric to evaluate object support.
    """

    # TODO: Expose these parameters to config file
    def __init__(self,
                 scene: Scene,
                 vlm: BaseVLM,
                 cfg: SupportMetricConfig,
                 **kwargs) -> None:
        """
        Initialize the metric.

        Args:
            scene: the scene to evaluate
            vlm: the VLM to use for evaluation
            cfg: the configuration for the metric
        """

        self.scene = scene
        self.vlm = vlm
        self.cfg = cfg
        
        self.vlm.reset()
        
        self.num_samples_per_square_meter = self.cfg.num_samples_per_square_meter
        self.min_num_samples = self.cfg.min_num_samples
        self.support_distance_threshold = self.cfg.support_distance_threshold
        self.epsilon = self.cfg.epsilon

        self.obj_descriptions = list(self.scene.obj_descriptions.values())
        self.image_paths_per_obj_id: dict[str, list[str]] = {}
        
        # Get the front and surroundings images for each object
        for obj_id in self.scene.get_obj_ids():
            self.image_paths_per_obj_id[obj_id] = [
                self.scene.get_obj_render_path(obj_id, "FRONT"),
                self.scene.get_obj_render_path(obj_id, "SURROUNDINGS")
            ]
        
    def _sample_points_in_triangles(self, points_0: np.ndarray, points_1: np.ndarray, points_2: np.ndarray, num_samples: int) -> np.ndarray:
        """
        Sample points in the triangles defined by the input points.

        Args:
            points_0: the first point of the triangles (N, 3)
            points_1: the second point of the triangles (N, 3)
            points_2: the third point of the triangles (N, 3)
            num_samples: the number of points to sample
        
        Returns:
            points: the sampled points in the triangles (N, num_samples, 3)
        """

        # Ensure the input points have the same shape
        assert points_0.shape == points_1.shape == points_2.shape

        # Compute the vectors representing the edges of the triangle
        v1 = points_1 - points_0
        v2 = points_2 - points_0

        # Sample points in [0, 1)
        u = np.random.rand(points_0.shape[0], num_samples)
        v = np.random.rand(points_0.shape[0], num_samples)

        # If the sum of u and v is greater than 1, then reflect the points back into the triangle
        outside_triangle_mask = u + v > 1
        u[outside_triangle_mask] = 1 - u[outside_triangle_mask]
        v[outside_triangle_mask] = 1 - v[outside_triangle_mask]

        # Compute the final point coordinates
        points = points_0[:, None, :] + u[:, :, None] * v1[:, None, :] + v[:, :, None] * v2[:, None, :]
        
        return points

    def run(self, verbose: bool = False) -> MetricResult:
        """
        Run the metric.

        Args:
            verbose: whether to visualize during the run
        
        Returns:
            result: the result of running the metric
        """

        evaluations = {}
        
        # ==========================================================================================
        # Determine the expected support type for each object
        
        obj_support_types: dict[str, str] = {}
        support_result_file = self.scene.output_dir / "obj_support_type_result.json"
        
        if support_result_file.exists() and self.cfg.use_existing_support_type_assessment:
            print(f"Using existing support type results...")
            with open(support_result_file, "r") as f:
                obj_support_types = json.load(f)
            print(f"Existing support type results loaded.\n")
        else:
            print(f"Creating new support type assessment...")
        
            for i, obj_id in enumerate(self.scene.get_obj_ids()):
                
                self.vlm.reset() # GPT can gives 500 error if not reset
                
                print(f"[{i+1}/{len(self.scene.get_obj_ids())}] Assessing support type for object {obj_id}...")
                
                response: SupportTypeAssessmentResponseFormat | str = self.vlm.send("obj_support_type",
                                                                                    image_paths=self.image_paths_per_obj_id[obj_id],
                                                                                    response_format=SupportTypeAssessmentResponseFormat)
            
                if type(response) is str:
                    warn("The response is not in the expected format.", RuntimeWarning)
                    obj_support_types[obj_id] = "ground" # Default to ground
                else:
                    obj_support_types[obj_id] = response.support_type
            
            with open(support_result_file, "w") as f:
                json.dump(obj_support_types, f, indent=4)
                
            print(f"New support type assessment saved to {support_result_file}\n")
        
        print("Using support type assessments:")
        for obj_id, support_type in obj_support_types.items():
            print(f" - {obj_id}: {support_type}")
        print()
        
        # ==========================================================================================
        # Create a ceiling object from the floor objects by translating it up
        t_floors = [t_arch for arch_id, t_arch in self.scene.t_architecture.items() if arch_id.startswith("floor")]
        t_floor = trimesh.util.concatenate(t_floors)
        t_ceiling = t_floor.copy()
        t_walls = [t_arch for arch_id, t_arch in self.scene.t_architecture.items() if arch_id.startswith("wall")]
        wall_height = np.max([t_wall.bounding_box.extents[2] for t_wall in t_walls])
        t_ceiling.apply_translation([0, 0, wall_height])
        
        # ==========================================================================================
        # Iterate over all objects in the scene
        for i, obj_id in enumerate(self.scene.get_obj_ids()):
            
            support_type = obj_support_types[obj_id]
            
            # Determine the gravity direction based on the support type
            match support_type:
                case "ground" | "object":
                    gravity_direction = np.array([0, 0, -1])
                case "ceiling":
                    gravity_direction = np.array([0, 0, 1])
                case "wall":
                    FRONT_VECTOR = np.array([0, -1, 0])
                    obj_matrix = self.scene.get_obj_matrix(obj_id)
                    obj_front_vector = np.asarray(obj_matrix)[:3, :3] @ FRONT_VECTOR
                    gravity_direction = -obj_front_vector # For wall objects, the gravity direction is the back of the object
            
            evaluations[obj_id] = {
                "support_type": support_type,
                "gravity_direction": gravity_direction.tolist(),
                "num_ray_origins": -1,
                "num_total_hit_points": -1,
                "num_valid_contact_points": -1,
                "min_distance": -1,
                "enough_hit_points_for_convex_hull": None,
                "supported": False,
            }

            # Move the object slightly in reverse of the gravity direction
            # Help avoid the corner case where no hit points are found due to the objects being too close to each other
            t_obj = self.scene.t_objs[obj_id].copy()
            t_obj.apply_translation(-self.cfg.epsilon * gravity_direction)

            # ==========================================================================================
            print(f"\nChecking support for object {obj_id}...")
            print(f"Support type: {support_type}, using gravity direction: {gravity_direction}")
            
            # Find locations to ray cast from
            # 1 - Get vertices close to the gravity side
            vertices = t_obj.vertices
            
            gravity_axis = np.argmax(np.abs(gravity_direction)) # NOTE: Assumes gravity direction is along one of the axes (0, 1, or 2 for x, y, z)
            if gravity_direction[gravity_axis] < 0:
                # If gravity direction is negative, use the minimum vertex coordinate as that is closest to the gravity side
                vertex_coordinate_in_gravity_direction = np.min(vertices[:, gravity_axis])
            else:
                # Otherwise, use the maximum vertex coordinate as that is closest to the gravity side
                vertex_coordinate_in_gravity_direction = np.max(vertices[:, gravity_axis])
            
            # Look at one axis coordinate for each vertex and pick those that are close to the minimum vertex coordinate in the gravity direction
            close_to_min_vertex_mask = np.isclose(vertices[:, gravity_axis], vertex_coordinate_in_gravity_direction, atol=self.cfg.support_distance_threshold)
            ray_origins = vertices[close_to_min_vertex_mask]
            
            # 2 - Sample points in the triangles facing the gravity direction
            faces = t_obj.faces
            
            # Only consider faces whose vertices are all close to the minimum vertex coordinate in the gravity direction
            face_vertices_close_mask = close_to_min_vertex_mask[faces]
            all_vertices_close_mask = np.all(face_vertices_close_mask, axis=1)  # Shape: (num_faces,)
            
            # Only consider faces whose normals are facing the gravity direction (i.e., the dot product with the gravity direction is close to 1)
            facing_gravity_direction_mask = abs(np.dot(t_obj.face_normals[all_vertices_close_mask], gravity_direction)) > self.cfg.normal_facing_threshold # Abs to handle inverted normals
            gravity_facing_faces = faces[all_vertices_close_mask][facing_gravity_direction_mask]
            
            # Sample points in the triangles of the gravity-facing faces if any exist
            if len(gravity_facing_faces) > 0:
                face_vertices = t_obj.vertices[gravity_facing_faces]
                max_face_area = np.max(trimesh.triangles.area(face_vertices))
                num_samples_per_face = max(self.cfg.min_num_samples, int(max_face_area * self.cfg.num_samples_per_square_meter))
                face_sampled_points = self._sample_points_in_triangles(face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2], num_samples_per_face)
                ray_origins = np.concatenate([ray_origins, face_sampled_points.reshape(-1, 3)], axis=0)
            
            evaluations[obj_id]["num_ray_origins"] = len(ray_origins)
            print(f"Number of ray origins: {len(ray_origins)}")

            # ==========================================================================================
            # Collect meshes to ray cast against
            other_meshes = [t_obj_mesh for t_obj_id_, t_obj_mesh in self.scene.t_objs.items() if t_obj_id_ != obj_id]
            other_meshes.extend(list(self.scene.t_architecture.values()))
            other_meshes.append(t_ceiling)
            
            # Ray cast from the vertices to the gravity direction
            all_hit_pts = []
            valid_contact_pts = []
            all_distances = []
            for target in other_meshes:
                
                # Do ray casting
                ray_origins = ray_origins.reshape(-1, 3)
                ray_directions = np.tile(gravity_direction, (ray_origins.shape[0], 1))
                ray_hit_pts, ray_hit_idxs, _ = target.ray.intersects_location(ray_origins, ray_directions, multiple_hits=False)
                all_hit_pts.extend(ray_hit_pts)
                
                # Skip this target object if no ray hits
                if len(ray_hit_pts) == 0:
                    continue
                
                # Filter out hit points that are within the distance threshold
                distances = np.linalg.norm(ray_hit_pts - ray_origins[ray_hit_idxs], axis=1)
                ray_hit_pts = ray_hit_pts[distances < self.cfg.support_distance_threshold]
                all_distances.extend(distances)
                valid_contact_pts.extend(ray_hit_pts)
            
            valid_contact_pts = np.asarray(valid_contact_pts)

            # Store statistics
            evaluations[obj_id]["num_total_hit_points"] = len(all_hit_pts)
            evaluations[obj_id]["num_valid_contact_points"] = len(valid_contact_pts)
            evaluations[obj_id]["min_distance"] = np.min(all_distances) if len(all_distances) > 0 else -1
            print(f"Total number of hit points: {len(all_hit_pts)}, valid hit points within threshold: {len(valid_contact_pts)}, min distance: {evaluations[obj_id]['min_distance']}")

            # ==========================================================================================
            # Determine if the object is supported, based on the support type
            if support_type in ["wall", "ceiling"]:
                # For wall and ceiling objects, the object is supported if there are any valid contact points
                
                evaluations[obj_id]["supported"] = (len(valid_contact_pts) > 0)
                print((
                    f"Type: {support_type.capitalize()} - "
                    "Valid " if evaluations[obj_id]["supported"] else "No valid "
                    "contact points found -> "
                    "Supported - O" if evaluations[obj_id]["supported"] else "Not supported - X"
                ))
            
            else:
                # For ground and object objects, the object is supported if the centroid projection is in the convex hull of the contact points
                
                # First check if there are enough hit points to create a convex hull
                enough_hit_points_for_convex_hull = (len(valid_contact_pts) >= 4) and (len(np.unique(valid_contact_pts, axis=0)) >= 3) 
                evaluations[obj_id]["enough_hit_points_for_convex_hull"] = enough_hit_points_for_convex_hull
                if not enough_hit_points_for_convex_hull:
                    evaluations[obj_id]["supported"] = False
                    print(f"{support_type.capitalize()} - Not enough valid contact points to build a convex hull -> Not supported - X")
                    continue
                
                # Create a convex hull from the valid contact points
                try:
                    contact_hull: trimesh.Trimesh = trimesh.convex.convex_hull(valid_contact_pts)
                except Exception as e:
                    evaluations[obj_id]["supported"] = False
                    print(f"{support_type.capitalize()} - Error creating convex hull: {e} -> Not supported - X")
                    continue
                centroid_projection_pt, _, _ = contact_hull.ray.intersects_location(t_obj.centroid[None, :], gravity_direction[None, :], multiple_hits=False)

                # Check if the centroid projection is in the convex hull
                evaluations[obj_id]["supported"] = (len(centroid_projection_pt) == 1)
                print((
                    f"{support_type.capitalize()} - "
                    "Centroid projection in contact point convex hull -> Supported - O" if evaluations[obj_id]["supported"] else
                    "Centroid projection not in contact point convex hull -> Not supported - X"
                ))

            # ==========================================================================================
            if verbose:
                
                visualize_scene = trimesh.Scene()
                visualize_scene.add_geometry(list([self.scene.t_architecture.values()]))
                visualize_scene.add_geometry(t_obj)
                
                if contact_hull:
                    contact_hull.visual.face_colors = [255, 0, 0, 255]
                    visualize_scene.add_geometry(contact_hull)
                
                if len(valid_contact_pts) > 0:
                    contact_points = trimesh.points.PointCloud(valid_contact_pts, colors=[0, 255, 0])
                    visualize_scene.add_geometry(contact_points)
                
                if len(centroid_projection_pt) > 0:
                    centroid_proj_point = trimesh.points.PointCloud(centroid_projection_pt, colors=[0, 0, 255])
                    visualize_scene.add_geometry(centroid_proj_point)
                
                visualize_scene.show()
        
        result = MetricResult(
            message=f"{sum([1 for s in evaluations.values() if s['supported']])}/{len(evaluations)} objects are supported",
            data=evaluations
        )
        
        print(f"\n{result.message}\n")
        
        return result
