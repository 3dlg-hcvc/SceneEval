import warnings
import shapely
import trimesh
import trimesh.repair
import numpy as np
from dataclasses import dataclass
from .scene_state import SceneState
from .config import SceneConfig
from .obj import Obj
from .architecture import Architecture
from assets import Retriever

# NOTE: prefix t_ for Trimesh related objects

# ----------------------------------------------------------------------------------------

@dataclass
class TrimeshConfig:
    """
    Configuration for Trimesh scenes.
    
    Attributes:
        arch_thickness: the thickness of the architecture elements
    """
    
    arch_thickness: float = 0.001

# ----------------------------------------------------------------------------------------

class TrimeshScene:
    def __init__(self,
                 retriever: Retriever,
                 scene_state: SceneState | None,
                 scene_cfg: SceneConfig,
                 trimesh_cfg: TrimeshConfig) -> None:
        """
        Initialize a Trimesh scene.

        Args:
            retriever: the mesh retriever to use for getting object files
            scene_state: the scene state to load
            scene_cfg: the scene configuration
            trimesh_cfg: the Trimesh configuration
        """
        
        self.retriever = retriever
        self.scene_state = scene_state
        self.scene_cfg = scene_cfg
        self.trimesh_cfg = trimesh_cfg
        
        self.t_scene = trimesh.Scene()
        self.t_objs: dict[str, trimesh.Trimesh] = {}
        self.obj_descriptions: dict[str, str] = {}
        self.inverse_obj_descriptions: dict[str, str] = {}
        self.t_architecture: dict[str, trimesh.Trimesh] = {}

        # Load the scene state
        if self.scene_state is not None:
            self.load(scene_state)

    def load(self, scene_state: SceneState) -> None:
        """
        Load a scene state.

        Args:
            scene_state: the scene state to load
        """

        # Load the objects
        for obj in scene_state.objs:
            self.load_obj(obj, self.scene_cfg.skip_missing_obj)
        
        # Load the architecture
        self.load_architecture(scene_state.architecture, self.scene_cfg.use_simple_architecture)
    
    def load_obj(self, obj: Obj, skip_missing: bool = False) -> None:
        """
        Load an object into the scene.

        Args:
            obj: the object to load
            skip_missing: whether to skip missing object files peacefully
        """

        # Get the object information from the mesh retriever
        asset_info = self.retriever.get_asset_info(obj.model_id)
        file_path = asset_info.file_path
        obj_description = asset_info.description
        extra_rotation_transform = asset_info.extra_rotation_transform
        
        # Check if the object file exists
        if not file_path.is_file():
            if skip_missing:
                warnings.warn(f"Object file '{file_path}' is missing.")
                return
            else:
                raise FileNotFoundError(f"Object file '{file_path}' is missing.")
        
        # Load the object file as a Trimesh mesh
        mesh = trimesh.load(file_path, force="mesh")
        
        # Apply extra rotation transform from the asset itself to normalize the mesh
        if extra_rotation_transform is not None:
            mesh.apply_transform(extra_rotation_transform)
        
        # Apply the object's transform from the scene state (how the object is placed in the scene)
        scene_transform = np.asarray(obj.transform.data).reshape(4, 4).T
        mesh.apply_transform(scene_transform)
        
        # Fix the mesh if it is inverted for objaverse objects
        if "objaverse" in obj.model_id:
            trimesh.repair.fix_inversion(mesh)

        # Rename the object for identification and add it to the scene
        obj_name = f"idx{obj.index}_{obj.model_id}"
        self.t_objs[obj_name] = mesh

        # Adjust the object description to be explicit about duplicates
        same_obj_description_count = sum([1 for desc in self.obj_descriptions.values() if desc.startswith(obj_description)])
        if same_obj_description_count > 0:
            obj_description += f" - instance {same_obj_description_count + 1}"
        self.obj_descriptions[obj_name] = obj_description
        self.inverse_obj_descriptions[obj_description] = obj_name

        self.t_scene.add_geometry(mesh, geom_name=obj_name)
    
    def load_architecture(self, architecture: Architecture = None, use_simple: bool = False) -> None:
        """
        Load an architecture into the scene.

        Args:
            architecture: the architecture to load
            use_simple: whether to use a simple architecture instead of the provided architecture
        """

        arch_thickness = self.trimesh_cfg.arch_thickness
        
        # Get the bounds of all objects
        scene_bounds: np.ndarray | None = self.t_scene.bounds
        
        
        # If there are no objects in the scene, set the bounds to zeros
        if scene_bounds is None:
            warnings.warn("No valid bounds for the scene. Are there any objects in the scene?")
            scene_bounds = np.array([[0, 0, 0], [0, 0, 0]])

        # Get the floor height as the minimum z value of all objects
        floor_height = scene_bounds.min(axis=0)[2]
        
        if use_simple or architecture is None:
            # Use a simple architecture - a rectangular floor
            
            # Compute arachitecture parameters
            scene_center = scene_bounds.mean(axis=0)
            scene_size = scene_bounds.ptp(axis=0)

            # Create the floor plane and position it
            floor = trimesh.creation.box((scene_size[0], scene_size[1], arch_thickness))
            floor.apply_translation([scene_center[0], scene_center[1], floor_height - arch_thickness / 2])
            
            # Rename the floor for identification and add it to the scene
            floor_name = "floor"
            self.t_architecture[floor_name] = floor
            self.t_scene.add_geometry(floor, geom_name=floor_name)

            # Prepare wall locations
            center_x, center_y = scene_center[0], scene_center[1]
            half_length_x, half_length_y = scene_size[0] / 2, scene_size[1] / 2
            wall_locations_xy = [
                (center_x, center_y + half_length_y),
                (center_x - half_length_x, center_y),
                (center_x, center_y - half_length_y),
                (center_x + half_length_x, center_y)
            ]
            
            # Create the walls
            for i, location_xy in enumerate(wall_locations_xy):
                wall_size = (
                    scene_size[i % 2] if i % 2 == 0 else arch_thickness,
                    arch_thickness if i % 2 == 0 else scene_size[i % 2],
                    self.scene_cfg.simple_arch_wall_height
                )
                wall = trimesh.creation.box(wall_size)
                wall.apply_translation([
                    location_xy[0], 
                    location_xy[1], 
                    floor_height - arch_thickness / 2 + self.scene_cfg.simple_arch_wall_height / 2
                ])

                wall_name = f"wall_{i}"
                self.t_architecture[wall_name] = wall
                self.t_scene.add_geometry(wall, geom_name=wall_name)
            
        else:
            # Use the provided architecture
            for element in architecture.elements:
                
                # Check the element type
                match element.type:
                    case "Floor":
                        # Create a floor plane by first creating a polygon and then extruding it
                        floor_points_2D = np.asarray(element.points)[..., :2]
                        floor_polygon = shapely.Polygon(floor_points_2D)
                        floor = trimesh.creation.extrude_polygon(floor_polygon, arch_thickness)
                        
                        # Translate the floor to the correct height
                        floor.apply_translation([0, 0, floor_height - arch_thickness / 2])
                        
                        # Rename the floor for identification and add it to the scene
                        floor_name = f"floor_{element.roomId.replace(' ', '_')}"
                        self.t_architecture[floor_name] = floor
                        self.t_scene.add_geometry(floor, geom_name=floor_name)
                        
                    case "Ceiling":
                        pass
                    
                    case "Wall":
                        # Create a wall given two end points of the wall
                        wall_from_to_points = element.points
                        wall_size = (
                            np.linalg.norm(np.array(wall_from_to_points[0]) - np.array(wall_from_to_points[1])),
                            arch_thickness,
                            element.height
                        )
                        wall = trimesh.creation.box(wall_size)

                        # Make holes in the wall if any
                        holes = {}
                        if len(element.holes) > 0:
                            # Treat the wall as a 2D object
                            # The hole is defined in the 2D wall space with the origin at the minimum x and z values of the wall
                            wall_min_2D_point = [wall.bounds[0][0], wall.bounds[0][2]]
                            
                            # Make holes in the wall using boolean difference
                            for hole_idx, hole in enumerate(element.holes):
                                # Create a box for cutting the hole
                                hole_size = (
                                    hole.box_max[0] - hole.box_min[0],
                                    0.25,
                                    hole.box_max[1] - hole.box_min[1]
                                )
                                hole_box = trimesh.creation.box(hole_size)
                                
                                # Translate the hole box to the center of the hole
                                hole_center = wall_min_2D_point + np.mean([hole.box_min, hole.box_max], axis=0)
                                hole_box.apply_translation([hole_center[0], 0, hole_center[1]])
                                
                                # Make the hole in the wall
                                try:
                                    # wall = trimesh.boolean.difference([wall, hole_box], engine="scad") # For trimesh 3.x
                                    wall = trimesh.boolean.difference([wall, hole_box], engine="manifold") # For trimesh 4.x
                                except Exception as e:
                                    warnings.warn(f"Failed to create hole in wall: {e}") # Usually due to invalid hole dimensions
                                    continue
                                
                                # --------------------------------------------------------------
                                # Additionally make the hole as a separate object
                                hole_type = hole.type.lower()
                                hole_name = f"{hole_type}_{hole_idx}_{element.id}"
                                
                                # Create a plane for the hole
                                hole_plane_size = (hole_size[0], arch_thickness, hole_size[2])
                                hole_plane = trimesh.creation.box(hole_plane_size)
                                
                                # Translate the hole plane to the center of the hole``
                                hole_plane.apply_translation([hole_center[0], 0, hole_center[1]])
                                
                                # Assign colors to the hole based on the hole type
                                match hole_type:
                                    case "door":
                                        hole_plane.visual.face_colors = (0.31, 0.76, 0.40, 1)
                                    case "window":
                                        hole_plane.visual.face_colors = (0.37, 0.76, 0.90, 1)
                                
                                # Store the hole of the wall
                                holes[hole_name] = hole_plane
                        
                        # Rotate the wall, rotation angle is based on the order of two points
                        if np.isclose(wall_from_to_points[1][0], wall_from_to_points[0][0]):
                            # Wall aligned with y-axis
                            if wall_from_to_points[1][1] > wall_from_to_points[0][1]:
                                rotation_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
                            else:
                                rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 0, 1])
                            wall.apply_transform(rotation_matrix)
                            for hole in holes.values():
                                hole.apply_transform(rotation_matrix)
                            
                        else:
                            # Wall aligned with x-axis
                            if wall_from_to_points[1][0] < wall_from_to_points[0][0]:
                                rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
                                wall.apply_transform(rotation_matrix)
                                for hole in holes.values():
                                    hole.apply_transform(rotation_matrix)
                        
                        # Translate the wall to the center of the two points and adjust the height
                        translation = np.mean([wall_from_to_points[0], wall_from_to_points[1]], axis=0)
                        translation += [0, 0, floor_height - arch_thickness / 2 + element.height / 2]
                        wall.apply_translation(translation)
                        for hole in holes.values():
                            hole.apply_translation(translation)
                        
                        # Rename the wall for identification and add it to the scene
                        wall_name = f"wall_{element.id}"
                        self.t_architecture[wall_name] = wall
                        self.t_scene.add_geometry(wall, geom_name=wall_name)
                        
                        # Add the holes to the scene
                        for hole_name, hole in holes.items():
                            self.t_architecture[hole_name] = hole
                            self.t_scene.add_geometry(hole, geom_name=hole_name)
                                                
    def show(self) -> None:
        """
        Show the scene.
        """

        self.t_scene.show()
