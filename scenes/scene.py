import pathlib
import bpy
import trimesh
import numpy as np
from mathutils import Matrix
from assets import Retriever
from .scene_state import SceneState
from .config import SceneConfig
from .blender_scene import BlenderScene, BlenderConfig
from .trimesh_scene import TrimeshScene, TrimeshConfig

# ----------------------------------------------------------------------------------------

class Scene:
    def __init__(self,
                 mesh_retriever: Retriever,
                 scene_state: SceneState,
                 scene_cfg: SceneConfig,
                 blender_cfg: BlenderConfig,
                 trimesh_cfg: TrimeshConfig,
                 output_dir: pathlib.Path) -> None:
        """
        Initialize a Scene object that contains both Blender and Trimesh scenes.

        Args:
            mesh_retriever: the mesh retriever to use for getting object files
            scene_state: the scene state to load
            scene_cfg: the scene configuration
            blender_cfg: the Blender configuration
            trimesh_cfg: the Trimesh configuration
            output_dir: the output directory for this scene
        """

        self.output_dir = output_dir
        self.blender_scene = BlenderScene(mesh_retriever, scene_state, scene_cfg, blender_cfg, output_dir)
        self.trimesh_scene = TrimeshScene(mesh_retriever, scene_state, scene_cfg, trimesh_cfg)
    
    @property
    def obj_descriptions(self) -> dict[str, str]:
        """
        Returns:
            The descriptions of the objects in the scene
        """

        return self.blender_scene.obj_descriptions

    @property
    def inverse_obj_descriptions(self) -> dict[str, str]:
        """
        Returns:
            The inverse mapping of object descriptions to object IDs
        """

        return self.blender_scene.inverse_obj_descriptions
    
    @property
    def b_objs(self) -> dict[str, bpy.types.Object]:
        """
        Returns:
            The objects in the Blender scene
        """

        return self.blender_scene.b_objs
    
    @property
    def b_architecture(self) -> dict[str, bpy.types.Object]:
        """
        Returns:
            The architecture objects in the Blender scene
        """

        return self.blender_scene.b_architecture
    
    @property
    def t_objs(self) -> dict[str, trimesh.Trimesh]:
        """
        Returns:
            The objects in the Trimesh scene
        """

        return self.trimesh_scene.t_objs

    @property
    def t_architecture(self) -> dict[str, trimesh.Trimesh]:
        """
        Returns:
            The architecture objects in the Trimesh scene
        """

        return self.trimesh_scene.t_architecture
    
    def get_obj_ids(self) -> list[str]:
        """
        Get the object IDs in the scene.

        Returns:
            obj_ids: the object IDs in the scene
        """

        return list(self.blender_scene.b_objs.keys())
    
    def get_arch_ids(self) -> list[str]:
        """
        Get the architecture element IDs in the scene.

        Returns:
            arch_ids: the architecture element IDs in the scene
        """

        return list(self.blender_scene.b_architecture.keys())
    
    def get_obj_matrix(self, obj_id: str) -> Matrix:
        """
        Get the 4x4 transformation matrix of the object.

        Args:
            obj_id: the object ID

        Returns:
            matrix: The 4x4 transformation matrix of the object
        """

        return self.blender_scene.b_objs[obj_id].matrix_world
    
    def get_arch_matrix(self, arch_id: str) -> Matrix:
        """
        Get the 4x4 transformation matrix of the architecture element.

        Args:
            arch_id: the architecture element ID

        Returns:
            matrix: The 4x4 transformation matrix of the architecture element
        """

        return self.blender_scene.b_architecture[arch_id].matrix_world
    
    def get_obj_z_rotation(self, obj_id: str) -> float:
        """
        Get the object's rotation around the z-axis (up axis).

        Args:
            obj_id: the object ID

        Returns:
            z_rotation: the object's rotation around the z-axis in radians
        """

        b_obj = self.b_objs[obj_id]
        b_obj.rotation_mode = "XYZ"
        return b_obj.rotation_euler.z
    
    def get_default_pose_t_obj(self, obj_id: str) -> trimesh.Trimesh:
        """
        Get the object in its default pose.

        Args:
            obj_id: the object ID
        
        Returns:
            new_t_obj: a copy of the object in its default pose
        """

        new_t_obj = self.t_objs[obj_id].copy()
        obj_world_matrix = self.get_obj_matrix(obj_id)
        obj_to_default_pose_matrix = obj_world_matrix.inverted()
        new_t_obj.apply_transform(obj_to_default_pose_matrix)
        
        return new_t_obj
    
    def get_default_pose_t_arch(self, arch_id: str) -> trimesh.Trimesh:
        """
        Get the architecture element in its default pose.

        Args:
            arch_id: the architecture element ID
        """
        
        new_t_arch = self.t_architecture[arch_id].copy()
        arch_world_matrix = self.get_arch_matrix(arch_id)
        arch_to_default_pose_matrix = arch_world_matrix.inverted()
        new_t_arch.apply_transform(arch_to_default_pose_matrix)
        
        return new_t_arch
    
    def get_default_pose_obj_bbox_center(self, obj_id: str) -> np.ndarray:
        """
        Get the center of the object's bounding box in its default pose.

        Args:
            obj_id: the object ID
        
        Returns:
            center: the center of the object's bounding box in its default pose
        """

        t_obj_default_pose = self.get_default_pose_t_obj(obj_id)
        center = t_obj_default_pose.bounds[0] + t_obj_default_pose.extents / 2
        return center
    
    def get_default_pose_arch_bbox_center(self, arch_id: str) -> np.ndarray:
        """
        Get the center of the architecture element's bounding box in its default pose.

        Args:
            arch_id: the architecture element ID
        
        Returns:
            center: the center of the architecture element's bounding box in its default pose
        """
            
        t_arch_default_pose = self.get_default_pose_t_arch(arch_id)
        center = t_arch_default_pose.bounds[0] + t_arch_default_pose.extents / 2
        return center
    
    def get_obj_bbox_center(self, obj_id: str) -> np.ndarray:
        """
        Get the center of the object's bounding box.

        Args:
            obj_id: the object ID
        
        Returns:
            center: the center of the object's bounding box
        """

        default_pose_obj_center = self.get_default_pose_obj_bbox_center(obj_id)
        obj_matrix = self.get_obj_matrix(obj_id)
        center = (np.asarray(obj_matrix) @ np.append(default_pose_obj_center, 1))[:3]
        return center
    
    def get_arch_bbox_center(self, arch_id: str) -> np.ndarray:
        """
        Get the center of the architecture element's bounding box.

        Args:
            arch_id: the architecture element ID
        
        Returns:
            center: the center of the architecture element's bounding box
        """
            
        default_pose_arch_center = self.get_default_pose_arch_bbox_center(arch_id)
        arch_matrix = self.get_arch_matrix(arch_id)
        center = (np.asarray(arch_matrix) @ np.append(default_pose_arch_center, 1))[:3]
        return center

    def get_default_pose_obj_bbox_extents(self, obj_id: str) -> np.ndarray:
        """
        Get the extents of the object's bounding box in its default pose.

        Args:
            obj_id: the object ID
        
        Returns:
            extents: the extents of the object's bounding box in its default pose
        """
            
        t_obj_default_pose = self.get_default_pose_t_obj(obj_id)
        return t_obj_default_pose.extents
    
    def get_default_pose_arch_bbox_extents(self, arch_id: str) -> np.ndarray:
        """
        Get the extents of the architecture element's bounding box in its default pose.

        Args:
            arch_id: the architecture element ID
        
        Returns:
            extents: the extents of the architecture element's bounding box in its default pose
        """
        
        t_arch_default_pose = self.get_default_pose_t_arch(arch_id)
        return t_arch_default_pose.extents
    
    def get_obj_render_path(self, obj_id: str, view: str = "FRONT") -> pathlib.Path:
        """
        Get the path to the rendered image of the object.

        Args:
            obj_id: the object ID
            view: the view to render from (FRONT, SIZE_REFERENCE, SURROUNDINGS, TOP)
        
        Returns:
            render_path: the path to the rendered image of the object
        """
        
        assert view in ["FRONT", "SIZE_REFERENCE", "SURROUNDINGS", "TOP"], "View must be one of FRONT, SIZE_REF, SURROUNDINGS, TOP."
        obj_render_subdir = self.blender_scene.blender_cfg.object_render_subdir
        image_format = self.blender_scene.blender_cfg.render_file_format.lower()
        
        file_name = f"{obj_id}.{image_format}"
        
        match view:
            case "FRONT":
                pass
            case "SIZE_REFERENCE":
                file_name = "size_" + file_name
            case "SURROUNDINGS":
                file_name = "surroundings_" + file_name
            case "TOP":
                file_name = "top_" + file_name
        
        render_path = self.output_dir / obj_render_subdir / file_name
        
        return render_path
    
    def show(self) -> None:
        """
        Show the scene in Trimesh.
        """

        self.trimesh_scene.show()
        