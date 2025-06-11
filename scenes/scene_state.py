import time
import json
import pathlib
import warnings
from scenes.architecture import Architecture
from scenes.obj import Obj

class SceneState:
    def __init__(self, source: pathlib.Path | dict = None) -> None:
        """
        Initialize a scene state object.

        Args:
            source: the source of the scene state dictionary
        """

        self.raw_json: dict = None
        self.name: str = None
        self.version: str = None
        self.id: str = None
        self.up: list[float] = None
        self.front: list[float] = None
        self.unit: float = None
        self.assetSource: str = None
        self.architecture: Architecture = None
        self.objs: list[Obj] = None

        # Load the scene state dictionary if provided
        if source is not None:
            self.load(source)

    def load(self, source: pathlib.Path | dict) -> None:
        """
        Load a scene state dictionary from a file or a dictionary.

        Args:
            source: the source of the scene state dictionary
        """

        # Load the scene state dictionary
        if isinstance(source, pathlib.Path):
            with open(source, "r") as f:
                scene_state_dict = json.load(f)
            self.name = source.stem
        elif isinstance(source, dict):
            scene_state_dict = source
            self.name = f"scene_state@{time.strftime('%y%m%d-%H%M%S')}"
        
        self.raw_json = scene_state_dict
        
        # Verify it is a scene state dictionary
        if scene_state_dict.get("format") != "sceneState":
            raise ValueError("The format of the dictionary is not 'sceneState'.")
        
        # Load the scene specification
        scene_spec: dict = scene_state_dict.get("scene", None)
        if scene_spec is None:
            raise ValueError("The dictionary does not contain a 'scene' key.")
        
        # Check version
        self.version = scene_spec.get("version", None)
        if self.version != "scene@1.0.2":
            warnings.warn(f"This module is developed for scene version 1.0.2, but the scene version is {self.version}.")
        
        # Load scene properties
        self.id = scene_spec.get("id", None)
        self.up = scene_spec.get("up", [0, 0, 1])
        self.front = scene_spec.get("front", [0, 1, 0])
        self.unit = scene_spec.get("unit", 1.0)
        self.assetSource = scene_spec.get("assetSource", None)

        # Load the architecture
        arch_spec: dict = scene_spec.get("arch", None)
        if arch_spec is None:
            warnings.warn("Not loading any architecture. Missing an 'arch' key.")
        else:
            self.architecture = Architecture(arch_spec)
        
        # Load the objects
        object_specs: dict = scene_spec.get("object", None)
        if object_specs is None:
            warnings.warn("Not loading any objects. Missing an 'object' key.")
        else:
            self.objs = [ Obj(obj_dict) for obj_dict in object_specs ]
        