class Obj:
    def __init__(self, obj_dict: dict = None) -> None:
        """
        Initialize an object object.

        Args:
            obj_dict: the object dictionary
        """

        self.model_id: str = None
        self.id: str = None
        self.parentId: str = None
        self.transform: _Transform = None
        self.index: int = None
        self.parentIndex: int = None
        self.holodeck_type: str = None
        self.holodeck_support_parent: int = None
    
        # Load the object dictionary if provided
        if obj_dict is not None:
            self.load(obj_dict)
    
    def load(self, obj_dict: dict) -> None:
        """
        Load an object dictionary.

        Args:
            obj_dict: the object dictionary
        """

        # Load the object properties
        self.model_id = obj_dict.get("modelId", None)
        self.id = obj_dict.get("id", None)
        self.parentId = obj_dict.get("parentId", None)
        self.transform = _Transform(obj_dict.get("transform", {}))
        self.index = obj_dict.get("index", None)
        self.parentIndex = obj_dict.get("parentIndex", None)
        self.holodeck_type = obj_dict.get("holodeckType", None)
        self.holodeck_support_parent = obj_dict.get("holodeckSupportParent", None)

class _Transform:
    def __init__(self, transform_dict: dict = None) -> None:
        """
        Initialize a transform object.

        Args:
            transform_dict: the transform dictionary
        """

        self.rows: int = None
        self.cols: int = None
        self.data: list[list[float]] = None
        self.rotation: list[float] = None
        self.translation: list[float] = None
        self.scale: list[float] = None

        # Load the transform dictionary if provided
        if transform_dict is not None:
            self.load(transform_dict)
    
    def load(self, transform_dict: dict) -> None:
        """
        Load a transform dictionary.

        Args:
            transform_dict: the transform dictionary
        """

        # Load the transform properties
        self.rows = transform_dict.get("rows", 4)
        self.cols = transform_dict.get("cols", 4)
        self.data = transform_dict.get("data", [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        self.rotation = transform_dict.get("rotation", [0.0, 0.0, 0.0, 1.0])
        self.translation = transform_dict.get("translation", [0.0, 0.0, 0.0])
        self.scale = transform_dict.get("scale", [1.0, 1.0, 1.0])
