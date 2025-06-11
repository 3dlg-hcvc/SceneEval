import warnings

class Architecture:
    def __init__(self, arch_dict: dict = None) -> None:
        """
        Initialize an architecture object.

        Args:
            arch_dict: the architecture dictionary
        """

        self.version: str = None
        self.up: list[float] = None
        self.front: list[float] = None
        self.coords2d: list[float] = None
        self.scaleToMeters: float = None
        self.defaults: dict = None
        self.id: str = None
        self.elements: list[_Element] = None
        self.regions: list[_Region] = None

        # Load the architecture dictionary if provided
        if arch_dict is not None:
            self.load(arch_dict)
    
    def load(self, arch_dict: dict) -> None:
        """
        Load an architecture dictionary.

        Args:
            arch_dict: the architecture dictionary
        """

        # Check version
        self.version = arch_dict.get("version", None)
        if self.version != "arch@1.0.2":
            warnings.warn(f"This module is developed for arch version 1.0.2, but the arch version is {self.version}.")

        # Load the architecture properties
        self.up = arch_dict.get("up", [0, 0, 1])
        self.front = arch_dict.get("front", [0, 1, 0])
        self.coords2d = arch_dict.get("coords2d", [0, 1])
        self.scaleToMeters = arch_dict.get("scaleToMeters", 1.0)
        self.defaults = arch_dict.get("defaults", None)
        self.id = arch_dict.get("id", None)

        # Load the architecture elements
        # Sort so that the elements are in the order of ["Floor", "Ceiling", "Wall_0", "Wall_1", ...]
        self.elements = [ _Element(element_dict) for element_dict in arch_dict.get("elements", []) ]
        self.elements.sort(key=lambda x: ["Floor", "Ceiling", "Wall"].index(x.type))

        # Load the architecture regions
        self.regions = [ _Region(region_dict) for region_dict in arch_dict.get("regions", []) ]

class _Hole:
    def __init__(self, hole_dict: dict = None) -> None:
        """
        Initialize an architecture hole object.

        Args:
            hole_dict: the hole dictionary
        """

        self.id: str = None
        self.type: str = None
        self.box_min: list[float] = None
        self.box_max: list[float] = None

        # Load the hole dictionary if provided
        if hole_dict is not None:
            self.load(hole_dict)
    
    def load(self, hole_dict: dict) -> None:
        """
        Load an architecture hole dictionary.

        Args:
            hole_dict: the hole dictionary
        """

        # Load the hole properties
        self.id = hole_dict.get("id", None)
        self.type = hole_dict.get("type", None)
        box = hole_dict.get("box", None)
        if box is not None:
            self.box_min = box["min"]
            self.box_max = box["max"]
        else:
            self.box_min = None
            self.box_max = None

class _Element:
    def __init__(self, element_dict: dict = None) -> None:
        """
        Initialize an architecture element object.

        Args:
            element_dict: the element dictionary
        """

        self.id: str = None
        self.type: str = None
        self.roomId: str = None
        self.points: list[list[float]] = None
        self.height: float = None
        self.depth: float = None
        self.materials: str[dict] = None
        self.holes: list[_Hole] = None

        # Load the element dictionary if provided
        if element_dict is not None:
            self.load(element_dict)
    
    def load(self, element_dict: dict) -> None:
        """
        Load an architecture element dictionary.

        Args:
            element_dict: the element dictionary
        """

        # Load the element properties
        self.id = element_dict.get("id", None)
        self.type = element_dict.get("type", None)
        self.roomId = element_dict.get("roomId", None)
        self.points = element_dict.get("points", [])
        self.height = element_dict.get("height", None)
        self.depth = element_dict.get("depth", None)
        self.materials = element_dict.get("materials", None)
        self.holes = [ _Hole(hole_dict) for hole_dict in element_dict.get("holes", []) ]

class _Region:
    def __init__(self, region_dict: dict = None) -> None:
        """
        Initialize an architecture region object.

        Args:
            region_dict: the region dictionary
        """

        self.id: str = None
        self.type: str = None
        self.walls: list[int] = None

        # Load the region dictionary if provided
        if region_dict is not None:
            self.load(region_dict)
    
    def load(self, region_dict: dict) -> None:
        """
        Load an architecture region dictionary.

        Args:
            region_dict: the region dictionary
        """

        # Load the region properties
        self.id = region_dict.get("id", None)
        self.type = region_dict.get("type", None)
        self.walls = region_dict.get("walls", [])
