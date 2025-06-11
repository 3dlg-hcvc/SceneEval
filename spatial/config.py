from dataclasses import dataclass, field
from typing import List
from numpy import inf
from omegaconf import DictConfig

# ======================================================================================== Spatial Relations Configurations

@dataclass
class FaceToRelationConfig:
    max_deviation_degrees: float = 30.0

@dataclass
class SideOfRelationConfig:
    no_contain: bool = True
    within_area_margin: float = 0.25

@dataclass
class SideRegionRelationConfig:
    no_contain: bool = False
    within_area_margin: float = 1e-9

@dataclass
class LongShortSideRelationConfig:
    no_contain: bool = False
    within_area_margin: float = 1e-9

@dataclass
class OnTopRelationConfig:
    no_contain: bool = True
    within_area_margin: float = 1e-9

@dataclass
class MiddleOfRelationConfig:
    gaussian_std: float = 0.25

@dataclass
class SurroundRelationConfig:
    distance_weight: float = 0.5
    angle_weight: float = 0.5

@dataclass
class DistanceScoreConfig:
    min_num_sample_points: int = 64
    gaussian_std: float = 0.25

@dataclass
class NextToRelationConfig:
    distance_range: List[float] = field(default_factory=lambda: [0.0, 0.5])

@dataclass
class NearRelationConfig:
    distance_range: List[float] = field(default_factory=lambda: [0.5, 1.5])

@dataclass
class AcrossFromRelationConfig:
    distance_range: List[float] = field(default_factory=lambda: [1.5, 4.0])

@dataclass
class FarRelationConfig:
    distance_range: List[float] = field(default_factory=lambda: [4.0, inf])

@dataclass
class SpatialRelationConfig:
    face_to: FaceToRelationConfig = field(default_factory=lambda: FaceToRelationConfig())
    side_of: SideOfRelationConfig = field(default_factory=lambda: SideOfRelationConfig())
    side_region: SideRegionRelationConfig = field(default_factory=lambda: SideRegionRelationConfig())
    long_short_side_of: LongShortSideRelationConfig = field(default_factory=lambda: LongShortSideRelationConfig())
    on_top: OnTopRelationConfig = field(default_factory=lambda: OnTopRelationConfig())
    middle_of: MiddleOfRelationConfig = field(default_factory=lambda: MiddleOfRelationConfig())
    surround: SurroundRelationConfig = field(default_factory=lambda: SurroundRelationConfig())
    distance_score: DistanceScoreConfig = field(default_factory=lambda: DistanceScoreConfig())
    next_to: NextToRelationConfig = field(default_factory=lambda: NextToRelationConfig())
    near: NearRelationConfig = field(default_factory=lambda: NearRelationConfig())
    across_from: AcrossFromRelationConfig = field(default_factory=lambda: AcrossFromRelationConfig())
    far: FarRelationConfig = field(default_factory=lambda: FarRelationConfig())
    
    def __post_init__(self):
        if isinstance(self.face_to, dict) or isinstance(self.face_to, DictConfig):
            self.face_to = FaceToRelationConfig(**self.face_to)
        if isinstance(self.side_of, dict) or isinstance(self.side_of, DictConfig):
            self.side_of = SideOfRelationConfig(**self.side_of)
        if isinstance(self.side_region, dict) or isinstance(self.side_region, DictConfig):
            self.side_region = SideRegionRelationConfig(**self.side_region)
        if isinstance(self.long_short_side_of, dict) or isinstance(self.long_short_side_of, DictConfig):
            self.long_short_side_of = LongShortSideRelationConfig(**self.long_short_side_of)
        if isinstance(self.on_top, dict) or isinstance(self.on_top, DictConfig):
            self.on_top = OnTopRelationConfig(**self.on_top)
        if isinstance(self.middle_of, dict) or isinstance(self.middle_of, DictConfig):
            self.middle_of = MiddleOfRelationConfig(**self.middle_of)
        if isinstance(self.surround, dict) or isinstance(self.surround, DictConfig):
            self.surround = SurroundRelationConfig(**self.surround)
        if isinstance(self.distance_score, dict) or isinstance(self.distance_score, DictConfig):
            self.distance_score = DistanceScoreConfig(**self.distance_score)
        if isinstance(self.next_to, dict) or isinstance(self.next_to, DictConfig):
            self.next_to = NextToRelationConfig(**self.next_to)
        if isinstance(self.near, dict) or isinstance(self.near, DictConfig):
            self.near = NearRelationConfig(**self.near)
        if isinstance(self.across_from, dict) or isinstance(self.across_from, DictConfig):
            self.across_from = AcrossFromRelationConfig(**self.across_from)
        if isinstance(self.far, dict) or isinstance(self.far, DictConfig):
            self.far = FarRelationConfig(**self.far)

# ======================================================================================== Architectural Relations Configurations

@dataclass
class ArchMiddleOfRoomRelationConfig:
    base_std_dev: float = 1.0
    obj_size_weight: float = 0.5
    ratio_weight: float = 1.0

@dataclass
class ArchNextToRelationConfig:
    distance_range: List[float] = field(default_factory=lambda: [0.0, 0.5])
    gaussian_std: float = 0.25

@dataclass
class ArchNearRelationConfig:
    distance_range: List[float] = field(default_factory=lambda: [0.5, 1.5])
    gaussian_std: float = 0.25

@dataclass
class ArchAcrossFromRelationConfig:
    distance_range: List[float] = field(default_factory=lambda: [1.5, 4.0])
    gaussian_std: float = 0.25

@dataclass
class ArchFarRelationConfig:
    distance_range: List[float] = field(default_factory=lambda: [4.0, inf])
    gaussian_std: float = 0.25

@dataclass
class ArchOnWallRelationConfig:
    distance_range: List[float] = field(default_factory=lambda: [0.0, 0.01])
    gaussian_std: float = 0.01
    no_contain: bool = True
    within_area_margin: float = 1e-9

@dataclass
class ArchAgainstWallRelationConfig:
    distance_range: List[float] = field(default_factory=lambda: [0.0, 0.3])
    gaussian_std: float = 0.1
    no_contain: bool = True
    within_area_margin: float = 1e-9

@dataclass
class ArchCornerOfRoomRelationConfig:
    base_distance_threshold: float = 0.8
    gaussian_std: float = 0.25
    perpendicular_threshold: float = 0.15

@dataclass
class ArchHangFromCeilingRelationConfig:
    distance_range: List[float] = field(default_factory=lambda: [0.0, 0.01])
    gaussian_std: float = 0.03

@dataclass
class ArchitecturalRelationConfig:
    middle_of_room: ArchMiddleOfRoomRelationConfig = field(default_factory=lambda: ArchMiddleOfRoomRelationConfig())
    next_to: ArchNextToRelationConfig = field(default_factory=lambda: ArchNextToRelationConfig())
    near: ArchNearRelationConfig = field(default_factory=lambda: ArchNearRelationConfig())
    across_from: ArchAcrossFromRelationConfig = field(default_factory=lambda: ArchAcrossFromRelationConfig())
    far: ArchFarRelationConfig = field(default_factory=lambda: ArchFarRelationConfig())
    on_wall: ArchOnWallRelationConfig = field(default_factory=lambda: ArchOnWallRelationConfig())
    against_wall: ArchAgainstWallRelationConfig = field(default_factory=lambda: ArchAgainstWallRelationConfig())
    corner_of_room: ArchCornerOfRoomRelationConfig = field(default_factory=lambda: ArchCornerOfRoomRelationConfig())
    hang_from_ceiling: ArchHangFromCeilingRelationConfig = field(default_factory=lambda: ArchHangFromCeilingRelationConfig())
    
    def __post_init__(self):
        if isinstance(self.middle_of_room, dict) or isinstance(self.middle_of_room, DictConfig):
            self.middle_of_room = ArchMiddleOfRoomRelationConfig(**self.middle_of_room)
        if isinstance(self.next_to, dict) or isinstance(self.next_to, DictConfig):
            self.next_to = ArchNextToRelationConfig(**self.next_to)
        if isinstance(self.near, dict) or isinstance(self.near, DictConfig):
            self.near = ArchNearRelationConfig(**self.near)
        if isinstance(self.across_from, dict) or isinstance(self.across_from, DictConfig):
            self.across_from = ArchAcrossFromRelationConfig(**self.across_from)
        if isinstance(self.far, dict) or isinstance(self.far, DictConfig):
            self.far = ArchFarRelationConfig(**self.far)
        if isinstance(self.on_wall, dict) or isinstance(self.on_wall, DictConfig):
            self.on_wall = ArchOnWallRelationConfig(**self.on_wall)
        if isinstance(self.against_wall, dict) or isinstance(self.against_wall, DictConfig):
            self.against_wall = ArchAgainstWallRelationConfig(**self.against_wall)
        if isinstance(self.corner_of_room, dict) or isinstance(self.corner_of_room, DictConfig):
            self.corner_of_room = ArchCornerOfRoomRelationConfig(**self.corner_of_room)
        if isinstance(self.hang_from_ceiling, dict) or isinstance(self.hang_from_ceiling, DictConfig):
            self.hang_from_ceiling = ArchHangFromCeilingRelationConfig(**self.hang_from_ceiling)
