from __future__ import annotations
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from warnings import warn
from dataclasses import dataclass

# ========================================================================================

@dataclass
class BoundingBoxConfig:
    """
    Configuration for the BoundingBox class.

    Attributes:
        epsilon: the epsilon for computation stability
        sample_points_per_unit_volume: the number of sample points per unit volume
        min_num_sample_points: the minimum number of sample points to sample
        presample_points: whether to pre-sample points inside the bounding box
    """
    
    epsilon: float = 1e-6
    sample_points_per_unit_volume: int = 5000
    min_num_sample_points: int = 64
    presample_points: bool = False

# ========================================================================================

class BoundingBox:
    def __init__(self,
                 centroid: npt.ArrayLike,
                 half_size: npt.ArrayLike,
                 coord_axes: npt.ArrayLike = None,
                 matrix_order: str = "C",
                 cfg: BoundingBoxConfig = None) -> None:
        """
        Initialize a oriented bounding box.

        Args:
            centroid: the centroid of the bounding box - 1x3 vector
            half_size: the half size of the bounding box - 1x3 vector
            coord_axes: the coordinate axes of the bounding box - 3x3 matrix
            matrix_order: the order of the matrix
            cfg: the configuration for the bounding box
        """

        self.centroid = np.asarray(centroid)
        self.half_size = np.asarray(half_size)
        
        if coord_axes is None:
            self.coord_axes = np.eye(3)
        else:
            self.coord_axes = np.asarray(coord_axes, order=matrix_order)
        
        # Use default config if none provided
        self.cfg = cfg if cfg is not None else BoundingBoxConfig()
        
        if cfg.presample_points:
            self.sampled_points = self.sample_points()
        else:
            self.sampled_points = None
    
    @property
    def full_size(self) -> np.ndarray:
        """
        Return:
            full_size: the full size of the bounding box
        """

        full_size = self.half_size * 2

        return full_size
    
    @property
    def volume(self) -> float:
        """
        Return:
            volume: the volume of the bounding box (with epsilon)
        """
        
        full_size_with_epsilon = np.maximum(self.full_size, [self.cfg.epsilon, self.cfg.epsilon, self.cfg.epsilon])
        volume = np.prod(full_size_with_epsilon)

        return volume

    @property
    def matrix(self) -> np.ndarray:
        """
        Return:
            matrix: the 4x4 transformation matrix of the bounding box with rotation, translation, and scale
        """
        matrix = np.eye(4)
        matrix[:3, :3] = self.coord_axes @ np.diag(self.half_size)
        matrix[:3, 3] = self.centroid

        return matrix

    @property
    def no_scale_matrix(self) -> np.ndarray:
        """
        Return:
            matrix: the 4x4 transformation matrix of the bounding box with rotation and translation only
        """
        matrix = np.eye(4)
        matrix[:3, :3] = self.coord_axes
        matrix[:3, 3] = self.centroid

        return matrix
    
    @property
    def min_corner(self) -> np.ndarray:
        """
        Return:
            min_corner: the minimum corner of the bounding box
        """
        
        if not np.allclose(self.coord_axes, np.eye(3)):
            warn("The min_corner is ill-defined for non-axis-aligned bounding box.", RuntimeWarning)

        min_corner = self.centroid - self.coord_axes @ self.half_size

        return min_corner
    
    @property
    def max_corner(self) -> np.ndarray:
        """
        Return:
            max_corner: the maximum corner of the bounding box
        """

        if not np.allclose(self.coord_axes, np.eye(3)):
            warn("The max_corner is ill-defined for non-axis-aligned bounding box.", RuntimeWarning)
        
        max_corner = self.centroid + self.coord_axes @ self.half_size

        return max_corner
    
    def sample_points(self) -> np.ndarray:
        """
        Sample points inside the bounding box.

        Return:
            points: the sampled points
        """

        if self.sampled_points is not None:
            return self.sampled_points

        num_points = int(self.volume * self.cfg.sample_points_per_unit_volume)
        
        num_points = max(num_points, self.cfg.min_num_sample_points)  # Ensure at least min_num_sample_points are sampled
        
        points = np.random.rand(num_points, 3) * 2 - 1      # Sample points in the [-1, 1] cube centered at the origin
        points = points * self.half_size                    # Scale the points to the bounding box size
        points = (self.coord_axes @ points.T).T             # Rotate the points to the bounding box orientation
        points = points + self.centroid                     # Translate the points to the bounding box centroid

        self.sampled_points = points

        return points

    def contains(self, points: npt.ArrayLike) -> np.ndarray:
        """
        Check if the bounding box contains the points.

        Args:
            points: the points to check

        Return:
            contains: the boolean array indicating whether the bounding box contains the points
        """

        points = np.asarray(points)
        points = points - self.centroid
        points = (self.coord_axes.T @ points.T).T
        contains = np.all(np.abs(points) <= self.half_size + self.cfg.epsilon, axis=1)

        return contains
    
    def at_side(self, points: npt.ArrayLike, side: str, no_contain: bool = False, within_area_margin: float = 0.25) -> np.ndarray:
        """
        Check if the points are at the side of the bounding box.

        Args:
            points: the points to check
            side: the side to check
            no_contain: if yes, the points must be outside the bounding box to be considered at the side
            within_area_margin: the margin outside the bounding box face area to be considered within the side

        Return:
            at_side: the boolean array indicating whether the points are at the side of the bounding box
        """

        assert side in ["+x", "-x", "+y", "-y", "+z", "-z"], "Invalid side."
        
        # Transform the points to the bounding box's local coordinate system
        points = np.asarray(points)
        points = points - self.centroid
        points = (self.coord_axes.T @ points.T).T
        
        # Check if the points are to the specified side of the bounding box
        # If no_contain is True, the points must be outside the bounding box to be considered at the side
        # Otherwise, all points to the side of the centroid are considered at the side
        match side:
            case "+x":
                reference_point = self.half_size[0] if no_contain else 0
                at_side = points[:, 0] >= reference_point - self.cfg.epsilon
            case "-x":
                reference_point = -self.half_size[0] if no_contain else 0
                at_side = points[:, 0] <= reference_point + self.cfg.epsilon
            case "+y":
                reference_point = self.half_size[1] if no_contain else 0
                at_side = points[:, 1] >= reference_point - self.cfg.epsilon
            case "-y":
                reference_point = -self.half_size[1] if no_contain else 0
                at_side = points[:, 1] <= reference_point + self.cfg.epsilon
            case "+z":
                reference_point = self.half_size[2] if no_contain else 0
                at_side = points[:, 2] >= reference_point - self.cfg.epsilon
            case "-z":
                reference_point = -self.half_size[2] if no_contain else 0
                at_side = points[:, 2] <= reference_point + self.cfg.epsilon
        
        # If a margin is specified, check if the points are within the area of the side
        # Otherwise, no checks regarding the side face area are performed
        if within_area_margin >= 0:
            match side:
                case "+x" | "-x":
                    y_range = self.half_size[1] * (1 + within_area_margin)
                    z_range = self.half_size[2] * (1 + within_area_margin)
                    within_y_range = np.abs(points[:, 1]) <= y_range
                    within_z_range = np.abs(points[:, 2]) <= z_range
                    within_area = within_y_range & within_z_range
                case "+y" | "-y":
                    x_range = self.half_size[0] * (1 + within_area_margin)
                    z_range = self.half_size[2] * (1 + within_area_margin)
                    within_x_range = np.abs(points[:, 0]) <= x_range
                    within_z_range = np.abs(points[:, 2]) <= z_range
                    within_area = within_x_range & within_z_range
                case "+z" | "-z":
                    x_range = self.half_size[0] * (1 + within_area_margin)
                    y_range = self.half_size[1] * (1 + within_area_margin)
                    within_x_range = np.abs(points[:, 0]) <= x_range
                    within_y_range = np.abs(points[:, 1]) <= y_range
                    within_area = within_x_range & within_y_range
            at_side = at_side & within_area

        return at_side

    def overlaps(self, other_bbox: BoundingBox) -> float:
        """
        Get the overlapping ratio of this bounding box with another bounding box.

        Args:
            other: the other bounding box

        Returns:
            ratio: the overlapping ratio of the two bounding boxes
        """

        sample_points = self.sample_points()
        contains = other_bbox.contains(sample_points)
        ratio = np.sum(contains) / len(sample_points)
        
        return ratio

    def visualize(self, ax: plt.Axes, color: str = "b") -> None:
        """
        Visualize the bounding box on a matplotlib axis.

        Args:
            ax: the matplotlib axis to visualize the bounding box
            color: the color of the bounding box
        """
        
        CORNERS = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]
        ])
        CORNERS = CORNERS * self.half_size
        CORNERS = (self.coord_axes @ CORNERS.T).T + self.centroid

        EDGES = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

        for edge in EDGES:
            ax.plot3D(CORNERS[edge, 0], CORNERS[edge, 1], CORNERS[edge, 2], color=color)
        
        front_pt = self.coord_axes @ np.array([0, -1, 0]) * self.half_size[1] + self.centroid
        ax.scatter3D([front_pt[0]], [front_pt[1]], [front_pt[2]], color=color)
        
        ax.set_aspect("equal", adjustable="box")
