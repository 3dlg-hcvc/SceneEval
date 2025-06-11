import trimesh
import numpy as np
from dataclasses import dataclass
from scenes import Scene
from .base import BaseMetric, MetricResult
from .registry import register_non_vlm_metric

# ----------------------------------------------------------------------------------------

@dataclass
class OutOfBoundMetricConfig:
    """
    Configuration for the out of bound metric.

    Attributes:
        threshold: the threshold ratio of points needed for the object to be considered in bound
        volume_sample_multiplier: multiplier for volume-based sampling density
        min_sample_points: minimum number of points to sample per object
    """

    threshold: float = 0.99
    volume_sample_multiplier: float = 5000.0
    min_sample_points: int = 1000

# ----------------------------------------------------------------------------------------

@register_non_vlm_metric(config_class=OutOfBoundMetricConfig)
class OutOfBoundMetric(BaseMetric):
    """
    Metric to evaluate object out of bound.
    """

    def __init__(self, scene: Scene, cfg: OutOfBoundMetricConfig, **kwargs) -> None:
        """
        Initialize the metric.

        Args:
            scene: the scene to evaluate
            cfg: the configuration for the metric
        """

        self.scene = scene
        self.cfg = cfg
        
        self.gravity_direction = np.array([0, 0, -1])

    def run(self, verbose: bool = False) -> MetricResult:
        """
        Run the metric.

        Args:
            verbose: whether to visualize during the run
        
        Returns:
            result: the result of running the metric
        """
        
        evaluations = {}
        
        if verbose:
            self.scene.show()
        
        t_floors = [t_arch for arch_id, t_arch in self.scene.t_architecture.items() if arch_id.startswith("floor")]
        t_floor = trimesh.util.concatenate(t_floors)
        
        for obj_id in self.scene.get_obj_ids():
            
            evaluations[obj_id] = {
                "num_sampled_points": 0,
                "num_out_of_bound": 0,
                "ratio_in_bound": 0,
                "out_of_bound": True
            }
        
            t_obj = self.scene.t_objs[obj_id]
            
            print(f"Checking out of bound for object {obj_id}...")
            
            # Sample points on the object mesh
            volume = t_obj.bounding_box_oriented.volume
            num_samples = int(max(volume * self.cfg.volume_sample_multiplier, self.cfg.min_sample_points))
            sample_points = t_obj.sample(num_samples)
            evaluations[obj_id]["num_sampled_points"] = sample_points.shape[0]
            print(f"Sampled {evaluations[obj_id]['num_sampled_points']} points")
            
            # Check if the points are out of bound by shooting rays from the points to the floor
            ray_origins = sample_points
            ray_directions = np.tile(self.gravity_direction, (ray_origins.shape[0], 1))
            ray_hit_pts, _, _ = t_floor.ray.intersects_location(ray_origins, ray_directions, multiple_hits=False)
            
            assert len(ray_hit_pts) <= len(sample_points)
            
            # Count the number of out of bound points and calculate the ratio
            evaluations[obj_id]["num_out_of_bound"] = len(sample_points) - len(ray_hit_pts)
            evaluations[obj_id]["ratio_in_bound"] = 1.0 - (evaluations[obj_id]["num_out_of_bound"] / len(sample_points))
            print(f"{evaluations[obj_id]['num_out_of_bound']} points are out of bound")
            
            # Check if the object is out of bound
            evaluations[obj_id]["out_of_bound"] =  evaluations[obj_id]["ratio_in_bound"] < self.cfg.threshold
            print(f"In-bound ratio: {evaluations[obj_id]['ratio_in_bound']}, threshold: {self.cfg.threshold}, out of bound: {evaluations[obj_id]['out_of_bound']}")
            
        result = MetricResult(
            message=f"{sum([1 for s in evaluations.values() if s['out_of_bound']])}/{len(evaluations)} objects are out of bound",
            data=evaluations
        )
        
        print(f"\n{result.message}\n")

        return result
