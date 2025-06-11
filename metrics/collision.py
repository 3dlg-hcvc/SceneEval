import trimesh
import numpy as np
from dataclasses import dataclass
from scenes import Scene
from .base import BaseMetric, MetricResult
from .registry import register_non_vlm_metric

# ----------------------------------------------------------------------------------------

@dataclass
class CollisionMetricConfig:
    """
    Configuration for the collision metric.

    Attributes:
        move_direction_amount: the distance to move objects when double-checking collisions
    """

    move_direction_amount: float = 0.005

# ----------------------------------------------------------------------------------------

@register_non_vlm_metric(config_class=CollisionMetricConfig)
class CollisionMetric(BaseMetric):
    """
    Metric to evaluate object collision.
    """

    def __init__(self, scene: Scene, cfg: CollisionMetricConfig, **kwargs) -> None:
        """
        Initialize the metric.

        Args:
            scene: the scene to evaluate
            cfg: the configuration for the metric
        """

        self.scene = scene
        self.cfg = cfg

    def run(self, verbose: bool = False) -> MetricResult:
        """
        Run the metric.

        Args:
            verbose: whether to visualize during the run
        
        Returns:
            result: the result of running the metric
        """
        
        collision_manager = trimesh.collision.CollisionManager()
        
        # Initialize the collision results
        collision_results = {
            obj_id: {
                "in_collision": False,
                "colliding_with": []
            }
        for obj_id in self.scene.get_obj_ids()}
        
        # For each object, check if it is in collision with any other object
        for i, obj_id in enumerate(self.scene.get_obj_ids()):
            
            # Add the object to the collision manager
            t_obj = self.scene.t_objs[obj_id]
            collision_manager.add_object(obj_id, t_obj)
            
            # Check for collision with each of the other objects
            for other_obj_id in self.scene.get_obj_ids()[i+1:]:
                
                # Add the other object to the collision manager and check for collision
                t_other_obj = self.scene.t_objs[other_obj_id]
                in_collision, contact_data = collision_manager.in_collision_single(t_other_obj, return_data=True)
                
                # If in collision, double check by separating the objects slightly and checking again
                if in_collision:
                    
                    # Get the contact point locations
                    contact_pts = np.asarray([contact.point for contact in contact_data])
                    
                    # Move the object slightly away from the other object
                    move_direction = t_other_obj.centroid - np.mean(contact_pts, axis=0)
                    move_direction /= np.linalg.norm(move_direction)
                    moved_t_other_obj = t_other_obj.copy()
                    moved_t_other_obj.apply_translation(move_direction * self.cfg.move_direction_amount)
                    
                    # Check for collision again
                    double_check_in_collision = collision_manager.in_collision_single(moved_t_other_obj)
                    
                    # If still in collision, add the other object to the collision results
                    if double_check_in_collision:
                        collision_results[obj_id]["in_collision"] = True
                        collision_results[obj_id]["colliding_with"].append(other_obj_id)
                        collision_results[other_obj_id]["in_collision"] = True
                        collision_results[other_obj_id]["colliding_with"].append(obj_id)
                    
                print((
                    f"Checked: {obj_id} and {other_obj_id} - 1st check: {in_collision}, 2nd check: {double_check_in_collision if in_collision else 'N/A'} -> "
                    f"{'Collision - O' if in_collision and double_check_in_collision else 'No Collision - X'}"
                ))
            
            # Remove the object from the collision manager after checking for collision with all other objects
            collision_manager.remove_object(obj_id)

        # Summarize the collision results
        num_obj_in_collision = sum(obj_result["in_collision"] for obj_result in collision_results.values())
        scene_in_collision = num_obj_in_collision > 0

        result = MetricResult(
            message=f"Scene is in collision: {scene_in_collision}, with {num_obj_in_collision}/{len(self.scene.get_obj_ids())} objects in collision.",
            data={
                "scene_in_collision": scene_in_collision,
                "num_obj_in_collision": num_obj_in_collision,
                "collision_results": collision_results
            }
        )

        print(f"\n{result.message}\n")

        return result
