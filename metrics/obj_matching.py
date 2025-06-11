from dataclasses import dataclass
from warnings import warn
from pydantic import BaseModel
from scenes import Scene, Annotation
from vlm import BaseVLM
from .base import BaseMetric, MetricResult
from .registry import register_vlm_metric

# ----------------------------------------------------------------------------------------

class MathcingAssessmentResponseFormat(BaseModel):
    provided_categories: list[str]
    matched: bool
    matched_category: str
    actual_category: str
    reason: str

@dataclass
class ObjMatchingResults:
    """
    Results of object matching.
    
    Attributes:
        per_category: the objects matched to each category
            - {category_1: {obj_id_1: reason_1, obj_id_2: reason_2, ...}, ...}
        not_matched_objs: the objects that were not matched
            - {obj_id_1: reason_1, obj_id_2: reason_2, ...}
        actual_categories: the actual categories of the objects
            - {obj_id_1: category_1, obj_id_2: category_2, ...}
    """
    per_category: dict[str, dict[str, str]]
    not_matched_objs: dict[str, str]
    actual_categories: dict[str, str]
    
    def to_dict(self):
        return {
            "per_category": self.per_category,
            "not_matched_objs": self.not_matched_objs,
            "actual_categories": self.actual_categories
        }
    
    def from_dict(data):
        return ObjMatchingResults(
            per_category=data["per_category"],
            not_matched_objs=data["not_matched_objs"],
            actual_categories=data["actual_categories"]
        )

# ----------------------------------------------------------------------------------------

@register_vlm_metric()
class ObjMatching(BaseMetric):
    """
    Metric to evaluate object matching.
    """

    def __init__(self, scene: Scene, annotation: Annotation, vlm: BaseVLM, **kwargs) -> None:
        """
        Initialize the metric.

        Args:
            scene: the scene to evaluate
            annotation: the annotation for the scene
            vlm: the VLM to use for matching
        """

        self.scene = scene
        self.annotation = annotation
        self.vlm = vlm

        self.vlm.reset()

        # Extract target categories from the obj_count field of the annotation
        self.target_categories = list(dict.fromkeys([spec.split(",")[-1] for spec in self.annotation.obj_count]))
        self.obj_ids = self.scene.get_obj_ids()

    def run(self, verbose: bool = False) -> MetricResult:
        """
        Run the metric.

        Args:
            verbose: whether to visualize during the run
        
        Returns:
            result: the result of running the metric
        """
        
        # Prepare prompt info
        prompt_info = {
            "target_categories": str(self.target_categories)
        }
        
        # Try to match each object to the target categories
        matching_result = ObjMatchingResults(
            per_category={category: {} for category in self.target_categories},
            not_matched_objs={},
            actual_categories={}
        )
        for i, obj_id in enumerate(self.obj_ids):
            
            print(f"Matching object {i+1}/{len(self.obj_ids)} ...")
            
            # Start with a clean VLM state
            self.vlm.reset()
            
            # Get the front image of the object
            image_path = self.scene.get_obj_render_path(obj_id, "FRONT")
            
            response: MathcingAssessmentResponseFormat | str = self.vlm.send("obj_matching",
                                                                             prompt_info=prompt_info,
                                                                             image_paths=[image_path],
                                                                             response_format=MathcingAssessmentResponseFormat)
            
            if type(response) is str:
                warn("The response is not in the expected format.", RuntimeWarning)
                matching_result.not_matched_objs[obj_id] = "The response is not in the expected format."
                continue
            
            if response.matched:
                matching_result.per_category[response.matched_category][obj_id] = response.reason
            else:
                matching_result.not_matched_objs[obj_id] = response.reason
            
            matching_result.actual_categories[obj_id] = response.actual_category
        
        # num_category_has_objs = len([category for category in matching_result if matching_result[category] != {}])
        num_category_has_objs = len([category for category in matching_result.per_category if matching_result.per_category[category] != {}])
        result = MetricResult(
            message=(
                f"Matched {num_category_has_objs}/{len(self.target_categories)} categories.\n"
                f"Matched {len(self.obj_ids) - len(matching_result.not_matched_objs)}/{len(self.obj_ids)} objects."
            ),
            data={
                "matching_result": matching_result
            }
        )
        
        print(f"\n{result.message}\n")

        return result
