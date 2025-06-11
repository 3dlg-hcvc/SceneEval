from warnings import warn
from pydantic import BaseModel
from scenes import Scene, Annotation
from vlm import BaseVLM
from .base import BaseMetric, MetricResult
from .obj_matching import ObjMatchingResults
from .registry import register_vlm_metric

# ----------------------------------------------------------------------------------------

class ObjAttributeAssessment(BaseModel):
    instance: int
    attribute: str
    satisfied: bool
    reason: str

class ObjAttributeMetricResponseFormat(BaseModel):
    category: str
    num_instances: int
    assessments: list[ObjAttributeAssessment]

# ----------------------------------------------------------------------------------------

@register_vlm_metric()
class ObjAttributeMetric(BaseMetric):
    """
    Metric to evaluate scene and object attributes against annotations.
    """

    def __init__(self,
                 scene: Scene,
                 annotation: Annotation,
                 vlm: BaseVLM,
                 matching_result: ObjMatchingResults,
                 **kwargs) -> None:
        """
        Initialize the metric.

        Args:
            scene: the scene to evaluate
            annotation: the annotation for the scene
            vlm: the VLM to use for evaluation
            matching_result: the object matching result
        """

        self.scene = scene
        self.annotation = annotation
        self.vlm = vlm
        self.matching_result = matching_result

        self.vlm.reset()

        self.obj_attribute_specs = self.annotation.obj_attr
        mentioned_categories = list(dict.fromkeys([spec.split(",")[-2] for spec in self.obj_attribute_specs]))
        self.image_paths_per_category: dict[str, list[str]] = {}
        
        # Get the front and size reference images for each object in the mentioned categories
        for category in mentioned_categories:
            front_images = [self.scene.get_obj_render_path(obj_id, "FRONT") for obj_id in self.matching_result.per_category[category]]
            size_reference_images = [self.scene.get_obj_render_path(obj_id, "SIZE_REFERENCE") for obj_id in self.matching_result.per_category[category]]
            self.image_paths_per_category[category] = [image for pair in zip(front_images, size_reference_images) for image in pair]
        
    def run(self, verbose: bool = False) -> MetricResult:
        """
        Run the metric.

        Args:
            verbose: whether to visualize during the run
        
        Returns:
            result: the result of running the metric
        """
        
        if len(self.obj_attribute_specs) == 0:
            return MetricResult(message="No object attribute requirements to evaluate.", data={})

        evaluations = {}
        for i, spec in enumerate(self.obj_attribute_specs):
            self.vlm.reset() # GPT can gives 500 error if not reset
            
            splitted_spec = spec.split(",")
            quantifier, quantity, category = splitted_spec[:3]
            attributes = splitted_spec[3:]

            print(f"[{i+1}/{len(self.obj_attribute_specs)}] Checking number of {category} with attributes {attributes} in the scene: {quantifier} {quantity}")

            num_objects_in_scene = len(self.matching_result.per_category[category])
            
            evaluations[spec] = {
                "category": category,
                "quantifier": quantifier,
                "quantity": quantity,
                "attributes": attributes,
                "num_category_in_scene": num_objects_in_scene,
                "count_satisfied": -1,
                "reasons": [],
                "satisfied": False
            }

            if num_objects_in_scene == 0:
                evaluations[spec]["satisfied"] = False
                print("No objects of this category - X\n")
                continue
            
            prompt_info = {
                "obj_count": str(num_objects_in_scene),
                "obj_category": category,
                "obj_attributes": str(attributes)
            }
            response: ObjAttributeMetricResponseFormat | str = self.vlm.send("obj_attribute",
                                                                             prompt_info,
                                                                             self.image_paths_per_category[category],
                                                                             ObjAttributeMetricResponseFormat)
            
            if type(response) is str:
                warn("The response is not in the expected format.", RuntimeWarning)
                continue
            
            evaluations[spec]["reasons"] = [assessment.reason for assessment in response.assessments]

            count_satisfied = sum([1 for assessment in response.assessments if assessment.satisfied])
            evaluations[spec]["count_satisfied"] = count_satisfied

            match quantifier:
                case "eq":
                    satisfied = count_satisfied == int(quantity)
                case "lt":
                    satisfied = count_satisfied < int(quantity)
                case "gt":
                    satisfied = count_satisfied > int(quantity)
                case "le":
                    satisfied = count_satisfied <= int(quantity)
                case "ge":
                    satisfied = count_satisfied >= int(quantity)
            evaluations[spec]["satisfied"] = satisfied
            
            print(f"Expected {quantifier} {quantity}, got {count_satisfied} - {'O' if satisfied else 'X'}\n")
        
        result = MetricResult(
            message=f"{sum([1 for s in evaluations.values() if s['satisfied']])}/{len(evaluations)} requirements are satisfied.",
            data=evaluations
        )

        print(f"\n{result.message}\n")

        return result
