from scenes import Annotation
from .base import BaseMetric, MetricResult
from .obj_matching import ObjMatchingResults
from .registry import register_non_vlm_metric

@register_non_vlm_metric()
class ObjCountMetric(BaseMetric):
    """
    Metric to evaluate object count.
    """

    def __init__(self, annotation: Annotation, matching_result: ObjMatchingResults, **kwargs) -> None:
        """
        Initialize the metric.

        Args:
            annotation: the annotation for the scene
            matching_result: the object matching result
        """

        self.annotation = annotation
        self.matching_result = matching_result

        self.obj_count_specs = self.annotation.obj_count

    def run(self, verbose: bool = False) -> MetricResult:
        """
        Run the metric.

        Args:
            verbose: whether to visualize during the run
        
        Returns:
            result: the result of running the metric
        """

        evaluations = {}
        for i, spec in enumerate(self.obj_count_specs):
            quantifier, quantity, category = spec.split(",")
            count_in_scene = len(self.matching_result.per_category[category])

            print(f"[{i+1}/{len(self.obj_count_specs)}] Checking number of {category} in the scene: {count_in_scene} {quantifier} {quantity} - ", end="")
            
            evaluations[spec] = {
                "category": category,
                "quantifier": quantifier,
                "quantity": quantity,
                "count_in_scene": count_in_scene,
                "satisfied": False
            }

            match quantifier:
                case "eq":
                    satisfied = count_in_scene == int(quantity)
                case "lt":
                    satisfied = count_in_scene < int(quantity)
                case "gt":
                    satisfied = count_in_scene > int(quantity)
                case "le":
                    satisfied = count_in_scene <= int(quantity)
                case "ge":
                    satisfied = count_in_scene >= int(quantity)
            evaluations[spec]["satisfied"] = satisfied
            
            print("O" if satisfied else "X")

        result = MetricResult(
            message=f"{sum([1 for s in evaluations.values() if s['satisfied']])}/{len(evaluations)} requirements are satisfied.",
            data=evaluations
        )

        print(f"\n{result.message}\n")

        return result
