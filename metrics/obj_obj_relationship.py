import pathlib
from warnings import warn
from dataclasses import dataclass, field
from itertools import combinations, product, chain
from pydantic import BaseModel
from omegaconf import DictConfig
from scenes import Scene, Annotation
from vlm import BaseVLM
from spatial import BoundingBox, BoundingBoxConfig, SpatialRelationEvaluator, SpatialRelationConfig
from .base import BaseMetric, MetricResult
from .obj_matching import ObjMatchingResults
from .registry import register_vlm_metric

# ----------------------------------------------------------------------------------------

class ObjObjRelationshipMappingAssessment(BaseModel):
    relationship: str
    anchor_object: str
    other_objects: list[str]
    other_object_counts: list[int]
    relationship_types: list[str] | None
    sides: list[str | None]
    reason: str

class ObjObjRelationshipMappingResponseFormat(BaseModel):
    assessments: list[ObjObjRelationshipMappingAssessment]
    
@dataclass
class ObjObjRelationshipMetricConfig:
    """
    Configuration for the object-object relationship metric.
    
    Attributes:
        relationship_satisfaction_threshold: the threshold for considering a relationship satisfied
        max_candidate_group_renders: the maximum number of candidate groups to render for visualization to avoid Blender freezing
        bounding_box: configuration for bounding boxes used in the metric
        spatial_relation: configuration for spatial relations used in the metric
    """
    
    relationship_satisfaction_threshold: float = 0.5
    max_candidate_group_renders: int = 50
    bounding_box: BoundingBoxConfig = field(default_factory=lambda: BoundingBoxConfig())
    spatial_relation: SpatialRelationConfig = field(default_factory=lambda: SpatialRelationConfig())
    
    def __post_init__(self):
        if isinstance(self.bounding_box, dict) or isinstance(self.bounding_box, DictConfig):
            self.bounding_box = BoundingBoxConfig(**self.bounding_box)
        if isinstance(self.spatial_relation, dict) or isinstance(self.spatial_relation, DictConfig):
            self.spatial_relation = SpatialRelationConfig(**self.spatial_relation)

# ----------------------------------------------------------------------------------------

@register_vlm_metric(config_class=ObjObjRelationshipMetricConfig)
class ObjObjRelationshipMetric(BaseMetric):
    """
    Metric to evaluate object-object relationships.
    """

    def __init__(self,
                 scene: Scene,
                 annotation: Annotation,
                 vlm: BaseVLM,
                 matching_result: ObjMatchingResults,
                 cfg: ObjObjRelationshipMetricConfig,
                 **kwargs) -> None:
        """
        Initialize the metric.

        Args:
            scene: the scene to evaluate
            annotation: the annotation for the scene
            vlm: the VLM to use for evaluation
            matching_result: the object matching results
            cfg: the configuration for the metric
        """

        self.scene = scene
        self.annotation = annotation
        self.vlm = vlm
        self.matching_result = matching_result
        self.cfg = cfg

        self.vlm.reset()

        self.obj_obj_relationship_specs = self.annotation.oo_rel
        self.spatial_relation_evaluator = SpatialRelationEvaluator(self.cfg.spatial_relation)
    
    def run(self, verbose: bool = False) -> MetricResult:
        """
        Run the metric.

        Args:
            verbose: whether to visualize during the run
        
        Returns:
            result: the result of running the metric
        """

        evaluations = {}
        
        # ==========================================================================================
        # Check if there are object-object relationship specs to evaluate
        if len(self.obj_obj_relationship_specs) == 0:
            result = MetricResult(
                message="No object-object relationship specs to evaluate.",
                data=evaluations
            )
            print(f"\n{result.message}\n")
            return result

        # ==========================================================================================
        # Check if objects specified in each spec are present in the scene
        # Specs that does not have all objects present in the scene are automatically unsatisfied
        filtered_specs = []
        for spec in self.obj_obj_relationship_specs:
            quantifier, quantity, relationship, anchor_obj_index = spec.split(",")[:4]
            obj_references = spec.split(",")[4:]
            
            evaluations[spec] = {
                "quantifier": quantifier,
                "quantity": quantity,
                "relationship": relationship,
                "anchor_obj_index": anchor_obj_index,
                "obj_references": obj_references,
                "all_objects_present": False,
                "mapped_relationship_types": None,
                "num_candidate_groups": 0,
                "child_relationship_results": [],
                "count_satisfied": -1,
                "satisfied": False
            }
            
            print(f"Checking object presence for {spec}...")
            
            # Extract the base categories and the number of times they are mentioned in the spec
            category_info = {}
            for obj_reference in obj_references:
                base_category = obj_reference.split(":")[0]
                if base_category not in category_info:
                    category_info[base_category] = 1
                else:
                    category_info[base_category] += 1
            
            # Check if all objects are present in the scene
            num_in_scene = [len(self.matching_result.per_category[base_category]) for base_category in category_info.keys()]
            if not all([num_in_scene[i] >= num_needed for i, num_needed in enumerate(category_info.values())]):
                evaluations[spec]["all_objects_present"] = False
                evaluations[spec]["satisfied"] = False
                print(f"Not all objects present in the scene - X\n")
                continue
            else:
                evaluations[spec]["all_objects_present"] = True
                filtered_specs.append(spec)
                print(f"All objects present in the scene\n")
        
        print(f"\n{len(filtered_specs)} out of {len(self.obj_obj_relationship_specs)} object-object relationship specs have all objects present in the scene. Checking relationships...\n")
        
        # ==========================================================================================
        # Check if there are any specs left to evaluate after filtering
        if len(filtered_specs) == 0:
            result = MetricResult(
                message="No object-object relationship specs have all objects present in the scene.",
                data=evaluations
            )
            print(f"\n{result.message}\n")
            return result
        
        # ==========================================================================================
        # Map remaining to-be-check specs to predefined relationships
        # Each spec can be mapped to multiple relationships, where all must be satisfied for the spec to be satisfied
        make_relationship_str = lambda spec: f"{spec[2]} - objects: {spec[4:]}, with the object with index: {spec[3]} being the anchor"
        prompt_info = {
            "relationships": str([make_relationship_str(spec.split(",")) for spec in filtered_specs])
        }
        response: ObjObjRelationshipMappingResponseFormat | str = self.vlm.send("obj_relationship_mapping",
                                                                                prompt_info=prompt_info,
                                                                                response_format=ObjObjRelationshipMappingResponseFormat)
        
        if type(response) is str:
            warn("The response is not in the expected format.", RuntimeWarning)

        # ==========================================================================================
        # Evaluate the relationships
        for spec, assessment in zip(filtered_specs, response.assessments):
            
            print(f"Evaluating {spec} ...")
            
            # --------------------------------------------------------------------------------------
            # Gather the mapped relationship types for the spec
            # Skip over relationships that are not implemented, assume they are satisfied
            if str(assessment.relationship_types) == str(None):
                evaluations[spec]["mapped_relationship_types"] = "Skipped"
                evaluations[spec]["satisfied"] = True
                warn(f"Could not find a suitable relationship type for {assessment.relationship}. Skipping...", RuntimeWarning)
                continue
            evaluations[spec]["mapped_relationship_types"] = assessment.relationship_types
            print(f"Relationship types: {assessment.relationship_types}")

            # --------------------------------------------------------------------------------------            
            # Find all suitable objects for the anchor and other objects based on the assessed categories
            # anthor_obj_ids: list of object ids for the anchor object
            # other_obj_ids: dict of lists of object ids for the other objects, keyed by the category
            anchor_category_obj_ids = list(self.matching_result.per_category[assessment.anchor_object].keys())
            other_categories_obj_ids: dict[str, list[str]] = {}
            for other_category in assessment.other_objects:
                other_categories_obj_ids[other_category] = list(self.matching_result.per_category[other_category].keys())
            
            # --------------------------------------------------------------------------------------
            # For each object, create a bounding box for it
            bboxes = {}
            all_ids = list(chain(anchor_category_obj_ids, *other_categories_obj_ids.values()))
            print(f"Creating bounding boxes for {len(all_ids)} objects...", end="")
            for obj_id in all_ids:
                obj_bbox_center = self.scene.get_obj_bbox_center(obj_id)
                obj_half_size = self.scene.get_default_pose_obj_bbox_extents(obj_id) / 2
                obj_coord_axes = self.scene.get_obj_matrix(obj_id).to_3x3()
                obj_bbox = BoundingBox(obj_bbox_center, obj_half_size, obj_coord_axes, cfg=self.cfg.bounding_box)
                bboxes[obj_id] = obj_bbox
            print("Done.")

            # --------------------------------------------------------------------------------------
            # Create all possible object combination groups for the current spec
            # The object choices are represented as indices in the object lists
            
            # Choices for the anchor object is all the objects in the anchor category because there can only be one anchor object
            # E.g., if there are 3 objects in the anchor category, the choices are [0, 1, 2]
            anchor_choices = list(range(len(anchor_category_obj_ids)))
            
            # Choices for the other objects are combinations of the objects in the corresponding categories based on the counts per category
            # E.g., if there are 4 objects for an other category and the count is 2, then there are 4C2 = 6 choices - [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            print(f"Creating object combination groups for {len(anchor_choices)} anchor objects and {len(other_categories_obj_ids)} other categories...")
            other_object_choices = {}
            for other_category, count in zip(other_categories_obj_ids.keys(), assessment.other_object_counts):
                # N choose "count" combinations for each category
                num_other_objs_for_choosing = len(other_categories_obj_ids[other_category])
                other_object_choices[other_category] = list(combinations(range(num_other_objs_for_choosing), count))
            
            # Create all possible groups based on choosing one choice from each category
            candidate_groups = list(product(anchor_choices, *other_object_choices.values()))
            print(f"Created {len(candidate_groups)} candidate groups.")
            
            # --------------------------------------------------------------------------------------
            # Evaluate the relationships for each group
            parent_relationship_candidate_results = []
            for group in candidate_groups:
                
                # Gather the anchor object and its bounding box for this group
                anchor_obj_id = anchor_category_obj_ids[group[0]]
                anchor_t_obj = self.scene.t_objs[anchor_obj_id]
                anchor_obj_bbox = bboxes[anchor_obj_id]
                
                # Gather the other objects and their bounding boxes for this group
                all_other_ids = []
                all_other_t_objs = []
                all_other_obj_bboxes = []
                for other_category, other_obj_indices in zip(other_categories_obj_ids.keys(), group[1:]):
                    for other_obj_index in other_obj_indices:
                        other_obj_id = other_categories_obj_ids[other_category][other_obj_index]
                        other_t_obj = self.scene.t_objs[other_obj_id]
                        other_obj_bbox = bboxes[other_obj_id]
                        all_other_ids.append(other_obj_id)
                        all_other_t_objs.append(other_t_obj)
                        all_other_obj_bboxes.append(other_obj_bbox)
                
                # --------------------------------------------------------------------------------------
                # The relationship in the spec may be mapped to multiple child relationships
                # Each child relationship must be satisfied for the parent relationship to be satisfied
                child_relationships_results = []
                for child_type_idx, child_relationship_type in enumerate(assessment.relationship_types):
                    
                    # Get the relationship check function
                    relation_check_function = getattr(self.spatial_relation_evaluator, child_relationship_type)
                    print("Using function:", relation_check_function.__name__)

                    # Prepare the parameters for the function
                    function_params = {
                        "reference_t_obj": anchor_t_obj,
                        "reference_bbox": anchor_obj_bbox,
                        "target_t_obj": all_other_t_objs[0], # For maintaining consistency in function signatures, put the first other object here even if there are multiple. But if there are multiple, this parameter should not be used in the function.
                        "target_bbox": all_other_obj_bboxes[0], # Same as above
                        "multiple_target_t_objs": all_other_t_objs,
                        "multiple_target_bboxes": all_other_obj_bboxes,
                        "side": assessment.sides[child_type_idx]
                    }
                    
                    # Evaluate the relationship using the function
                    relationship_score = relation_check_function(**function_params)
                    satisfied = bool(relationship_score > self.cfg.relationship_satisfaction_threshold) # bool() to convert from numpy.bool_ for JSON serialization
                    child_relationships_results.append(satisfied)
                    
                    print((f" --- Spec: {spec}, Candidate Group: {evaluations[spec]['num_candidate_groups']}, "
                           f"Reference: {anchor_obj_id}, Target(s): {all_other_ids}, "
                           f"Child Relationships: {assessment.relationship_types}, "
                           f"Evaluating: {child_relationship_type}, Score: {relationship_score:.2f}"))
                
                # --------------------------------------------------------------------------------------
                # For this group, results for all child relationships are collected
                # Now, evaluate the parent relationship based on whether all child relationships are satisfied
                evaluations[spec]["child_relationship_results"].append(child_relationships_results)
                parent_relationship_satisfied = all(child_relationships_results)
                parent_relationship_candidate_results.append(parent_relationship_satisfied)
                    
                print(f"> Child relationship results: {child_relationships_results} -> Parent relationship satisfied: {parent_relationship_satisfied}\n")
                
                # --------------------------------------------------------------------------------------
                # Render this group of objects for visualization
                if evaluations[spec]["num_candidate_groups"] < self.cfg.max_candidate_group_renders: # Limit the number of groups to render as Blender freezes after certain number of renders
                    self.scene.blender_scene.render_selected_objs_global_top([anchor_obj_id, *all_other_ids], pathlib.Path(f"oorel/{spec.replace(',', '_')}/candidate_group_{evaluations[spec]['num_candidate_groups']}.png"))
                evaluations[spec]["num_candidate_groups"] += 1
            
            # --------------------------------------------------------------------------------------
            # All groups have been evaluated for this spec
            # Check if the relationship is satisfied based on the quantifier and quantity
            quantifier, quantity = spec.split(",")[:2]
            count_satisfied = sum(parent_relationship_candidate_results)
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
    