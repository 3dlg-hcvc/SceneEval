import pathlib
from warnings import warn
from dataclasses import dataclass, field
from pydantic import BaseModel
from omegaconf import DictConfig
from scenes import Scene, Annotation
from vlm import BaseVLM
from spatial import BoundingBox, BoundingBoxConfig, ArchitecturalRelationEvaluator, ArchitecturalRelationConfig
from .base import BaseMetric, MetricResult
from .obj_matching import ObjMatchingResults
from .registry import register_vlm_metric

# ----------------------------------------------------------------------------------------

class ObjArchRelationshipMappingAssessment(BaseModel):
    relationship: str
    target_object: str
    architectural_element_type: str
    relationship_type: str | None
    specific_floors: list[str]
    reason: str

class ObjArchRelationshipMappingResponseFormat(BaseModel):
    assessments: list[ObjArchRelationshipMappingAssessment]
    
@dataclass
class ObjArchRelationshipMetricConfig:
    """
    Configuration for the object-architecture relationship metric.
    
    Attributes:
        relationship_satisfaction_threshold: the threshold for considering a relationship satisfied
        bounding_box: configuration for bounding boxes used in the metric
        arch_relation: configuration for architectural relations used in the metric
    """
    
    relationship_satisfaction_threshold: float = 0.5
    bounding_box: BoundingBoxConfig = field(default_factory=lambda: BoundingBoxConfig())
    arch_relation: ArchitecturalRelationConfig = field(default_factory=lambda: ArchitecturalRelationConfig())
    
    def __post_init__(self):
        if isinstance(self.bounding_box, dict) or isinstance(self.bounding_box, DictConfig):
            self.bounding_box = BoundingBoxConfig(**self.bounding_box)
        if isinstance(self.arch_relation, dict) or isinstance(self.arch_relation, DictConfig):
            self.arch_relation = ArchitecturalRelationConfig(**self.arch_relation)

# ----------------------------------------------------------------------------------------

@register_vlm_metric(config_class=ObjArchRelationshipMetricConfig)
class ObjArchRelationshipMetric(BaseMetric):
    """
    Metric to evaluate object-architecture relationships.
    """
    
    def __init__(self,
                 scene: Scene,
                 annotation: Annotation,
                 vlm: BaseVLM,
                 matching_result: ObjMatchingResults,
                 cfg: ObjArchRelationshipMetricConfig,
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

        self.obj_arch_relationship_specs = self.annotation.oa_rel
        self.spatial_relation_evaluator = ArchitecturalRelationEvaluator(self.cfg.arch_relation)
        
    def _prepare_arch_element_data(self) -> tuple[dict, dict]:
        """
        Prepare architectural element data for evaluation.
        
        Returns:
            arch_data: a dictionary containing architectural element data
            opening_data: a dictionary containing opening data
        """
        
        # Collect floor trimesh objects and create BoundingBoxes for them
        floor_bboxes = {}
        floor_t_objs = {}
        floor_ids = [arch_id for arch_id in self.scene.get_arch_ids() if arch_id.startswith("floor")]
        for floor_id in floor_ids:
            floor_bbox_center = self.scene.get_arch_bbox_center(floor_id)
            floor_bbox_half_size = self.scene.get_default_pose_arch_bbox_extents(floor_id) / 2
            floor_coord_axes = self.scene.get_arch_matrix(floor_id).to_3x3()
            floor_bbox = BoundingBox(floor_bbox_center, floor_bbox_half_size, floor_coord_axes, cfg=self.cfg.bounding_box)
            floor_bboxes[floor_id] = floor_bbox
            floor_t_objs[floor_id] = self.scene.t_architecture[floor_id]
        
        # Collect wall trimesh objects and create BoundingBoxes for them
        wall_bboxes = {}
        wall_t_objs = {}
        wall_ids = [arch_id for arch_id in self.scene.get_arch_ids() if arch_id.startswith("wall")]
        for wall_id in wall_ids:
            wall_bbox_center = self.scene.get_arch_bbox_center(wall_id)
            wall_bbox_half_size = self.scene.get_default_pose_arch_bbox_extents(wall_id) / 2
            wall_coord_axes = self.scene.get_arch_matrix(wall_id).to_3x3()
            wall_bbox = BoundingBox(wall_bbox_center, wall_bbox_half_size, wall_coord_axes, cfg=self.cfg.bounding_box)
            wall_bboxes[wall_id] = wall_bbox
            wall_t_objs[wall_id] = self.scene.t_architecture[wall_id]
        
        # Collect door trimesh objects and create BoundingBoxes for them
        door_bboxes = {}
        door_t_objs = {}
        door_ids = [arch_id for arch_id in self.scene.get_arch_ids() if arch_id.startswith("door")]
        for door_id in door_ids:
            door_bbox_center = self.scene.get_arch_bbox_center(door_id)
            door_bbox_half_size = self.scene.get_default_pose_arch_bbox_extents(door_id) / 2
            door_coord_axes = self.scene.get_arch_matrix(door_id).to_3x3()
            door_bbox = BoundingBox(door_bbox_center, door_bbox_half_size, door_coord_axes, cfg=self.cfg.bounding_box)
            door_bboxes[door_id] = door_bbox
            door_t_objs[door_id] = self.scene.t_architecture[door_id]
        
        # Collect window trimesh objects and create BoundingBoxes for them
        window_bboxes = {}
        window_t_objs = {}
        window_ids = [arch_id for arch_id in self.scene.get_arch_ids() if arch_id.startswith("window")]
        for window_id in window_ids:
            window_bbox_center = self.scene.get_arch_bbox_center(window_id)
            window_bbox_half_size = self.scene.get_default_pose_arch_bbox_extents(window_id) / 2
            window_coord_axes = self.scene.get_arch_matrix(window_id).to_3x3()
            window_bbox = BoundingBox(window_bbox_center, window_bbox_half_size, window_coord_axes, cfg=self.cfg.bounding_box)
            window_bboxes[window_id] = window_bbox
            window_t_objs[window_id] = self.scene.t_architecture[window_id]
        
        arch_data = {
            "floor_bboxes": floor_bboxes,
            "floor_t_objs": floor_t_objs,
            "wall_bboxes": wall_bboxes,
            "wall_t_objs": wall_t_objs,
        }
        
        opening_data = {
            "door_bboxes": door_bboxes,
            "door_t_objs": door_t_objs,
            "window_bboxes": window_bboxes,
            "window_t_objs": window_t_objs,
        }
        
        return arch_data, opening_data
    
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
        # Check if there are object-architecture relationship specs to evaluate
        if len(self.obj_arch_relationship_specs) == 0:
            result = MetricResult(
                message="No object-architecture relationship specs to evaluate.",
                data=evaluations
            )
            print(f"\n{result.message}\n")
            return result
        
        # ==========================================================================================
        # Prepare all architectural element data for evaluation
        arch_data, opening_data = self._prepare_arch_element_data()
        
        # ==========================================================================================
        # Check if objects specified in each spec are present in the scene
        # Specs that does not have all objects present in the scene are automatically unsatisfied
        filtered_specs = []
        for spec in self.obj_arch_relationship_specs:
            quantifier, quantity, relationship, obj_reference = spec.split(",")[:4]
            arch_reference = spec.split(",")[4:]
            
            evaluations[spec] = {
                "quantifier": quantifier,
                "quantity": quantity,
                "relationship": relationship,
                "obj_reference": obj_reference,
                "arch_references": arch_reference,
                "obj_present": False,
                "num_floors": len(arch_data["floor_bboxes"]),
                "num_walls": len(arch_data["wall_bboxes"]),
                "num_doors": len(opening_data["door_bboxes"]),
                "num_windows": len(opening_data["window_bboxes"]),
                "mapped_relationship_type": None,
                "num_candidates": 0,
                "count_satisfied": -1,
                "satisfied": False
            }
            
            print(f"Checking object presence for {spec}...")
            
            # Extract the base category of the object
            base_category = obj_reference.split(":")[0]
            
            # Check if there is at least one object of the base category in the scene
            num_in_scene = len(self.matching_result.per_category[base_category])
            if num_in_scene == 0:
                evaluations[spec]["obj_present"] = False
                evaluations[spec]["satisfied"] = False
                print(f"Missing object category in scene: {base_category} - X\n")
            else:
                evaluations[spec]["obj_present"] = True
                filtered_specs.append(spec)
                print(f"Specified object presents in the scene\n")
                
        print(f"\n{len(filtered_specs)} out of {len(self.obj_arch_relationship_specs)} object-architecture relationship specs have all objects present in the scene. Checking relationships...\n")
        
        # ==========================================================================================
        # Check if there are any specs left to evaluate after filtering
        if len(filtered_specs) == 0:
            result = MetricResult(
                message="No object-architecture relationship specs have all objects present in the scene.",
                data=evaluations
            )
            print(f"\n{result.message}\n")
            return result
        
        # ==========================================================================================
        # Map remaining to-be-check specs to predefined relationships
        make_relationship_str = lambda spec: f"{spec[2]} - object: {spec[3]}, with respect to architectural element: {spec[4:]}"
        prompt_info = {
            "relationships": str([make_relationship_str(spec.split(",")) for spec in filtered_specs]),
            "floor_ids": str([floor_id for floor_id in arch_data["floor_bboxes"].keys()]),
        }
        response: ObjArchRelationshipMappingResponseFormat | str = self.vlm.send("arch_relationship_mapping",
                                                                                         prompt_info=prompt_info,
                                                                                         response_format=ObjArchRelationshipMappingResponseFormat)
        
        if type(response) is str:
            warn("The response is not in the expected format.", RuntimeWarning)
        
        # ==========================================================================================
        # Evaluate the relationships
        for spec, assessment in zip(filtered_specs, response.assessments):
            
            # --------------------------------------------------------------------------------------
            # Gather the mapped relationship types for the spec
            # Skip over relationships that are not implemented, assume they are satisfied
            relationship_type = assessment.relationship_type
            if str(relationship_type) == str(None):
                evaluations[spec]["mapped_relationship_type"] = "Skipped"
                evaluations[spec]["satisfied"] = True
                warn(f"Could not find a suitable relationship type for {assessment.relationship}. Skipping...", RuntimeWarning)
                continue
            evaluations[spec]["mapped_relationship_type"] = relationship_type
            
            # --------------------------------------------------------------------------------------
            # Get relationship check function and find all suitable objects for the relationship
            relation_check_function = getattr(self.spatial_relation_evaluator, relationship_type)

            obj_ids = list(self.matching_result.per_category[assessment.target_object].keys())
            
            # --------------------------------------------------------------------------------------
            # Evaluate the relationship for each object choice
            relationship_candidate_results = []
            for obj_id in obj_ids:
                
                # Create a BoundingBox for the object for evaluating spatial relationships
                obj_bbox_center = self.scene.get_obj_bbox_center(obj_id)
                obj_half_size = self.scene.get_default_pose_obj_bbox_extents(obj_id) / 2
                obj_coord_axes = self.scene.get_obj_matrix(obj_id).to_3x3()
                obj_bbox = BoundingBox(obj_bbox_center, obj_half_size, obj_coord_axes, cfg=self.cfg.bounding_box)
                
                # Prepare the specific floors for the relationship check
                # If no specific floors are specified, use all floors
                if len(assessment.specific_floors) == 0:
                    specified_floor_t_objs = list(arch_data["floor_t_objs"].values())
                    specified_floor_bboxes = list(arch_data["floor_bboxes"].values())
                else:
                    specified_floor_t_objs = [arch_data["floor_t_objs"][floor_id] for floor_id in assessment.specific_floors]
                    specified_floor_bboxes = [arch_data["floor_bboxes"][floor_id] for floor_id in assessment.specific_floors]
                
                # Prepare the parameters for the function
                function_params = {
                    "architectural_element_type": assessment.architectural_element_type,
                    "obj_t_obj": self.scene.t_objs[obj_id],
                    "floor_t_objs": specified_floor_t_objs,
                    "wall_t_objs": list(arch_data["wall_t_objs"].values()),
                    "door_t_objs": list(opening_data["door_t_objs"].values()),
                    "window_t_objs": list(opening_data["window_t_objs"].values()),
                    "obj_bbox": obj_bbox,
                    "floor_bboxes": specified_floor_bboxes,
                    "wall_bboxes": list(arch_data["wall_bboxes"].values()),
                    "door_bboxes": list(opening_data["door_bboxes"].values()),
                    "window_bboxes": list(opening_data["window_bboxes"].values()),
                }
                
                # Evaluate the relationship using the function
                relationship_score = relation_check_function(**function_params)
                candidate_satisfied = bool(relationship_score > self.cfg.relationship_satisfaction_threshold)
                relationship_candidate_results.append(candidate_satisfied)
                
                # Render the object for visualization
                self.scene.blender_scene.render_selected_objs_global_top([obj_id], pathlib.Path(f"oarel/{spec.replace(',', '_')}/candidate_obj_{evaluations[spec]['num_candidates']}.png"))
                evaluations[spec]["num_candidates"] += 1
                
                print(f"Spec: {spec}, Object: {obj_id}, Architecture type: {assessment.architectural_element_type}, Relationship: {assessment.relationship_type}, Score: {relationship_score:.2f}")

            # --------------------------------------------------------------------------------------
            # All candidate objects have been evaluated for this spec
            # Check if the relationship is satisfied based on the quantifier and quantity
            quantifier, quantity = spec.split(",")[:2]
            count_satisfied = sum(relationship_candidate_results)
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
