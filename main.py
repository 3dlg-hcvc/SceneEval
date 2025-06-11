import os
import random
import pathlib
import hydra
import json
import numpy as np
from natsort import natsorted
from dataclasses import dataclass
from dotenv import load_dotenv
from omegaconf import DictConfig, open_dict

from scenes import *
from metrics import MetricRegistry, ObjMatching, ObjMatchingResults
from assets import Retriever
from semantic_colors import apply_semantic_colors
from vlm import VLMRegistry, BaseVLM

load_dotenv()

# ========================================================================================

@dataclass
class EvaluationConfig:
    metrics: list[str]
    output_dir: str
    save_blend_file: bool
    vlm: str
    use_existing_matching: bool
    use_empty_matching_result: bool
    support_metric_use_existing_support_type_assessment: bool
    no_eval: bool
    verbose: bool

@dataclass
class InputConfig:
    root_dir: str
    scene_methods: list[str]
    method_use_simple_architecture: list[str]
    scene_mode: str
    scene_range: list[int]
    scene_list: list[int]
    annotation_file: str

@dataclass
class RenderConfig:
    normal_render_tasks: list[str] | None = None
    semantic_render_tasks: list[str] | None = None
    semantic_color_reference: str | None = None

@dataclass
class EvaluationPlan:
    evaluation_cfg: EvaluationConfig
    input_cfg: InputConfig
    render_cfg: RenderConfig
    
    def __post_init__(self):
        self.evaluation_cfg = EvaluationConfig(**self.evaluation_cfg)
        self.input_cfg = InputConfig(**self.input_cfg)
        self.render_cfg = RenderConfig(**self.render_cfg)

# ========================================================================================

def _fetch_scene_state_files(evaluation_plan: EvaluationPlan) -> dict[str, list[pathlib.Path]]:
    """
    Fetch scene state files for the given methods based on the evaluation plan.
    
    Args:
        evaluation_plan: the evaluation plan containing input configurations.
        
    Returns:
        scenes_per_method: a dictionary mapping method names to lists of scene state files.
    """
    
    scenes_per_method = {}
    
    for method in evaluation_plan.input_cfg.scene_methods:
        method_dir = pathlib.Path(evaluation_plan.input_cfg.root_dir) / method
        scene_files = natsorted(list(method_dir.expanduser().resolve().glob("*.json")))
        id_to_file = {int(scene_file.stem.split("_")[-1]): scene_file for scene_file in scene_files}
        
        match evaluation_plan.input_cfg.scene_mode:
            case "all":
                scene_files = scene_files
            case "range":
                scene_range = evaluation_plan.input_cfg.scene_range
                scene_files = [id_to_file[scene_id] for scene_id in range(scene_range[0], scene_range[1]) if scene_id in id_to_file]
            case "list":
                scene_list = evaluation_plan.input_cfg.scene_list
                scene_files = [id_to_file[scene_id] for scene_id in scene_list if scene_id in id_to_file]
                
        scenes_per_method[method] = scene_files
    
    return scenes_per_method
    
def _render_scene(scene: Scene, render_tasks: list[str]) -> None:
    for render_task in render_tasks:
        match render_task:
            case "scene_top":
                scene.blender_scene.render_scene_from_top()
            case "obj_solo":
                scene.blender_scene.render_all_objs_front_solo()
            case "obj_size":
                scene.blender_scene.render_all_objs_front_size_reference()
            case "obj_surroundings":
                scene.blender_scene.render_all_objs_front_surroundings()
            case "obj_global_top":
                scene.blender_scene.render_all_objs_global_top()

def _get_obj_matching(scene: Scene,
                      annotation: Annotation,
                      vlm: BaseVLM,
                      use_existing_matching: bool) -> ObjMatchingResults:
    """
    Get object matching results for the scene and annotation using a VLM.
    
    Args:
        scene: the scene to evaluate
        annotation: the annotation for the scene
        vlm: the VLM to use for object matching
        use_existing_matching: whether to use existing matching results if available
        
    Returns:
        matching_result: the object matching results
    """
    
    # Match object descriptions to target categories in annotation - used by other metrics
    matching_result_file = scene.output_dir / f"obj_matching_result.json"
    
    if use_existing_matching and matching_result_file.exists():
        print("\nUsing existing object matching result...")
        with open(matching_result_file, "r") as f:
            matching_result = ObjMatchingResults.from_dict(json.load(f))
        print("Existing object matching loaded.\n")
    else:
        print("\nCreating new object matching...")
        obj_matching = ObjMatching(scene, annotation, vlm)
        obj_matching_result = obj_matching.run()
        matching_result: ObjMatchingResults = obj_matching_result.data["matching_result"]
        
        # Save the matching result
        with open(matching_result_file, "w") as f:
            json.dump(matching_result.to_dict(), f, indent=4)
    
        print(f"New object matching done. Saved to: {matching_result_file}\n")
        
    return matching_result

# ========================================================================================

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:

    # Set random seeds
    if cfg.seed:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        print(f"\nRandom seed set to: {cfg.seed}\n")
    else:
        print("\nNo random seed set.\n")

    # Localize paths with environment variables
    blender_42_dir = pathlib.Path(os.getenv("BLENDER_42_DIR"))
    cfg.blender.environment_map = str(blender_42_dir / cfg.blender.environment_map)

    # ----------------------------------------------------------------------------------------

    # Load evaluation plan
    evaluation_plan = EvaluationPlan(**cfg.evaluation_plan)
    
    if evaluation_plan.render_cfg.semantic_render_tasks and not evaluation_plan.evaluation_cfg.no_eval:
        input((
            "\nNote:\n"
            "Semantic rendering tasks are set, but evaluation is not skipped.\n"
            "Future renderings from metrics will all be in semantic colors.\n"
            "Press Enter to continue or Ctrl+C to abort.\n"
        ))
    
    # Fetch all scene state files that are to be evaluated
    scenes_per_method = _fetch_scene_state_files(evaluation_plan)
    print(f"\nEvaluating scenes for methods: {list(scenes_per_method.keys())}")
    for method, scene_files in scenes_per_method.items():
        print(f"{method} scenes:")
        [print(f" - {scene_file}") for scene_file in scene_files]
    print()

    # Load annotations
    annotations = Annotations(evaluation_plan.input_cfg.annotation_file)

    # Print the metrics to run
    metrics_to_run = list(evaluation_plan.evaluation_cfg.metrics) if evaluation_plan.evaluation_cfg.metrics else []
    print("\nRunning metrics:")
    [print(f" - {metric}") for metric in metrics_to_run]
    print()
    
    # Load metric configurations
    with open_dict(cfg):
        cfg.metrics.SupportMetric.use_existing_support_type_assessment = evaluation_plan.evaluation_cfg.support_metric_use_existing_support_type_assessment
    metric_configs = MetricRegistry.load_all_configs(cfg.metrics, metrics_to_run)
    
    # Load a mesh retriever for the asset datasets used by the methods
    all_asset_datasets = set()
    for method in evaluation_plan.input_cfg.scene_methods:
        all_asset_datasets.update(cfg.models[method].asset_datasets)
    dataset_cfgs = {asset: cfg.assets[asset] for asset in all_asset_datasets}
    mesh_retriever = Retriever(dataset_cfgs)
    
    # Load a VLM for object matching and other tasks
    vlm_config = getattr(cfg.vlms, evaluation_plan.evaluation_cfg.vlm)
    if cfg.seed:
        with open_dict(vlm_config):
            vlm_config.seed = cfg.seed
    vlm = VLMRegistry.instantiate_vlm(evaluation_plan.evaluation_cfg.vlm, vlm_config)

    # ----------------------------------------------------------------------------------------
    
    # Load Blender and scene configurations
    scene_cfg = SceneConfig(**cfg.scene)
    blender_cfg = BlenderConfig(**cfg.blender)
    trimesh_cfg = TrimeshConfig(**cfg.trimesh)
    
    # Load scene states and do the evaluations
    for method in scenes_per_method.keys():
        
        print(f"Evaluting: {method} - {len(scenes_per_method[method])} scenes")
        
        # Adjust whether to use simple architecture for all scenes of the method based on the evaluation plan
        scene_cfg.use_simple_architecture = method in evaluation_plan.input_cfg.method_use_simple_architecture
        
        # Evaluate each scene for the method
        method_scene_files = scenes_per_method[method]
        for i, method_scene_file in enumerate(method_scene_files):

            print(f"--- {method} ({i+1}/{len(method_scene_files)}) --- {method_scene_file}\n")
            print(f"*** Load scene with simple architecture? -> {scene_cfg.use_simple_architecture} ***\n")

            # ---------------------------------------------------
            
            # Load scene state, if it fails, load an empty scene
            try:
                scene_state = SceneState(method_scene_file)
            except Exception as e:
                print(f"Error loading scene: {method_scene_file} - error: {e}")
                scene_state = SceneState(pathlib.Path("./input/empty_scene.json")) # TODO: Make this a config
                if hasattr(method_scene_file, "stem"):
                    scene_state.name = method_scene_file.stem
            
            # Create the output directory
            output_dir: pathlib.Path = pathlib.Path(evaluation_plan.evaluation_cfg.output_dir) / method / scene_state.name
            output_dir.mkdir(parents=True, exist_ok=True)
                
            # Create the scene with output directory
            scene = Scene(mesh_retriever, scene_state, scene_cfg, blender_cfg, trimesh_cfg, output_dir)
            
            # ---------------------------------------------------
            
            # Save the Blender scene for reference if configured
            if evaluation_plan.evaluation_cfg.save_blend_file:
                scene.blender_scene.save_blend()
            
            # ---------------------------------------------------
            
            # Render as requested
            if evaluation_plan.render_cfg.normal_render_tasks:
                _render_scene(scene, evaluation_plan.render_cfg.normal_render_tasks)
                
            # If no_eval and not semantic_render, can skip the rest
            if evaluation_plan.evaluation_cfg.no_eval and not evaluation_plan.render_cfg.semantic_render_tasks:
                continue
            
            # ---------------------------------------------------
            
            # Get the corresponding annotation
            scene_file_id = scene_state.name.split("_")[-1]
            annotation = annotations[int(scene_file_id)]
            
            # ---------------------------------------------------
            
            # Use empty matching result if configured
            # Useful if only running metrics that do not require object matching (e.g., collision)
            if evaluation_plan.evaluation_cfg.use_empty_matching_result:
                print("Using empty matching result as configured.")
                matching_result = ObjMatchingResults(per_category={}, not_matched_objs=[], actual_categories={})
            else:
                matching_result = _get_obj_matching(scene, annotation, vlm, evaluation_plan.evaluation_cfg.use_existing_matching)
            
            print("Using object matching:")
            for category, obj_ids in matching_result.per_category.items():
                print(f" - {category}: {obj_ids}")
            print()
            
            # ---------------------------------------------------
            
            # Apply semantic colors, render, and save the scene
            if evaluation_plan.render_cfg.semantic_render_tasks:
                color_reference_path = pathlib.Path(evaluation_plan.render_cfg.semantic_color_reference.replace("*", scene_file_id)) if evaluation_plan.render_cfg.semantic_color_reference else None
                apply_semantic_colors(scene, matching_result, vlm, color_reference_path)
                _render_scene(scene, evaluation_plan.render_cfg.semantic_render_tasks)
                if evaluation_plan.evaluation_cfg.save_blend_file:
                    scene.blender_scene.save_blend("scene_semantic_colors.blend")
                
            # ---------------------------------------------------

            # Skip evaluation if no_eval is set
            if evaluation_plan.evaluation_cfg.no_eval:
                continue
                
            # Initialize output json
            output_json = {
                "method": method,
                "scene_id": scene_file_id,
                "description": annotation.description,
                "obj_ids": scene.get_obj_ids(),
                "object_descriptions": [scene.obj_descriptions[obj_id] for obj_id in scene.get_obj_ids()],
                "object_matching_per_category": matching_result.per_category,
                "not_matched_objects": matching_result.not_matched_objs,
                "metrics": metrics_to_run,
                "results": {}
            }

            # Run metrics
            common_metric_params = {
                "scene": scene,
                "annotation": annotation,
                "vlm": vlm,
                "matching_result": matching_result,
                "output_dir": output_dir
            }
            for metric_name in metrics_to_run:
                print(f"----- Running metric: {metric_name}")
                metric_instance = MetricRegistry.instantiate_metric(metric_name, metric_configs, **common_metric_params)
                
                result = metric_instance.run(evaluation_plan.evaluation_cfg.verbose)
                output_json["results"][metric_name] = {
                    "message": result.message,
                    "data": result.data
                }
        
                # Save results up to this point
                output_file = output_dir / f"eval_result.json"
                with open(output_file, "w") as f:
                    json.dump(output_json, f, indent=4)
                    
                print()

            print(f"All done. Results saved to: {output_file}\n")

if __name__ == "__main__":
    main()
