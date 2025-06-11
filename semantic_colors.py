import json
import pathlib
import colorsys
import bpy
import numpy as np
from pydantic import BaseModel
from tqdm import tqdm
from vlm import GPT
from scenes import Scene
from metrics.obj_matching import ObjMatchingResults

def _compute_pairwise_distances(scene: Scene) -> dict[str, dict[str, float]]:
    """
    Compute pairwise distances between bounding boxes of objects in the scene.
    
    Args:
        scene: the scene
        
    Returns:
        pairwise_distances: a dictionary of pairwise distances between objects
    """
    
    pairwise_distances = {obj_id: {} for obj_id in scene.t_objs.keys()}
    
    t_objs = scene.t_objs
    for obj_id_1, t_obj_1 in tqdm(t_objs.items(), desc="Computing pairwise distances", leave=True):
        for obj_id_2, t_obj_2 in tqdm(t_objs.items(), position=1, leave=False):
            if obj_id_1 == obj_id_2:
                continue
            
            # Get the bounding boxes minimum and maximum vertices
            bbox_1 = t_obj_1.bounding_box.vertices
            bbox_2 = t_obj_2.bounding_box.vertices
            bbox_1_min = np.min(bbox_1, axis=0)
            bbox_1_max = np.max(bbox_1, axis=0)
            bbox_2_min = np.min(bbox_2, axis=0)
            bbox_2_max = np.max(bbox_2, axis=0)
            
            # Compute the distance between the bounding boxes
            dx = max(0, max(bbox_1_min[0] - bbox_2_max[0], bbox_2_min[0] - bbox_1_max[0]))
            dy = max(0, max(bbox_1_min[1] - bbox_2_max[1], bbox_2_min[1] - bbox_1_max[1]))
            dz = max(0, max(bbox_1_min[2] - bbox_2_max[2], bbox_2_min[2] - bbox_1_max[2]))
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            
            pairwise_distances[obj_id_1][obj_id_2] = distance
            pairwise_distances[obj_id_2][obj_id_1] = distance
            
    return pairwise_distances
    
def _get_semantic_color(idx: int) -> tuple[float]:
    """
    Get the semantic color for the given index in RGB.

    Args:
        idx: the index
    
    Returns:
        r: the red value
        g: the green value
        b: the blue value
    """

    # Hue
    h = (-1.88 * idx) % (2 * np.pi)
    if h < 0:
        h += 2 * np.pi
    h /= 2 * np.pi
    
    # Saturation
    s = 0.8 + 0.2 * np.sin(0.1 * idx)
    
    # Lightness
    L_VALUES = [0.5, 0.6, 0.45, 0.55, 0.35, 0.4]
    l = L_VALUES[idx % len(L_VALUES)]
    
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    
    return r, g, b

def _get_obj_bbox_volume(b_obj: bpy.types.Object) -> float:
    """
    Compute the bounding box volume of a Blender object.

    Args:
        b_obj: the Blender object
    
    Returns:
        volume: the bounding box volume of the object
    """
    
    # Collect all vertices of the object and its children
    all_vertices = []
    if b_obj.type == "MESH":
        vertices = [b_obj.matrix_world @ vert.co for vert in b_obj.data.vertices]
        all_vertices.extend(vertices)
    
    for b_obj_child in b_obj.children_recursive:
        if b_obj_child.type == "MESH":
            vertices = [b_obj_child.matrix_world @ vert.co for vert in b_obj_child.data.vertices]
            all_vertices.extend(vertices)
    
    # Compute the bounding box volume
    all_vertices = np.asarray(all_vertices)
    min_corner = np.min(all_vertices, axis=0)
    max_corner = np.max(all_vertices, axis=0)
    volume = np.prod(max_corner - min_corner)
    
    return volume

def _get_color_idx_from_obj_volume(volume: float) -> int:
    """
    Get the color index from the object volume.

    Args:
        volume: the volume of the object
    
    Returns:
        idx: the color index
    """
    
    # Volume steps in cubic meters
    VOLUME_STEPS = [
        1.25e-4,    # 120 cm^3 (5cm cube)
        1e-3,       # 1000 cm^3 (10cm cube)
        3.375e-3,   # 3375 cm^3 (15cm cube)
        8e-3,       # 8000 cm^3 (20cm cube)
        1.5625e-2,  # 15625 cm^3 (25cm cube)
        2.7e-2,     # 27000 cm^3 (30cm cube)
        4.2875e-2,  # 42875 cm^3 (35cm cube)
        6.4e-2,     # 64000 cm^3 (40cm cube)
        9.1125e-2,  # 91125 cm^3 (45cm cube)
        1.25e-1,    # 0.125 m^3 (50cm cube)
        2.16e-1,    # 0.216 m^3 (60cm cube)
        3.43e-1,    # 0.343 m^3 (70cm cube)
        5.12e-1,    # 0.512 m^3 (80cm cube)
        7.29e-1,    # 0.729 m^3 (90cm cube)
        1.0,        # 1.0 m^3 (1m cube)
        1.953125,   # 1.953125 m^3 (1.25m cube)
        3.375,      # 3.375 m^3 (1.5m cube)
        5.359375,   # 5.359375 m^3 (1.75m cube)
        8,          # 8 m^3 (2m cube)
        15.625,     # 15.625 m^3 (2.5m cube)
        27,         # 27 m^3 (3m cube)
    ]
    
    # Get the color index based on the volume
    for i, volume_step in enumerate(VOLUME_STEPS):
        if volume < volume_step:
            return i
    else:
        return len(VOLUME_STEPS) - 1

def _apply_color_to_b_obj(b_obj: bpy.types.Object, r: float, g: float, b: float, material_name: str) -> None:
    """
    Apply a RGB color to a Blender object.

    Args:
        b_obj: the Blender object
        r: the red value
        g: the green value
        b: the blue value
        material_name: the name of the material to apply
    """
    
    # Create a new material with the color
    new_material = bpy.data.materials.new(name=material_name)
    new_material.use_nodes = True
    new_material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (r, g, b, 1)
    
    # Apply the material to the object and its children
    b_obj.active_material = new_material
    for b_obj_child in b_obj.children_recursive:
        b_obj_child.active_material = new_material

# ==============================================================================================================

class CategoryMatchingAssessment(BaseModel):
    input_category: str
    matched: bool
    matched_reference_category: str
    reason: str
    
class CategoryMatchingResponseFormat(BaseModel):
    assessments: list[CategoryMatchingAssessment]

def apply_semantic_colors(scene: Scene,
                          matching_result: ObjMatchingResults,
                          vlm: GPT,
                          color_reference_path: pathlib.Path = None) -> None:
    """
    Apply semantic colors to all objects in a Blender scene based on the object matching result.
    This is irreversible.
    
    Args:
        scene: the scene
        matching_result: the object matching result
        vlm: the VLM to use for category matching
        color_reference_path: the path to the color reference JSON file
    """
    
    obj_bbox_volumes = {obj_id: _get_obj_bbox_volume(b_obj) for obj_id, b_obj in scene.b_objs.items()}
    
    pairwise_distances = _compute_pairwise_distances(scene)
    
    # Initialize the color choices
    color_choices = {category: -1 for category in matching_result.actual_categories.values()}

    # Group objects by category
    category_to_obj_ids = {}
    for obj_id, actual_category in matching_result.actual_categories.items():
        category_to_obj_ids.setdefault(actual_category, []).append(obj_id)
    
    # New run, no color reference
    if color_reference_path is None:
        
        print("No color reference available, assigning colors from scratch ...")

        # Assign colors to each category based on the maximum bounding box volume within the category
        for category, obj_ids in category_to_obj_ids.items():
            max_bbox_volume = max([obj_bbox_volumes[obj_id] for obj_id in obj_ids])
            
            all_close_to_obj_ids = set()
            for obj_id in obj_ids:
                close_to_obj_ids = set([obj_id_ for obj_id_, distance in pairwise_distances[obj_id].items() if distance < 1])
                all_close_to_obj_ids |= close_to_obj_ids
            all_close_to_categories = set([matching_result.actual_categories[obj_id] for obj_id in all_close_to_obj_ids])
            avoid_color_idxs = set([color_choices[category_] for category_ in all_close_to_categories if color_choices[category_] != -1])
            
            color_idx = _get_color_idx_from_obj_volume(max_bbox_volume)
            while color_idx in avoid_color_idxs:
                color_idx += 1
            
            # Get the RGB color from the color index
            color_choices[category] = color_idx
            r, g, b = _get_semantic_color(color_idx)
            
            # Apply the same color to all objects in the same category
            for obj_id in obj_ids:
                obj_id_num = obj_id.split("_")[0][3:]
                b_obj = scene.b_objs[obj_id]
                _apply_color_to_b_obj(b_obj, r, g, b, f"semantic_material_{obj_id_num}")
    
    # Color reference is available
    else:
        
        print(f"Color reference available at {color_reference_path}, using it to assign colors ...")
        
        # Load the color reference
        with open(color_reference_path, "r") as f:
            color_reference = json.load(f)
        
        # First match the categories to the ones in the color reference
        # This takes care of the cases where similar categories are named differently
        vlm.reset()
        prompt_info = {
            "reference_categories": str(list(color_reference.keys())),
            "input_categories": str(list(category_to_obj_ids.keys()))
        }
        response: CategoryMatchingResponseFormat | str = vlm.send("category_matching_for_semantic_color",
                                                                  prompt_info=prompt_info,
                                                                  response_format=CategoryMatchingResponseFormat)
        if isinstance(response, str):
            raise ValueError(f"The response is not in the expected format: {response}")

        # Extract the matched categories
        matched_categories = {assessment.input_category: assessment.matched_reference_category
                              for assessment in response.assessments if assessment.matched}

        # Color objects based on the categories
        remaining_categories = set()
        for category, obj_ids in category_to_obj_ids.items():
            
            # If the category exists exactly in the color reference, assign the same color
            if category in color_reference:
                color_idx = color_reference[category]
            else:
                # If the category is matched to a reference category, assign the same color
                if category in matched_categories:
                    color_idx = color_reference[matched_categories[category]]
                
                # If the category is not matched, assign a new color based on the maximum bounding box volume
                else:
                    remaining_categories.add(category)
                    continue
            
            # Get the RGB color from the color index
            color_choices[category] = color_idx
            r, g, b = _get_semantic_color(color_idx)
            
            # Apply the same color to all objects in the same category
            for obj_id in obj_ids:
                obj_id_num = obj_id.split("_")[0][3:]
                b_obj = scene.b_objs[obj_id]
                _apply_color_to_b_obj(b_obj, r, g, b, f"semantic_material_{obj_id_num}")

        # Now assign colors to the remaining categories
        for category in remaining_categories:
            max_bbox_volume = max([obj_bbox_volumes[obj_id] for obj_id in category_to_obj_ids[category]])
            
            all_close_to_obj_ids = set()
            for obj_id in category_to_obj_ids[category]:
                close_to_obj_ids = set([obj_id_ for obj_id_, distance in pairwise_distances[obj_id].items() if distance < 1])
                all_close_to_obj_ids |= close_to_obj_ids
            all_close_to_categories = set([matching_result.actual_categories[obj_id] for obj_id in all_close_to_obj_ids])
            avoid_color_idxs = set([color_choices[category_] for category_ in all_close_to_categories if color_choices[category_] != -1])
            
            color_idx = _get_color_idx_from_obj_volume(max_bbox_volume)
            while color_idx in avoid_color_idxs:
                color_idx += 1
            
            # Get the RGB color from the color index
            color_choices[category] = color_idx
            r, g, b = _get_semantic_color(color_idx)
            
            # Apply the same color to all objects in the same category
            for obj_id in category_to_obj_ids[category]:
                obj_id_num = obj_id.split("_")[0][3:]
                b_obj = scene.b_objs[obj_id]
                _apply_color_to_b_obj(b_obj, r, g, b, f"semantic_material_{obj_id_num}")
    
    # Turn on the flag so that future renders are in a separate folder
    scene.blender_scene.applied_semantic_colors = True
    
    # Save the category-color reference
    output_dir = scene.output_dir
    with open(output_dir / "semantic_color_reference.json", "w") as f:
        json.dump(color_choices, f)
    print(f"Saved the semantic color reference to {output_dir / 'semantic_color_reference.json'}\n")
