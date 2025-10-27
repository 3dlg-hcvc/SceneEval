import sys
import argparse
import subprocess
import shutil
import gc
import gzip
import bpy
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted

# ====================================================================================== Download Dataset

def _check_subprocess_return_code(return_code: int) -> None:
    """
    Check the return code of a subprocess call and handle it accordingly.
    
    Args:
        return_code: the return code from the subprocess call
    """
    
    if return_code != 0 and return_code != -11:  # -11 is a common return code for Blender subprocesses (Segfault)
        choice = input("\nIssue occurred while running the previous step. Do you want to continue? (y/n): ").strip().lower()
        if choice != 'y':
            print("Exiting ...\n")
            sys.exit(1)
        else:
            print("Continuing despite the issue...\n")

def install_packages() -> None:
    """
    Install the required packages for downloading the dataset used by Holodeck.
    """
    
    print("\n*** Installing 'objathor' and 'attrs' ***\n")
    finished_process = subprocess.run([sys.executable, "-m",
                                       "pip", "install", "objathor", "attrs"])
    _check_subprocess_return_code(finished_process.returncode)
    print("\n > Finished installing 'objathor' and 'attrs'.\n")
    
    print("\n*** Installing ai2thor (specific version as per Holodeck requirements) ***\n")
    finished_process = subprocess.run([sys.executable, "-m",
                                       "pip", "install", "--extra-index-url", "https://ai2thor-pypi.allenai.org",
                                       "ai2thor==0+8524eadda94df0ab2dbb2ef5a577e4d37c712897"])
    
    _check_subprocess_return_code(finished_process.returncode)
    print("\n > Finished installing ai2thor.\n")

def download_objathor(data_dir: Path) -> None:
    """
    Download the objathor dataset to the specified root directory.
    
    Args:
        data_dir: the directory where the dataset will be downloaded and extracted
    """
    
    # Remove the existing download directory if it exists (usually indicates a failed download)
    existing_download_dir = data_dir / "2023_09_23"
    if existing_download_dir.is_dir():
        shutil.rmtree(existing_download_dir, ignore_errors=True)
        print(f"\nRemoved existing data directory: {existing_download_dir}. (This is usually due to a failed download.)\n")
    
    print("\n*** Downloading objathor assets ***\n")
    finished_process = subprocess.run([sys.executable, "-m",
                                       "objathor.dataset.download_assets",
                                       "--version", "2023_09_23",
                                       "--path", str(data_dir)])
    _check_subprocess_return_code(finished_process.returncode)
    print("\n > Finished downloading objathor assets.\n")
    
    print("\n*** Downloading objathor annotations ***\n")
    finished_process = subprocess.run([sys.executable, "-m",
                                       "objathor.dataset.download_annotations",
                                       "--version", "2023_09_23",
                                       "--path", str(data_dir)])
    _check_subprocess_return_code(finished_process.returncode)
    print("\n > Finished downloading objathor annotations.\n")

def fix_directory_structure(data_dir: Path, dataset_dir_name: str = "objathor-assets") -> None:
    """
    Fix the directory structure of the downloaded objathor dataset to what SceneEval expects.
    
    Args:
        data_dir: the directory where the dataset was downloaded and extracted
        dataset_dir_name: the name of the directory where the assets will be stored
    """
    
    print("\n*** Fixing objathor directory structure ***\n")
    
    # The directory where the data was downloaded is named with the version Holodeck uses
    downloaded_dir = data_dir / "2023_09_23"
    
    # Remove the lock files
    for lock_file in downloaded_dir.glob("*.lock"):
        lock_file.unlink()
    
    # Extract the annotations.json.gz file to annotations.json then remove the .gz file
    annotations_gz = downloaded_dir / "annotations.json.gz"
    with gzip.open(annotations_gz, "rb") as f_in:
        with open(downloaded_dir / "annotations.json", "wb") as f_out:
            f_out.write(f_in.read())
    annotations_gz.unlink()
    
    # Move the annotations.json file to under the "assets" directory
    annotations_file = downloaded_dir / "annotations.json"
    assets_dir = downloaded_dir / "assets"
    annotations_file.rename(assets_dir / "annotations.json")
    
    # Rename the "assets" directory
    objathor_assets_dir = data_dir / dataset_dir_name
    assets_dir.rename(objathor_assets_dir)
    
    # Remove the downloaded directory
    shutil.rmtree(downloaded_dir, ignore_errors=True)
    
    # Remove all ._* files (macOS metadata files) throughout the objathor assets directory
    for file in objathor_assets_dir.glob("**/._*"):
        file.unlink()
    
    print("\n > Finished fixing objathor directory structure.\n")

# ====================================================================================== Asset Extraction

def _cleanup_blender_data() -> None:
    """
    Clean up Blender data blocks to prevent memory accumulation.
    This function removes orphaned meshes, materials, images, and other data blocks
    that are no longer referenced by any objects in the scene.
    """
    
    # Remove all orphaned data blocks
    for collection_name in ["meshes", "materials", "images", "textures", "node_groups"]:
        collection = getattr(bpy.data, collection_name)
        for item in collection:
            if item.users == 0:
                collection.remove(item)
    
    # Force garbage collection in Blender
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    
    # Reset the scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Garbage collection to free up memory
    gc.collect()
    
def _extract_pkl_gz_asset(file_path: Path) -> bpy.types.Object:
    """
    Extract an object from a .pkl.gz file and assign the textures to the object in Blender.
    
    Args:
        file_path: the path to the .pkl.gz file
    
    Returns:
        obj: the object in Blender
    """
    
    # Reset the scene
    _cleanup_blender_data()
    
    # Load the object data from the .pkl.gz file
    with gzip.open(file_path, "rb") as f:
        obj_data = pickle.load(f)
    
    # ----------------- Mesh -----------------
     
    mesh = bpy.data.meshes.new(name="obj")
    obj = bpy.data.objects.new("obj", mesh)
    bpy.context.collection.objects.link(obj)
    
    triangles = np.array(obj_data["triangles"]).reshape((-1, 3))
    vertices = [[v["x"], v["z"], v["y"]] for v in obj_data["vertices"]]
    
    mesh.from_pydata(vertices, [], triangles)
    mesh.update()
    
    # ----------------- UV -----------------
    
    uvs = [[uv["x"], uv["y"]] for uv in obj_data["uvs"]]
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")
    
    uv_layer = mesh.uv_layers["UVMap"]
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            loop = mesh.loops[loop_index]
            uv = uvs[loop.vertex_index]
            uv_layer.data[loop_index].uv = uv
            
    mesh.update()
            
    # ----------------- Material -----------------
    
    material = bpy.data.materials.new(name="AlbedoMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    
    principled_bsdf = nodes.get("Principled BSDF")
    normal_map = nodes.new("ShaderNodeNormalMap")
    
    # Texture
    texture_node = nodes.new("ShaderNodeTexImage")
    texture_node.image = bpy.data.images.load(str(file_path.parent / "albedo.jpg"))
    material.node_tree.links.new(texture_node.outputs["Color"], principled_bsdf.inputs["Base Color"])
    
    # Normal map
    normal_texture_node = nodes.new("ShaderNodeTexImage")
    normal_texture_node.image = bpy.data.images.load(str(file_path.parent / "normal.jpg"))
    normal_texture_node.image.colorspace_settings.name = "Non-Color"
    material.node_tree.links.new(normal_texture_node.outputs["Color"], normal_map.inputs["Color"])
    material.node_tree.links.new(normal_map.outputs["Normal"], principled_bsdf.inputs["Normal"])
    
    # Emission
    emission_texture_node = nodes.new("ShaderNodeTexImage")
    emission_texture_node.image = bpy.data.images.load(str(file_path.parent / "emission.jpg"))
    material.node_tree.links.new(emission_texture_node.outputs["Color"], principled_bsdf.inputs["Emission Color"])
    principled_bsdf.inputs["Emission Strength"].default_value = 1.0
    
    obj.data.materials.append(material)
    
    mesh.update()
    
    # ----------------- Rotations -----------------
    
    rotation_angle = np.deg2rad(-obj_data["yRotOffset"] + 180) # +180 to face -Y as forward in Blender
    obj.rotation_euler = (0, 0, rotation_angle)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    
    return obj

def process_assets(objathor_dir: Path, start_idx: int = None, end_idx: int = None, process_id: int = None) -> None:
    """
    Process assets in the specified directory within a given range, extracting them from .pkl.gz files and exporting them as .glb files.
    
    Args:
        objathor_dir: the directory containing the objathor assets
        start_idx: start index of the range to process (inclusive)
        end_idx: end index of the range to process (exclusive)
        process_id: ID of the process (for logging purposes, can be None)
    """
    
    asset_dirs = natsorted([d for d in objathor_dir.glob("*") if d.is_dir()])
    
    # Apply range filtering if specified
    if start_idx is not None and end_idx is not None:
        asset_dirs = asset_dirs[start_idx:end_idx]
        print(f"\n*** Processing objathor assets [{start_idx}:{end_idx}] ({len(asset_dirs)} assets) ***\n")
    else:
        print(f"\n*** Processing objathor assets (all {len(asset_dirs)} assets) ***\n")
    
    for data_dir in tqdm(asset_dirs, desc=f"{('P' + str(process_id)) if process_id is not None else ''}"):
        file_path = data_dir / f"{data_dir.stem}.pkl.gz"
        output_path = data_dir / f"{data_dir.stem}.glb"
        
        if not file_path.is_file():
            print(f"File not found: {file_path}")
            continue
        
        obj = _extract_pkl_gz_asset(file_path)
        bpy.ops.export_scene.gltf(filepath=str(output_path), export_format="GLB", use_selection=False)
        
    print(f"\n > Finished processing objathor assets range [{start_idx}:{end_idx}).\n" if start_idx is not None else "\n > Finished processing all objathor assets.\n")

def process_assets_parallel(objathor_dir: Path, num_processes: int) -> None:
    """
    Process assets in parallel using multiple subprocess workers.
    
    Args:
        objathor_dir: the directory containing the objathor assets
        num_processes: number of parallel processes to use
    """
    
    asset_dirs = natsorted([d for d in objathor_dir.glob("*") if d.is_dir()])
    total_num_assets = len(asset_dirs)
    assets_per_process = total_num_assets // num_processes + (total_num_assets % num_processes > 0)
    
    print(f"\n*** Processing objathor assets in parallel using {num_processes} processes, each processing {assets_per_process} assets ***\n")
    
    # Calculate ranges for each process
    processes = []
    for i in range(num_processes):
        
        # Calculate start and end indices for this process
        start_idx = i * assets_per_process
        end_idx = min((i + 1) * assets_per_process, total_num_assets)
        print(f"Starting process {i}: assets [{start_idx}:{end_idx}]")
        
        # Create subprocess command
        cmd = [
            sys.executable, __file__,
            "--data-dir", str(objathor_dir.parent),
            "--dataset-dir-name", objathor_dir.name,
            "--num-processes", str(num_processes),
            "--subprocess-id", str(i),
            "--subprocess-assets-per-process", str(assets_per_process),
            "--subprocess-total-num-assets", str(total_num_assets)
        ]
        
        process = subprocess.Popen(cmd)
        processes.append(process)
    
    # Wait for all processes to complete
    return_codes = []
    for i, process in enumerate(processes):
        return_code = process.wait()
        return_codes.append(return_code)
    
    # Check return codes of all processes
    for i, return_code in enumerate(return_codes):
        print(f"Process {i} completed with return code: {return_code}")
        _check_subprocess_return_code(return_code)
    
    # Check if all assets were processed
    num_glb_files = len(list(objathor_dir.glob("**/*.glb")))
    print(f"\n > Finished parallel processing objathor assets. Total assets processed: {num_glb_files} out of {total_num_assets}.\n")
    if num_glb_files < total_num_assets:
        print(f"Warning: Not all assets were processed. Check for errors in the subprocess logs.\n")

# ======================================================================================

def main(data_dir: Path, dataset_dir_name: str, num_processes: int):
    """
    Download and process the objathor dataset for SceneEval.
    
    Args:
        data_dir: the directory where the dataset will be downloaded and extracted
        dataset_dir_name: the name of the directory where the objathor assets will be stored
        num_processes: number of parallel processes to use for asset extraction
    """
    
    # Download the dataset
    install_packages()
    download_objathor(data_dir)
    fix_directory_structure(data_dir, dataset_dir_name)
    
    # Process the assets
    objathor_dir = data_dir / dataset_dir_name
    process_assets_parallel(objathor_dir, num_processes)

def subprocess_main(data_dir: Path, dataset_dir_name: str, process_id: int, assets_per_process: int, total_num_assets: int):
    """
    Entry point for the subprocess to process a range of assets.
    
    Args:
        data_dir: the directory where the dataset is stored
        dataset_dir_name: the name of the directory where the objathor assets are stored
        process_id: the ID of this process (used to calculate range)
        assets_per_process: number of assets to process per subprocess
        total_num_assets: total number of assets to process
    """
    
    objathor_dir = data_dir / dataset_dir_name
    start_idx = process_id * assets_per_process
    end_idx = min((process_id + 1) * assets_per_process, total_num_assets)
    
    process_assets(objathor_dir, start_idx, end_idx, process_id)

# ======================================================================================

if __name__ == "__main__":
    DEFAULT_DATA_DIR = "_data"
    DEFAULT_DATASET_DIR_NAME = "objathor-assets"
    
    parser = argparse.ArgumentParser(description="Download and process objathor dataset for SceneEval.")
    parser.add_argument("--data-dir", type=Path, default=Path("./_data"), help="Directory to store the dataset.")
    parser.add_argument("--dataset-dir-name", type=str, default="objathor-assets", help="Name of the directory where the objathor assets will be stored.")
    parser.add_argument("--num-processes", type=int, default=5, help="Number of parallel processes to use for asset extraction.")
    parser.add_argument("--subprocess-id", type=int, help="Process ID for subprocess range processing. (Do not use manually, this is set by the script itself)")
    parser.add_argument("--subprocess-assets-per-process", type=int, help="Number of assets to process per subprocess (Do not use manually, this is set by the script itself)")
    parser.add_argument("--subprocess-total-num-assets", type=int, help="Total number of assets to process (Do not use manually, this is set by the script itself)")
    args = parser.parse_args()
    
    # -----------------------------------------------------------------------------------
    # Subprocess flow
    
    # Check if this is a subprocess call for range processing
    if args.subprocess_id is not None:
        print(f"Starting subprocess {args.subprocess_id} for range processing...")
        subprocess_main(args.data_dir, args.dataset_dir_name, args.subprocess_id, args.subprocess_assets_per_process, args.subprocess_total_num_assets)
        sys.exit(0)
        
    # -----------------------------------------------------------------------------------
    # Main process flow
    
    INFO = f"""
    This scripts downloads and processes the objathor dataset for use in SceneEval.
    It installs the required packages, downloads the dataset, fixes the directory structure, and extracts the assets.
    Make sure you have sufficient disk space (Peak size: ~50 GB) and a stable internet connection before running this script.
    
    The dataset will be stored in '{args.data_dir}' under the directory named '{args.dataset_dir_name}'.
    You are using {args.num_processes} parallel processes for asset extraction (Recommended: >= 5).
    
    Continue with the download and extraction? (y/n): """
    
    choice = input(INFO).strip().lower()
    if choice != 'y':
        print("Exiting ...\n")
        sys.exit(0)
    
    main(data_dir=args.data_dir, dataset_dir_name=args.dataset_dir_name, num_processes=args.num_processes)

    print("\n > Successfully downloaded and processed the objathor dataset for SceneEval.\n")
    if str(args.data_dir) != DEFAULT_DATA_DIR:
        print(f"Warning: You specified a custom data directory: {args.data_dir}. You may need to adjust configs accordingly in SceneEval.")
    if args.dataset_dir_name != DEFAULT_DATASET_DIR_NAME:
        print(f"Warning: You specified a custom dataset directory name: {args.dataset_dir_name}. You may need to adjust configs accordingly in SceneEval.")
