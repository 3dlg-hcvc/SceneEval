import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted

def check_gltf_transform_installed():
    """
    Check if gltf-transform is installed.
    """
    
    try:
        subprocess.run(["gltf-transform", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_ktx_software_installed():
    """
    Check if ktx software is installed.
    """
    
    try:
        subprocess.run(["ktx", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_git_lfs():
    """
    Install Git LFS if not already installed.
    """
    
    try:
        subprocess.run(["git", "lfs", "install"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error installing Git LFS:", e)
        sys.exit(1)

def download_hssd(destination_path: Path, clone_method: str = "https"):
    """
    Download the HSSD dataset from HuggingFace.

    Args:
        destination_path: the path where the dataset should be downloaded.
        clone_method: the method to use for cloning (ssh or https).
    """
    
    if clone_method == "ssh":
        clone_url = "git@hf.co:datasets/hssd/hssd-models"
    else:
        clone_url = "https://huggingface.co/datasets/hssd/hssd-models"
    
    save_path = destination_path / "hssd-models"

    try:
        subprocess.run(["git", "clone", clone_url, str(save_path)], check=True)
    except subprocess.CalledProcessError as e:
        print("Error cloning HSSD models:", e)
        sys.exit(1)
    
    # Move relevant files to destination path and clean up
    shutil.move(save_path / "objects", destination_path / "glb")
    shutil.rmtree(save_path, ignore_errors=True)

def download_hssd_decomposed(destination_path: Path, clone_method: str = "https"):
    """
    Download the HSSD decomposed dataset from HuggingFace.

    Args:
        destination_path: the path where the decomposed dataset should be downloaded.
        clone_method: the method to use for cloning (ssh or https).
    """
    
    if clone_method == "ssh":
        clone_url = "git@hf.co:datasets/hssd/hssd-hab.git"
    else:
        clone_url = "https://huggingface.co/datasets/hssd/hssd-hab"

    save_path = destination_path / "hssd-hab"
    
    try:
        subprocess.run(["git", "clone", clone_url, str(save_path)], check=True)
    except subprocess.CalledProcessError as e:
        print("Error cloning HSSD models (decomposed):", e)
        sys.exit(1)
    
    # Move relevant files to destination path and clean up
    shutil.move(save_path / "metadata" / "fpmodels-with-decomposed.csv", destination_path / "fpmodels.csv")
    shutil.move(save_path / "objects" / "decomposed", destination_path / "decomposed")
    shutil.rmtree(save_path, ignore_errors=True)
    
    # Remove non-asset files
    for item in destination_path.rglob("*"):
        if item.is_file() and item.name.count(".") > 1:
            item.unlink()

def decompress_ktx2_files(download_destination_path: Path, remove_original: bool = True):
    """
    Decompress all downloaded assets to not use KHR_texture_basisu compression.
    This is required as Blender does not support loading assets with the KHR_texture_basisu extension.

    Args:
        download_destination_path: the path where the dataset is stored.
        remove_original: whether to remove the original .ktx2 files after decompression.
    """

    glb_files = natsorted(list(download_destination_path.rglob("*.glb")))
    for glb_file in tqdm(glb_files, desc="Decompressing .glb files"):
        print()
        file_name = glb_file.name
        temp_output_path = glb_file.parent / f"{glb_file.stem}_decompressed.glb"
        subprocess.run(["gltf-transform", "ktxdecompress", str(glb_file), str(temp_output_path)], check=True)
        if remove_original:
            glb_file.unlink()
        else:
            glb_file.rename(glb_file.parent / f"{file_name}_original.glb")
        temp_output_path.rename(glb_file)

if __name__ == "__main__":
    DEFAULT_DATA_DIR = "_data"
    DEFAULT_DATASET_DIR_NAME = "hssd"
    
    parser = argparse.ArgumentParser(description="Download HSSD dataset for SceneEval.")
    parser.add_argument("--data-dir", type=Path, default=Path("./_data"), help="Directory to store the dataset.")
    parser.add_argument("--dataset-dir-name", type=str, default="hssd", help="Name of the directory where the hssd assets will be stored.")
    parser.add_argument("--clone-method", type=str, choices=["ssh", "https"], default="https", help="Method to clone from HuggingFace (ssh or https).")
    parser.add_argument("--glb-decompress-keep-original", action="store_true", help="Whether to keep original .glb files after decompressing KHR_texture_basisu compression.")
    args = parser.parse_args()
    
    INFO = f"""
    This scripts downloads the HSSD dataset for use in SceneEval.
    It installs Git LFS if not already installed, downloads the dataset, and fixes the directory structure.
    
    Prerequisites:
    1. Agree to the HSSD dataset license on HuggingFace:
       https://huggingface.co/datasets/hssd/hssd-models
       
    2. Set up to clone repos from HuggingFace using SSH or HTTPS:
      - For SSH: set up SSH keys on your machine and add the public key to your HuggingFace account
        - See https://huggingface.co/docs/hub/en/security-git-ssh
      - For HTTPS: prepare to enter your HuggingFace access token with write permissions when prompted
        - See https://huggingface.co/docs/hub/en/security-tokens
    
    3. Ensure that both 'gltf-transform' and 'ktx' software are installed and accessible in your PATH.
      - gltf-transform via npm: 'npm install -g @gltf-transform/cli'
        - See https://www.npmjs.com/package/@gltf-transform/cli
      - ktx from the KhronosGroup/KTX-Software repository
        - See https://github.com/KhronosGroup/KTX-Software/releases

    The current clone method is set to '{args.clone_method}' (Change --clone-method to switch).
    The current setting for decompressing .glb files is to {'keep original files' if args.glb_decompress_keep_original else 'remove original files'} after decompression. (Change --glb-decompress-keep-original to switch).
    
    Make sure you have sufficient disk space (Peak size: ~80 GB) and a stable internet connection before running this script.
    The dataset will be stored in '{args.data_dir}' under the directory named '{args.dataset_dir_name}'.
    
    Continue with the download and extraction? (y/n): """
    
    choice = input(INFO).strip().lower()
    if choice != 'y':
        print("Exiting ...\n")
        exit(0)
        
    if not check_gltf_transform_installed():
        print("Error:\n'gltf-transform' is not installed or not found in PATH. Please install it before running this script.\nYou can install it via npm with 'npm install -g @gltf-transform/cli'.")
        sys.exit(1)
    if not check_ktx_software_installed():
        print("Error:\n'ktx' software is not installed or not found in PATH. Please install it before running this script.\nYou can find it in the KhronosGroup/KTX-Software repository.")
        sys.exit(1)
    
    destination_path = args.data_dir / args.dataset_dir_name
    destination_path.mkdir(parents=True, exist_ok=True)
    
    print("\n > Installing Git LFS ...")
    install_git_lfs()
    
    print("\n > Downloading HSSD dataset ...")
    download_hssd(destination_path=destination_path, clone_method=args.clone_method)
    print("\n -- 1 / 2 completed.")
    download_hssd_decomposed(destination_path=destination_path, clone_method=args.clone_method)
    print("\n -- 2 / 2 completed.")
    print("\n > Successfully downloaded HSSD dataset for SceneEval.\n")
    
    print(" > Decompressing downloaded .glb files to remove KHR_texture_basisu compression ...")
    decompress_ktx2_files(download_destination_path=destination_path, remove_original=not args.glb_decompress_keep_original)
    print("\n > Decompression completed.\n")

    if str(args.data_dir) != DEFAULT_DATA_DIR:
        print(f"Warning: You specified a custom data directory: {args.data_dir}. You may need to adjust configs accordingly in SceneEval.")
    if args.dataset_dir_name != DEFAULT_DATASET_DIR_NAME:
        print(f"Warning: You specified a custom dataset directory name: {args.dataset_dir_name}. You may need to adjust configs accordingly in SceneEval.")
