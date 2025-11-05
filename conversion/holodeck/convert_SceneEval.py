import time
import json
import argparse
import subprocess
from pathlib import Path
from uuid import uuid4

COMMAND_FILE = "bridgeio_command.txt"
STATE_FILE = "bridgeio_state.txt"
RESULTS_FILE = "bridgeio_results.json"

# ===================================================================

SCENE_STATE_JSON_BASE = {
  "format": "sceneState",
  "scene": {
    "arch": {
      "coords2d": [
        0,
        1
      ],
      "defaults": {
        "Ceiling": {
          "depth": 0.05
        },
        "Floor": {
          "depth": 0.05
        },
        "Wall": {
          "depth": 0.1,
          "extraHeight": 0.035
        }
      },
      "elements": [],
      "front": [
        0,
        1,
        0
      ],
      "holes": [],
      "id": "",
      "images": [],
      "materials": [],
      "regions": [
        {
          "id": "bedroom",
          "type": "Other",
          "walls": []
        }
      ],
      "scaleToMeters": 1,
      "textures": [],
      "up": [
        0,
        0,
        1
      ],
      "version": "arch@1.0.2"
    },
    "assetSource": [
      "objaverse"
    ],
    "front": [
      0,
      1,
      0
    ],
    "id": "",
    "modifications": [],
    "object": [],
    "unit": 1.0,
    "up": [
      0,
      0,
      1
    ],
    "version": "scene@1.0.2"
  },
  "selected": []
}

SCENE_STATE_ARCH_ELEMENT_TEMPLATE = {
    "id": 0,
    "type": "",
    "roomId": "",
    "height": -1,
    "depth": 0.05,
    "points": [[], []],
    "holes": [
    {
        "box": {
            "max": [],
            "min": []
        },
        "id": "",
        "type": ""
    }
    ],
    "materials": [
    {
        "diffuse": "#888899",
        "name": "Walldrywall4Tiled"
    }
    ]
}

# SCENE_STATE_OBJECT_TEMPLATE = {
#     "id": "0",
#     "modelId": "",
#     "index": 0,
#     "parentId": "",
#     "parentIndex": 0,
#     "transform": {
#         "rows": 4,
#         "cols": 4,
#         "data": [],
#         "rotation": [],
#         "scale": [],
#         "translation": []
#     }
# }

# ===================================================================

class UnityController:
    """
    A controller to communicate with a Unity instance via file-based commands and states.
    """
    
    def __init__(self, command_file: Path, state_file: Path, results_file: Path, timeout: int = 30) -> None:
        """
        Initializes the UnityController with file paths and timeout.
        
        Args:
            command_file: path to the command file
            state_file: path to the state file
            results_file: path to the results file
            timeout: maximum time to wait for state changes in seconds
        """
        
        self.command_file = command_file
        self.state_file = state_file
        self.results_file = results_file
        self.timeout = timeout
        
        if not self.command_file.exists():
            self.command_file.touch()
        self.command_file.write_text("")  # Clear any existing command

    def send_command(self, command: str) -> None:
        """
        Sends a command to Unity by writing to the command file.
        """
        
        self.command_file.write_text(command)
        print(f"<-- Sent command: {command}")
    
    def wait_for_state(self, target_state: str, ignore_timeout=False) -> None:
        """
        Blocks until the Unity process writes the target state to the state file or timeout is reached.
        
        Args:
            target_state: the desired state to wait for
            ignore_timeout: if True, return even if timeout is reached, otherwise raise an error on timeout
        """
        
        start = time.time()
        
        print(f"> Waiting for Unity: {target_state}")
        
        while time.time() - start < self.timeout:
            if self.state_file.exists():
                state = self.state_file.read_text().strip()
                if state == target_state:
                    print(f"--> Unity is now {state}")
                    return
            time.sleep(0.5)
            
        print(f"> Timeout waiting for {target_state}")
        if not ignore_timeout:
            raise TimeoutError(f"Unity did not reach state {target_state} within {self.timeout} seconds")

        return

    def make_idle(self) -> None:
        """
        Sends the STOP command to Unity and waits for it to reach the STOPPED state.
        """
        
        self.send_command("STOP")
        self.wait_for_state("STOPPED")

    def get_processed_object_info(self, scene_json: Path) -> list[dict]:
        """
        Imports a scene json into Unity and retrieves processed object info.
        
        Args:
            scene_json: path to the scene json file to be processed
        
        Returns:
            A list of dictionaries representing processed objects.
        """
        
        print(f"Processing scene: {scene_json}")

        # Launch connect_to_unity.py (from Holodeck) to load the scene into Unity
        # It waits until Unity is in PLAY mode
        p = subprocess.Popen(["python", "connect_to_unity.py", "--scene", str(scene_json)])
        
        # Switch Unity to PLAY mode
        self.send_command("PLAY")
        
        # connect_to_unity.py will exit once Unity is in PLAY mode and the scene is loaded
        # Wait for it to complete
        while p.poll() is None:
            time.sleep(1)
        print(f" > Loaded scene into Unity")

        # Request Unity to process the scene and write results
        self.send_command("PROCESS")
        self.wait_for_state("PROCESSED")

        # Read the results file
        with self.results_file.open("r") as f:
            object_info = json.load(f)

        # Stop Unity play mode to prepare for next scene
        self.make_idle()

        return object_info["objs"]

# -------------------------------------------------------------------

def gather_holodeck_scene_jsons(holodeck_output_dir: Path) -> list[Path]:
    """
    Gathers all scene JSON file paths from the Holodeck output directory.
    
    Args:
        holodeck_output_dir: directory containing Holodeck output scene JSONs

    Returns:
        A list of paths to the scene JSON files.
    """
    
    scene_jsons = []
    for scene_dir in holodeck_output_dir.iterdir():
        if scene_dir.is_dir():
            json_files = list(scene_dir.glob("*.json"))
            scene_jsons.extend(json_files)

    return scene_jsons

def get_arch_info(scene_json: Path) -> list[dict]:
    """
    Extracts arch element information from a Holodeck scene JSON file into scene state format.
    
    Args:
        scene_json: path to the Holodeck scene JSON file
    
    Returns:
        A list of dictionaries representing arch elements in scene state format.
    """
    
    with scene_json.open("r") as f:
        holodeck_scene = json.load(f)
    
    windows_info = holodeck_scene.get("windows", [])
    doors_info = holodeck_scene.get("doors", [])
    
    arch_info = []
    for floor_info in holodeck_scene["rooms"]:
        arch_element = SCENE_STATE_ARCH_ELEMENT_TEMPLATE.copy()
        arch_element["id"] = f"floor|{floor_info['id']}"
        arch_element["type"] = "Floor"
        arch_element["roomId"] = floor_info["id"]
        arch_element["depth"] = 0.05
        arch_element["points"] = [
            [floor_point["x"], floor_point["z"], floor_point["y"]] for floor_point in floor_info["floorPolygon"]
        ]
        
        arch_element.pop("height")
        arch_element.pop("holes")
        
        arch_info.append(arch_element)
        
    for wall_info in holodeck_scene["walls"]:
        arch_element = SCENE_STATE_ARCH_ELEMENT_TEMPLATE.copy()
        arch_element["id"] = wall_info["id"]
        arch_element["type"] = "Wall"
        arch_element["roomId"] = wall_info["roomId"]
        arch_element["height"] = wall_info["height"]
        arch_element["depth"] = 0.025
        arch_element["points"] = [
            point + [0.0] for point in wall_info["segment"]
        ]
        
        holes = []
        for opening_info in windows_info + doors_info:
            wall0_id = opening_info["wall0"]
            wall1_id = opening_info["wall1"]
            
            # If this opening is not on this wall, skip
            if not arch_element["id"] in [wall0_id, wall1_id]:
                continue
            
            # Determine if we need to mirror the coordinates of the opening
            # Convention: holePolygon X is defined w.r.t wall0's local frame.
            # If this opening is being applied to wall1 (the opposite face), mirror X across wall width.
            mirror_x = (arch_element["id"] == wall1_id)
            
            # Mirror coordinates if needed
            p0 = opening_info["holePolygon"][0]
            p1 = opening_info["holePolygon"][1]
            width = wall_info.get("width")

            if mirror_x and width is not None:
                x0 = width - p0["x"]
                x1 = width - p1["x"]
            else:
                x0 = p0["x"]
                x1 = p1["x"]

            # Y is vertical in wall-local coords; no mirroring needed
            y0, y1 = p0["y"], p1["y"]

            # Compute min/max after optional mirroring
            min_x, max_x = (x0, x1) if x0 <= x1 else (x1, x0)
            min_y, max_y = (y0, y1) if y0 <= y1 else (y1, y0)

            holes.append({
                "box": {
                    "min": [min_x, min_y],
                    "max": [max_x, max_y]
                },
                "id": opening_info["id"],
                "type": "Window" if opening_info in windows_info else "Door"
            })
            
        arch_element["holes"] = holes

        arch_info.append(arch_element)

    return arch_info

def save_scene_state(arch_info: list[dict], object_info: list[dict], output_path: Path) -> None:
    """
    Saves a scene state JSON file combining arch and object information.
    
    Args:
        arch_info: list of arch element dictionaries
        object_info: list of object dictionaries
        output_path: path to save the scene state JSON file
    """
    
    # Create scene state structure
    scene_state = SCENE_STATE_JSON_BASE.copy()
    scene_state["scene"]["arch"]["elements"] = arch_info
    scene_state["scene"]["object"] = object_info
    
    # Generate a unique ID for the scene
    uuid = str(uuid4())
    scene_state["scene"]["arch"]["id"] = uuid
    scene_state["scene"]["id"] = uuid
    
    # Save to file
    with output_path.open("w") as f:
        json.dump(scene_state, f, indent=2)
    
    print(f"Saved scene state to {output_path}")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--holodeck_output_dir", type=Path, default=Path("./data/scenes"), help="Directory containing Holodeck output scene JSONs")
    parser.add_argument("--unity_temp_dir", type=Path, default=Path("./ai2thor/unity/Temp"), help="Temp directory of the ai2thor Unity project")
    parser.add_argument("--output_dir", type=Path, default=Path("./data/converted_scenes"), help="Directory to save processed scene state JSONs")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds for Unity state changes")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check that Unity temp directory exists, which indicates Unity is running
    if not args.unity_temp_dir.exists():
        raise RuntimeError(f"Unity temp directory does not exist: {args.unity_temp_dir}. This directory only exists if the ai2thor Unity project is actively running. Please start Unity and try again.")
    
    # Create UnityController instance
    controller = UnityController(
        command_file=args.unity_temp_dir / COMMAND_FILE,
        state_file=args.unity_temp_dir / STATE_FILE,
        results_file=args.unity_temp_dir / RESULTS_FILE,
        timeout=args.timeout
    )

    # Gather all scene JSONs from Holodeck output directory that need to be processed
    all_scene_jsons = gather_holodeck_scene_jsons(args.holodeck_output_dir)
    print(f" > Found {len(all_scene_jsons)} scenes to process")
    
    # First, ensure Unity is not in play mode
    controller.make_idle()
    
    for scene_json in all_scene_jsons:
        
        arch_info = get_arch_info(scene_json)
        object_info = controller.get_processed_object_info(scene_json)
        
        output_scene_state_path = args.output_dir / f"{scene_json.parent.stem}.json"
        save_scene_state(arch_info, object_info, output_scene_state_path)

if __name__ == "__main__":
    main()
