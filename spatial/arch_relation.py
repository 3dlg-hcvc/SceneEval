import trimesh
import numpy as np
from .bounding_box import BoundingBox
from .config import ArchitecturalRelationConfig

SIDE_MAP = {
    "left": "-x",
    "right": "+x",
    "front": "-y",
    "back": "+y",
    "top": "+z",
    "bottom": "-z"
}

class ArchitecturalRelationEvaluator:
    
    def __init__(self, cfg: ArchitecturalRelationConfig) -> None:
        """
        Initialize the evaluator with a configuration.
        
        Args:
            cfg: the configuration for the architectural relations
        """
        
        self.cfg = cfg
        self.gravity_direction = np.array([0, 0, -1])
        self.front_vector = np.array([0, -1, 0])

    def inside_room(self, obj_bbox: BoundingBox, floor_t_objs: list[trimesh.Trimesh], **kwargs) -> float:
        """
        Calculate the score of how much the object is inside the room defined by a floor object.
        For example, a chair is inside a room.
        
        Args:
            obj_bbox: the bounding box of the target object
            floor_t_objs: all floor elements to check
            
        Returns:
            score: the score in [0.0, 1.0]
        """
        
        sample_points = obj_bbox.sample_points()
        ray_origins = sample_points
        ray_directions = np.tile(self.gravity_direction, (len(sample_points), 1))
        
        scores = []
        for floor_t_obj in floor_t_objs:
            ray_hit_points, _, _ = floor_t_obj.ray.intersects_location(ray_origins, ray_directions, multiple_hits=False)
            floor_score = len(ray_hit_points) / len(sample_points)
            scores.append(floor_score)
        
        score = np.max(scores) if len(scores) > 0 else 0.0
        
        return score

    def middle_of_room(self, obj_bbox: BoundingBox, floor_bboxes: list[BoundingBox], **kwargs) -> float:
        """
        Calculate the score of how much the object is in the middle of the room defined by a floor object.
        For example, a table is in the middle of a room.
        
        Args:
            obj_bbox: the bounding box of the target object
            floor_bboxes: the bounding boxes of the floor elements
            
        Returns:
            score: the score in [0.0, 1.0]
        """
        
        scores = []
        for floor_bbox in floor_bboxes:
            # Calculate the 2D distance between the centers of the bounding boxes without considering height
            floor_bbox_center_2d = floor_bbox.centroid[:2]
            obj_bbox_center_2d = obj_bbox.centroid[:2]
            distance = np.linalg.norm(floor_bbox_center_2d - obj_bbox_center_2d)
            
            # The standard deviation depends on the size of the room and the object
            room_size = np.mean(floor_bbox.full_size[:2])
            obj_size = np.max(obj_bbox.full_size[:2])
            ratio = 1 - obj_size / room_size
            std_dev = self.cfg.middle_of_room.base_std_dev * (self.cfg.middle_of_room.obj_size_weight * obj_size + self.cfg.middle_of_room.ratio_weight * ratio)
            
            # The score is the Gaussian of the distance
            floor_score = np.exp(-distance ** 2 / (2 * std_dev ** 2))
            scores.append(floor_score)
        
        score = np.max(scores) if len(scores) > 0 else 0.0
        
        return score

    def _distance_score(self, obj_t_obj: trimesh.Trimesh, arch_t_obj: trimesh.Trimesh, obj_bbox: BoundingBox, range: list[float], gaussian_std: float = 0.25) -> float:
        """
        Generalized function for calulating the score of how much the target object is a certain distance from the reference object.
        
        Args:
            obj_t_obj: the target object
            arch_t_obj: the reference object
            obj_bbox: the target bounding box
            range: the range for calculating the score (score is 1.0 when the distance is within the range, decrease gradually to 0.0 when outside)
            gaussian_std: the standard deviation of the Gaussian function
        """
        
        sample_points = obj_bbox.sample_points()
        closest_points, distances, _ = arch_t_obj.nearest.on_surface(sample_points)
        min_distance = min(distances)
        
        # Score is 1.0 when the distance is within the range
        # Then gradually decreases to 0.0 when the distance is outside the range using a Gaussian
        if range[0] < min_distance < range[1]:
            score = 1.0
        else:
            distance_to_range = np.minimum(np.abs(min_distance - range[0]), np.abs(min_distance - range[1]))
            score = np.exp(-distance_to_range ** 2 / (2 * gaussian_std ** 2))
        
        return score

    def next_to(self,
                architectural_element_type: str,
                obj_t_obj: trimesh.Trimesh,
                wall_t_objs: list[trimesh.Trimesh],
                door_t_objs: list[trimesh.Trimesh],
                window_t_objs: list[trimesh.Trimesh],
                obj_bbox: BoundingBox,
                **kwargs) -> float:
        """
        Calculate the score of how much the object is next to an architectural element.
        The distance for next to is [0.0, 0.5].
        For example, a chair is next to a wall.
        
        Args:
            architectural_element_type: the type of the architectural element (wall, door, window)
            obj_t_obj: the target object
            wall_t_objs: all wall elements to check
            door_t_objs: all door elements to check
            window_t_objs: all window elements to check
            obj_bbox: the bounding box of the target object
        
        Returns:
            score: the score in [0.0, 1.0]
        """
        
        assert architectural_element_type in ["wall", "door", "window"], f"Invalid architectural element type: {architectural_element_type}"
                
        match architectural_element_type:
            case "wall":
                arch_t_objs = wall_t_objs
            case "door":
                arch_t_objs = door_t_objs
            case "window":
                arch_t_objs = window_t_objs
        
        scores = []
        for arch_t_obj in arch_t_objs:
            distance_score = self._distance_score(obj_t_obj, arch_t_obj, obj_bbox, self.cfg.next_to.distance_range, self.cfg.next_to.gaussian_std)
            scores.append(distance_score)
            
        score = np.max(scores) if len(scores) > 0 else 0.0
        
        return score

    def near(self,
             architectural_element_type: str,
             obj_t_obj: trimesh.Trimesh,
             wall_t_objs: list[trimesh.Trimesh],
             door_t_objs: list[trimesh.Trimesh],
             window_t_objs: list[trimesh.Trimesh],
             obj_bbox: BoundingBox,
             **kwargs) -> float:
        """
        Calculate the score of how much the object is near an architectural element.
        The distance for near is [0.5, 1.5].
        For example, a table is near a wall.
        
        Args:
            architectural_element_type: the type of the architectural element (wall, door, window)
            obj_t_obj: the target object
            wall_t_objs: all wall elements to check
            door_t_objs: all door elements to check
            window_t_objs: all window elements to check
            obj_bbox: the bounding box of the target object
        
        Returns:
            score: the score in [0.0, 1.0]
        """
        
        assert architectural_element_type in ["wall", "door", "window"], f"Invalid architectural element type: {architectural_element_type}"
                
        match architectural_element_type:
            case "wall":
                arch_t_objs = wall_t_objs
            case "door":
                arch_t_objs = door_t_objs
            case "window":
                arch_t_objs = window_t_objs
        
        scores = []
        for arch_t_obj in arch_t_objs:
            distance_score = self._distance_score(obj_t_obj, arch_t_obj, obj_bbox, self.cfg.near.distance_range, self.cfg.near.gaussian_std)
            scores.append(distance_score)
        
        score = np.max(scores) if len(scores) > 0 else 0.0
        
        return score

    def across_from(self,
                    architectural_element_type: str,
                    obj_t_obj: trimesh.Trimesh,
                    wall_t_objs: list[trimesh.Trimesh],
                    door_t_objs: list[trimesh.Trimesh],
                    window_t_objs: list[trimesh.Trimesh],
                    obj_bbox: BoundingBox,
                    **kwargs) -> float:
        """
        Calculate the score of how much the object is across from an architectural element.
        The distance for across from is [1.5, 4.0].
        For example, a sofa is across from a wall.
        
        Args:
            architectural_element_type: the type of the architectural element (wall, door, window)
            obj_t_obj: the target object
            wall_t_objs: all wall elements to check
            door_t_objs: all door elements to check
            window_t_objs: all window elements to check
            obj_bbox: the bounding box of the target object
        
        Returns:
            scores: the scores for each architecture element object in [0.0, 1.0]
        """
        
        assert architectural_element_type in ["wall", "door", "window"], f"Invalid architectural element type: {architectural_element_type}"
                
        match architectural_element_type:
            case "wall":
                arch_t_objs = wall_t_objs
            case "door":
                arch_t_objs = door_t_objs
            case "window":
                arch_t_objs = window_t_objs
        
        scores = []
        for arch_t_obj in arch_t_objs:
            distance_score = self._distance_score(obj_t_obj, arch_t_obj, obj_bbox, self.cfg.across_from.distance_range, self.cfg.across_from.gaussian_std)
            scores.append(distance_score)
        
        score = np.max(scores) if len(scores) > 0 else 0.0
            
        return score

    def far(self,
            architectural_element_type: str,
            obj_t_obj: trimesh.Trimesh,
            wall_t_objs: list[trimesh.Trimesh],
            door_t_objs: list[trimesh.Trimesh],
            window_t_objs: list[trimesh.Trimesh],
            obj_bbox: BoundingBox,
            **kwargs) -> float:
        """
        Calculate the score of how much the object is far from an architectural element.
        The distance for far is [4.0, +inf].
        For example, a chair is far from a wall.
        
        Args:
            architectural_element_type: the type of the architectural element (wall, door, window)
            obj_t_obj: the target object
            wall_t_objs: all wall elements to check
            door_t_objs: all door elements to check
            window_t_objs: all window elements to check
            obj_bbox: the bounding box of the target object
        
        Returns:
            scores: the scores for each architecture element object in [0.0, 1.0]
        """
        
        assert architectural_element_type in ["wall", "door", "window"], f"Invalid architectural element type: {architectural_element_type}"
                
        match architectural_element_type:
            case "wall":
                arch_t_objs = wall_t_objs
            case "door":
                arch_t_objs = door_t_objs
            case "window":
                arch_t_objs = window_t_objs
        
        scores = []
        for arch_t_obj in arch_t_objs:
            distance_score = self._distance_score(obj_t_obj, arch_t_obj, obj_bbox, self.cfg.far.distance_range, self.cfg.far.gaussian_std)
            scores.append(distance_score)
        
        score = np.max(scores) if len(scores) > 0 else 0.0
        
        return score

    def on_wall(self,
                obj_t_obj: trimesh.Trimesh,
                wall_t_objs: list[trimesh.Trimesh],
                obj_bbox: BoundingBox,
                wall_bboxes: list[BoundingBox],
                **kwargs) -> float:
        """
        Calculate the score of how much the object is on a wall.
        An object is considered on a wall if it is in front of a wall within a certain distance.
        The distance for on wall is [0.0, 0.01].
        For example, a mirror is on a wall.
        
        Args:
            obj_t_obj: the target object
            wall_t_objs: all wall elements to check
            obj_bbox: the bounding box of the target object
            wall_bboxes: the bounding boxes of the wall elements
        
        Returns:
            score: the score in [0.0, 1.0]
        """
        
        sample_points = obj_bbox.sample_points()
        scores = []
        for wall_t_obj, wall_bbox in zip(wall_t_objs, wall_bboxes):
            at_side = wall_bbox.at_side(sample_points, SIDE_MAP["front"], no_contain=self.cfg.on_wall.no_contain, within_area_margin=self.cfg.on_wall.within_area_margin)
            at_side_score = np.sum(at_side) / len(at_side)
            distance_score = self._distance_score(obj_t_obj, wall_t_obj, obj_bbox, self.cfg.on_wall.distance_range, self.cfg.on_wall.gaussian_std)
            combined_score = at_side_score * distance_score
            scores.append(combined_score)
            
            # print(f"[on_wall] At side: {at_side_score}, Distance: {distance_score}")
        
        score = np.max(scores) if len(scores) > 0 else 0.0
        
        return score

    def against_wall(self,
                     obj_t_obj: trimesh.Trimesh,
                     wall_t_objs: list[trimesh.Trimesh],
                     obj_bbox: BoundingBox,
                     wall_bboxes: list[BoundingBox],
                     **kwargs) -> float:
        """
        Calculate the score of how much the object is against a wall within a certain distance.
        The distance for in front of is [0.0, 0.3].
        For example, a sofa is in front of a wall.
        
        Args:
            obj_t_obj: the target object
            wall_t_objs: all wall elements to check
            obj_bbox: the bounding box of the target object
            wall_bboxes: the bounding boxes of the wall elements
        
        Returns:
            score: the score in [0.0, 1.0]
        """
        
        sample_points = obj_bbox.sample_points()
        scores = []
        for wall_t_obj, wall_bbox in zip(wall_t_objs, wall_bboxes):
            at_side = wall_bbox.at_side(sample_points, SIDE_MAP["front"], no_contain=self.cfg.against_wall.no_contain, within_area_margin=self.cfg.against_wall.within_area_margin)
            at_side_score = np.sum(at_side) / len(at_side)
            distance_score = self._distance_score(obj_t_obj, wall_t_obj, obj_bbox, self.cfg.against_wall.distance_range, self.cfg.against_wall.gaussian_std)
            combined_score = at_side_score * distance_score
            scores.append(combined_score)
                    
        score = np.max(scores) if len(scores) > 0 else 0.0
        
        return score

    def corner_of_room(self,
                       obj_t_obj: trimesh.Trimesh,
                       floor_t_objs: list[trimesh.Trimesh],
                       wall_t_objs: list[trimesh.Trimesh],
                       obj_bbox: BoundingBox,
                       floor_bboxes: list[BoundingBox],
                       wall_bboxes: list[BoundingBox],
                       **kwargs) -> float:
        """
        Calculate the score of how much the object is in the corner of a room.
        An object is considered in the corner of a room if it is in front of two walls.
        The distance for corner of room depends on the object size and the room area.
        For example, a chair is in the corner of a room.
        
        Args:
            obj_t_obj: the target object
            floor_t_objs: all floor elements to check
            wall_t_objs: all wall elements to check
            obj_bbox: the bounding box of the target object
            floor_bboxes: the bounding boxes of the floor elements
            wall_bboxes: the bounding boxes of the wall elements
        
        Returns:
            score: the score in [0.0, 1.0]
        """
        
        room_scores = []
        for floor_t_obj, floor_bbox in zip(floor_t_objs, floor_bboxes):
            
            distance_range = (0.0, self.cfg.corner_of_room.base_distance_threshold)
            
            # Check if the object is inside the room
            in_room = self.inside_room(obj_bbox, [floor_t_obj])
            if in_room < 0.5:
                continue
            
            # Determine the walls for this floor
            wall_indices = []
            for i, wall_bbox in enumerate(wall_bboxes):
                wall_front_vector = wall_bbox.no_scale_matrix[:3, :3] @ self.front_vector
                wall_front_vector = wall_front_vector / np.linalg.norm(wall_front_vector)
                
                ray_origin = wall_bbox.centroid + wall_front_vector * 0.05
                
                ray_hit, _, _ = floor_t_obj.ray.intersects_location([ray_origin], [self.gravity_direction], multiple_hits=False)
                
                if len(ray_hit) > 0:
                    wall_indices.append(i)
            
            # Skip if there are not enough walls
            if len(wall_indices) < 2:
                continue
            
            # Compute the direction to the walls
            wall_directions = []
            sample_points = obj_bbox.sample_points()
            for wall_index in wall_indices:
                wall_t_obj = wall_t_objs[wall_index]
                closest_points, _, _ = wall_t_obj.nearest.on_surface(sample_points)
                directions = closest_points - sample_points
                mean_direction = np.mean(directions, axis=0)
                mean_direction /= np.linalg.norm(mean_direction)
                wall_directions.append(mean_direction)
            
            # Compute the distance score for each wall
            wall_scores = []
            for wall_index in wall_indices:
                wall_t_obj = wall_t_objs[wall_index]
                wall_bbox = wall_bboxes[wall_index]

                distance_score = self._distance_score(obj_t_obj, wall_t_obj, obj_bbox, distance_range, self.cfg.corner_of_room.gaussian_std)
                combined_score = distance_score
                wall_scores.append(combined_score)
            
            # Check all possible pairs of walls
            best_score = -np.inf
            for i in range(len(wall_indices) - 1):
                for j in range(i + 1, len(wall_indices)):
                    
                    # Check if the two walls are roughly perpendicular
                    wall_1_direction, wall_2_direction = wall_directions[i], wall_directions[j]
                    wall_1_direction = wall_1_direction / np.linalg.norm(wall_1_direction)
                    wall_2_direction = wall_2_direction / np.linalg.norm(wall_2_direction)
                    dot_product = np.dot(wall_1_direction, wall_2_direction)
                    
                    # Skip if the walls are not perpendicular
                    if not abs(dot_product) < self.cfg.corner_of_room.perpendicular_threshold:
                        continue
                    
                    # Compute the score for the corner of the room
                    wall_score = wall_scores[i] * wall_scores[j]
                    best_score = max(best_score, wall_score)
            
            room_scores.append(best_score)
        
        score = np.max(room_scores) if len(room_scores) > 0 else 0.0
            
        return score

    def hang_from_ceiling(self,
                          obj_t_obj: trimesh.Trimesh,
                          floor_t_objs: list[trimesh.Trimesh],
                          obj_bbox: BoundingBox,
                          wall_bboxes: list[BoundingBox],
                          **kwargs) -> float:
        """
        Calculate the score of how much the object is hanging from the ceiling.
        An object is considered hanging from the ceiling if it is close to the ceiling.
        The distance for hanging from ceiling is [0.0, 0.01].
        For example, a chandelier is hanging from the ceiling.
        
        Args:
            obj_t_obj: the target object
            floor_t_objs: all floor elements
            wall_bboxes: the bounding boxes of the wall elements
        
        Returns:
            score: the score in [0.0, 1.0]
        """
        
        # Create a ceiling object from the floor object by translating it up
        t_floor = trimesh.util.concatenate(floor_t_objs)
        t_ceiling = t_floor.copy()
        wall_height = np.max([bbox.full_size[2] for bbox in wall_bboxes])
        t_ceiling.apply_translation([0, 0, wall_height])
        
        score = self._distance_score(obj_t_obj, t_ceiling, obj_bbox, self.cfg.hang_from_ceiling.distance_range, self.cfg.hang_from_ceiling.gaussian_std)
        
        return score
