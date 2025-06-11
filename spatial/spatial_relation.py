import trimesh
import numpy as np
from .bounding_box import BoundingBox
from .config import SpatialRelationConfig

SIDE_MAP = {
    "left": "-x",
    "right": "+x",
    "front": "-y",
    "back": "+y",
    "top": "+z",
    "bottom": "-z"
}

class SpatialRelationEvaluator:
    
    def __init__(self, cfg: SpatialRelationConfig) -> None:
        """
        Initialize the spatial relation with the configuration.

        Args:
            SpatialRelationConfig: the configuration for the spatial relations
        """
        
        self.cfg = cfg
        self.front_vector = np.array([0, -1, 0])
        self.up_vector = np.array([0, 0, 1])
    
    def inside_of(self, target_bbox: BoundingBox, reference_bbox: BoundingBox, **kwargs) -> float:
        """
        Calculate the score of how much the target bounding box is inside of the reference bounding box.
        For example, a book is inside of a bookshelf.

        Args:
            target_bbox: the target bounding box
            reference_bbox: the reference bounding box

        Returns:
            score: the score in [0.0, 1.0]
        """

        score = target_bbox.overlaps(reference_bbox)
                
        return score

    def outside_of(self, target_bbox: BoundingBox, reference_bbox: BoundingBox, **kwargs) -> float:
        """
        Calculate the score of how much the target bounding box is outside of the reference bounding box.
        For example, a toy is outside of a toy box.

        Args:
            target_bbox: the target bounding box
            reference_bbox: the reference bounding box

        Returns:
            score: the score in [0.0, 1.0]
        """

        score = 1.0 - target_bbox.overlaps(reference_bbox)
        
        return score

    def face_to(self, reference_t_obj: trimesh.Trimesh, target_bbox: BoundingBox, reference_bbox: BoundingBox, **kwargs) -> float:
        """
        Calculate the score of how much the target bounding box is facing towards the reference bounding box.
        For example, a sofa is facing towards a TV.

        Args:
            reference_t_obj: the reference object
            target_bbox: the target bounding box
            reference_bbox: the reference bounding box
        
        Returns:
            score: the score in [0.0, 1.0]
        """
        
        # Get the front vector of the reference bounding box
        target_bbox_front = target_bbox.coord_axes @ self.front_vector
        
        # Sample points from the target bounding box and shoot rays towards the front
        ray_origins = target_bbox.sample_points()
        ray_directions = np.tile(target_bbox_front, (len(ray_origins), 1))
        ray_hits, ray_idxs, _ = reference_t_obj.ray.intersects_location(ray_origins, ray_directions, multiple_hits=False)
        
        # If there are no hits, the score is 0.0 (facing away)
        if len(ray_hits) == 0:
            return 0.0
        
        # Calculate the mean hit point and the direction from the target centroid to the mean hit point
        mean_hit_point = np.mean(ray_hits, axis=0)
        target_to_reference_direction = mean_hit_point - target_bbox.centroid
        
        # Ignore the height component and normalize the vectors
        target_bbox_front_2D = target_bbox_front[:2]
        target_bbox_front_2D /= (np.linalg.norm(target_bbox_front_2D) + target_bbox.epsilon)
        target_to_reference_direction_2D = target_to_reference_direction[:2]
        target_to_reference_direction_2D /= (np.linalg.norm(target_to_reference_direction_2D) + target_bbox.epsilon)
        
        # The score is calculated based on the deviation of the front vector to the direction to the mean hit point
        dot_product = np.dot(target_bbox_front_2D, target_to_reference_direction_2D)
        deviation = np.rad2deg(np.arccos(dot_product))
        score = np.clip(1 - deviation / self.cfg.face_to.max_deviation_degrees, 0.0, 1.0)
        
        return score

    def side_of(self, target_bbox: BoundingBox, reference_bbox: BoundingBox, side: str, **kwargs) -> float:
        """
        Calculate the score of how much the target bounding box is to the side of the reference bounding box.
        For example, a chair is to the left of a table.

        Args:
            target_bbox: the target bounding box
            reference_bbox: the reference bounding box
            side: the side to check (left, right, front, back, top, bottom)
        
        Returns:
            score: the score in [0.0, 1.0]
        """

        assert side in SIDE_MAP, f"Invalid side: {side}, accepted values: {list(SIDE_MAP.keys())}"
        
        sample_points = target_bbox.sample_points()
        at_side = reference_bbox.at_side(sample_points,
                                         SIDE_MAP[side],
                                         no_contain=self.cfg.side_of.no_contain,
                                         within_area_margin=self.cfg.side_of.within_area_margin)
        score = np.sum(at_side) / len(sample_points)
        
        # _debug_visualize([target_bbox, reference_bbox], [sample_points[at_side], sample_points[~at_side]], ["r", "k"], ["g", "b"], f"Side of: {side}, score: {score}")

        return score

    def side_region(self, target_bbox: BoundingBox, reference_bbox: BoundingBox, side: str, **kwargs) -> float:
        """
        Calculate the score of how much the target bounding box is inside the reference bounding box to the side.
        For example, a book is inside a bookshelf to the left.

        Args:
            target_bbox: the target bounding box
            reference_bbox: the reference bounding box
            side: the side to check (left, right, front, back, top, bottom)
        
        Returns:
            score: the score in [0.0, 1.0]
        """

        assert side in SIDE_MAP, f"Invalid side: {side}, accepted values: {list(SIDE_MAP.keys())}"
        
        sample_points = target_bbox.sample_points() 
        contains = reference_bbox.contains(sample_points)
        at_side = reference_bbox.at_side(sample_points,
                                         SIDE_MAP[side],
                                         no_contain=self.cfg.side_region.no_contain,
                                         within_area_margin=self.cfg.side_region.within_area_margin)
        at_side_region = at_side & contains
        score = np.sum(at_side_region) / len(sample_points)
        
        return score

    def long_short_side_of(self, target_bbox: BoundingBox, reference_bbox: BoundingBox, side: str, **kwargs) -> float:
        """
        Calculate the score of how much the target bounding box is on a long or short side of the reference bounding box.
        For example, a chair at the long side of a table.

        Args:
            target_bbox: the target bounding box
            reference_bbox: the reference bounding box
            side: the side to check (long or short)
        
        Returns:
            score: the score in [0.0, 1.0]
        """

        assert side in ["long", "short"], f"Invalid side: {side}, accepted values: ['long', 'short']"
                
        # If there is not a clear long or short side, return 0.0
        if np.allclose(reference_bbox.full_size[0], reference_bbox.full_size[1]):
            return 0.0

        # Determine the long and short sides of the reference bounding box
        match side:
            case "long":
                # If the reference bounding box is longer in the y-axis (a.k.a. depth), putting something on the left or right sides is considered long
                # If the reference bounding box is longer in the x-axis (a.k.a. width), putting something on the front or back sides is considered long
                target_sides = "x" if reference_bbox.full_size[0] < reference_bbox.full_size[1] else "y"            
            case "short":
                # Reverse of the above
                target_sides = "x" if reference_bbox.full_size[0] > reference_bbox.full_size[1] else "y"

        sample_points = target_bbox.sample_points()

        # Find sample points that are at the target sides of the reference bounding box
        no_contain = self.cfg.long_short_side_of.no_contain
        within_area_margin = self.cfg.long_short_side_of.within_area_margin
        target_side_a = reference_bbox.at_side(sample_points, f"+{target_sides}", no_contain=no_contain, within_area_margin=within_area_margin)
        target_side_b = reference_bbox.at_side(sample_points, f"-{target_sides}", no_contain=no_contain, within_area_margin=within_area_margin)
        
        # Take the maximum score of the two sides
        score_positive = np.sum(target_side_a) / len(sample_points)
        score_negative = np.sum(target_side_b) / len(sample_points)
        score = np.max([score_positive, score_negative])
        
        return score

    def on_top(self, target_bbox: BoundingBox, reference_bbox: BoundingBox, **kwargs) -> float:
        """
        Calculate the score of how much the target bounding box is on top of the reference bounding box and within the area.
        For example, a book is on top of a table.

        Args:
            target_bbox: the target bounding box
            reference_bbox: the reference bounding box
        
        Returns:
            score: the score in [0.0, 1.0]
        """
        
        sample_points = target_bbox.sample_points()
        on_top = reference_bbox.at_side(sample_points,
                                        "+z",
                                        no_contain=self.cfg.on_top.no_contain,
                                        within_area_margin=self.cfg.on_top.within_area_margin)
        score = np.sum(on_top) / len(sample_points)

        return score

    def middle_of(self, target_bbox: BoundingBox, reference_bbox: BoundingBox, **kwargs) -> float:
        """
        Calculate the score of how much the target bounding box is at the 2D middle of the reference bounding box.
        For example, a chair is in the 2D middle of a room.

        Args:
            target_bbox: the target bounding box
            reference_bbox: the reference bounding box
        
        Returns:
            score: the score in [0.0, 1.0]
        """

        # Calculate the 2D distance between the centers of the bounding boxes without considering height
        reference_bbox_center_2d = reference_bbox.centroid[:2]
        target_bbox_center_2d = target_bbox.centroid[:2]
        distance = np.linalg.norm(reference_bbox_center_2d - target_bbox_center_2d)

        # The score is the Gaussian of the distance
        score = np.exp(-distance ** 2 / (2 * self.cfg.middle_of.gaussian_std ** 2))

        return score

    def surround(self, multiple_target_bboxes: list[BoundingBox], reference_bbox: BoundingBox, **kwargs) -> float:
        """
        Calculate the score of how much the multiple target bounding boxes surround the reference bounding box.
        For example, four chairs surround a table.

        Args:
            multiple_target_bboxes: the list of target bounding boxes
            reference_bbox: the reference bounding box
        
        Returns:
            score: the score in [0.0, 1.0]
        """
                
        num_ring_bboxes = len(multiple_target_bboxes)
        
        # If there are not enough target bounding boxes for surround, return 0.0
        if num_ring_bboxes < 3:
            return 0.0

        # Get the vectors from target (ring) bounding boxes to the reference (center) bounding box
        target_bbox_centroids = np.array([bbox.centroid for bbox in multiple_target_bboxes])
        reference_bbox_centroid = reference_bbox.centroid
        reference_to_target_directions = target_bbox_centroids - reference_bbox_centroid
        
        # Find the distances and normalized directions
        distances = np.linalg.norm(reference_to_target_directions, axis=1)
        normalized_directions = reference_to_target_directions / distances[:, np.newaxis]
        
        # For each target, find the index of the two closest neighbours in terms of angle to the positive and negative sides
        neighbour_infos = []
        for i in range(num_ring_bboxes):
            to_neighbour_angles = []
            for j in range(num_ring_bboxes):
                
                # Skip the same object
                if i == j:
                    continue
                
                # Calculate the angle between the two directions
                cross_product = np.cross(normalized_directions[i], normalized_directions[j])
                dot_product = np.dot(normalized_directions[i], normalized_directions[j])
                angle = np.arccos(dot_product)
                
                # Use the cross product to determine the side of the angle
                side_test = np.dot(self.up_vector, cross_product)
                if side_test < 0:
                    angle = -angle
                
                to_neighbour_angles.append((j, angle))
            
            # Find the closest positive and negative neighbours
            positive_angles = [(idx, angle) for idx, angle in to_neighbour_angles if angle > 0]
            if len(positive_angles) > 0:
                closest_positive_index, _ = min(positive_angles, key=lambda x: x[1])
                to_positive_neighbour_angle = [angle for idx, angle in positive_angles if idx == closest_positive_index][0]
            else:
                closest_positive_index = None
                to_positive_neighbour_angle = None
            
            negative_angles = [(idx, angle) for idx, angle in to_neighbour_angles if angle < 0]
            if len(negative_angles) > 0:
                closest_negative_index, _ = max(negative_angles, key=lambda x: x[1])
                to_negative_neighbour_angle = [angle for idx, angle in negative_angles if idx == closest_negative_index][0]
            else:
                closest_negative_index = None
                to_negative_neighbour_angle = None
            
            neighbour_infos.append({
                "positive_neighbour_idx": closest_positive_index,
                "to_positive_neighbour_angle": to_positive_neighbour_angle,
                "negative_neighbour_idx": closest_negative_index,
                "to_negative_neighbour_angle": to_negative_neighbour_angle
            })
        
        # Find the ideal distance and angle
        ideal_distance = np.median(distances)
        ideal_angle = 2 * np.pi / num_ring_bboxes
        
        # Calculate the deviations from the median distance
        distance_deviations = np.abs(distances - ideal_distance)
        distance_deviation_ratios = np.clip(distance_deviations / ideal_distance, 0.0, 1.0)
        
        # Calculate the angle deviations
        angle_deviation_ratios = []
        for i, neighbour_info in enumerate(neighbour_infos):
            if neighbour_info["positive_neighbour_idx"] is not None:
                neighbour_angle = neighbour_info["to_positive_neighbour_angle"]
            else:
                neighbour_angle = neighbour_info["to_negative_neighbour_angle"]
            angle_deviation = np.abs(neighbour_angle - ideal_angle)
            angle_deviation_ratio = np.clip(angle_deviation / ideal_angle, 0.0, 1.0)
            angle_deviation_ratios.append(angle_deviation_ratio)
        
        # The score depends on the deviations from the mean distance and the angle deviations for each target
        # Each target contributes to a portion of the score
        portion = 1 / num_ring_bboxes
        scores = []
        for i in range(num_ring_bboxes):
            distance_score = (1 - distance_deviation_ratios[i]) ** 2 * self.cfg.surround.distance_weight
            angle_score = (1 - angle_deviation_ratios[i]) ** 2 * self.cfg.surround.angle_weight
            scores.append(portion * (distance_score + angle_score))
        score = np.sum(scores)
            
        return score

    def _distance_score(self,
                        target_t_obj: trimesh.Trimesh,
                        reference_t_obj: trimesh.Trimesh,
                        target_bbox: BoundingBox,
                        reference_bbox: BoundingBox,
                        range: list[float]) -> float:
        """
        Generalized function for calculating the score of how much the target object is a certain distance from the reference object.

        Args:
            target_t_obj: the target object
            reference_t_obj: the reference object
            target_bbox: the target bounding box
            reference_bbox: the reference bounding box
            range: the range for calculating the score (score is 1.0 when the distance is within the range, decrease gradually to 0.0 when outside)
        
        Returns:
            score: the score in [0.0, 1.0]
        """

        if target_bbox.volume < reference_bbox.volume:
            obj_to_sample_points_from = target_t_obj
            comparison_object = reference_t_obj
            num_sample_points = int(target_bbox.volume * target_bbox.sample_points_per_unit_volume)
        else:
            obj_to_sample_points_from = reference_t_obj
            comparison_object = target_t_obj
            num_sample_points = int(reference_bbox.volume * reference_bbox.sample_points_per_unit_volume)
        
        num_sample_points = max(num_sample_points, self.cfg.distance_score.min_num_sample_points)

        # NOTE:
        # This limits the number of sample points to prevent memory explosion
        # num_sample_points = min(num_sample_points, 1024)

        oriented_bbox: trimesh.primitives.Box = obj_to_sample_points_from.bounding_box_oriented
        sample_points = oriented_bbox.sample_volume(num_sample_points)
        closest_points, distances, _ = comparison_object.nearest.on_surface(sample_points)
        min_distance = np.min(distances)

        # Score is 1.0 when the distance is within the range
        # Then gradually decreases to 0.0 when the distance is outside the range using a Gaussian
        if range[0] < min_distance < range[1]:
            score = 1.0
        else:
            distance_to_range = np.minimum(np.abs(min_distance - range[0]), np.abs(min_distance - range[1]))
            score = np.exp(-distance_to_range ** 2 / (2 * self.cfg.distance_score.gaussian_std ** 2))

        return score

    def next_to(self,
                target_t_obj: trimesh.Trimesh,
                reference_t_obj: trimesh.Trimesh,
                target_bbox: BoundingBox,
                reference_bbox: BoundingBox,
                **kwargs) -> float:
        """
        Calculate the score of how much the target object is next to the reference object.
        The distance range for next to is [0.0, 0.5] (default).
        For example, a nightstand is next to a bed.

        Args:
            target_t_obj: the target object
            reference_t_obj: the reference object
            target_bbox: the target bounding box
            reference_bbox: the reference bounding box
        
        Returns:
            score: the score in [0.0, 1.0]
        """
        
        score = self._distance_score(target_t_obj, reference_t_obj, target_bbox, reference_bbox, self.cfg.next_to.distance_range)
        return score

    def near(self,
            target_t_obj: trimesh.Trimesh,
            reference_t_obj: trimesh.Trimesh,
            target_bbox: BoundingBox,
            reference_bbox: BoundingBox,
            **kwargs) -> float:
        """
        Calculate the score of how much the target object is near the reference object.
        The distance range for near is [0.5, 1.5] (default).
        For example, a sofa is near a TV.

        Args:
            target_t_obj: the target object
            reference_t_obj: the reference object
            target_bbox: the target bounding box
            reference_bbox: the reference bounding box
        
        Returns:
            score: the score in [0.0, 1.0]
        """
        
        score = self._distance_score(target_t_obj, reference_t_obj, target_bbox, reference_bbox, self.cfg.near.distance_range)
        return score

    def across_from(self,
                    target_t_obj: trimesh.Trimesh,
                    reference_t_obj: trimesh.Trimesh,
                    target_bbox: BoundingBox,
                    reference_bbox: BoundingBox,
                    **kwargs) -> float:
        """
        Calculate the score of how much the target object is across from the reference object.
        The distance range for across from is [1.5, 4] (default).
        For example, a lamp is across the room from a sofa.

        Args:
            target_t_obj: the target object
            reference_t_obj: the reference object
            target_bbox: the target bounding box
            reference_bbox: the reference bounding box
        
        Returns:
            score: the score in [0.0, 1.0]
        """
        
        score = self._distance_score(target_t_obj, reference_t_obj, target_bbox, reference_bbox, self.cfg.across_from.distance_range)
        return score

    def far(self,
            target_t_obj: trimesh.Trimesh,
            reference_t_obj: trimesh.Trimesh,
            target_bbox: BoundingBox,
            reference_bbox: BoundingBox,
            **kwargs) -> float:
        """
        Calculate the score of how much the target object is far from the reference object.
        The distance range for far is [4.0, +inf] (default).
        For example, a sofa is far from a TV.

        Args:
            target_t_obj: the target object
            reference_t_obj: the reference object
            target_bbox: the target bounding box
            reference_bbox: the reference bounding box
        
        Returns:
            score: the score in [0.0, 1.0]
        """
        
        score = self._distance_score(target_t_obj, reference_t_obj, target_bbox, reference_bbox, self.cfg.far.distance_range)
        return score
