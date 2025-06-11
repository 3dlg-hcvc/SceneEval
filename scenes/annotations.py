from dataclasses import dataclass
from pathlib import Path
from csv import DictReader

@dataclass
class Annotation:
    """
    An annotation for a scene.

    Attributes:
        id: the ID of the scene
        difficulty: the difficulty of the scene
        description: the description of the scene
        obj_count: the number of objects in the scene
        obj_attr: the attributes of the objects in the scene
        oo_rel: the object-object relationships in the scene
        oa_rel: the object-architecture relationships in the scene
    """

    id: str
    difficulty: str
    description: str
    obj_count: list[str]
    obj_attr: list[str]
    oo_rel: list[str]
    oa_rel: list[str]

class Annotations:
    def __init__(self, file_path: Path):
        """
        Initialize the annotations object.

        Args:
            file_path: the path to the annotations file
        """
        
        self.file_path = file_path
        with open(file_path, "r") as file:
            self.raw_data = list(DictReader(file))
        
        self.parsed_data: list[Annotation] = self._extract(self.raw_data)

    def __len__(self) -> int:
        """
        Get the number of annotations.
        """

        return len(self.parsed_data)

    def __getitem__(self, index: int) -> Annotation:
        """
        Get an annotation by index.
        """

        return self.parsed_data[index]
    
    def _extract(self, raw_data: list[dict[str, str]]) -> list[Annotation]:
        """
        Extract the data from the raw data.
        """

        FIELDS = [
            "ID",
            "Difficulty",
            "Description",
            "ObjCount",
            "ObjAttr",
            "OORel",
            "OARel",
        ]

        def _parser(string: str) -> list[str] | str:
            """
            Helper function to parse a string in the annotations file.

            Args:
                string: the string to parse
            """

            if string == "":
                return []
            strings = string.strip("\n").split(";")
            
            filtered_strings = []
            for s in strings:
                s = s.strip()
                
                if s == "":
                    continue

                filtered_strings.append(s)
            
            return filtered_strings
        
        # Put the raw data into a dictionary by column
        annotations = []
        for row in raw_data:
            data = {field.lower(): _parser(row[field]) for field in FIELDS}
            for field in ["id", "description", "difficulty"]:
                data[field] = data[field][0]
            
            annotation = Annotation(*data.values())
            annotations.append(annotation)

        return annotations
