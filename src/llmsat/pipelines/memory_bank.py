from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Union
import json
from pathlib import Path

class Role(Enum):
    LEADER = "leader"
    MEMBER = "member"

@dataclass
class Algorithm:
    function_name: str
    description: str 
    role: Role        
    
    id: str
    parent_id: Optional[List[str]] = None
    parent_algorithm_description: Optional[List[str]] = None
    
    raw_par2_score: Optional[float] = None
    normalized_par2_score: Optional[float] = None

    # Analysis on why leader/member is good
    analysis: Optional[str] = None

    def save_to_json(self, directory: Union[str, Path]) -> None:
        """
        saves the Algorithm instance as a json in the specified directory, and filename will be '{self.id}.json'
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        file_path = dir_path / f"{self.id}.json"
        data = asdict(self)
        data['role'] = self.role.value

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
