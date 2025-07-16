from typing import Dict
from dataclasses import dataclass

@dataclass
class SystemConfiguration:
    """
    Configuration dataclass for the LSI-IR system
    """
    n_components: int = 100
    metric: str = "tf-idf" # Options: 'bool', 'freq', 'tf-idf'
    preprocessing_config: Dict = None
    
    def __post_init__(self):
        if self.preprocessing_config is None:
            self.preprocessing_config = {
                'text_columns': ['W'],
                'lowercase': True,
                'remove_punct': True,
                'remove_stop': True,
                'lemmatize': True,
                'remove_num': True,
                'allowed_pos': ['NOUN', 'PROPN', 'ADJ', 'VERB', 'ADV']
            }