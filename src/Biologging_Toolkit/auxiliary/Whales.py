import pandas as pd
import os
from typing import Union, List

class Whales():
    """
    A class to handle cetacean annotations.

    Attributes:
        depid: The deployment ID of the animal being analyzed.
        analysis_length: The length of the dive portion (in seconds) that is analyzed to be considered as a drift dive.
    """
    idx = 1

    def __init__(self,
                 depid : Union[List, str],
                 *,
                 annotation_path : Union[List, str] = [''],
                 ):
        """
        Initializes the DriftDives object with the given deployment ID, dataset path, and analysis length.
        Args:
            depid: The deployment ID of the animals being analyzed.
            annotation_path: The path to the annotation data
        """

        self.depid = depid
        self.annotation_path = annotation_path
        if isinstance(self.depid, List):
            if isinstance(self.annotation_path, str) :
                self.annotation_path = [self.annotation_path] * len(self.depid)
            assert len(self.depid) == len(self.annotation_path), "Please provide paths for each depid"
        else:
            self.depid = [self.depid]
            self.annotation_path = [self.annotation_path]


