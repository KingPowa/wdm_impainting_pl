import os
import pandas as pd
import numpy as np
from abc import abstractmethod
from typing import Any

class SourceFile():

    def __init__(self, file_path: str):
        self.file_path = file_path

    @property
    def basename(self):
        return os.path.basename(self.file_path)
    
    @property
    def filename(self):
        return os.path.splitext(self.basename)[0]
    
    @property
    def modality(self):
        # Default: inferred
        return self.filename.split("_")[-1]
    
    @abstractmethod
    def get_paths(self) -> list[tuple[str, Any, str]]:
        raise NotImplementedError
    
    def exists(self):
        return os.path.exists(self.file_path)
    
class Masterfile(SourceFile):

    def __init__(self, file_path: str, modality: str = None):
        super(Masterfile, self).__init__(file_path=file_path)
        self.modality_prm = modality
        self.internal_path = f"path_{modality}" if modality else "path"

    @property
    def modality(self):
        return self.modality_prm if self.modality_prm else SourceFile.modality.fget(self)
    
    def get_subset_paths(self, pths):
        df = pd.read_csv(self.file_path)
        return self.__make_dict(df.loc[df[self.internal_path].isin(pths), [self.internal_path, 'age', 'sex']].values)
    
    def get_paths(self):
        return self.__make_dict(pd.read_csv(self.file_path)[[self.internal_path, 'age', 'sex']].values)
    
    def __make_dict(self, vals: np.ndarray):
        # may be useful...
        return vals


class Nifti(SourceFile):

    def __init__(self, file_path: str, cov_file: str | Masterfile = None):
        super(Nifti, self).__init__(file_path=file_path)
        if cov_file is None and not self.file_path.endswith('sv'):
            FileNotFoundError("Covariate file not provided")
        self.cov_file = cov_file

    def exists(self):
        return super().exists() and (os.path.exists(self.cov_file) if self.cov_file else True)
    
    def get_paths(self):
        if self.cov_file:
            # Grab the paths in file_path and check against the cov file
            with open(self.file_path, 'r') as f:
                pths = [pth.strip() for pth in f.readlines()]
            if isinstance(self.cov_file, str):
                masterfile = Masterfile(self.cov_file)
                return masterfile.get_subset_paths(pths)
            else:
                masterfile: Masterfile = self.cov_file
                return masterfile.get_subset_paths(pths)
        else:
            masterfile = Masterfile(self.filename)
            return masterfile.get_paths()
            
            

    
    
