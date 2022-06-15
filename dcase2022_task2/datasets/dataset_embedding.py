from typing import List, Union

from torch import Tensor
from torch.utils.data import Dataset

from .utils import *

class Dataset(Dataset):
    
    def __init__(
        self,
        data_path: str,
        machine: str,
        train: bool, 
        source: Union[bool, List[bool]],
        normal: Union[bool, List[bool]],
        sections: Union[int, List[int]],
        max_audios: int,
    ) -> None:

        self.info = filter_info(
            data_path=data_path, 
            machines=machine, 
            train=train, 
            source=source, 
            normal=normal,
            sections=sections,
            max_audios=max_audios,
            sort=True,
        )
        self.embeddings, self.n_chunks_audio = load_embeddings(
            info_list=self.info,
        )

        
    def __len__(self) -> int:
        return self.n_chunks_audio*len(self.info)


    def __getitem__(self, idx: int) -> Tensor:
        return self._get_embed(idx), self._get_path(idx)


    def _get_info(self, idx) -> Tensor:
        idx = idx//self.n_chunks_audio
        return self.info[idx]

    def _get_path(self, idx) -> Tensor:
        return self._get_info(idx)['path']
    

    def _get_embed(self, idx) -> Tensor:
        return self.embeddings[idx]