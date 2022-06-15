import math
from typing import List, Union
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset

from .utils import *

AUDIO_DURATION = 10     # seconds

class Dataset(Dataset):
    
    def __init__(
        self,
        data_path: str,
        train: bool, 
        source: Union[bool, List[bool]],
        normal: Union[bool, List[bool]],
        sections: Union[int, List[int]],
        stats_path: str,
        max_audios: int,
        target_sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        n_columns: int,
        hop_columns: int,
        db: bool,
    ) -> None:

        self.info = filter_info(
            data_path=data_path,
            train=train, 
            source=source, 
            normal=normal,
            sections=sections,
            max_audios=max_audios,
        )

        self.values, self.n_classes =  MACHINES, len(MACHINES)

        spectrograms = load_spectrograms(
            info_list=self.info,
            target_sample_rate=target_sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            db=db,

        ) 

        self.spectrograms = normalize_spectrograms(
            spectrograms=spectrograms,
            stats_path=stats_path,
        )

        self.n_columns = n_columns
        n_columns_audio = math.ceil(target_sample_rate*AUDIO_DURATION/hop_length)
        self.n_chunks_audio = math.ceil((n_columns_audio-n_columns)/hop_columns)
        
    #======Dataset methods========================================================= 
    def __len__(self) -> int:
        return self.n_chunks_audio*len(self.info)

    def __getitem__(self, idx: int) -> Tensor:
        return self._get_spec(idx), self._get_labels(idx)

    def _get_spec(self, idx) -> Tensor:
        spec = self.spectrograms[math.floor(idx/self.n_chunks_audio)]
        pos_ini = idx % self.n_chunks_audio
        return spec[:,:,pos_ini:pos_ini+self.n_columns]

    def _get_labels(self, idx):
        idx = idx//self.n_chunks_audio
        label = self.values.index(self.info[idx]['machine'])
        label_onehot = np.zeros(self.n_classes, dtype='int8')
        label_onehot[label]=1
        return label_onehot