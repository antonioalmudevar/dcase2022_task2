import os
import h5py
import itertools
from pathlib import Path
from typing import List, Union

import torch
from torch import Tensor
import torchaudio


DEV_SECTIONS = [0,1,2]
EVAL_SECTIONS = [3,4,5]
SECTIONS = [0,1,2,3,4,5]
MACHINES = ['ToyCar','ToyTrain','bearing','fan','gearbox','slider','valve']
DATASETS = ["dev", "eval", "all"]

attributes = {
    'bearing':      ['vel',     'loc',      'f-n'   ],
    'fan':          ['m-n',     'f-n',      'n-lv'  ],
    'gearbox':      ['volt',    'wt',       'id'    ],
    'slider':       ['vel',     'ac',       'f-n'   ],
    'ToyCar':       ['car',     'spd',      'noise' ],
    'ToyTrain':     ['car',     'spd',      'noise' ],
    'valve':        ['pat',     'panel',    'v1pat' ],
}

sections_attribute = {
    'bearing':      {'vel':[0,3], 'loc':[1,4], 'f-n':[2,5]},
    'fan':          {'m-n':[0,3], 'f-n':[1,4], 'n-lv':[2,5]},
    'gearbox':      {'volt':[0,3], 'wt':[1,4], 'id' :[2,5]},
    'slider':       {'vel':[0,3], 'ac':[1,4], 'f-n':[2,5]},
    'ToyCar':       {'car':[0,1,2,3,4,5], 'spd':[0,1,2,3,4,5], 'noise':[0,1,2,3,4,5]},
    'ToyTrain':     {'car':[0,1,2,3,4,5], 'spd':[0,1,2,3,4,5], 'noise':[0,1,2,3,4,5]},
    'valve':        {'pat':[0,1,3,4], 'panel':[1,4], 'v1pat':[2,5]},
}

attributes_section = {
    'bearing':      {0:'vel', 1:'loc', 2:'f-n', 3:'vel', 4:'loc', 5:'f-n'},
    'fan':          {0:'m-n', 1:'f-n', 2:'n-lv', 3:'m-n', 4:'f-n', 5:'n-lv'},
    'gearbox':      {0:'volt', 1:'wt', 2:'id', 3:'volt', 4:'wt', 5:'id'},
    'slider':       {0:'vel', 1:'ac', 2:'f-n', 3:'vel', 4:'ac', 5:'f-n'},
    'ToyCar':       {0:'car', 1:'spd', 2:'noise', 3:'car', 4:'spd', 5:'noise'},
    'ToyTrain':     {0:'car', 1:'spd', 2:'noise', 3:'car', 4:'spd', 5:'noise'},
    'valve':        {0:'pat', 1:'panel', 2:'v1pat', 3:'pat', 4:'panel', 5:'v1pat'},
}


#======Filter info=========================================================
def filter_info_train(
    data_path: str,
    machines: Union[str, List[str]]=MACHINES,
    train: Union[bool, List[bool]]=[True, False],
    source: Union[bool, List[bool]]=[True, False],
    normal: Union[bool, List[bool]]=[True, False],
    sections: Union[int, List[int]]=SECTIONS,
    attribute: str=None,
    max_audios: Union[int, float, str]=float("inf")
) -> List[dict]:

    machines = [machines] if isinstance(machines, str) else machines
    train = [train] if isinstance(train, bool) else train
    source = [source] if isinstance(source, bool) else source
    normal = [normal] if isinstance(normal, bool) else normal
    sections = [sections] if isinstance(sections, int) else sections
    max_audios = float(max_audios) if isinstance(max_audios, str) else max_audios

    info = []
    for machine, train in itertools.product(machines, train):
        path = data_path/machine/("train" if train else "test")
        cont_audios = 0
        for fn in sorted(os.listdir(path)):
            vec = ".".join(fn.split(".")[:-1]).split("_")
            att_audio = [vec[i] for i in range(6,len(vec),2)]
            if (
                int(vec[1]) in sections and
                ((vec[2]=='source') in source) and
                ((vec[4]=='normal') in normal) and
                (attribute in att_audio or attribute is None) and
                cont_audios < max_audios
            ):
                cont_audios += 1
                info.append({
                    'machine': machine,
                    'section': int(vec[1]),
                    'source': True if vec[2]=='source' else False,
                    'train': True if vec[3]=='train' else False,
                    'normal': True if vec[4]=='normal' else False,
                    'path': str(data_path/machine/vec[3]/fn),
                    'value': vec[6+2*att_audio.index(attribute)+1] \
                        if attribute is not None else None
                })
    return info


#======Filter info=========================================================
def filter_info(
    data_path: str,
    machines: Union[str, List[str]]=MACHINES,
    train: Union[bool, List[bool]]=[True, False],
    source: Union[bool, List[bool]]=[True, False],
    normal: Union[bool, List[bool]]=[True, False],
    sections: Union[int, List[int]]=SECTIONS,
    max_audios: Union[int, float, str]=float("inf"),
    sort: bool = False,
) -> List[dict]:

    machines = [machines] if isinstance(machines, str) else machines
    train = [train] if isinstance(train, bool) else train
    source = [source] if isinstance(source, bool) else source
    normal = [normal] if isinstance(normal, bool) else normal
    sections = [sections] if isinstance(sections, int) else sections
    max_audios = float(max_audios) if isinstance(max_audios, str) else max_audios

    info = []
    for machine, train in itertools.product(machines, train):
        path = data_path/machine/("train" if train else "test")
        cont_audios = 0
        list_dir = sorted(os.listdir(path)) if sort else os.listdir(path)
        for fn in list_dir:
            vec = ".".join(fn.split(".")[:-1]).split("_")
            section = int(vec[1])
            if section in EVAL_SECTIONS and not train:
                if (
                    section in sections and
                    cont_audios < max_audios
                ):
                    cont_audios += 1
                    info.append({
                        'machine': machine,
                        'section': section,
                        'path': str(data_path/machine/"test"/fn),
                    })
            else:
                if (
                    section in sections and
                    ((vec[2]=='source') in source) and
                    ((vec[4]=='normal') in normal) and
                    cont_audios < max_audios
                ):
                    cont_audios += 1
                    info.append({
                        'machine': machine,
                        'section': section,
                        'source': True if vec[2]=='source' else False,
                        'train': True if vec[3]=='train' else False,
                        'normal': True if vec[4]=='normal' else False,
                        'path': str(data_path/machine/vec[3]/fn),
                    })
    return info


def filter_info_eval(
    data_path: str,
    machines: Union[str, List[str]]=MACHINES,
    sections: Union[int, List[int]]=SECTIONS,
    max_audios: Union[int, float, str]=float("inf")
) -> List[dict]:

    machines = [machines] if isinstance(machines, str) else machines
    sections = [sections] if isinstance(sections, int) else sections
    max_audios = float(max_audios) if isinstance(max_audios, str) else max_audios

    info = []
    for machine in machines:
        path = data_path/machine/"test"
        cont_audios = 0
        for fn in os.listdir(path):
            vec = ".".join(fn.split(".")[:-1]).split("_")
            if (
                int(vec[1]) in sections and
                cont_audios < max_audios
            ):
                cont_audios += 1
                info.append({
                    'machine': machine,
                    'section': int(vec[1]),
                    'path': str(data_path/machine/"test"/fn),
                })
    return info

#======Get values of attributes=========================================================
def get_values_attributes(
    data_path,
    machine: str,
    train: Union[bool, List[bool]],
    source: Union[bool, List[bool]],
    sections: Union[int, List[int]],
    attribute: str,
):
    train = [train] if isinstance(train, bool) else train
    source = [source] if isinstance(source, bool) else source
    sections = [sections] if isinstance(sections, int) else sections

    values = set()
    for train, section, source in itertools.product(train, sections, source):
        path = data_path/machine/("train" if train else "test")
        filename = "section_{section}_{source}_{train}_normal".format(
            section = "{:02d}".format(section), 
            source = "source" if source else "target",
            train = "train" if train else "test"
        )
        for fn in sorted(os.listdir(path)):
            if fn.startswith(filename):
                vec = fn.split(".")[0].split("_")
                n = len(vec)
                for i in range(6, n, 2):
                    if vec[i] == attribute:
                        values.add(vec[i+1])
    values = list(values)
    n_classes = len(values)
    return values, n_classes


#======Raw Transformations=========================================================
def process_raw(
    signal: Tensor, 
    sample_rate: int,
    target_sample_rate: int
) -> Tensor:
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        signal = resampler(signal)
    signal = (signal-torch.mean(signal))/torch.std(signal)
    return signal


#======Raw to Spectrogram=========================================================
def raw_to_spec(
    signal: Tensor,
    target_sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    db: bool,
) -> Tensor:
    if n_mels > 0:
        transformation = torchaudio.transforms.MelSpectrogram(
            sample_rate = target_sample_rate,
            n_fft = n_fft, 
            hop_length = hop_length, 
            n_mels = n_mels,
        )
        signal = transformation(signal)
    else:
        transformation = torchaudio.transforms.Spectrogram(
            n_fft = n_fft, 
            hop_length = hop_length,
        )
        signal = transformation(signal)[:n_fft//2,:]
    if db:
        power_to_db = torchaudio.transforms.AmplitudeToDB()
        signal = power_to_db(signal)
    return signal


#======Spectrogram=========================================================
def spectrogram(
    signal: Tensor, 
    sample_rate: int,
    target_sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    db: bool,
) -> Tensor:
    signal = process_raw(signal, sample_rate, target_sample_rate)
    signal = raw_to_spec(signal, target_sample_rate, n_fft, hop_length, n_mels, db)
    return signal


def normalize_spectrograms(
    spectrograms: List[Tensor], 
    stats_path: str
) -> List[Tensor]:
    path = Path(__file__).resolve().parents[0]
    stats_path = "{}/stats/{}.h5".format(path, stats_path)
    with h5py.File(stats_path, 'r') as f:
        mean, std = f['mean'][()], f['std'][()]
    for i in range(len(spectrograms)):
        spectrograms[i] = ((spectrograms[i].permute(2,1,0)-mean)/(std+1e-6)).permute(2,1,0)
    return spectrograms


def load_spectrograms(
    info_list: List[dict],
    target_sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    db: bool,
) -> List[Tensor]:
    specs = []
    for info in info_list:
        signal, sample_rate = torchaudio.load(info['path'])
        spec = spectrogram(
            signal=signal, 
            sample_rate=sample_rate, 
            target_sample_rate=target_sample_rate, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            n_mels=n_mels, 
            db=db)
        specs.append(spec)
    return specs


def save_training_stats(
    data_path: str,
    machines: str,
    sections: Union[int, List[int]],
    target_sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    db: bool,
    stats_path: str,
):
    path = Path(__file__).resolve().parents[0]
    Path(path/"stats").mkdir(parents=True, exist_ok=True)
    stats_path = "{}/stats/{}.h5".format(path, stats_path)
    info = filter_info(data_path=data_path, machines=machines, sections=sections)
    specs = load_spectrograms(info, target_sample_rate, n_fft, hop_length, n_mels, db)
    mean = torch.stack(specs).mean()
    std = torch.stack(specs).std()
    with h5py.File(stats_path, 'w') as f:
        f.create_dataset('mean', data=mean)
        f.create_dataset('std', data=std)


#======Embeddings=========================================================
def load_embeddings(
    info_list: List[dict],
) -> List[Tensor]:
    embeds = []
    for info in info_list:
        with h5py.File(info['path'], 'r') as f:
            embeddings = f['embeddings'][()]
            n_chunks_audio = embeddings.shape[0]
        embeds.extend([embedding for embedding in embeddings])
    return embeds, n_chunks_audio