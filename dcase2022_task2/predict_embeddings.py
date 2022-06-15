from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

from .datasets import *
from .embeddings import get_embeddings_extractor

DATA_PATH = Path(__file__).resolve().parents[1]/"data"

def predict(args):

    print("--------------------")
    print("|PREDICT EMBEDDINGS|")
    print("--------------------")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True


    #======Read args========================================================= 
    dir_name = "_".join([str(i) for i in args.sections])
    path = Path(__file__).resolve().parents[1]
    with open(path/("configs/data/"+args.config_data+".yaml"), 'r') as f:
        cfg_data = yaml.load(f, yaml.FullLoader)
    with open(path/("configs/classifier/"+args.config_class+".yaml"), 'r') as f:
        cfg_class = yaml.load(f, yaml.FullLoader)
    models_dir = path/"models"/args.config_data/args.config_class/dir_name
    preds_dir = path/"predictions"/args.config_data/args.config_class/dir_name/\
        (str(args.embed_epochs)+"_embed_epochs")


    #======Load Embeddings Extractor=========================================================
    embeddings_net = get_embeddings_extractor(
        n_mels=cfg_data['n_mels'],
        n_columns=cfg_data['n_columns'],
        **cfg_class['embeddings_extractor'],
    ).to(device=device, dtype=torch.float)
    embed_path =  models_dir/("embeddings_extractor_"+str(args.embed_epochs)+".pt")
    embeddings_net.load_state_dict(torch.load(embed_path)['embeddings'])
    
    '''''
    embeddings_net_10 = get_embeddings_extractor(
        n_mels=cfg_data['n_mels'],
        n_columns=cfg_data['n_columns'],
        **cfg_class['embeddings_extractor'],
    ).to(device=device, dtype=torch.float)
    embed_path =  models_dir/("embeddings_extractor_10.pt")
    embeddings_net_10.load_state_dict(torch.load(embed_path)['embeddings'])
    beta = 0.5
    params_10 = embeddings_net_10.named_parameters()
    params = embeddings_net.named_parameters()
    dict_params = dict(params)
    for name1, param1 in params_10:
        if name1 in dict_params:
            dict_params[name1].data.copy_(beta*param1.data + (1-beta)*dict_params[name1].data)
    embeddings_net.load_state_dict(dict_params)
    '''''
    embeddings_net = torch.nn.DataParallel(embeddings_net)
    embeddings_net.eval()

    for machine in MACHINES:

        print(machine)

        Path(preds_dir/machine/"train").mkdir(parents=True, exist_ok=True)
        Path(preds_dir/machine/"test").mkdir(parents=True, exist_ok=True)

        #======Load Train Dataset=========================================================
        train_dataset = DatasetPredict(
            data_path=DATA_PATH,
            machine=machine,
            train=True, 
            source=[True, False], 
            normal=True, 
            sections=EVAL_SECTIONS,
            stats_path=args.config_data+"_"+dir_name,
            **cfg_data,
        )
        train_loader = DataLoader(
            dataset=train_dataset, 
            shuffle=False,
            batch_size=train_dataset.n_chunks_audio,
            num_workers=4,
            pin_memory=True
        )

        #======Load Test Dataset=========================================================
        test_dataset = DatasetPredict(
            data_path=DATA_PATH,
            machine=machine,
            train=False,
            source=[True, False], 
            normal=[True, False],  
            sections=EVAL_SECTIONS,
            stats_path=args.config_data+"_"+dir_name,
            **cfg_data
        )
        test_loader = DataLoader(
            dataset=test_dataset, 
            shuffle=False,
            batch_size=test_dataset.n_chunks_audio,
            num_workers=4,
            pin_memory=True
        )

        #======Train Embeddings========================================================= 
        for signal, path in train_loader:
            signal = signal.to(device=device, dtype=torch.float, non_blocking=True)
            path = preds_dir/machine/"train"/(path[0].split("/")[-1][:-4]+".h5")
            with torch.no_grad():
                embeddings = embeddings_net(signal).detach()
            with h5py.File(path, 'w') as f:
                f.create_dataset("embeddings", data=embeddings.cpu().numpy())

        #======Test Embeddings========================================================= 
        for signal, path in test_loader:
            signal = signal.to(device=device, dtype=torch.float, non_blocking=True)
            path = preds_dir/machine/"test"/(path[0].split("/")[-1][:-4]+".h5")
            with torch.no_grad():
                embeddings = embeddings_net(signal).detach()
            with h5py.File(path, 'w') as f:
                f.create_dataset("embeddings", data=embeddings.cpu().numpy())