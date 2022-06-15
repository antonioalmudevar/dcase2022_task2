from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .datasets import *
from .utils import auc

__all__ = ["evaluate_1", "evaluate_2", "evaluate_3"]

DATA_PATH = Path(__file__).resolve().parents[1]/"data"


def get_idx(sections, info):
    idx = {}
    for section in sections:
        normal = [i for i in range(len(info)) if 
            info[i]['normal'] and info[i]['section'] == section]
        anomaly = [i for i in range(len(info)) if
            not info[i]['normal'] and info[i]['section'] == section]
        normal_source = [i for i in range(len(info)) if 
            info[i]['normal'] and info[i]['source'] and info[i]['section'] == section]
        normal_target = [i for i in range(len(info)) if 
            info[i]['normal'] and not info[i]['source'] and info[i]['section'] == section]
        anomaly_source = [i for i in range(len(info)) if
            not info[i]['normal'] and info[i]['source'] and info[i]['section'] == section]
        anomaly_target = [i for i in range(len(info)) if
            not info[i]['normal'] and not info[i]['source'] and info[i]['section'] == section]
        idx[section] = {
            'normal': normal,
            'anomaly': anomaly,
            'normal_source': normal_source,
            'normal_target': normal_target,
            'anomaly_source': anomaly_source,
            'anomaly_target': anomaly_target
        }
    return idx


def get_aucs(preds_dir, machine, section):

    #======Load Train Source Dataset=========================================================
    train_source_dataset = DatasetEmbedding(
        data_path=preds_dir,
        machine=machine,
        train=True, 
        source=True, 
        normal=True, 
        sections=section,
        max_audios=float("inf"),
    )
    train_source_loader = DataLoader(
        dataset=train_source_dataset, 
        shuffle=False,
        batch_size=1024,
        num_workers=4,
        pin_memory=True
    )

    #======Load Train Target Dataset=========================================================
    train_target_dataset = DatasetEmbedding(
        data_path=preds_dir,
        machine=machine,
        train=True, 
        source=False, 
        normal=True, 
        sections=section,
        max_audios=float("inf"),
    )
    train_target_loader = DataLoader(
        dataset=train_target_dataset, 
        shuffle=False,
        batch_size=1024,
        num_workers=4,
        pin_memory=True
    )

    #======Load Test Dataset=========================================================
    test_dataset = DatasetEmbedding(
        data_path=preds_dir,
        machine=machine,
        train=False, 
        source=[True, False], 
        normal=[True, False],  
        sections=section,
        max_audios=float("inf"),
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        shuffle=False,
        batch_size=1024,
        num_workers=4,
        pin_memory=True
    )

    #======Train Embeddings========================================================= 
    train_embeddings = []
    for embeddings, _ in train_source_loader:
        train_embeddings.extend(embeddings)
    for embeddings,_ in train_target_loader:
        train_embeddings.extend(embeddings)  
    train_embeddings = torch.stack(train_embeddings).numpy()
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    knn.fit(train_embeddings, np.zeros(train_embeddings.shape[0]))

    #======Test Embeddings=========================================================
    cd_as = []
    for embeddings, _ in test_loader:
        cd_as.extend(knn.kneighbors(embeddings)[0])
    cd_as = Tensor(cd_as).reshape(-1, test_dataset.n_chunks_audio)
    cd_as = cd_as.mean(dim=1)

    #======Logs and Scores=========================================================
    idx = get_idx(sections=[section], info=test_dataset.info)
    auc_source = auc(cd_as[idx[section]['normal_source']], cd_as[idx[section]['anomaly_source']])
    auc_target = auc(cd_as[idx[section]['normal_target']], cd_as[idx[section]['anomaly_target']])
    pauc = auc(cd_as[idx[section]['normal']], cd_as[idx[section]['anomaly']], p=0.1)
    
    return auc_source, auc_target, pauc


def evaluate_1(args):
    print("---------------------")
    print("|EVALUATION SYSTEM 1|")
    print("---------------------")

    #======Read args========================================================= 
    path = Path(__file__).resolve().parents[1]
    score = 0

    dir_name = "_".join([str(i) for i in [0,1,2,3,4,5]])
    embed_epochs = 15
    preds_dir = path/"predictions"/args.config_data/args.config_class/dir_name\
        /(str(embed_epochs)+"_embed_epochs")

    for machine in MACHINES:

        print(machine)
        print("\t\t\t| AUC Source\t| AUC Target\t| pAUC \t\t\t|".format(machine))
        arith_mean = {'auc_source': 0, 'auc_target': 0, 'pauc': 0}
        harm_mean = {'auc_source': 0, 'auc_target': 0, 'pauc': 0}

        for section in DEV_SECTIONS:
            auc_source, auc_target, pauc = get_aucs(preds_dir, machine, section)
            arith_mean['auc_source'] = arith_mean['auc_source'] + auc_source
            arith_mean['auc_target'] = arith_mean['auc_target'] + auc_target
            arith_mean['pauc'] = arith_mean['pauc'] + pauc
            harm_mean['auc_source'] = harm_mean['auc_source'] + 1 / auc_source
            harm_mean['auc_target'] = harm_mean['auc_target'] + 1 / auc_target
            harm_mean['pauc'] = harm_mean['pauc'] + 1 / pauc
            score += (1 / auc_source + 1 / auc_target + 1 / pauc)
            print(" Section {}\t| {:.4f}\t\t| {:.4f}\t\t| {:.4f}\t\t|".format(
                section, auc_source, auc_target, pauc))
        print("-------------------------------------------------------------")
        print(" Arith mean\t| {:.4f}\t\t| {:.4f}\t\t| {:.4f}\t\t|".format(
            arith_mean['auc_source']/len(DEV_SECTIONS), 
            arith_mean['auc_target']/len(DEV_SECTIONS),
            arith_mean['pauc']/len(DEV_SECTIONS)
        ))
        print(" Harm mean\t| {:.4f}\t\t| {:.4f}\t\t| {:.4f}\t\t|".format(
            len(DEV_SECTIONS)/harm_mean['auc_source'], 
            len(DEV_SECTIONS)/harm_mean['auc_target'],
            len(DEV_SECTIONS)/harm_mean['pauc']
        ))
        print("=============================================================\n")
                
    score = 3*len(MACHINES)*len(DEV_SECTIONS)/score
    print("\nFINAL SCORE: {:.4f}\n".format(score))


def evaluate_2(args):
    print("---------------------")
    print("|EVALUATION SYSTEM 2|")
    print("---------------------")

    #======Read args========================================================= 
    path = Path(__file__).resolve().parents[1]
    score = 0

    dir_name = "_".join([str(i) for i in [0,1,2]])
    embed_epochs = 30
    preds_dir = path/"predictions"/args.config_data/args.config_class/dir_name\
        /(str(embed_epochs)+"_embed_epochs")

    for machine in MACHINES:

        print(machine)
        print("\t\t\t| AUC Source\t| AUC Target\t| pAUC \t\t\t|".format(machine))
        arith_mean = {'auc_source': 0, 'auc_target': 0, 'pauc': 0}
        harm_mean = {'auc_source': 0, 'auc_target': 0, 'pauc': 0}

        for section in DEV_SECTIONS:
            auc_source, auc_target, pauc = get_aucs(preds_dir, machine, section)
            arith_mean['auc_source'] = arith_mean['auc_source'] + auc_source
            arith_mean['auc_target'] = arith_mean['auc_target'] + auc_target
            arith_mean['pauc'] = arith_mean['pauc'] + pauc
            harm_mean['auc_source'] = harm_mean['auc_source'] + 1 / auc_source
            harm_mean['auc_target'] = harm_mean['auc_target'] + 1 / auc_target
            harm_mean['pauc'] = harm_mean['pauc'] + 1 / pauc
            score += (1 / auc_source + 1 / auc_target + 1 / pauc)
            print(" Section {}\t| {:.4f}\t\t| {:.4f}\t\t| {:.4f}\t\t|".format(
                section, auc_source, auc_target, pauc))
        print("-------------------------------------------------------------")
        print(" Arith mean\t| {:.4f}\t\t| {:.4f}\t\t| {:.4f}\t\t|".format(
            arith_mean['auc_source']/len(DEV_SECTIONS), 
            arith_mean['auc_target']/len(DEV_SECTIONS),
            arith_mean['pauc']/len(DEV_SECTIONS)
        ))
        print(" Harm mean\t| {:.4f}\t\t| {:.4f}\t\t| {:.4f}\t\t|".format(
            len(DEV_SECTIONS)/harm_mean['auc_source'], 
            len(DEV_SECTIONS)/harm_mean['auc_target'],
            len(DEV_SECTIONS)/harm_mean['pauc']
        ))
        print("=============================================================\n")
                
    score = 3*len(MACHINES)*len(DEV_SECTIONS)/score
    print("\nFINAL SCORE: {:.4f}\n".format(score))


def evaluate_3(args):
    print("---------------------")
    print("|EVALUATION SYSTEM 3|")
    print("---------------------")

    #======Read args========================================================= 
    path = Path(__file__).resolve().parents[1]
    score = 0

    embeddings_sections = {
        'ToyCar': {0:[0,1,2], 1:[0,1,2], 2:[0,1,2], 3:[3,4,5], 4:[3,4,5], 5:[3,4,5]},
        'ToyTrain': {0:[0,1,2], 1:[0,1,2], 2:[0,1,2], 3:[3,4,5], 4:[3,4,5], 5:[3,4,5]},
        'bearing': {0:[0,1,2,3,4,5], 1:[0,1,2,3,4,5], 2:[0,1,2,3,4,5], \
            3:[0,1,2,3,4,5], 4:[0,1,2,3,4,5], 5:[0,1,2,3,4,5]},
        'fan': {0:[0,3], 1:[1,4], 2:[2,5], 3:[0,3], 4:[1,4], 5:[2,5]},
        'gearbox': {0:[0,1,2,3,4,5], 1:[0,1,2,3,4,5], 2:[0,1,2,3,4,5], \
            3:[0,1,2,3,4,5], 4:[0,1,2,3,4,5], 5:[0,1,2,3,4,5]},
        'slider': {0:[0,1,2], 1:[0,1,2], 2:[0,1,2], 3:[3,4,5], 4:[3,4,5], 5:[3,4,5]},
        'valve': {0:[0,3], 1:[1,4], 2:[2,5], 3:[0,3], 4:[1,4], 5:[2,5]},
    }

    embed_epochs_machine = {
        'ToyCar':   {0:30, 1:30, 2:30, 3:30, 4:30, 5:30},
        'ToyTrain': {0:25, 1:25, 2:25, 3:25, 4:25, 5:25},
        'bearing':  {0:15, 1:15, 2:15, 3:15, 4:15, 5:15},
        'fan':      {0:16, 1:16, 2:25, 3:16, 4:16, 5:25},
        'gearbox':  {0:10, 1:10, 2:10, 3:10, 4:10, 5:10},
        'slider':   {0:25, 1:25, 2:25, 3:25, 4:25, 5:25},
        'valve':    {0:21, 1:25, 2:25, 3:21, 4:25, 5:25},
    }

    for machine in MACHINES:

        print(machine)
        print("\t\t\t| AUC Source\t| AUC Target\t| pAUC \t\t\t|".format(machine))
        arith_mean = {'auc_source': 0, 'auc_target': 0, 'pauc': 0}
        harm_mean = {'auc_source': 0, 'auc_target': 0, 'pauc': 0}

        for section in DEV_SECTIONS:

            dir_name = "_".join([str(i) for i in embeddings_sections[machine][section]])
            embed_epochs = embed_epochs_machine[machine][section]
            preds_dir = path/"predictions"/args.config_data/args.config_class/dir_name\
                /(str(embed_epochs)+"_embed_epochs")

            auc_source, auc_target, pauc = get_aucs(preds_dir, machine, section)
            arith_mean['auc_source'] = arith_mean['auc_source'] + auc_source
            arith_mean['auc_target'] = arith_mean['auc_target'] + auc_target
            arith_mean['pauc'] = arith_mean['pauc'] + pauc
            harm_mean['auc_source'] = harm_mean['auc_source'] + 1 / auc_source
            harm_mean['auc_target'] = harm_mean['auc_target'] + 1 / auc_target
            harm_mean['pauc'] = harm_mean['pauc'] + 1 / pauc
            score += (1 / auc_source + 1 / auc_target + 1 / pauc)
            print(" Section {}\t| {:.4f}\t\t| {:.4f}\t\t| {:.4f}\t\t|".format(
                section, auc_source, auc_target, pauc))
        print("-------------------------------------------------------------")
        print(" Arith mean\t| {:.4f}\t\t| {:.4f}\t\t| {:.4f}\t\t|".format(
            arith_mean['auc_source']/len(DEV_SECTIONS), 
            arith_mean['auc_target']/len(DEV_SECTIONS),
            arith_mean['pauc']/len(DEV_SECTIONS)
        ))
        print(" Harm mean\t| {:.4f}\t\t| {:.4f}\t\t| {:.4f}\t\t|".format(
            len(DEV_SECTIONS)/harm_mean['auc_source'], 
            len(DEV_SECTIONS)/harm_mean['auc_target'],
            len(DEV_SECTIONS)/harm_mean['pauc']
        ))
        print("=============================================================\n")
                
    score = 3*len(MACHINES)*len(DEV_SECTIONS)/score
    print("\nFINAL SCORE: {:.4f}\n".format(score))