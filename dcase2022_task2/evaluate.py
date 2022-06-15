from pathlib import Path

import yaml
from sklearn.neighbors import KNeighborsClassifier
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .datasets import *
from .utils.measures import *

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


def evaluate(args):
    print("------------")
    print("|EVALUATION|")
    print("------------")

    #======Read args========================================================= 
    path = Path(__file__).resolve().parents[1]
    with open(path/("configs/data/"+args.config_data+".yaml"), 'r') as f:
        cfg_data = yaml.load(f, yaml.FullLoader)
    dir_name = "_".join([str(i) for i in args.sections])
    preds_dir = path/"predictions"/args.config_data/args.config_class/dir_name\
        /(str(args.embed_epochs)+"_embed_epochs")

    score, score_source, score_target = 0, 0, 0

    for machine in ['fan', 'valve']:

        print(machine)
        print("\t\t\t| AUC Source\t| AUC Target\t| pAUC \t\t\t|".format(machine))
        arith_mean = {'auc_source': 0, 'auc_target': 0, 'pauc': 0}
        harm_mean = {'auc_source': 0, 'auc_target': 0, 'pauc': 0}

        for section in DEV_SECTIONS:

            #======Load Train Source Dataset=========================================================
            train_source_dataset = DatasetEmbedding(
                data_path=preds_dir,
                machine=machine,
                train=True, 
                source=True, 
                normal=True, 
                sections=section,
                max_audios=200,
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
                max_audios=10,
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
                max_audios=cfg_data['max_audios'],
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
            for embeddings,_ in train_source_loader:
                train_embeddings.extend(embeddings)
            for embeddings,_ in train_target_loader:
                train_embeddings.extend(embeddings)  
            train_embeddings = torch.stack(train_embeddings).numpy()
            knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
            knn.fit(train_embeddings, np.zeros(train_embeddings.shape[0]))

            #======Test Embeddings=========================================================
            cd_as = []
            for embeddings, path_audio in test_loader:
                cd_as.extend(knn.kneighbors(embeddings)[0])
                #cd_as.extend(cosine_distance(train_embeddings, embeddings))
            cd_as = Tensor(cd_as).reshape(-1, test_dataset.n_chunks_audio)
            cd_as = cd_as.mean(dim=1)

            #======Logs and Scores=========================================================
            idx = get_idx(sections=[section], info=test_dataset.info)
            auc_source = auc(cd_as[idx[section]['normal_source']], cd_as[idx[section]['anomaly_source']])
            auc_target = auc(cd_as[idx[section]['normal_target']], cd_as[idx[section]['anomaly_target']])
            pauc = auc(cd_as[idx[section]['normal']], cd_as[idx[section]['anomaly']], p=0.1)
            arith_mean['auc_source'] = arith_mean['auc_source'] + auc_source
            arith_mean['auc_target'] = arith_mean['auc_target'] + auc_target
            arith_mean['pauc'] = arith_mean['pauc'] + pauc
            harm_mean['auc_source'] = harm_mean['auc_source'] + 1 / auc_source
            harm_mean['auc_target'] = harm_mean['auc_target'] + 1 / auc_target
            harm_mean['pauc'] = harm_mean['pauc'] + 1 / pauc
            score_source += \
                1 / auc(cd_as[idx[section]['normal_source']], cd_as[idx[section]['anomaly_source']])
            score_target += \
                1 / auc(cd_as[idx[section]['normal_target']], cd_as[idx[section]['anomaly_target']])
            score += (
                1 / auc(cd_as[idx[section]['normal_source']], cd_as[idx[section]['anomaly_source']]) +\
                1 / auc(cd_as[idx[section]['normal_target']], cd_as[idx[section]['anomaly_target']]) +\
                1 / auc(cd_as[idx[section]['normal']], cd_as[idx[section]['anomaly']], p=0.1)
            )
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
    score_source = len(MACHINES)*len(DEV_SECTIONS)/score_source
    score_target = len(MACHINES)*len(DEV_SECTIONS)/score_target

    print("\nFINAL SCORE: {:.4f}\t".format(score))
    print("FINAL SCORE SOURCE: {:.4f}\t".format(score_source))
    print("FINAL SCORE TARGET: {:.4f}\t".format(score_target))