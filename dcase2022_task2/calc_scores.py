from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .datasets import *
from .utils.measures import *

DATA_PATH = Path(__file__).resolve().parents[1]/"data"

__all__ = ["calc_scores_1", "calc_scores_2", "calc_scores_3"]

def get_scores(preds_dir, machine, section):

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
    cd_as, paths = [], []
    for embeddings, path_audio in test_loader:
        cd_as.extend(knn.kneighbors(embeddings)[0])
        paths.extend(path_audio)
        #cd_as.extend(cosine_distance(train_embeddings, embeddings))
    paths = [paths[i] for i in range(0,len(paths),test_dataset.n_chunks_audio)]
    paths = [path_h5.split("/")[-1][:-2]+"wav" for path_h5 in paths]
    cd_as = Tensor(cd_as).reshape(-1, test_dataset.n_chunks_audio)
    cd_as = cd_as.mean(dim=1).tolist()
    
    return dict(zip(paths, cd_as))


def calc_scores_1(args):
    print("---------------------------")
    print("|CALCULATE SCORES SYSTEM 1|")
    print("---------------------------")

    #======Read args========================================================= 
    path = Path(__file__).resolve().parents[1]
    csv_dir = path/"results"/args.config_data/args.config_class/args.dataset/\
        "Almudevar_UZ_task2_1"
    Path(csv_dir).mkdir(parents=True, exist_ok=True)

    dir_name = "_".join([str(i) for i in [0,1,2,3,4,5]])
    embed_epochs = 15
    preds_dir = path/"predictions"/args.config_data/args.config_class/dir_name\
        /(str(embed_epochs)+"_embed_epochs")

    sections = EVAL_SECTIONS if args.dataset=="eval" else DEV_SECTIONS
    for machine in MACHINES:
        for section in sections:
            l = get_scores(preds_dir, machine, section)
            csv_file = "anomaly_score_{}_section_{:02d}_test.csv".format(machine, section)
            with open(csv_dir/csv_file, 'w') as f:
                for key in l.keys():
                    f.write("%s,%s\n" % (key, l[key]))


def calc_scores_2(args):
    print("---------------------------")
    print("|CALCULATE SCORES SYSTEM 2|")
    print("---------------------------")

    #======Read args========================================================= 
    path = Path(__file__).resolve().parents[1]
    csv_dir = path/"results"/args.config_data/args.config_class/args.dataset/\
        "Almudevar_UZ_task2_1"
    Path(csv_dir).mkdir(parents=True, exist_ok=True)

    dir_name = "_".join([str(i) for i in [3,4,5]]) if args.dataset=="eval" else\
        "_".join([str(i) for i in [0,1,2]])
    embed_epochs = 30
    preds_dir = path/"predictions"/args.config_data/args.config_class/dir_name\
        /(str(embed_epochs)+"_embed_epochs")

    sections = EVAL_SECTIONS if args.dataset=="eval" else DEV_SECTIONS
    for machine in MACHINES:
        for section in sections:
            l = get_scores(preds_dir, machine, section)
            csv_file = "anomaly_score_{}_section_{:02d}_test.csv".format(machine, section)
            with open(csv_dir/csv_file, 'w') as f:
                for key in l.keys():
                    f.write("%s,%s\n" % (key, l[key]))


def calc_scores_3(args):
    print("---------------------------")
    print("|CALCULATE SCORES SYSTEM 3|")
    print("---------------------------")

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
        'ToyCar': {0:30, 1:30, 2:30, 3:30, 4:30, 5:30},
        'ToyTrain': {0:25, 1:25, 2:25, 3:25, 4:25, 5:25},
        'bearing': {0:15, 1:15, 2:15, 3:15, 4:15, 5:15},
        'fan': {0:16, 1:16, 2:25, 3:16, 4:16, 5:25},
        'gearbox': {0:10, 1:10, 2:10, 3:10, 4:10, 5:10},
        'slider': {0:25, 1:25, 2:25, 3:25, 4:25, 5:25},
        'valve': {0:21, 1:25, 2:25, 3:21, 4:25, 5:25},
    }

    #======Read args========================================================= 
    path = Path(__file__).resolve().parents[1]
    csv_dir = path/"results"/args.config_data/args.config_class/args.dataset/\
        "Almudevar_UZ_task2_1"
    Path(csv_dir).mkdir(parents=True, exist_ok=True)

    sections = EVAL_SECTIONS if args.dataset=="eval" else DEV_SECTIONS
    for machine in MACHINES:
        for section in sections:
            dir_name = "_".join([str(i) for i in embeddings_sections[machine][section]])
            embed_epochs = embed_epochs_machine[machine][section]
            preds_dir = path/"predictions"/args.config_data/args.config_class/dir_name\
                /(str(embed_epochs)+"_embed_epochs")
            l = get_scores(preds_dir, machine, section)
            csv_file = "anomaly_score_{}_section_{:02d}_test.csv".format(machine, section)
            with open(csv_dir/csv_file, 'w') as f:
                for key in l.keys():
                    f.write("%s,%s\n" % (key, l[key]))