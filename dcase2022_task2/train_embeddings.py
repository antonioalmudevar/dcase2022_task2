from pathlib import Path

import yaml
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .datasets import *
from .embeddings import get_embeddings_extractor
from .classifiers import get_classifier
from .utils import *

DATA_PATH = Path(__file__).resolve().parents[1]/"data"

def train(args):
    print("\n-------------------------------")
    print("|TRAINING  MACHINES CLASSIFIER|")
    print("-------------------------------")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    #======Read and process args========================================================= 
    dir_name = "_".join([str(i) for i in args.sections])
    path = Path(__file__).resolve().parents[1]
    with open(path/("configs/data/"+args.config_data+".yaml"), 'r') as f:
        cfg_data = yaml.load(f, yaml.FullLoader)
    with open(path/("configs/classifier/"+args.config_class+".yaml"), 'r') as f:
        cfg_class = yaml.load(f, yaml.FullLoader)
    models_dir = path/"models"/args.config_data/args.config_class/dir_name
    plots_dir = path/"plots"/"tsne"/args.config_data/args.config_class/dir_name
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    #======Save Training Data Statistics=========================================================
    save_training_stats(
        data_path=DATA_PATH,
        machines=MACHINES,
        sections=args.sections,
        target_sample_rate=cfg_data['target_sample_rate'],
        n_fft=cfg_data['n_fft'],
        hop_length=cfg_data['hop_length'],
        n_mels=cfg_data['n_mels'],
        db=cfg_data['db'],
        stats_path=args.config_data+"_"+dir_name,
    )

    #======Load Train Dataset=========================================================
    train_dataset = DatasetTrain(
        data_path=DATA_PATH,
        train=True, 
        source=True, 
        normal=True, 
        sections=args.sections,
        stats_path=args.config_data+"_"+dir_name,
        **cfg_data,
    )
    train_loader = DataLoader(
        dataset=train_dataset, 
        shuffle=True,
        batch_size=cfg_class['training']['batch_size'],
        num_workers=4,
        pin_memory=True
    )

    #======Load Validation Dataset=========================================================
    val_dataset = DatasetTrain(
        data_path=DATA_PATH,
        train=False, 
        source=[True, False], 
        normal=[True, False],  
        sections=DEV_SECTIONS,
        stats_path=args.config_data+"_"+dir_name,
        **cfg_data
    )
    val_loader = DataLoader(
        dataset=val_dataset, 
        shuffle=False,
        batch_size=cfg_class['training']['batch_size'],
        num_workers=4,
        pin_memory=True
    )

    #======Create models=========================================================
    embeddings_net = get_embeddings_extractor(
        n_mels=cfg_data['n_mels'],
        n_columns=cfg_data['n_columns'],
        **cfg_class['embeddings_extractor'],
    ).to(device=device, dtype=torch.float)

    class_net = get_classifier(
        embedding_dim=cfg_class['embeddings_extractor']['embedding_dim'],
        n_classes=train_dataset.n_classes,
        **cfg_class['classifier'],
    ).to(device=device, dtype=torch.float)

    parameters = list(embeddings_net.parameters())+list(class_net.parameters())
    optimizer = torch.optim.Adam(parameters)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer=optimizer,
        base_lr=cfg_class['training']['min_lr'],
        max_lr=cfg_class['training']['max_lr'],
        step_size_up=cfg_class['training']['step_size'],
        cycle_momentum=False
    )
    nepochs = cfg_class['training']['nepochs']

    previous_models = os.listdir(models_dir)
    if len(previous_models)>0:
        checkpoint = torch.load(models_dir/previous_models[-1])
        ini_epoch = checkpoint['epoch']
        embeddings_net.load_state_dict(checkpoint['embeddings'])
        class_net.load_state_dict(checkpoint['class'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        ini_epoch = 0

    embeddings_net = torch.nn.DataParallel(embeddings_net)
    class_net = torch.nn.DataParallel(class_net)

    '''''
    #======Print Previous Losses========================================================= 
    print("Epoch\t| Training Loss\t\t| Validation Loss\t|")
    for epoch in range(0, ini_epoch):
        print("{}/{}\t| {:.4f}\t\t\t| {:.4f}\t\t\t| ".format(
            epoch, nepochs, checkpoint[epoch]['train_loss'], checkpoint[epoch]['val_loss']))
    '''''
            
    #======Training========================================================= 
    for epoch in range(ini_epoch+1, nepochs+1):
        train_loss, val_loss = 0, 0
        val_embeddings = []
        #======Training Epoch=======
        embeddings_net.train(), class_net.train()
        for signal, labels in train_loader:
            signal = Variable(signal).to(device=device, dtype=torch.float, non_blocking=True)
            labels = Variable(labels).to(device=device, dtype=torch.float, non_blocking=True)
            signal, labels = mixup_data(signal, labels,alpha=cfg_class['training']['mixup_factor'])
            #======Forward=======
            pred_embeddings = embeddings_net(signal)
            pred_labels = class_net(pred_embeddings, labels)
            #======Losses=======
            loss = torch.mean(cross_entropy(pred_labels, labels))
            train_loss += loss.detach()
            #======Backward=======
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            for param in parameters:
                param.grad = None
        scheduler.step()
        train_loss /= len(train_loader)

        #======Validation Epoch=======
        embeddings_net.eval(), class_net.eval()
        for signal, labels in val_loader:
            signal = Variable(signal).to(device=device, dtype=torch.float, non_blocking=True)
            labels = Variable(labels).to(device=device, dtype=torch.float, non_blocking=True)
            with torch.no_grad():
                #======Forward=======
                pred_embeddings = embeddings_net(signal)
                if args.plot_tsne:  # Memory problem in some cases
                    val_embeddings.extend(pred_embeddings)
                pred_labels = class_net(pred_embeddings, labels)
                #======Losses=======
                loss = torch.mean(cross_entropy(pred_labels, labels))
                val_loss += loss.detach()
        val_loss /= len(val_loader)

        #======Logs=======
        print("{}/{}\t| {:.4f}\t\t\t| {:.4f}\t\t\t| ".format(
            epoch, nepochs, train_loss, val_loss))
        
        #======Plot t-SNE=======
        if args.plot_tsne:
            val_embeddings = torch.stack(val_embeddings).\
                reshape(-1, val_dataset.n_chunks_audio, \
                    cfg_class['embeddings_extractor']['embedding_dim']).\
                        mean(dim=1).cpu().numpy()
            labels = [info['machine'] for info in val_dataset.info]
            plot_tsne(val_embeddings, labels, len(MACHINES), plots_dir/("epoch"+str(epoch)))
        
        #======Save checkpoint=======
        if epoch % 5 == 0:
            embed_path =  models_dir/("embeddings_extractor_"+str(epoch)+".pt")
            checkpoint = {
                'embeddings': embeddings_net.module.state_dict(), 
                'class': class_net.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, embed_path)