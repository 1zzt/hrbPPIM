import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from dataset import PPIMIDataset
from torch_geometric.loader import DataLoader, DataListLoader
import torch
import torch.nn as nn
from model import *
# from utils import EvalMeter
import copy
import math
from torch_geometric.nn import DataParallel
import datetime
import numpy as np
from utils import *
from tqdm import tqdm
import argparse
import tempfile
tempfile.tempdir = "/home/tmp"

now_time = datetime.datetime.now()



seed = 17
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


parser = argparse.ArgumentParser(description='PyTorch implementation of MultiPPIMI')
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--emb', type=int, default=256)
parser.add_argument('--heads', type=int, default=4)



args = parser.parse_args()
eval_setting = args.eval_setting
dataseed = args.dataseed

batch_size = args.batch_size
num_epochs = args.num_epochs

emb = args.emb
heads = args.heads

device = torch.device(f'cuda:{args.device}')



def run_a_train_epoch(device, epoch, model, data_loader, criterion, optimizer):
    model.train()
    tbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for _, data in tbar:

        data = data.to(device)
        y = data.y.float().to(device)
        
        optimizer.zero_grad()

        output = model(data).float()

        loss =  criterion(output.view(-1), y.view(-1))


        loss.backward()
        optimizer.step()

        tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item()  :.3f}')
        # tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item()  :.3f}  DSLoss={main_loss.item()  :.3f}  AUX_loss={cl_loss.item()  :.3f} ')


def run_an_eval_epoch(model, data_loader, criterion):
    model.eval()
    running_loss = AverageMeter()
    with torch.no_grad():
        preds =  torch.Tensor()
        trues = torch.Tensor()

        for _, data in tqdm(enumerate(data_loader)):

            data = data.to(device)
            y = data.y.float().to(device)
        
            logits = model(data).float()

            loss =  criterion(logits.view(-1), y.view(-1))

            # logits = torch.sigmoid(logits)
            preds = torch.cat((preds, logits.cpu()), 0)
            trues = torch.cat((trues, y.view(-1, 1).cpu()), 0)
            running_loss.update(loss.item(), y.size(0))
        preds, trues = preds.numpy().flatten(), trues.numpy().flatten()
    val_loss =  running_loss.get_average()
    return preds, trues, val_loss

for fold in range(5):


    train_dataset = PPIMIDataset('train', fold)
    valid_dataset = PPIMIDataset( 'val', fold)
    test_dataset = PPIMIDataset('test', fold)



    train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size= batch_size, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size= batch_size, shuffle=False, drop_last=False)


    model = GPANN(emb=emb, heads=heads).to(device) # 输入特征维度，隐藏特征维度，输出特征维度

    optimizer = torch.optim.AdamW(model.parameters(), lr= 1e-4)

    criterion = nn.BCEWithLogitsLoss()


    best_model = None
    best_roc_auc = 0.0
    patience = 20
    counter = 0         

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}')

        run_a_train_epoch(device, epoch, model, train_dataloader, criterion, optimizer)


        P, G, _ = run_an_eval_epoch(model, valid_dataloader, criterion)
        current_roc_auc, current_aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels = \
            performance_evaluation(P, G)

        print('Val AUC:\t{}'.format(current_roc_auc))
        print('Val AUPR:\t{}'.format(current_aupr))

        if current_roc_auc > best_roc_auc:
            best_roc_auc = current_roc_auc
            best_model = copy.deepcopy(model)
            counter = 0
            print('Validation AUC improved, saving best model.')
        else:
            counter += 1
            print(f'No improvement. Early stop counter: {counter}/{patience}')

        if counter >= patience:
            print('Early stopping triggered.')
            break


    P, G, _ = run_an_eval_epoch(best_model, test_dataloader, criterion)
    roc_auc, aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels = \
        performance_evaluation(P, G)

    print('Fold {} Test Results:'.format(fold))
    print('Test AUC:\t{}'.format(roc_auc))
    print('Test AUPR:\t{}'.format(aupr))