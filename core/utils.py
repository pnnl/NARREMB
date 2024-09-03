'''
Helper functions for modeling and logging
'''

import torch 
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import os
import csv
import json
from models import *
from datasets import *
from losses import *

def build_tokenizer(model_name):
    '''
    Returns the tokenizer object corresponding to the pre-trained model

    token_type - not needed if using AutoTokenizer
    model_name - path to pre-trained language model
    '''
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    return tokenizer


def build_optimizer(opt_type, model, lr, l2):
    '''
    Returns an optimizer object

    opt_type - name of optimizer. Currently, only Adam is supported
    model - Pytorch model object
    lr - learning rate
    l2 - weight decay
    '''
    if opt_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    else:
        optimizer= None

    return optimizer


def build_model(model_type, model_name, static_model, mode, dropout, norm, pooling, new_dim):
    '''
    Returns a PyTorch model object

    model_type - type of model: first_pass (regression), first_pass_class (classification), or contrastive
    mmodel_name - path to pre-trained language model
    '''
    if model_type == 'regression':
        _model = RegressionModel(model_name=model_name, mode=mode, norm=norm, dropout=dropout, pooling=pooling)
    elif model_type == 'classification':
        _model = ClassModel(model_name=model_name, mode=mode, norm=norm, dropout=dropout, pooling=pooling)
    elif model_type == 'contrastive':
        _model = ContrastiveModel(model_name=model_name, norm=norm, dropout=dropout, pooling=pooling)
    elif model_type == 'edcontrastive':
        _model = EDContrastiveModel(model_name=model_name, norm=norm, dropout=dropout, pooling=pooling, new_dim=new_dim)
    elif model_type == 'edclassification':
        _model = EDClassModel(model_name=model_name, norm=norm, dropout=dropout, pooling=pooling, new_dim=new_dim)
    elif model_type == 'blendcontrastive':
        _model = BlendContrastiveModel(model_name=model_name, static_model=static_model, norm=norm, dropout=dropout, pooling=pooling, new_dim=new_dim)
    else:
        _model = None

    return _model


def build_criterion(loss_type, reduction, margin=None):
    '''
    Returns a loss function object

    loss_type - the loss function to use
    reduction - specifies the reduction to apply. Default is mean: the sum of output will be divided by the number of elements
    margin - threshold value for contrastive loss
    '''
    if loss_type == "mae":
        _criterion = torch.nn.L1Loss(reduction=reduction)
    elif loss_type == "mse":
        _criterion = torch.nn.MSELoss(reduction=reduction)
    elif loss_type == "poisson":
        _criterion = torch.nn.PoissonNLLLoss(log_input=False, reduction=reduction)
    elif loss_type == "bce":
        _criterion = torch.nn.BCELoss()
    elif loss_type == 'cosine':
        _criterion = CosineLoss(margin=margin)
    elif loss_type == 'euclidean':
        _criterion = ContrastiveLoss(metric=loss_type, margin=margin)
    else:
        _criterion = None
    
    return _criterion


def build_datasets(dset_path, phases, sp_thres):
    '''
    Returns a dictionary of dataset objects

    dset_path - path to train, val and test data
    phases - list of phases. E.g., [train, val]
    sp_thres - shortest path length threshold 
    '''
    datasets = {}
    for phase in phases:
        datasets[phase] = NetworkTwitterDataset(dset_path, phase, sp_thres)

    return datasets


def build_dataloaders(datasets, phases, batch_size):
    '''
    Returns a dictionary of DataLoaders

    datasets - dictionary of dataset objects
    phases - list of phases
    batch_size - number of samples per batch
    '''
    dataloaders = {}
    for phase in phases:
        dataloaders[phase] = DataLoader(datasets[phase], batch_size=batch_size,
                                        shuffle=True, num_workers=0)

    return dataloaders


def get_track_value(prev_loss,track_metric=None,list_metrics=None):
    '''
    Helper function to track metric performance during training. 
    It returns the current value of the metric and 
    a condition: True if new value is better than previous value, otherwise False

    prev_loss - previous value of the metric of interest
    track_metric - metric to track
    list_metric - list of metric scores for current epoch
    '''
    track_loss = None
    if (track_metric == "loss"):
        track_loss = list_metrics[0]
        condition = track_loss < prev_loss
    elif (track_metric == 'acc'):
        track_loss = list_metrics[1]
        condition = track_loss > prev_loss
    elif (track_metric == 'prec'):
        track_loss = list_metrics[2]
        condition = track_loss > prev_loss
    elif (track_metric == 'recall'):
        track_loss = list_metrics[3]
        condition = track_loss > prev_loss
    elif (track_metric == "mae"):
        track_loss = list_metrics[4]
        condition = track_loss < prev_loss
    elif (track_metric == "mse"):
        track_loss = list_metrics[5]
        condition = track_loss < prev_loss
    elif (track_metric == "r2"):
        track_loss = list_metrics[6]
        condition = track_loss > prev_loss
    elif (track_metric == "corr"):
        track_loss = list_metrics[7]
        condition = track_loss < prev_loss
    elif (track_metric == "group_corr"):
        track_loss = list_metrics[8]
        condition = track_loss < prev_loss
    else:
        raise Exception("Sorry, invalid tracking metric")
    return track_loss, condition


class CustomLogger():
    '''
    Logger Class to log performance progress for Regression and Contrastive models. 
    To avoid creating an additional logger class, dummy regression metrics will still be computed for contrastive models.
    '''
    def __init__(self, phases, log_root, run_id, device):

        metric_head = ['epoch', 'loss', 'mae', 'mse', 'r2', 'corr', 'group_corr']
        sample_head = ['epoch', 'train', 'tweet1', 'tweet2', 'pred', 'gt']

        self.metrics_dict = dict()
        self.samples_dict = dict()

        self.phases = phases
        self.log_root = log_root
        self.run_id = run_id

        self.epoch_loss_sum = 0 # sum of loss over batches
        self.epoch_mae_sum = 0 # sum of mae (not averaged) over batches
        self.epoch_mse_sum = 0 # sum of mse (not averaged) over batches
        self.epoch_r2 = 0 # R2 score
        self.epoch_num_samples = 0 # total number of samples processed
        
        self.y_pred = torch.empty(0) # tensor of predictions
        self.y_true = torch.empty(0) # tensor of ground truth values
        
        self.sims = torch.empty(0) # tensor for storing pairwise cosine similarity values

        for phase in self.phases:
            tmp_metrics = list()
            tmp_metrics.append(metric_head)

            tmp_samples = list()
            tmp_samples.append(sample_head)

            self.metrics_dict[phase] = tmp_metrics
            self.samples_dict[phase] = tmp_samples
            

    def log_epoch_metrics(self, epoch, phase, loss, mae, mse, r2, corr, group_corr):
        '''
        Append metrics of interest to logger
        '''
        self.metrics_dict[phase].append([epoch, loss, mae, mse, r2, corr, group_corr])

    def start_epoch_logger(self):
        '''
        Reset logger metrics and variables of interest each epoch
        '''
        self.epoch_loss_sum = 0 
        self.epoch_mae_sum = 0 
        self.epoch_mse_sum = 0
        self.epoch_num_samples = 0
        self.y_pred = torch.empty(0)
        self.y_true = torch.empty(0)
        self.sims = torch.empty(0)

    def add_epoch_totals(self, loss, mae, mse, num_samples):
        '''
        Accumulate errors for metrics of interest
        '''
        self.epoch_loss_sum += loss * num_samples
        self.epoch_mae_sum += mae
        self.epoch_mse_sum += mse
        self.epoch_num_samples += num_samples


    def get_metrics(self, pred, labels, cos_sim, phase, round_=False):
        '''
        Returns metrics of interest

        pred - tensor of predictions
        labels - tensor of ground truth values
        cos_sim - tensor of cosine similarities for tweet pairs
        phase - whether train, val, or test
        round_ - if True, round predictions
        '''
        pred = pred.detach() # We detach since these tensors do not require gradients
        labels = labels.detach()
        cos_sim = cos_sim.detach()
        
        if round_:
            pred = torch.round(pred) # Round predictions
        
        # Compute MSE sum
        mse_sum = ((labels - pred)**2).sum().item()
        # Compute MAE sum
        mae_sum = torch.abs(labels - pred).sum().item()
        num_samples = len(labels)
        
        if phase != "train":
            # Append current batch predictions and ground truth
            self.y_pred = torch.cat((self.y_pred, pred.to("cpu")))
            self.y_true = torch.cat((self.y_true, labels.to("cpu")))
            self.sims = torch.cat((self.sims, cos_sim.to("cpu")))
        
        return num_samples, mae_sum, mse_sum
    
    
    def get_r2_score(self, is_test=False):
        '''
        Returns R2 score for val or test splits

        is_test - flag to indicate whether we are in test or val phase
        '''
        if is_test:
            pred = self.y_pred.view(-1).cpu().numpy()
            labels = self.y_true.view(-1).cpu().numpy()
            r2 = r2_score(labels, pred)
        else: 
            r2 = np.inf
        return r2
    
    def get_corr(self, is_test=False):
        '''
        Returns Pearson correlation between cosine similarity and shortest paths for val or test splits

        is_test - flag to indicate whether we are in test or val phase
        '''
        if is_test:
            labels = self.y_true.view(-1).cpu().numpy()
            cos_sim = self.sims.view(-1).cpu().numpy()
            corr, _ = pearsonr(labels, cos_sim)
        else:
            corr = np.inf
        return corr
    
    
    def get_group_corr(self, is_test=False):
        '''
        Returns Pearson correlation between median cosine similarities per shortest path lengths for val or test splits

        is_test - flag to indicate whether we are in test or val phase
        '''

        sp = []
        cos = []
        if is_test:
            labels = self.y_true.view(-1).cpu().numpy()
            cos_sim = self.sims.view(-1).cpu().numpy()

            mask = labels <= 8 # Ignore paths with more than 8 hops away
            labels = labels[mask]
            cos_sim = cos_sim[mask]

            for i in np.unique(labels): # Iterate through each path length
                mask = labels == i
                t = cos_sim[mask]
                cos.append(np.median(t)) # Compute the median cosine similarity score
                sp.append(i)
            corr, _ = pearsonr(sp, cos)
        else:
            corr = np.inf
        return corr


    def get_percentile_corr(self, is_test=False):
        '''
        Returns a list of Pearson correlations between cosine similarity across multiple percentiles 
        per shortest path length for val and test splits
        
        is_test - flag to indicate whether we are in test or val phase
        '''
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]

        if is_test:
            labels = self.y_true.view(-1).cpu().numpy()
            cos_sim = self.sims.view(-1).cpu().numpy()

            cos_percentiles = []
            shortest_paths = np.tile(np.unique(labels), (len(percentiles), 1))

            for i in np.unique(labels): # Obtain cosine similiaries per short path
                mask = labels == i
                t = cos_sim[mask]
                cos_percentiles.append(np.percentile(t, percentiles))

            cos_percentiles = np.array(cos_percentiles).T 
            corr_percentiles = [pearsonr(row, cos_percentiles[i])[0] for i, row in enumerate(shortest_paths)]
        else:
            corr_percentiles = [np.inf]*len(percentiles)
        return corr_percentiles       


    def get_running_metrics(self):
        '''
        Returns metrics of interest at each iteration
        '''
        running_loss = self.epoch_loss_sum / self.epoch_num_samples
        running_mae = self.epoch_mae_sum / self.epoch_num_samples
        running_mse = self.epoch_mse_sum / self.epoch_num_samples

        return running_loss, running_mae, running_mse


    def save_metrics(self):
        '''
        Saves logger for all epochs
        '''
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        for phase in self.phases:
            metric_fn = f"{self.run_id}_{phase}_metrics.csv"
            with open(os.path.join(self.log_root, metric_fn), 'w') as f:
                # create the csv writer
                writer = csv.writer(f)
                # write a row to the csv file
                writer.writerows(self.metrics_dict[phase])         
                

class CustomLoggerClass():
    '''
    Logger Class to log performance progress for classification models. 
    '''
    def __init__(self, phases, log_root, run_id, device):

        metric_head = ['epoch', 'loss', 'acc', 'precision', 'recall', 'corr', 'group_corr']

        sample_head = ['epoch', 'train', 'tweet1', 'tweet2', 'pred', 'url1', 'url2', 'label']

        self.metrics_dict = dict()
        self.samples_dict = dict()

        self.phases = phases
        self.log_root = log_root
        self.run_id = run_id

        self.epoch_loss = 0
        self.epoch_correct = 0
        self.epoch_total = 0
        self.epoch_TP = 0
        self.epoch_FP = 0
        self.epoch_FN = 0
        
        self.y_true = torch.empty(0) # tensor to store ground truth values
        self.sims = torch.empty(0) # tensor to store cosine similarities

        for phase in self.phases:
            tmp_metrics = list()
            tmp_metrics.append(metric_head)

            tmp_samples = list()
            tmp_samples.append(sample_head)

            self.metrics_dict[phase] = tmp_metrics
            self.samples_dict[phase] = tmp_samples

    def log_epoch_metrics(self, epoch, phase, loss, acc, precision, recall, corr, group_corr):
        self.metrics_dict[phase].append([epoch, loss, acc, precision, recall, corr, group_corr])

    def start_epoch_logger(self):
        self.epoch_loss = 0
        self.epoch_correct = 0
        self.epoch_total = 0
        self.epoch_TP = 0
        self.epoch_FP = 0
        self.epoch_FN = 0
        self.y_true = torch.empty(0)
        self.sims = torch.empty(0)

    def add_epoch_totals(self, num_correct, total, TP, FP, FN, loss):
        self.epoch_loss += loss
        self.epoch_correct += num_correct
        self.epoch_total += total
        self.epoch_TP += TP
        self.epoch_FP += FP
        self.epoch_FN += FN

   
    def get_metrics(self, pred, labels, y_true, cos_sim, phase):
        '''
        Returns classification metrics

        pred - prediction labels
        labels - ground truth labels
        y_true - ground truth shortest path lengths
        cos_sim - tensor of cosine similarity values
        '''
        pred = pred.detach()
        labels = labels.detach()
        pred = (pred > 0.5).float()
        pred_1s = (pred == 1)
        label_1s = (labels == 1)

        # acc
        total = len(labels)
        num_correct = (pred_1s == label_1s).float().sum().item()

        # TruePos, FalsePos, FalseNeg
        TP = (pred_1s & label_1s).sum().item()
        FP = (pred_1s & ~label_1s).sum().item()
        FN = (~pred_1s & label_1s).sum().item()
        
        if phase != "train":
            # Append predictions and ground truth
            self.y_true = torch.cat((self.y_true, y_true.to("cpu")))
            self.sims = torch.cat((self.sims, cos_sim.to("cpu")))

        return num_correct, total, TP, FP, FN


    def get_running_metrics(self):
        # Work-around for divide by 0
        try:
            running_precision = self.epoch_TP / (self.epoch_TP + self.epoch_FP)
        except:
            running_precision = 0
        try:
            running_recall = self.epoch_TP / (self.epoch_TP + self.epoch_FN)
        except:
            running_recall = 0

        running_accuracy = self.epoch_correct / self.epoch_total
        running_loss = self.epoch_loss / self.epoch_total

        return running_loss, running_accuracy, running_precision, running_recall
    

    def get_corr(self, is_test=False):
        if is_test:
            labels = self.y_true.view(-1).cpu().numpy()
            cos_sim = self.sims.view(-1).cpu().numpy()
            corr, _ = pearsonr(labels, cos_sim)
        else:
            corr = np.inf
        return corr
    

    def get_group_corr(self, is_test=False):

        sp = []
        cos = []
        if is_test:
            labels = self.y_true.view(-1).cpu().numpy()
            cos_sim = self.sims.view(-1).cpu().numpy()

            mask = labels <= 8
            labels = labels[mask]
            cos_sim = cos_sim[mask]

            for i in np.unique(labels):
                mask = labels == i
                t = cos_sim[mask]
                cos.append(np.median(t))
                sp.append(i)
            corr, _ = pearsonr(sp, cos)
        else:
            corr = np.inf
        return corr


    def save_metrics(self):

        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        for phase in self.phases:
            metric_fn = f"{self.run_id}_{phase}_metrics.csv"
            with open(os.path.join(self.log_root, metric_fn), 'w') as f:
                # create the csv writer
                writer = csv.writer(f)

                # write a row to the csv file
                writer.writerows(self.metrics_dict[phase])