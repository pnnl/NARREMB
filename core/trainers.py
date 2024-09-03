import numpy as np
import os
from datetime import datetime
import time
import pickle
from models import *
from datasets import *
from utils import *
  

def base_trainer(params):
    '''
    Function takes a dictionary of parameters and trains the models
    '''
    
    phases = params['phases']
    device = params['device']
    data_path = params["data_root"]
    log_path = params["log_root"]
    log_rate = params["log_rate"]
    batch_rate = params["batch_rate"]

    batch_size = params["batch_size"]
    max_epochs = params['max_epochs']
    patience = params["patience"]
    dropout = params['dropout']
    opt_type = params["optimizer"]
    lr = params["lr"]
    l2 = params["l2_decay"]
    mode = params['mode']
    loss_func = params["loss_function"]
    loss_reduction = params["loss_reduction"]

    model_type = params['model_type']
    model_name = params["model_name"]
    model_path = params["model_root"]
    save_results = params["save_results"]
    track_metric = params["track_metric"]
    norm = params["norm"] 
    margin = params["margin"]
    sp_thres = params["sp_thres"]
    pooling = params['pooling']
    
    if 'new_dim' not in params:
        new_dim = 50
    else:
        new_dim = params["new_dim"]
        
    if 'static_model' not in params:
        static_model = model_name
    else:
        static_model = params['static_model']
    
    stop = False # Flag for early stopping
    running_patience = 0 # To keep track of early stopping patience
    running_loss = running_acc = running_precision = running_recall = running_mae = running_mse = r2 = None # Define variables for tracking metrics
    
    if (track_metric == "r2") or (track_metric == "acc"): # Assign large negative value if track metric is r2 or accuracy
        best_loss = -torch.inf
    else:
        best_loss = torch.inf
    
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S-%f') # Assign a unique ID to the model
    print("ID:", run_id)
    
    model_path = os.path.join(model_path, model_type)
    model_path = os.path.join(model_path, run_id)
    log_path = os.path.join(log_path, model_type)
    log_path = os.path.join(log_path, run_id)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print("Path created:", model_path)
    else:
        print("Path already exists:", model_path)
        
    with open(os.path.join(model_path, run_id+"_params_dict.pkl"), "wb") as f:
        pickle.dump(params, f)
    
    if torch.cuda.is_available():
        cur_d = torch.cuda.current_device()
        print("Device #:{} and name:{}".format(str(cur_d), torch.cuda.get_device_name(cur_d)))
    else:
        print("Device: CPU")

    print("Building Model...")
    cos_similarity = torch.nn.CosineSimilarity(dim=1) # Function variable to compute row-wise cosine similarity
    tokenizer = build_tokenizer(model_name)
    model = build_model(model_type=model_type,
                        model_name=model_name,
                        static_model=static_model,
                        mode=mode,
                        norm=norm,
                        dropout=dropout, 
                        pooling=pooling, 
                        new_dim=new_dim)
    model.to(device)

    criterion = build_criterion(loss_func, loss_reduction, margin) # Build loss function
    optimizer = build_optimizer(opt_type,
                                model,
                                lr=lr,
                                l2=l2)
    

    print("Building Datasets...")
    dsets = build_datasets(data_path, phases, sp_thres)
    dataloaders = build_dataloaders(dsets, phases, batch_size=batch_size)
    
    if ("classification" in model_type): # Build logger
        custom_logger = CustomLoggerClass(phases, log_path, run_id, device)
    else:
        custom_logger = CustomLogger(phases, log_path, run_id, device)
    
    print("Start training...")
    for epoch in range(max_epochs):
        if stop:
            break
        
        for phase in phases:

            if phase == "train":
                model.train()
            else:
                model.eval()

            custom_logger.start_epoch_logger() 
            track_metrics = [] # To keep scores of all possible metrics to track during the current epoch

            with torch.set_grad_enabled(phase=="train"): # Disable grads if val or test phase

                t1_epoch = time.time()
                for ii, (u1id, u2id, t1id, t2id, t1, t2, y_true, labels) in enumerate(dataloaders[phase]):

                    t1_batch = time.time()
                    t1 = list(t1)
                    t2 = list(t2)

                    tokens1 = tokenizer(t1, padding='max_length', max_length=100,
                                        truncation=True, return_tensors='pt')
                    tokens2 = tokenizer(t2, padding='max_length', max_length=100,
                                        truncation=True, return_tensors='pt')

                    tokens1, tokens2, y_true, labels = tokens1.to(device), tokens2.to(device), y_true.float().to(device), labels.float().to(device)

                    y_pred = model(tokens1, tokens2)
                    
                    # Extract tweet embeddings and run cosine similarity
                    embs1 = model(tokens1, embed=True)
                    embs2 = model(tokens2, embed=True)
                    cos_sim = cos_similarity(embs1.detach(), embs2.detach())

                    if "classification" in model_type: # Compute classification
                        loss = criterion(y_pred, labels)
                    elif "contrastive" in model_type: # Compute contrastive loss
                        loss = criterion(*y_pred, labels)
                    else:
                        loss = criterion(y_pred, y_true) # Compute regression loss

                    if phase == "train": # Backprop
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if "classification" in model_type: # Compute running loss and metrics for classification
                        num_correct, total, TP, FP, FN = custom_logger.get_metrics(y_pred, labels, y_true, cos_sim, phase)
                        custom_logger.add_epoch_totals(num_correct, total, TP, FP, FN, loss.detach().cpu().numpy())
                        running_loss, running_acc, running_precision, running_recall = custom_logger.get_running_metrics()
                    elif "contrastive" in model_type: # Compute running loss and metrics for contrastive
                        num_samples, mae, mse = custom_logger.get_metrics(y_true, y_true, cos_sim, phase, round_=True)
                        custom_logger.add_epoch_totals(loss.detach().item(), mae, mse, num_samples)
                        running_loss, running_mae, running_mse = custom_logger.get_running_metrics()
                    else: # Compute running loss and metrics for regression
                        num_samples, mae, mse = custom_logger.get_metrics(y_pred, y_true, cos_sim, phase, round_=True)
                        custom_logger.add_epoch_totals(loss.detach().item(), mae, mse, num_samples)
                        running_loss, running_mae, running_mse = custom_logger.get_running_metrics()

                    t2_batch = time.time() - t1_batch
                    if ii % batch_rate == 0:
                        print(f"Phase: {phase}, Batch: {ii} / {len(dataloaders[phase])}, Time per batch: {t2_batch:0.3f} seconds...", flush=True)

                # Log metric results per finished epoch 
                t2_epoch = time.time() - t1_epoch
                if epoch % log_rate == 0:
                    corr = custom_logger.get_corr(is_test=(phase!='train')) # compute correlation of val/test samples
                    group_corr = custom_logger.get_group_corr(is_test=(phase!='train')) # compute median correlation of val/test samples

                    if "classification" in model_type: # Log metrics for classification
                        custom_logger.log_epoch_metrics(epoch, phase, 
                                                        running_loss,
                                                        running_acc,
                                                        running_precision,
                                                        running_recall, 
                                                        corr, 
                                                        group_corr)
                    else: # Log metrics for contrastive and regression tasks
                        r2 = custom_logger.get_r2_score(is_test=(phase!="train"))
                        custom_logger.log_epoch_metrics(epoch, 
                                                        phase, 
                                                        running_loss, 
                                                        running_mae, 
                                                        running_mse, 
                                                        r2, 
                                                        corr, 
                                                        group_corr)

                # Append all possible tracking metrics for the current epoch
                track_metrics.extend([running_loss, running_acc,
                                      running_precision, running_recall, 
                                      running_mae, running_mse, 
                                      r2, corr, group_corr])

                if phase == "val":
                    track_loss, condition = get_track_value(best_loss, track_metric=track_metric, list_metrics=track_metrics)
                        
                    if condition: # Save best model on validation
                        best_loss = track_loss
                        running_patience = 0 # Reset patience
                        current_model = f"{run_id}_model_{model_type}_lr_{lr}_l2_{l2}_drop_{dropout}.pt"
                        torch.save(model.state_dict(), os.path.join(model_path, current_model))
                        print(f"Model saved at epoch: {epoch} with loss {best_loss}")
                    else: # Increment patience
                        running_patience += 1
                    if running_patience == patience: # Finish training if no more improvements
                        stop = True
                        print("No more model improvements...")
                
                if "classification" in model_type:
                    output = f"Epoch: {epoch}, Time: {t2_epoch:0.3f} seconds, Phase: {phase}, Batch: {ii} / {len(dataloaders[phase])}, Loss: {running_loss:.3f}, Acc: {running_acc:.3f}, Pred: {running_precision:.3f}, Recall: {running_recall:.3f}, Corr: {corr:0.4f}"
                else:
                    output = f"Epoch: {epoch}, Time: {t2_epoch:0.3f} seconds, Phase: {phase}, Batch: {ii} / {len(dataloaders[phase])}, Loss: {running_loss:.3f}, MAE: {running_mae:.3f}, MSE: {running_mse:.3f}, R2: {r2:0.4f}, Corr: {corr:0.4f}"

                print(output, flush=True)

    if save_results: # Save model parameters used for the model
        with open(os.path.join(model_path, run_id+"_params_dict.pkl"), "wb") as f:
            pickle.dump(params, f)
        custom_logger.save_metrics()