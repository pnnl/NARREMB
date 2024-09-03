import sys
sys.path.append('../core')
from trainers import *

params = {
    'phases': ['train', 'val'],
    'device': 'cuda', # cpu or cuda
    'mode': 'concat', # technique to join embeddings (concat, sub or mean)
    'log_rate': 1, # frequency for saving logs
    'batch_rate': 500, # batch frequency for showing performance
    'model_type': 'edcontrastive', # model type
    'log_root': '', # path to save model
    'model_root': '', # path to save model
    'data_root': '', # path to data splits
    'checkpoint_rate': 1, 
    'max_epochs': 250, # number of epochs
    'optimizer': 'adam', 
    'lr': 5e-8, # learning rate
    'l2_decay': 1e-3, # weight decay
    'dropout': 0.2, 
    'model_name': 'sentence-transformers/all-distilroberta-v1', # pre-trained model from hugging face
    'save_results': True,
    'batch_size': 16,
    'loss_function': 'cosine',
    'loss_reduction': 'mean', # returns average of loss across batches
    'patience': 20, # patience for early stopping
    'track_metric': 'loss', # metric of interest for early stopping
    'norm': True, # If True, it normalizes embeddings with Euclidean norm
    'margin': 0.0, # For contrastive learning, the value of margin threshold
    'sp_thres': 4, # The shortest path length threshold. Sps < sp_thres are close else distant
    'pooling':'mean', # The pooling strategy to aggregate tokens
    'new_dim':100 # Number of dimensions for trainable transformer
}

if __name__ == "__main__":
    base_trainer(params)