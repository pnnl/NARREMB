'''
Classes for implemented models

RegressionModel - regression
ClassModel - classification
ContrastiveModel - contrastive learning
'''

import torch
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F

'''
Pooling Helper Functions/Classes
'''
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # hidden states shape -> (B_size, seq_length, D_in)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cls_pooling(model_output):
    token_embeddings = model_output[0] # last hidden state
    cls_embeddings = token_embeddings[:, 0] # First token
    return cls_embeddings

class WeightedLayerPooling(torch.nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else torch.nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        device = all_layer_embedding.device
        weight_factor = self.layer_weights.to(device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor.to(device)*all_layer_embedding).sum(dim=0) / self.layer_weights.to(device).sum()
        return weighted_average


def build_pooling(model_output, attention_mask, num_hidden_layers=6, pooling='mean'):
    
    if pooling == 'mean':
        embs = mean_pooling(model_output, attention_mask)
    elif pooling == 'cls':
        embs = cls_pooling(model_output)
    elif pooling == 'weight':
        layer_start = 4
        all_hidden_states = torch.stack(model_output[2])
        pooler = WeightedLayerPooling(num_hidden_layers, layer_start=layer_start, layer_weights=None)
        embs = pooler(all_hidden_states)
        embs = embs[:, 0]
    else:
        print('Pooling strategy not defined...Defaulting to mean')
        embs = mean_pooling(model_output, attention_mask)
        
    return embs

    
class ContrastiveModel(torch.nn.Module):
    
    def __init__(self, 
                 model_name="sentence-transformers/bert-base-nli-mean-tokens",
                 mode="concat",
                 norm=False,
                 D_in=768, 
                 dropout=0.1,
                 pooling="mean"):
        
        super(ContrastiveModel, self).__init__()
        
        self.mode = mode
        self.model_name = model_name
        self.norm = norm
        self.pooling = pooling
        
        # Load Config and Update it
        config = AutoConfig.from_pretrained(model_name)
        config.update({'output_hidden_states':True, 'attention_probs_dropout_prob':dropout, 'hidden_dropout_prob':dropout})
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size

        # Load pretrained BERT
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
            
    def forward(self, t1, t2=None, embed=False):
        '''
        t1 and t2 are the tokenized tweets
        embed if True return embeddings
        '''
        
        emb1 = self.transformer(**t1) # shape -> (B_size, seq_length, D_in)
        emb1 = build_pooling(emb1, t1['attention_mask'], num_hidden_layers=self.num_hidden_layers, pooling=self.pooling)

        if self.norm:
            emb1 = F.normalize(emb1, p=2, dim=1) # Normalize input embeddings
        
        if embed:
            return emb1
        
        emb2 = self.transformer(**t2)
        emb2 = build_pooling(emb2, t2['attention_mask'], num_hidden_layers=self.num_hidden_layers, pooling=self.pooling)
        if self.norm:
            emb2 = F.normalize(emb2, p=2, dim=1) # Normalize input embeddings
        
        return emb1, emb2


class EDContrastiveModel(torch.nn.Module):
    '''
    Contrastive model that concatenates static pre-trained embeddings with extra learnable dimensions
    '''
        
    def __init__(self, 
                 model_name="sentence-transformers/bert-base-nli-mean-tokens",
                 mode="concat",
                 norm=False,
                 D_in=768, 
                 dropout=0.1,
                 pooling="mean",
                 new_dim=50):
        
        super(EDContrastiveModel, self).__init__()
        
        self.mode = mode
        self.model_name = model_name
        self.norm = norm
        self.pooling = pooling
        self.new_dim = new_dim
        
        # Load Config and Update it
        config = AutoConfig.from_pretrained(model_name)
        config.update({'output_hidden_states':True, 'attention_probs_dropout_prob':dropout, 'hidden_dropout_prob':dropout})
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size

        # Load pretrained BERT
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        
        # Load pretrained BERT and freeze
        self.frozen_transformer = AutoModel.from_pretrained(model_name,config=config)
        for param in self.frozen_transformer.base_model.parameters():
            param.requires_grad = False
        
        # Linear layer to reduce pretrained embeddings representation
        self.linear1 = torch.nn.Linear(self.hidden_size, self.new_dim)
            
    def forward(self, t1, t2=None, embed=False):
        '''
        t1 and t2 are the tokenized tweets
        embed if True return embeddings
        '''
        
        emb1 = self.transformer(**t1) # shape -> (B_size, seq_length, D_in)
        emb1 = build_pooling(emb1, t1['attention_mask'], num_hidden_layers=self.num_hidden_layers, pooling=self.pooling)
        emb1 = self.linear1(emb1)
        
        static_emb1 = self.frozen_transformer(**t1)
        static_emb1 = build_pooling(static_emb1, t1['attention_mask'], num_hidden_layers=self.num_hidden_layers, pooling=self.pooling)
        if self.norm:
            emb1 = F.normalize(emb1, p=2, dim=1) # Normalize input embeddings
            static_emb1 = F.normalize(static_emb1, p=2, dim=1)
        
        concat_emb1 = torch.cat((static_emb1, emb1), -1)
        if embed:
            return concat_emb1
        
        emb2 = self.transformer(**t2) # shape -> (B_size, seq_length, D_in)
        emb2 = build_pooling(emb2, t2['attention_mask'], num_hidden_layers=self.num_hidden_layers, pooling=self.pooling)
        emb2 = self.linear1(emb2)
        
        static_emb2 = self.frozen_transformer(**t2)
        static_emb2 = build_pooling(static_emb2, t2['attention_mask'], num_hidden_layers=self.num_hidden_layers, pooling=self.pooling)
        if self.norm:
            emb2 = F.normalize(emb2, p=2, dim=1) # Normalize input embeddings
            static_emb2 = F.normalize(static_emb2, p=2, dim=1)
        
        concat_emb2 = torch.cat((static_emb2, emb2), -1)
        return concat_emb1, concat_emb2
