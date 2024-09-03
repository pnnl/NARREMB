'''
Contrastive Loss Functions

ContrastiveLoss - to be used for Euclidean loss function
CosineLoss - uses Cosine loss implementation from PyTorch
'''

import torch
import torch.nn.functional as F
from torch.nn import CosineEmbeddingLoss


class ContrastiveLoss(torch.nn.Module):
    '''Adapted from: https://innovationincubator.com/siamese-neural-network-with-pytorch-code-example/'''
    def __init__(self, margin=2.0, metric='euclidean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.metric = metric

    def forward(self, output1, output2, label):
        # Flatten tensor
        label = label.view(-1) 
        
        # Find the eucledian distance of two output feature vectors
        if self.metric == 'euclidean':
            distance = F.pairwise_distance(output1, output2)
        elif self.metric == 'cosine':
            distance = 1 - F.cosine_similarity(output1, output2)
        # perform contrastive loss calculation with the distance
        loss_contrastive = torch.mean((label) * torch.pow(distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))

        return loss_contrastive


class CosineLoss(torch.nn.Module):

    def __init__(self, margin=0.5):
        super(CosineLoss, self).__init__()
        self.margin = margin
        self.cosine_loss = CosineEmbeddingLoss(margin=self.margin)
        
    def forward(self, output1, output2, label):
        
        label = torch.where(label == 1., 1., -1.).view(-1)

        loss_constrastive = self.cosine_loss(output1, output2, label)
        return loss_constrastive