import torch
import torch.nn as nn
import torch.nn.functional as F

    
# Only takes one multi-class label 

class Classifier_mc(nn.Module):
    def __init__(self, z_dim, num_mc_classes):
        super(Classifier_mc, self).__init__()
        self.categorical_params = nn.Linear(z_dim, num_mc_classes)
        
    def forward(self, x):
        return self.categorical_params(x)

class CondPrior_mc(nn.Module):
    def __init__(self, num_mc_classes):
        super(CondPrior_mc, self).__init__()
        
        self.dim = 1
        self.num_mc_classes = num_mc_classes
        
        self.diag_loc = [nn.Parameter(torch.zeros(self.dim)) for i in range(self.num_mc_classes)]
        self.diag_scale = [nn.Parameter(torch.ones(self.dim)) for i in range(self.num_mc_classes)]


    def forward(self, x):
        
        loc = torch.zeros(self.dim)
        scale = torch.ones(self.dim)
        
        for i in range(self.num_mc_classes):
            loc = torch.add(loc,x[i]*self.diag_loc[i])  
            scale = torch.add(scale,x[i]*self.diag_scale[i])
            
        return loc/torch.sum(x), torch.clamp(F.softplus(scale/torch.sum(x)), min=1e-3)


