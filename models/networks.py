import torch
import torch.nn as nn
import torch.nn.functional as F




'''
This part has been implemented for our project in Probabilistic Graphical Model: 
We enable CCVAE to embed binary attribute but also multi-classes attributes (hair colors for instance)
All the details and results are in the project report.
'''

# Only takes one multi-class label for now
class Classifier_mc(nn.Module):
    def __init__(self, z_dim, num_mc_classes):
        super(Classifier_mc, self).__init__()
        self.categorical_params = nn.Linear(z_dim, num_mc_classes)
        
    def forward(self, x):
        return self.categorical_params(x)

class CondPrior_mc(nn.Module):
    def __init__(self, num_mc_classes, num_mc_labels):
        super(CondPrior_mc, self).__init__()
        
        self.dim = num_mc_labels
        self.num_mc_classes = num_mc_classes
        
        self.diag_loc = nn.ParameterList([nn.Parameter(torch.zeros(self.dim)) for i in range(self.num_mc_classes)])
        self.diag_scale = nn.ParameterList([nn.Parameter(torch.ones(self.dim)) for i in range(self.num_mc_classes)])


    def forward(self, x):
        
        x_ = x.int()
        loc = torch.zeros((x.shape[0],1))
        scale = torch.ones((x.shape[0],1))
        for i, label in enumerate(x_):
            loc[i,:] = self.diag_loc[int(label)]
            scale[i,:] =  self.diag_scale[int(label)]
        return loc.view(-1, 1), torch.clamp(F.softplus(scale.view(-1, 1)), min=1e-3)


'''
Codes forked from the Github of the paper
'''

           
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    
class CELEBAEncoder(nn.Module):
    def __init__(self, z_dim, hidden_dim=256, *args, **kwargs):
        super().__init__()
        # setup the three linear transformations used
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), 
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), 
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), 
            nn.ReLU(True),
            nn.Conv2d(128, hidden_dim, 4, 1),
            nn.ReLU(True),
            View((-1, hidden_dim*1*1))
        )

        self.locs = nn.Linear(hidden_dim, z_dim)
        self.scales = nn.Linear(hidden_dim, z_dim)


    def forward(self, x):
        hidden = self.encoder(x)
        return self.locs(hidden), torch.clamp(F.softplus(self.scales(hidden)), min=1e-3)
     
        
class CELEBADecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim=256, *args, **kwargs):
        super().__init__()
        # setup the two linear transformations used
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),  
            View((-1, hidden_dim, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 128, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        m = self.decoder(z)
        return m


class Diagonal(nn.Module):
    def __init__(self, dim):
        super(Diagonal, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))

    def forward(self, x):
        return x * self.weight + self.bias

class Classifier(nn.Module):
    def __init__(self, dim):
        super(Classifier, self).__init__()
        self.dim = dim
        self.diag = Diagonal(self.dim)

    def forward(self, x):
        return self.diag(x)

class CondPrior(nn.Module):
    def __init__(self, dim):
        super(CondPrior, self).__init__()
        self.dim = dim
        self.diag_loc_true = nn.Parameter(torch.zeros(self.dim))
        self.diag_loc_false = nn.Parameter(torch.zeros(self.dim))
        self.diag_scale_true = nn.Parameter(torch.ones(self.dim))
        self.diag_scale_false = nn.Parameter(torch.ones(self.dim))

    def forward(self, x):
        loc = x * self.diag_loc_true + (1 - x) * self.diag_loc_false
        scale = x * self.diag_scale_true + (1 - x) * self.diag_scale_false
        return loc, torch.clamp(F.softplus(scale), min=1e-3)


