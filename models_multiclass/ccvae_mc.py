import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import torch.distributions as dist
import os
from utils.multiclass_dataset_cached import CELEBA_MULTI_LABELS 
from .networks_mc import (Classifier_mc, CondPrior_mc)
from models.networks import (Classifier, CondPrior,
                       CELEBADecoder, CELEBAEncoder)

CELEBA_VERY_EASY_LABELS = ['Arched_Eyebrows', 'Bags_Under_Eyes','Bushy_Eyebrows', 'Chubby','Eyeglasses', 'Heavy_Makeup', 'Male', \
                      'No_Beard', 'Pale_Skin', 'Smiling', 'Wavy_Hair', 'Wearing_Necktie', 'Young']

HAIR_ATTRIBUTES = ['Bald','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair']

def compute_kl(locs_q, scale_q, locs_p=None, scale_p=None):
    """
    Computes the KL(q||p)
    """
    if locs_p is None:
        locs_p = torch.zeros_like(locs_q)
    if scale_p is None:
        scale_p = torch.ones_like(scale_q)

    dist_q = dist.Normal(locs_q, scale_q)
    dist_p = dist.Normal(locs_p, scale_p)
    return dist.kl.kl_divergence(dist_q, dist_p).sum(dim=-1)

def img_log_likelihood(recon, xs):
        return dist.Laplace(recon, torch.ones_like(recon)).log_prob(xs).sum(dim=(1,2,3))

class mc_CCVAE(nn.Module):
    """
    CCVAE
    
    num_mc_labels: number of labels with multiple classes
    num_mc classes: number of classes for a multi-class label
    
    For now, only one label can have multi-classes !
    
    """
    def __init__(self, z_dim, num_binary_classes,
                 im_shape, use_cuda, prior_fn_binary, prior_fn_mc, num_mc_labels = 1, num_mc_classes = 5):
        super(mc_CCVAE, self).__init__()
        self.z_dim = z_dim
        
        # Added
        self.z_binary_classify = num_binary_classes   
        self.z_mc_classify =  num_mc_labels 
        self.z_style = z_dim - num_binary_classes - num_mc_labels
        
        self.im_shape = im_shape
        self.use_cuda = use_cuda
        
        # Added
        self.num_binary_classes = num_binary_classes
        self.num_mc_labels = num_mc_labels
        self.num_mc_classes = num_mc_classes
        
        self.ones = torch.ones(1, self.z_style)
        self.zeros = torch.zeros(1, self.z_style)
        
        # Added
        self.y_prior_params = prior_fn_binary()
        self.y_mc_prior_params = prior_fn_mc()

        self.encoder = CELEBAEncoder(self.z_dim)
        self.decoder = CELEBADecoder(self.z_dim)
        
        self.classifier_binary = Classifier(self.num_binary_classes)
        self.cond_prior_binary = CondPrior(self.num_binary_classes)
        
        # Added
        self.classifier_mc = Classifier_mc(self.z_mc_classify,self.num_mc_classes)
        self.cond_prior_mc = CondPrior_mc(self.num_mc_classes)

        if self.use_cuda:
            self.ones = self.ones.cuda()
            self.zeros = self.zeros.cuda()
            self.y_prior_params = self.y_prior_params.cuda()
            self.y_mc_prior_params = self.y_mc_prior_params.cuda()
            self.cuda()

    def unsup(self, x):
        bs = x.shape[0]
        #inference
        post_params = self.encoder(x)
        z = dist.Normal(*post_params).rsample()
        
        # Added
        zc_binary, zc_mc, zs = z.split([self.z_binary_classify, self.z_mc_classify, self.z_style], 1)
        
        qyzc_binary = dist.Bernoulli(logits=self.classifier_binary(zc_binary))
        y_b = qyzc_binary.sample()
        log_qy_b = qyzc_binary.log_prob(y_b).sum(dim=-1)
        
        # Added
        qyzc_mc = dist.Categorical(logits=self.classifier_mc(zc_mc))
        y_mc = qyzc_mc.sample()
        log_qy_mc = qyzc_mc.log_prob(y_mc).sum(dim=-1)

        # compute params binary
        locs_p_zc, scales_p_zc = self.cond_prior_binary(y_b)
        
        # compute params mc
        locs_p_zc_mc, scales_p_zc_mc = self.cond_prior_mc(y_mc)
        
        # Concatenate all
        prior_params = (torch.cat([locs_p_zc, locs_p_zc_mc, self.zeros.expand(bs, -1)], dim=1), 
                        torch.cat([scales_p_zc, scales_p_zc_mc, self.ones.expand(bs, -1)], dim=1))
        kl = compute_kl(*post_params, *prior_params)
        
        #compute log probs for x and y
        recon = self.decoder(z)
        log_py_binary = dist.Bernoulli(self.y_prior_params.expand(bs, -1)).log_prob(y_b).sum(dim=-1)
        log_py_mc = dist.Categorical(self.y_mc_prior_params.expand(bs, -1)).log_prob(y_mc).sum(dim=-1)
        elbo = (img_log_likelihood(recon, x) + log_py_binary 
                + log_py_mc - kl - log_qy_b - log_qy_mc).mean()
        return -elbo

    def sup(self, x, y_b, y_mc):
        bs = x.shape[0]
        #inference
        post_params = self.encoder(x)
        z = dist.Normal(*post_params).rsample()
        
        # Added
        zc_binary, zc_mc, zs = z.split([self.z_binary_classify, self.z_mc_classify, self.z_style], 1)
        
        # No need to sample since we have a label
        qyzc_binary = dist.Bernoulli(logits=self.classifier_binary(zc_binary))
        log_qyzc_b = qyzc_binary.log_prob(y_b).sum(dim=-1)
        
        qyzc_mc = dist.Categorical(logits=self.classifier_mc(zc_mc))
        log_qyzc_mc = qyzc_mc.log_prob(y_mc).sum(dim=-1)

        # compute params binary
        locs_p_zc, scales_p_zc = self.cond_prior_binary(y_b)
        
        # compute params mc
        locs_p_zc_mc, scales_p_zc_mc = self.cond_prior_mc(y_mc)
        
        # Concatenate all
        prior_params = (torch.cat([locs_p_zc, locs_p_zc_mc, self.zeros.expand(bs, -1)], dim=1), 
                        torch.cat([scales_p_zc, scales_p_zc_mc, self.ones.expand(bs, -1)], dim=1))
        kl = compute_kl(*post_params, *prior_params)

        #compute log probs for x and y
        recon = self.decoder(z)
        
        log_py_binary = dist.Bernoulli(self.y_prior_params.expand(bs, -1)).log_prob(y_b).sum(dim=-1)
        log_py_mc = dist.Categorical(self.y_mc_prior_params.expand(bs, -1)).log_prob(y_mc).sum(dim=-1)
        
        log_qyx = self.classifier_loss(x, y_b, y_mc)
        log_pxz = img_log_likelihood(recon, x)

        # we only want gradients wrt to params of qyz, so stop them propogating to qzx
        log_qyzc_b_ = dist.Bernoulli(logits=self.classifier_binary(zc_binary.detach())).log_prob(y_b).sum(dim=-1)
        log_qyzc_mc_ = dist.Categorical(logits=self.classifier_mc(zc_mc.detach())).log_prob(y_mc).sum(dim=-1)
        w = torch.exp(log_qyzc_b_ + log_qyzc_mc_ - log_qyx)
        elbo = (w * (log_pxz - kl - log_qyzc_mc - log_qyzc_b) + log_py_binary + log_py_mc  + log_qyx).mean()
        return -elbo

    def classifier_loss(self, x, y_b, y_mc, k=100):
        """
        Computes the classifier loss.
        """
        zc_b, zc_mc, _ = dist.Normal(*self.encoder(x)).rsample(torch.tensor([k])).split(
            [self.z_binary_classify, self.z_mc_classify, self.z_style], -1)
        logits_b = self.classifier_binary(zc_b.view(-1, self.z_binary_classify))
        logits_mc = self.classifier_mc(zc_mc.view(-1, self.z_mc_classify))
        d_b = dist.Bernoulli(logits=logits_b)
        d_mc = dist.Categorical(logits=logits_mc)
        y_b = y_b.expand(k, -1, -1).contiguous().view(-1, self.num_binary_classes)
        y_mc = y_mc.expand(k, -1, -1).contiguous().view(-1, self.num_mc_labels)
        lqy_z_b = d_b.log_prob(y_b).view(k, x.shape[0], self.num_binary_classes).sum(dim=-1)
        lqy_z_mc = d_mc.log_prob(y_mc).view(k, x.shape[0], self.num_mc_labels).sum(dim=-1)
        lqy_x = torch.logsumexp(lqy_z_b + lqy_z_mc, dim=0) - np.log(k)
        return lqy_x

    def reconstruct_img(self, x):
        return self.decoder(dist.Normal(*self.encoder(x)).rsample())

    def classifier_acc(self, x, y_b=None, y_mc = None, k=1):
        zc_b, zc_mc, _ = dist.Normal(*self.encoder(x)).rsample(torch.tensor([k])).split(
            [self.z_binary_classify, self.z_mc_classify, self.z_style], -1)
        
        logits_b = self.classifier_binary(zc_b.view(-1, self.z_binary_classify)).view(-1, self.num_binary_classes)
        y_b = y_b.expand(k, -1, -1).contiguous().view(-1, self.num_binary_classes)
        
        logits_mc = self.classifier_mc(zc_mc.view(-1, self.z_mc_classify)).view(-1, self.num_mc_labels)
        y_mc = y_mc.expand(k, -1, -1).contiguous().view(-1, self.num_mc_labels)
        
        preds_mc = torch.round(torch.sigmoid(logits_mc))
        preds_b = torch.round(torch.sigmoid(logits_b))
        
        acc = (preds_b.eq(y_b)).float().mean() + (preds_mc.eq(y_mc)).float().mean()
        
        return acc

    def save_models(self, path='./data'):
        torch.save(self.encoder, os.path.join(path,'encoder.pt'))
        torch.save(self.decoder, os.path.join(path,'decoder.pt'))
        torch.save(self.classifier, os.path.join(path,'classifier.pt'))
        torch.save(self.cond_prior, os.path.join(path,'cond_prior.pt'))

    def accuracy(self, data_loader, *args, **kwargs):
        acc = 0.0
        
        # VOIR COMMENT LE DATA LOADER EST FAIT POUR EXTRAIRE SEPAREMMENT LES 2 Y 
        for (x, y) in data_loader:
            if self.use_cuda:
                x, y = x.cuda(), y.cuda()
            batch_acc = self.classifier_acc(x, y_b, y_mc)
            acc += batch_acc
        return acc / len(data_loader)


    def mc_latent_walk(self, image, save_dir):
        """
        Does latent walk between all possible classes
        """
    

        z_ = dist.Normal(*self.encoder(image.unsqueeze(0))).sample()
        mc_z_id = self.z_binary_classify
        mult = 8

        z = z_.clone()
        z = z.expand(10, -1).contiguous()
        y = torch.zeros(1, self.num_mc_labels)
        if self.use_cuda:
            y = y.cuda()
            
        locs = []
        scales = []
        
        for i in range(self.num_mc_classes):
            y.fill_(float(i))
            locs_, scales_ = self.cond_prior_mc(y)
            locs.append(locs_)
            scales.append(scales_)
            
        for i in range(self.num_mc_classes):
            for j in range(self.num_mc_classes):   
                    if i != j:
                         sign = torch.sign(locs[i, :] - locs[j, :])
                         z_lim_1 = (locs[i, :] - mult * sign * scales[i,:]).item() 
                         z_lim_2 = (locs[j, :] - mult * sign * scales[j,:]).item() 
                         range_ = torch.linspace(z_lim_1, z_lim_2, 10)
                         z[:, mc_z_id] = range_
                         imgs = self.decoder(z).view(-1, *self.im_shape)
                         grid = make_grid(imgs, nrow=10)
                         save_image(grid, 
                                    os.path.join(save_dir, "./images_mc/mc_latent_walk_between_%s_and_%s.png"
                                                            % (HAIR_ATTRIBUTES[i],HAIR_ATTRIBUTES[j])))
                        
            
            
    
    
        
        


