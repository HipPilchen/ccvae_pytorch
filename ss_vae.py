import argparse

import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from utils import multiclass_dataset_cached as mdc
from utils.dataset_cached import setup_data_loaders, CELEBA_EASY_LABELS
from utils.multiclass_dataset_cached import CELEBA_MULTI_LABELS
from utils.dataset_cached import CELEBACached
from models.ccvae import CCVAE, mc_CCVAE

import numpy as np
import os

NUM_MC_LABELS = 1

def main(args):
    """
    run inference for SS-VAE
    :param args: arguments for SS-VAE
    :return: None
    """

    im_shape = (3, 64, 64)

    # change root for mutlilabel (already dropped the nans within celebaCached)
    if args.mc:
        data_loaders = mdc.setup_data_loaders(args.cuda,
                                        args.batch_size,
                                        cache_data=True,
                                        sup_frac=args.sup_frac,
                                        root='./data/datasets/celeba',
                                        multi_class=True)
        if args.pruned >0:
            data_loaders = {name:mdc.random_prune_dataloader(dl, args.pruned) for name,dl in data_loaders.items()}
        
        cc_vae = mc_CCVAE(z_dim=args.z_dim,
                   num_binary_classes=len(CELEBA_MULTI_LABELS)-NUM_MC_LABELS,
                   im_shape=im_shape,
                   use_cuda=args.cuda)
    else:
        data_loaders = setup_data_loaders(args.cuda,
                                        args.batch_size,
                                        cache_data=True,
                                        sup_frac=args.sup_frac,
                                        root='./data/datasets/celeba')
        if args.pruned >0:
            data_loaders = {name:mdc.random_prune_dataloader(dl, args.pruned) for name,dl in data_loaders.items()}
            
        cc_vae = CCVAE(z_dim=args.z_dim,
                num_classes=len(CELEBA_EASY_LABELS),
                im_shape=im_shape,
                use_cuda=args.cuda,
                prior_fn=data_loaders['test'].dataset.prior_fn)

    

    # ENCODER_PATH = './pretrained_weights/weights_f90_80epoch_mc/encoder.pt'
    # DECODER_PATH = './pretrained_weights/weights_f90_80epoch_mc/decoder.pt'
    # COND_PRIOR_BIN_PATH = './pretrained_weights/weights_f90_80epoch_mc/cond_prior_binary.pt'
    # CLASSIFIER_BIN_PATH = './pretrained_weights/weights_f90_80epoch_mc/classifier_binary.pt'
    # COND_PRIOR_MC_PATH = './pretrained_weights/weights_f90_80epoch_mc/cond_prior_mc.pt'
    # CLASSIFIER_MC_PATH = './pretrained_weights/weights_f90_80epoch_mc/classifier_mc.pt'
    # # Load weights for the encoder
    # encoder_checkpoint = torch.load(ENCODER_PATH)
    # cc_vae.encoder.load_state_dict(encoder_checkpoint.state_dict())

    # # Load weights for the decoder
    # decoder_checkpoint = torch.load(DECODER_PATH)
    # cc_vae.decoder.load_state_dict(decoder_checkpoint.state_dict())

    # # Load weights for the classifier
    # classifier_binary_checkpoint = torch.load(CLASSIFIER_BIN_PATH)
    # cc_vae.classifier_binary.load_state_dict(classifier_binary_checkpoint.state_dict())

    # # Load weights for the conditional prior
    # cond_prior_binary_checkpoint = torch.load(COND_PRIOR_BIN_PATH)
    # cc_vae.cond_prior_binary.load_state_dict(cond_prior_binary_checkpoint.state_dict())

    # # Load weights for the classifier
    # classifier_mc_checkpoint = torch.load(CLASSIFIER_MC_PATH)
    # cc_vae.classifier_mc.load_state_dict(classifier_mc_checkpoint.state_dict())

    # # Load weights for the conditional prior
    # cond_prior_mc_checkpoint = torch.load(COND_PRIOR_MC_PATH)
    # cc_vae.cond_prior_mc.load_state_dict(cond_prior_mc_checkpoint.state_dict())

    optim = torch.optim.Adam(params=cc_vae.parameters(), lr=args.learning_rate)

    # run inference for a certain number of epochs
    for epoch in range(0, args.num_epochs):

        # # # compute number of batches for an epoch
        if args.sup_frac == 1.0: # fully supervised
            batches_per_epoch = len(data_loaders["sup"])
            period_sup_batches = 1
            sup_batches = batches_per_epoch
        elif args.sup_frac > 0.0: # semi-supervised
            sup_batches = len(data_loaders["sup"])
            unsup_batches = len(data_loaders["unsup"])
            batches_per_epoch = sup_batches + unsup_batches
            period_sup_batches = int(batches_per_epoch / sup_batches)
        elif args.sup_frac == 0.0: # unsupervised
            sup_batches = 0.0
            batches_per_epoch = len(data_loaders["unsup"])
            period_sup_batches = np.Inf
        else:
            assert False, "Data frac not correct"

        # initialize variables to store loss values
        epoch_losses_sup = 0.0
        epoch_losses_unsup = 0.0

        # setup the iterators for training data loaders
        if args.sup_frac != 0.0:
            sup_iter = iter(data_loaders["sup"])
        if args.sup_frac != 1.0:
            unsup_iter = iter(data_loaders["unsup"])

        # count the number of supervised batches seen in this epoch
        ctr_sup = 0

        for i in tqdm(range(batches_per_epoch)):
            # whether this batch is supervised or not
            is_supervised = (i % period_sup_batches == 0) and ctr_sup < sup_batches
            # extract the corresponding batch
            if is_supervised:
                (xs, ys) = next(sup_iter)
                if args.mc:
                    y_b = ys[:,:-NUM_MC_LABELS]
                    if NUM_MC_LABELS == 1:
                        y_mc = ys[:,-NUM_MC_LABELS].unsqueeze(-1)
                    else:
                        y_mc = ys[:,-NUM_MC_LABELS]  
                ctr_sup += 1
            else:
                (xs, ys) = next(unsup_iter)
                

            if args.cuda:
                xs, ys = xs.cuda(), ys.cuda()
                if args.mc:
                    y_b, y_mc = y_b.cuda(), y_mc.cuda()

            if is_supervised:
                if not args.mc:
                    loss = cc_vae.sup(xs, ys)
                else:
                    loss =  cc_vae.sup(xs, y_b, y_mc)
                epoch_losses_sup += loss.detach().item()
            else:
                loss = cc_vae.unsup(xs)
                epoch_losses_unsup += loss.detach().item()

            loss.backward()
            # print("Grad de mc:", cc_vae.cond_prior_mc.diag_loc[1].grad)
            # print("Grad de binary:", cc_vae.cond_prior_binary.diag_scale_true.grad)
            optim.step()
            optim.zero_grad()

            
        if args.sup_frac != 0.0: # Only compute the accuracy if we have a fraction of supervised learning   
            with torch.no_grad():
                validation_accuracy = cc_vae.accuracy(data_loaders['valid'])
        else:
            validation_accuracy = np.nan
        with torch.no_grad():
            # save some reconstructions
            img = mdc.CELEBACached.fixed_imgs
            if args.cuda:
                img = img.cuda()
  
            recon = cc_vae.reconstruct_img(img).view(-1, *im_shape)
            save_image(make_grid(recon, nrow=8), './data/output/recon.png')
            save_image(make_grid(img, nrow=8), './data/output/img.png')
        
        print("[Epoch %03d] Sup Loss %.3f, Unsup Loss %.3f, Val Acc %.3f" % 
                (epoch, epoch_losses_sup, epoch_losses_unsup, validation_accuracy))
    cc_vae.save_models(args.data_dir)
    test_acc = cc_vae.accuracy(data_loaders['test'])
    print("Test acc %.3f" % test_acc)
    if args.mc:
        cc_vae.mc_latent_walk(img[5], './data/output')
    else:
        cc_vae.latent_walk(img[5], './data/output')
    return 

def parser_args(parser):
    parser.add_argument('--cuda', action='store_true',
                        help="use GPU(s) to speed up training")
    parser.add_argument('-n', '--num-epochs', default=200, type=int,
                        help="number of epochs to run") 
    parser.add_argument('-sup', '--sup-frac', default=0.9,
                        type=float, help="supervised fractional amount of the data i.e. "
                                         "how many of the images have supervised labels."
                                         "Should be a multiple of train_size / batch_size")
    parser.add_argument('-zd', '--z_dim', default=45, type=int,
                        help="size of the tensor representing the latent variable z "
                             "variable (handwriting style for our MNIST dataset)")
    parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float,
                        help="learning rate for Adam optimizer")
    parser.add_argument('-bs', '--batch-size', default=200, type=int,
                        help="number of images (and labels) to be considered in a batch")
    parser.add_argument('--data_dir', type=str, default='./pretrained_weights',
                        help='Data path')
    parser.add_argument('-multi_class','--mc', type=bool, default=False,
                        help='Perform CCVAE with multi-classes labels')
    parser.add_argument('-pruned','--pruned', type=float, default=0,
                        help='Prune ratio (to keep)')
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parser_args(parser)
    args = parser.parse_args()

    main(args)

