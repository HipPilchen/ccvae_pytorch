import errno
import os
import PIL
from functools import reduce
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
import torchvision.transforms as transforms


def split_celeba(X, y, sup_frac, validation_num):
    """
    splits celeba
    """

    # validation set is the last 10,000 examples
    X_valid = X[-validation_num:]
    y_valid = y[-validation_num:]

    X = X[0:-validation_num]
    y = y[0:-validation_num]

    if sup_frac == 0.0:
        return None, None, X, y, X_valid, y_valid

    if sup_frac == 1.0:
        return X, y, None, None, X_valid, y_valid

    split = int(sup_frac * len(X))
    X_sup = X[0:split]
    y_sup = y[0:split]
    X_unsup = X[split:]
    y_unsup = y[split:]

    return X_sup, y_sup, X_unsup, y_unsup, X_valid, y_valid

def compute_uniform_prior_multiclass(dict_class_nclass:dict):
    prior = []
    dict_index_class = {}
    i=0

    for label, n_classes in dict_class_nclass.items():
        uniform_distrib = [1/n_classes for i in range(n_classes)]
        uniform_tensor = torch.tensor(uniform_distrib).unsqueeze(0)

        prior.append(torch.distributions.categorical.Categorical(probs= uniform_tensor))
        dict_index_class[i] = label
        i+=1

    dict_class_index = {v:k for k,v in dict_index_class.items()}
    return prior, dict_class_index, dict_index_class

# 41 labels
CELEBA_LABELS = ['5_o_Clock_Shadow', 'Arched_Eyebrows','Attractive','Bags_Under_Eyes','Bangs','Big_Lips','Big_Nose','Blurry','Bushy_Eyebrows', \
                 'Chubby', 'Double_Chin','Eyeglasses','Goatee','Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open','Mustache','Narrow_Eyes', 'No_Beard', 'Oval_Face', \
                 'Pale_Skin','Pointy_Nose','Receding_Hairline','Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', \
                'Wearing_Necklace', 'Wearing_Necktie', 'Young', 'Hair_MULTI']

# 18 labels
CELEBA_EASY_LABELS = ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bangs', 'Black_Hair', 'Blond_Hair','Brown_Hair','Bushy_Eyebrows', 'Chubby','Eyeglasses', 'Heavy_Makeup', 'Male', \
                      'No_Beard', 'Pale_Skin', 'Receding_Hairline', 'Smiling', 'Wavy_Hair', 'Wearing_Necktie', 'Young']
# 16 labels 
# Take care to put multi-class labels at the end of the dict
CELEBA_MULTI_LABELS = {'Arched_Eyebrows': 2, 'Bags_Under_Eyes': 2, 'Bangs': 2, 'Bushy_Eyebrows': 2, 'Chubby': 2, 'Eyeglasses': 2, 'Heavy_Makeup': 2, 'Male': 2, 
                      'No_Beard': 2, 'Pale_Skin': 2, 'Receding_Hairline': 2, 'Smiling': 2, 'Wavy_Hair': 2, 'Wearing_Necktie': 2, 'Young': 2, 'Hair_MULTI': 5} # Multi-label: has 5 possible rankings

class CELEBACached(CelebA):
    """
    a wrapper around CelebA to load and cache the transformed data
    once at the beginning of the inference
    """
    # static class variables for caching training data
    train_data_sup, train_labels_sup = None, None
    train_data_unsup, train_labels_unsup = None, None
    train_data, test_labels = None, None
    prior, dict_class_index, dict_index_class = compute_uniform_prior_multiclass(CELEBA_MULTI_LABELS)
    fixed_imgs = None
    validation_size = 20000
    data_valid, labels_valid = None, None

    def prior_fn(self):
        return CELEBACached.prior

    def clear_cache():
        CELEBACached.train_data, CELEBACached.test_labels = None, None

    def __init__(self, mode, sup_frac=None, *args, **kwargs):
        super(CELEBACached, self).__init__(split='train' if mode in ["sup", "unsup", "valid"] else 'test', *args, **kwargs)
        self.sub_label_inds = [i for i in range(len(CELEBA_LABELS)) if CELEBA_LABELS[i] in CELEBA_MULTI_LABELS.keys()] 
        self.mode = mode
        self.transform = transforms.Compose([
                                transforms.Resize((64, 64)),
                                transforms.ToTensor()
                            ])

        assert mode in ["sup", "unsup", "test", "valid"], "invalid train/test option values"

        if mode in ["sup", "unsup", "valid"]:

            if CELEBACached.train_data is None:
                print("Splitting Dataset")

                CELEBACached.train_data = self.filename
                CELEBACached.train_targets = self.attr

                CELEBACached.train_data_sup, CELEBACached.train_labels_sup, \
                    CELEBACached.train_data_unsup, CELEBACached.train_labels_unsup, \
                    CELEBACached.data_valid, CELEBACached.labels_valid = \
                    split_celeba(CELEBACached.train_data, CELEBACached.train_targets,
                                 sup_frac, CELEBACached.validation_size)

            if mode == "sup":
                self.data, self.targets = CELEBACached.train_data_sup, CELEBACached.train_labels_sup
                probs_prior = []
                dataset_size = self.targets.shape[0]
                for label_idx, label_name in CELEBACached.dict_index_class.items():
                    n_classes = CELEBA_MULTI_LABELS[label_name]
                    probs_label = []
                    for i in range (n_classes):
                        probs_label.append(torch.sum(self.targets[:, label_idx] == i).item() / dataset_size)
                    probs_prior.append(probs_label)

                CELEBACached.prior = [torch.distributions.categorical.Categorical(probs= torch.tensor(lab_prior).unsqueeze(0)) for lab_prior in probs_prior]
            elif mode == "unsup":
                self.data = CELEBACached.train_data_unsup
                # making sure that the unsupervised labels are not available to inference
                self.targets = CELEBACached.train_labels_unsup * np.nan
            else:
                self.data, self.targets = CELEBACached.data_valid, CELEBACached.labels_valid

        else:
            self.data = self.filename # list with all the filename in the dataset ("image00001.jpg", ...)
            self.targets = self.attr

        # create a batch of fixed images
        if CELEBACached.fixed_imgs is None:
            temp = []
            for i, f in enumerate(self.data[:64]):
                temp.append(self.transform(PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", f))))
            CELEBACached.fixed_imgs = torch.stack(temp, dim=0)

    def __getitem__(self, index):
        
        X = self.transform(PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.data[index])))

        target = self.targets[index].float()
        target = target[self.sub_label_inds]
        
        return X, target

    def __len__(self):
        return len(self.data)


def setup_data_loaders(use_cuda, batch_size, sup_frac=1.0, root=None, cache_data=False, download=False, multi_class=False, **kwargs):
    """
        helper function for setting up pytorch data loaders for a semi-supervised dataset
    :param use_cuda: use GPU(s) for training
    :param batch_size: size of a batch of data to output when iterating over the data loaders
    :param sup_frac: fraction of supervised data examples
    :param cache_data: saves dataset to memory, prevents reading from file every time
    :param kwargs: other params for the pytorch data loader
    :return: three data loaders: (supervised data for training, un-supervised data for training,
                                  supervised data for testing)
    """

    if root is None:
        root = get_data_directory(__file__)
    if 'num_workers' not in kwargs:
        kwargs = {'num_workers': 4, 'pin_memory': True}

    cached_data = {}
    loaders = {}

    #clear previous cache
    CELEBACached.clear_cache()

    if sup_frac == 0.0:
        modes = ["unsup", "test"]
    elif sup_frac == 1.0:
        modes = ["sup", "test", "valid"]
    else:
        modes = ["unsup", "test", "sup", "valid"]
        
    for mode in modes:
        cached_data[mode] = CELEBACached(mode, sup_frac, root=root, download=download, check_itr=download, multi_class = multi_class)
        loaders[mode] = DataLoader(cached_data[mode], batch_size=batch_size, shuffle=True, **kwargs)
    return loaders

def random_prune_dataloader(original_dataloader, prune_ratio):
    """
    Randomly prunes the dataset in a DataLoader.

    Args:
    - original_dataloader (DataLoader): The original DataLoader.
    - prune_ratio (float): The ratio of data to keep after pruning (0 to 1).

    Returns:
    - pruned_dataloader (DataLoader): The pruned DataLoader.
    """
    assert 0 <= prune_ratio < 1, "Enter a value between 0 and 1 (strict) for the pruning ration"
    # Get the original dataset
    original_dataset = original_dataloader.dataset

    # Get the number of samples to keep after pruning
    num_samples_to_keep = int(len(original_dataset) * prune_ratio)

    # Randomly select indices to keep
    indices_to_keep = torch.randperm(len(original_dataset))[:num_samples_to_keep]

    # Create a new dataset with the randomly selected indices
    pruned_dataset = torch.utils.data.Subset(original_dataset, indices_to_keep)

    # Create a new dataloader using the pruned dataset
    pruned_dataloader = DataLoader(pruned_dataset, batch_size=original_dataloader.batch_size, shuffle=True)

    return pruned_dataloader