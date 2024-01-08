# Capturing label characteristics in VAEs

This is the repository of the  final project done for the MVA Master's course "Probabilistic Graph Models" (Prof. Latouche and Prof. Mattei). We extend and run additional experiments on the model developed in [Capturing Label Chacteristics in VAEs ICLR 2021](https://openreview.net/pdf?id=wQRlSUZ5V7B). More precisely, you will find:
- A tutorial to create multi-class labels, *multi_class_celebA.ipynb*, (e.g., $Hair \in [Brown \space Hair, Blond \space Hair, Bald, Black \space Hair]$). The transformed data is already present in the *data* folder
- Tutorials to setup the training (*tutorial\_data\_loader.ipynb*) in a binary class or multi-class fashion
- Adapted code of *CCVAE* model for multi-class training
- Several notebooks (test multi-class CCVAE and tests on latent traversal with CCVAE) and the *report.pdf* that analyzes and discusses the results

## Original Repository

Pytorch repository for [Capturing Label Chacteristics in VAEs ICLR 2021](https://openreview.net/pdf?id=wQRlSUZ5V7B).  We kindly ask that you cite our work if you plan to use this codebase:

    @inproceedings{
        joy2021capturing,
        title={Capturing Label Characteristics in {\{}VAE{\}}s},
        author={Tom Joy and Sebastian Schmon and Philip Torr and Siddharth N and Tom Rainforth},
        booktitle={International Conference on Learning Representations},
        year={2021},
        url={https://openreview.net/forum?id=wQRlSUZ5V7B}
    }

 ### Usage

 Ensure that CelebA is in the directory `data/datasets/celeba`, such that the path `data/datasets/celeba/celeba/img_align_celeba/*` is accessable. 

 To train, run:

>   `python ss_vae.py -sup <sup-frac> --cuda>`

where `<sup-frac>` is the fraction of supervised data (e.g. 0.004, 0.06, 0.2, 1.0).
