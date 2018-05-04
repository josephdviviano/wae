Wasserstein Auto-Encoders (WAE) and Variational Auto-Encoders (VAE) for Celeb A and MNIST Datasets
--------------------------------------------------------------------------------------------------

code built on top of [this implementation](https://github.com/wohlert/semi-supervised-pytorch)
and [this one](https://github.com/bhpfelix/Variational-Autoencoder-PyTorch/blob/master/src/vanila_vae.py).

+ `preprocess.py` -- generates preprocessed data (from `raw/`, to `data/`).
+ `wae.py` -- trains WAE model.
+ `inception.py` -- InceptionV3 model used for Frechet Inception Distance measure (FID).
+ `fid_score.py` -- Methods for calculating FID score on pairs of images.

Pytorch replication of the results presented in [Tolstikhin, Bousquet, Gelly, Schoelkopf (2017)](https://arxiv.org/abs/1711.01558).


**resources**

+ https://github.com/tolstikhin/wae
    + Official tensorflow implementation.
+ https://github.com/paruby/Wasserstein-Auto-Encoders
    + clean WIP code for WAE (being used for some new paper).
+ https://github.com/schelotto/Wasserstein_Autoencoders
    + Another simple implementation.
+ https://github.com/maitek/waae-pytorch/blob/master/WAAE.py
    + this is WAE in an adversarial setting -- not quite what we need but some aspects of this code are very clean.
+ https://github.com/wsnedy/WAE_Pytorch/blob/master/wae_for_mnist.py
    + simple WAE implementation in pytorch for MNIST.
+ https://github.com/wohlert/semi-supervised-pytorch/blob/master/examples/notebooks/Variational%20Autoencoder.ipynb
    + nice explaination of normal VAE.
+ https://github.com/sbarratt/inception-score-pytorch
    + inception score
+ https://github.com/mseitzer/pytorch-fid
    + frechet inception distance


** todo **

+ FID measure
+ Blur measure
+
