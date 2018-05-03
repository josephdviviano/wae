#!/usr/bin/env python

import os
import sys
import scandir
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import save_image

from utils import load_data, load_mnist

CHCKDIR = '/data/milatmp1/vivianoj/wae_checkpoints/'
CUDA = torch.cuda.is_available()

DATASET = 'mnist' # 'mnist' / 'celeb'
LOSS = 'wae-mmd'     # 'vae', 'wae-gan', 'wae-mmd'

# data options
if DATASET == 'celeb':
    DIMS = 64  # dependent on script that generates preprocessed data
    N_Z = 64             # mnist = 8, celeb = 64
    N_CHAN = 3
    N_DATA = 20000
    N_TEST = 1000
    BATCH = 64
    SIGMA = 2             # celeba = 2, mnist = 1

    if LOSS == 'wae-mmd':
        LAMBDA = 100          # celeb gan = 1, celeb mdd = 100, mnist = 10
        LR_VAE = 0.001
        LR_DIS = None
        RECON = nn.MSELoss(size_average = False)
    elif LOSS == 'wae-gan':
        LAMBDA = 1
        LR_VAE = 0.0003      # 0.0003 for wae-gan
        LR_DIS = 0.001
        RECON = nn.MSELoss(size_average = False)
    elif LOSS == 'vae':
        LAMBDA = None
        LR_VAE = 0.0001      # 0.0003 for wae-gan
        LR_DIS = None
        RECON =  nn.BCELoss(size_average = False)

    DATADIR = '/u/vivianoj/data/celeba/data/'

    im_data = load_data('celeba_data.npy', N_DATA, DATADIR, BATCH, (N_CHAN, DIMS))
    im_test = load_data('celeba_test.npy', N_TEST, DATADIR, BATCH, (N_CHAN, DIMS))

elif DATASET == 'mnist':
    DIMS = 28  # dependent on script that generates preprocessed data
    N_Z = 8             # mnist = 8, celeb = 64
    N_CHAN = 1
    N_DATA = -1
    N_TEST = -1
    BATCH = 8
    SIGMA = 1            # celeba = 2, mnist = 1
    LAMBDA = 10          # celeb gan = 1, celeb mdd = 100, mnist = 10
    LR_VAE = 0.003
    RECON =  nn.BCELoss(size_average = False)

    if LOSS == 'wae-mmd':
        LR_DIS = None
    elif LOSS == 'wae-gan':
        LR_DIS = 0.0005
    elif LOSS == 'vae':
        LR_DIS = None

    DATADIR = '/u/vivianoj/data/celeba/data/'
    CUDA = torch.cuda.is_available()

    im_data, im_test = load_mnist(batch_size=BATCH)

# model options
SCALEBULLSHIT = 0.05
EPOCHS = 100

class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(VAE, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.relu = nn.LeakyReLU(0.2)

        if DATASET == 'celeb':
            self.conv_filt = 5
            self.size = 4
        elif DATASET == 'mnist':
            self.conv_filt = 4
            self.size = 3

        # encoder
        self.e1 = nn.Sequential(nn.Conv2d(nc, ndf, self.conv_filt, 2, 2),
            nn.BatchNorm2d(ndf), self.relu)

        self.e2 = nn.Sequential(nn.Conv2d(ndf, ndf*2, self.conv_filt, 2, 2),
            nn.BatchNorm2d(ndf*2), self.relu)

        self.e3 = nn.Sequential(nn.Conv2d(ndf*2, ndf*4, self.conv_filt, 2, 2),
            nn.BatchNorm2d(ndf*4), self.relu)

        self.e4 = nn.Sequential(nn.Conv2d(ndf*4, ndf*8, self.conv_filt, 2, 2),
            nn.BatchNorm2d(ndf*8), self.relu)

        self.fc1 = nn.Linear(ndf*8*self.size*self.size, latent_variable_size)
        self.fc2 = nn.Linear(ndf*8*self.size*self.size, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf*8*self.size*self.size)

        self.d2 = nn.Sequential(nn.ConvTranspose2d(ngf*8, ngf*4, self.conv_filt-1, 2, 1),
            nn.BatchNorm2d(ngf*4, 1.e-3), self.relu)

        if DATASET == 'celeb':

            self.d3 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, self.conv_filt-1, 2, 1),
                nn.BatchNorm2d(ngf*2, 1.e-3), self.relu)

            self.d4 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, self.conv_filt-1, 2, 1),
                nn.BatchNorm2d(ngf, 1.e-3), self.relu)

            self.d5 = nn.Sequential(nn.ConvTranspose2d(ngf, nc, self.conv_filt-1, 2, 1),
                nn.Sigmoid())

        elif DATASET == 'mnist':
            self.d3 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, self.conv_filt-2, 2, 1),
                nn.BatchNorm2d(ngf*2, 1.e-3), self.relu)

            self.d4 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, self.conv_filt-1, 2, 1),
                nn.BatchNorm2d(ngf, 1.e-3), self.relu)

            self.d5 = nn.Sequential(nn.ConvTranspose2d(ngf, nc, self.conv_filt-2, 2, 1),
                nn.Sigmoid())


    def encoder(self, x):
        """ encoder architecture: note 2 read-out heads for mu and logvar """
        h1 = self.e1(x)
        h2 = self.e2(h1)
        h3 = self.e3(h2)
        h4 = self.e4(h3)
        h4 = h4.view(-1, self.ndf*8*self.size*self.size)
        #print(h1.size())
        #print(h2.size())
        #print(h3.size())
        #print(h4.size())

        return(self.fc1(h4), self.fc2(h4))

    def reparametrize(self, mu, logvar):
        """ generates a normal distribution with the specified mean and std"""
        std = logvar.mul(0.5).exp_()    # converts log(var) to std

        # with MDD, standard deviation blows up to produce NaNs without tiny
        # learning rates, so clamp std to prevent z_encoded from naning out
        std = torch.clamp(std, -50, 50)

        #print('mu={}, std={}'.format(torch.max(mu).data[0], torch.max(std).data[0]))

        # z is a gaussian with std and mean=mu. begin with eps (a stock gauss)
        if CUDA:
            eps = Variable(torch.cuda.FloatTensor(std.size()).normal_())
        else:
            eps = Variable(torch.FloatTensor(std.size()).normal_())

        z = eps.mul(std).add_(mu)

        return(z)

    def decoder(self, z):
        """ decoder architecture """
        h1 = self.d1(z)
        h1 = h1.view(-1, self.ngf*8, self.size, self.size)
        h2 = self.d2(h1)
        h3 = self.d3(h2)
        h4 = self.d4(h3)
        h5 = self.d5(h4)
        #print(z.size())
        #print(h1.size())
        #print(h2.size())
        #print(h3.size())
        #print(h4.size())
        #print(h5.size())

        return(self.d5(h4))

    def encode(self, x):
        """ feeds data through encoder and reparameterization to produce z """
        mu, logvar = self.encoder(x.view(-1, self.nc, DIMS, DIMS))
        z = self.reparametrize(mu, logvar)
        return(z)

    def decode(self, z):
        """ feeds z through decoder """
        return(self.decoder(z))

    def sample_pz(self, n):
        """
        samples noise from a gaussian distribution with mean=0 and experiment-
        specific standard-deviation (SIGMA)
        """
        return(Variable(torch.normal(torch.zeros(n, N_Z), std=SIGMA).cuda()))

    def forward(self, x):
        """ passes data through encode, reparameterization, and decode """
        mu, logvar = self.encoder(x.view(-1, self.nc, DIMS, DIMS))
        z = self.reparametrize(mu, logvar)
        res = self.decoder(z)
        return(res, mu, logvar)


class Discriminator(nn.Module):
    def __init__(self, latent_variable_size):
        super(Discriminator, self).__init__()

        self.latent_variable_size = latent_variable_size
        self.hid_size = 512

        # 3 hidden-layer discriminator
        # TODO: add sigmoid at end? Also, we're using 2 output nodes now...
        self.linear = nn.Sequential(
            nn.Linear(self.latent_variable_size, self.hid_size), nn.ReLU(),
            nn.Linear(self.hid_size, self.hid_size), nn.ReLU(),
            nn.Linear(self.hid_size, self.hid_size), nn.ReLU(),
            nn.Linear(self.hid_size, self.hid_size), nn.ReLU(),
            nn.Linear(self.hid_size, 2)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0099999)
                m.bias.data.zero_()

    def forward(self, z):
        return(self.linear(z))

## TODO: move this somewhere else (models should go in models.py)
# instantiate models
vae = VAE(nc=N_CHAN, ngf=128, ndf=128, latent_variable_size=N_Z)
dis = Discriminator(N_Z)

if CUDA:
    vae = vae.cuda()
    dis = dis.cuda()

# instantiate optimizers
opt_vae = torch.optim.Adam(vae.parameters(), lr=LR_VAE, betas=(0.5, 0.999))
#opt_vae = torch.optim.SGD(vae.parameters(), lr=LR_VAE)
opt_dis = torch.optim.Adam(dis.parameters(), lr=0.0005, betas=(0.5, 0.999))


def calc_blur(X):
    """
    https://github.com/tolstikhin/wae/blob/master/wae.py -- line 344
    Keep track of the min blurriness / batch for each test loop
    """

    # RGB case
    if X.size(1) == 3:
        X = torch.mean(X, 1, keepdim=True)

    lap_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    lap_filter = lap_filter.reshape([1, 1, 3, 3])
    lap_filter = Variable(torch.from_numpy(lap_filter).float())

    if CUDA:
        lap_filter = lap_filter.cuda()

    # valid padding (i.e., no padding)
    conv = F.conv2d(X, lap_filter, padding=0, stride=1)

    # smoothness is the variance of the convolved image
    var = torch.var(conv)

    return(var)


def ae_loss(recon_x, X):
    return(RECON(recon_x, X))


def kld_loss(mu, logvar):
    """
    https://arxiv.org/abs/1312.6114 (Appendix B)
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    kld = torch.sum(kld_element).mul_(-0.5)

    return(kld)


def wae_gan_loss(z_encoded, z_sample):
    """
    https://github.com/wsnedy/WAE_Pytorch/blob/master/wae_for_mnist.py
    https://myurasov.github.io/2017/09/24/wasserstein-gan-keras.html
    """
    # cross entropy loss, as discriminator has 2 outputs to distinguish real
    # (pz) and fake (qz) samples.
    ce = nn.CrossEntropyLoss()

    logits_pz = dis(z_encoded) # is 0 when correct
    logits_qz = dis(z_sample)  # is 1 when correct

    # this is used in the WAE loss function, i.e., when is pz 1?
    # we use [:, 1] to make our targets (ones_like or zeros_like) a single col
    loss_penalty = ce(logits_pz, torch.ones_like(logits_pz[:, 1]).long())

    # this is used to train the Discriminator
    loss_pz = ce(logits_pz, torch.zeros_like(logits_pz[:, 1]).long())
    loss_qz = ce(logits_qz, torch.ones_like(logits_qz[:, 1]).long())
    loss_discrim = LAMBDA * (loss_pz + loss_qz)

    return (loss_discrim, logits_pz, logits_qz), loss_penalty


def isnan(x):
    return(x != x)


def wae_mmd_loss(x, y):
    """
    inverse multiquadratic kernel for MMD

    C = 2.0 * N_Z * (sigma ** 2)
    return(C / (C + torch.mean(z_pair[0] - z_pair[1]) ** 2))

    https://github.com/tolstikhin/wae/blob/master/wae.py -- line 226
    https://discuss.pytorch.org/t/maximum-mean-discrepancy-mmd-and-radial-basis-function-rbf/1875
    https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
    """

    # x = z_encoded # y = z_sampled
    norm_x = torch.sum(x**2, 1, keepdim=True)
    norm_y = torch.sum(y**2, 1, keepdim=True)

    dot_xx = torch.mm(x, x.t())
    dot_yy = torch.mm(y, y.t())
    dot_xy = torch.mm(x, y.t())

    dist_xx = norm_x + norm_x.transpose(1, 0) - 2.0 * dot_xx
    dist_yy = norm_y + norm_y.transpose(1, 0) - 2.0 * dot_yy
    dist_xy = norm_y + norm_x.transpose(1, 0) - 2.0 * dot_xy

    # kernel: k(x, y) = C / (C + ||x - y||^2)
    # expand: xx + yy - 2xy
    C_init = 2.0 * N_Z * (SIGMA ** 2)

    mmd = 0.0

    for scale in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        C = C_init * scale

        res_xx = (C / (C + dist_xx + 1e-8))
        dims = res_xx.size(0)
        # NB: in original code he does NOT use the abs ... is it a bug?
        #     otherwise it will make the entire matrix negative while 0-ing diag
        mask = Variable(torch.abs(1.0-torch.eye(dims)).cuda())
        res_xx = torch.mul(res_xx, mask)
        res_xx = 1.0 * torch.sum(res_xx) / (BATCH*(BATCH-1))     # no diag

        res_yy = (C / (C + dist_yy + 1e-8))
        dims = res_yy.size(0)
        # NB: in original code he does NOT use the abs ... is it a bug?
        #     otherwise it will make the entire matrix negative while 0-ing diag
        mask = Variable(torch.abs(1.0-torch.eye(dims)).cuda())
        res_yy = torch.mul(res_yy, mask)
        res_yy = 1.0 * torch.sum(res_yy) / (BATCH*(BATCH-1))     # no diag

        res_xy = C / (C + dist_xy + 1e-8)
        res_xy = 2.0 * torch.sum(res_xy) / BATCH**2              # keep diag

        mmd += (res_xx + res_yy - res_xy)


    if isnan(mmd.data[0]):
        import IPython; IPython.embed()

    return(mmd)


def calc_loss(X, recon, mu, logvar, method='vae', verbose=False):
    """
    key observation: do we SUBTRACT or ADD LAMBDA*PENALTY/MMD?
    in the paper he always adds LAMBDA*PENALTY/MMD...
    """

    try:
        assert method in ['vae', 'wae-gan', 'wae-mmd']
    except:
        print('incorrect LOSS specified: {}'.format(method))
        sys.exit(1)

    loss_gan = None # default value, if not using wae-gan

    loss_recon = ae_loss(recon, X)
    loss_recon = loss_recon / BATCH # normalize loss_recon by BATCH

    if method == 'vae':
        kld = kld_loss(mu, logvar)
        loss = loss_recon + kld

        if verbose:
            print('recon: {}, kld: {}'.format(loss_recon.data[0], kld.data[0]))

    elif method == 'wae-gan':
        z_encoded = vae.encode(X)
        z_sampled = vae.sample_pz(X.size(0)) # always normal distribution
        loss_gan, loss_penalty = wae_gan_loss(z_encoded, z_sampled)

        # normalize loss for discriminator by LAMBDA/BATCH
        loss_recon = loss_recon * SCALEBULLSHIT
        loss_gan = LAMBDA * loss_gan[0] / BATCH
        loss = loss_recon + (LAMBDA * loss_penalty) # NB: RECON + LAMBDA*PENALTY

        # loss_penalty is the misclassification rate os z_encoded by discrim
        if verbose:
            print('z_encoded {} : {} : {}'.format(
                torch.min(z_encoded).data[0],
                torch.mean(z_encoded).data[0],
                torch.max(z_encoded).data[0]))
            print('recon: {}, penalty: {}, disc: {}'.format(
                loss_recon.data[0], loss_penalty.data[0], loss_gan.data[0]))

    elif method == 'wae-mmd':
        # DOES BATCH SIZE NEED TO MATCH LATENT SPACE SIZE??
        z_encoded = vae.encode(X)
        z_sampled = vae.sample_pz(X.size(0)) # always normal distribution
        loss_mmd = wae_mmd_loss(z_encoded, z_sampled)

        # loss_mmd is already normalized by BATCH**2, see wae_mmd_loss
        loss_recon = loss_recon * SCALEBULLSHIT
        loss = loss_recon + (LAMBDA * loss_mmd) # NB: RECON - LAMBDA*MMD

        if verbose:
            print('z_encoded {} : {} : {}'.format(
                torch.min(z_encoded).data[0],
                torch.mean(z_encoded).data[0],
                torch.max(z_encoded).data[0]))
            print('recon: {}, mmd: {}'.format(loss_recon.data[0], loss_mmd.data[0]))

    return(loss, loss_gan)


def train(ep, data):
    vae.train()
    train_loss = 0

    for batch, X in enumerate(data):

        if CUDA:
            X = X.cuda()
        X = Variable(X)

        # inserts single channel dimensions for black and white images
        if DATASET == 'mnist':
            X = X.unsqueeze(1)

        recon, mu, logvar = vae.forward(X)

        loss, loss_gan = calc_loss(X, recon, mu, logvar, method=LOSS)

        # optimize VAE / WAE
        opt_vae.zero_grad()
        loss.backward(retain_graph=True)
        opt_vae.step()
        train_loss += loss.data[0]

        if LOSS == 'wae-gan':
            # optimize discriminator
            opt_dis.zero_grad()
            loss_dis = loss_gan[0]
            loss_dis.backward()
            opt_dis.step()
        else:
            opt_dis.zero_grad()
            opt_dis.step()

        if batch % 100 == 0:
            print('[ep={}/batch={}]: loss={:.4f}'.format(
                ep, batch, loss.data[0]))

    return(train_loss)


def test(ep, data):
    vae.eval()
    test_loss = 0

    for batch, X in enumerate(data):

        if CUDA:
            X = X.cuda()
        X = Variable(X)

        recon, mu, logvar = vae(X)
        loss, loss_gan = calc_loss(X, recon, mu, logvar, method=LOSS, verbose=True)
        test_loss += loss.data[0]

        # TODO: do something with the blur
        blur = calc_blur(recon)

        import IPython; IPython.embed()
        save_image(X.data, 'img/{}_ep_{}_data.jpg'.format(LOSS, ep), nrow=8, padding=2)
        save_image(recon.data, 'img/{}_ep_{}_recon.jpg'.format(LOSS, ep), nrow=8, padding=2)

    print('[{}] test loss: {:.4f}'.format(ep, test_loss))

    return(test_loss)


def main():

    for ep in range(EPOCHS):
        train_loss = train(ep, im_data)
        test_loss  = test(ep,  im_test)
        torch.save(vae.state_dict(), os.path.join(CHCKDIR,
            '{}_ep_{}_train_loss_{:.4f}_test_loss_{:.4f}.pth'.format(
            LOSS, ep, train_loss, test_loss)))

if __name__ == '__main__':
    main()

