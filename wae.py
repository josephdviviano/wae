#!/usr/bin/env python
import os
import sys
import scandir
import time

from PIL import Image
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

DIMS = 64  # dependent on script that generates preprocessed data
N_CHAN = 3
N_DATA = 20000
N_TEST = 1000
N_Z = 64 # mnist=8, celeba=64
LAMBDA = 1 # celeba gan = 1, celeba mdd = 100, mnist = 10
SIGMA = 2 # celebaa = 2, mnist = 1

BATCH = 50
EPOCHS = 100

DATADIR = '/u/vivianoj/data/celeba/data/'
CUDA = torch.cuda.is_available()
RECON = nn.BCELoss()
RECON.size_average = False

LOSS = 'wae-gan' # 'vae', 'wae-gan', 'wae-mmd'


def data_celeb(n=-1):
    ''' n = 1 will load all the data '''
    im_data = []

    if n == -1:
        files = os.listdir(DATADIR)
    else:
        files = os.listdir(DATADIR)[:n]

    for i, f in enumerate(files):
        if i % 1000 == 0:
            print('loaded {}/{}'.format(i, n))

        im = Image.open(os.path.join(DATADIR, f))
        im = np.asarray(im.convert('RGB').getdata()).astype(np.float)
        mins = np.min(im)
        maxs = np.max(im)
        im = (im - mins)/(maxs - mins)*2 - 1

        # reshape image
        im = im.transpose((1,0))
        im = im.reshape(N_CHAN, DIMS, DIMS)

        im_data.append(im)

    return(im_data)


def load_data(name, n):
    """
    im_data is a list of n x 4096 x 3
    name.npy is n x 3 x 64 x 64
    """
    if not os.path.isfile(name):
        im_data = data_celeb(n=n)
        np.save(name, im_data)
    else:
        im_data = np.load(name)

    im_data = torch.Tensor(im_data)
    im_data = torch.utils.data.DataLoader(im_data, batch_size=BATCH, num_workers=2)

    return(im_data)


class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(VAE, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        # TODO: redo using nn.Sequentials (1 per layer)
        self.e1 = nn.Conv2d(nc, ndf, 5, 2, 2)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 5, 2, 2)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 5, 2, 2)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 5, 2, 2)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.fc1 = nn.Linear(ndf*8*4*4, latent_variable_size)
        self.fc2 = nn.Linear(ndf*8*4*4, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf*8*4*4)

        self.d2 = nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1)
        self.bn6 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.d3 = nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1)
        self.bn7 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.d4 = nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1)
        self.bn8 = nn.BatchNorm2d(ngf, 1.e-3)

        self.d5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1)

        self.relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        """ encoder architecture: note 2 read-out heads for mu and logvar """
        h1 = self.relu(self.bn1(self.e1(x)))
        h2 = self.relu(self.bn2(self.e2(h1)))
        h3 = self.relu(self.bn3(self.e3(h2)))
        h4 = self.relu(self.bn4(self.e4(h3)))
        h4 = h4.view(-1, self.ndf*8*4*4)
        #print(h1.size())
        #print(h2.size())
        #print(h3.size())
        #print(h4.size())

        return(self.fc1(h4), self.fc2(h4))

    def reparametrize(self, mu, logvar):
        """ generates a normal distribution with the specified mean and std"""
        std = logvar.mul(0.5).exp_() # converts log(var) to std

        # z is a gaussian with std and mean=mu. begin with eps (a stock gauss)
        if CUDA:
            eps = Variable(torch.cuda.FloatTensor(std.size()).normal_())
        else:
            eps = Variable(torch.FloatTensor(std.size()).normal_())

        z = eps.mul(std).add_(mu)

        return(z)

    def decoder(self, z):
        """ decoder architecture """
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*8, 4, 4)
        h2 = self.relu(self.bn6(self.d2(h1)))
        h3 = self.relu(self.bn7(self.d3(h2)))
        h4 = self.relu(self.bn8(self.d4(h3)))
        out = self.sigmoid(self.d5(h4))
        #print(h1.size())
        #print(h2.size())
        #print(h3.size())
        #print(h4.size())
        #print(out.size())

        return(out)

    def encode(self, x):
        """ feeds data through encoder and reparameterization to produce z """
        mu, logvar = self.encoder(x.view(-1, self.nc, DIMS, DIMS))
        z = self.reparametrize(mu, logvar)
        return(z)

    def decode(self, z):
        """ feeds z through decoder """
        return(self.decoder(z))

    def sample_pz(self):
        """ samples noise from a gaussian distribution. is it always mean=0?"""
        # is this always the same distribution??
        noise = Variable(torch.normal(torch.zeros(BATCH, N_Z), std=SIGMA).cuda())
        return(noise)

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

    # valid padding (i.e., no padding)
    conv = F.conv2d(Variable(X), Variable(lap_filter), padding=0, stride=1)

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


def kernel(z_pair, sigma=1):
    """ inverse multiquadratic kernel for MMD """
    C = 2.0 * N_Z * (sigma ** 2)
    return(C / (C + torch.mean(z_pair[0] - z_pair[1]) ** 2))


def wae_mmd_loss(x, y):
    """
    useful:
    https://github.com/tolstikhin/wae/blob/master/wae.py -- line 226
    https://discuss.pytorch.org/t/maximum-mean-discrepancy-mmd-and-radial-basis-function-rbf/1875
    https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/

    maybe:
    https://github.com/schelotto/Wasserstein_Autoencoders/blob/master/wae_mmd.py
    https://github.com/paruby/Wasserstein-Auto-Encoders/blob/master/models.py -- line 383
    """

    # x = z_encoded
    # y = z_sampled
    SIGMA = 1.0 # variance, hard coded at 1.0

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

        res1 = (C / (C + dist_xx)) + (C / (C + dist_yy))
        res1 = torch.mul(res1, Variable(1.0-torch.eye(BATCH).cuda())) # 0 diag
        res1 = 1.0 * torch.sum(res1) / (BATCH*(BATCH-1)) # diagonal not included

        res2 = C / (C + dist_xy)
        res2 = 2.0 * torch.sum(res2) / BATCH**2 # keep diagonal for this term

        mmd += res1 - res2

    return(mmd)


def calc_loss(X, recon, mu, logvar, method='vae'):

    try:
        assert method in ['vae', 'wae-gan', 'wae-mmd']
    except:
        print('incorrect LOSS specified: {}'.format(method))
        sys.exit(1)

    loss_gan = None # default value, if not using wae-gan

    loss_recon = ae_loss(recon, X)
    loss_recon = -loss_recon

    if method == 'vae':
        kld = kld_loss(mu, logvar)
        loss = loss_recon + kld

    elif method == 'wae-gan':
        z_encoded = vae.encode(X)
        z_sampled = vae.sample_pz() # always normal distribution
        loss_gan, loss_penalty = wae_gan_loss(z_encoded, z_sampled)
        loss = loss_recon - LAMBDA * torch.log(loss_penalty)

        # loss is penalty from gan_loss (misclassification rate)
        print('recon: {}, gan: {}'.format(loss_recon.data[0], loss_penalty.data[0]))

    elif method == 'wae-mmd':
        z_encoded = vae.encode(X)
        z_sampled = vae.sample_pz() # always normal distribution
        loss_mmd = wae_mmd_loss(z_encoded, z_sample)
        loss = loss_recon + LAMBDA * loss_mmd

    return(loss, loss_gan)


def train(epoch, data):
    vae.train()
    train_loss = 0

    for batch, X in enumerate(data):

        if CUDA:
            X = X.cuda()
        X = Variable(X)

        recon, mu, logvar = vae.forward(X)

        loss, loss_gan = calc_loss(X, recon, mu, logvar, method=LOSS)

        # optimize VAE / WAE
        opt_vae.zero_grad()
        loss.backward(retain_graph=True)
        opt_vae.step()
        train_loss += loss.data[0]

        # optimize discriminator
        if LOSS == 'wae-gan':
            opt_dis.zero_grad()
            loss_dis = loss_gan[0]
            loss_dis.backward()
            opt_dis.step()

        if batch % 10 == 0:
            print('[ep={}/batch={}]: loss={:.4f}'.format(
                ep, batch, loss.data[0]))

    return(train_loss / (len(data)*BATCH*N_CHAN*DIMS*DIMS))


def test(epoch, data):
    vae.eval()
    test_loss = 0

    for batch, X in enumerate(data):

        if CUDA:
            X = X.cuda()
        X = Variable(X)

        recon, mu, logvar = vae(X)

        loss, loss_gan = calc_loss(X, recon, mu, logvar, method=LOSS)

        test_loss += loss.data[0]

        # TODO: do something with the blur
        blur = calc_blur(recon)

        save_image(X.data, 'img/epoch_{}_data.jpg'.format(epoch), nrow=8, padding=2)
        save_image(recon.data, 'img/epoch_{}_recon.jpg'.format(epoch), nrow=8, padding=2)

    test_loss /= (len(data)*BATCH*N_CHAN*DIMS*DIMS)
    print('[{}] test loss: {:.4f}'.format(epoch, test_loss))

    return(test_loss)


im_data = load_data('celeba_data.npy', N_DATA)
im_test = load_data('celeba_test.npy', N_TEST)

vae = VAE(nc=N_CHAN, ngf=128, ndf=128, latent_variable_size=N_Z)
dis = Discriminator(N_Z)

if CUDA:
    vae = vae.cuda()
    dis = dis.cuda()

opt_vae = torch.optim.Adam(vae.parameters(), lr=0.0003, betas=(0.5, 0.999))
opt_dis = torch.optim.Adam(dis.parameters(), lr=0.0005, betas=(0.5, 0.999))

for ep in range(EPOCHS):
    train_loss = train(ep, im_data)
    test_loss = test(ep, im_test)
    torch.save(vae.state_dict(), 'checkpoints/ep_{}_train_loss_{:.4f}_test_loss_{:.4f}.pth'.format(
        ep, train_loss, test_loss))

