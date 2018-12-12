import torch
import torch.nn as nn
from utils import save_images, load_model
from torchvision.utils import save_image
import config
import WAE
import VAE
import os

def traverse_latent(model,images1,images2,num_images=10):
   mu1,logvar1 = model.encode(images1)
   mu2,logvar2 = model.encode(images2)
   z1 = model.reparameterize(mu1,logvar1)
   z2 = model.reparameterize(mu2,logvar2)

   alphas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
   results = None

   for alpha in alphas:
       z = alpha*z1 + (1-alpha)*z2

       recon_x = model.decode(z)
       if type(results) == type(None):
          results = recon_x.data
       else:
           results = torch.cat((results,recon_x.data),dim=0)
   path = './latent_trav/{}/{}/'.format(model.confs['dataset'],model.confs['type'])
   if not os.path.isdir(path):
       os.makedirs(path,)
   save_image(results, '{}/trav_full.jpg'.format(path,alpha), nrow=num_images,padding=2)

