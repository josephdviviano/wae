#!/usr/bin/env python

from PIL import Image
import numpy as np
import os
import torch

def data_celeb(dimentions):
    ''' n = 1 will load all the data '''
    im_data = []
    n_chan, dims = dimentions[0], dimentions[1]

    if n == -1:
        files = os.listdir(directory)
    else:
        files = os.listdir(directory)[:n]

    for i, f in enumerate(files):
        if i % 1000 == 0:
            print('loaded {}/{}'.format(i, n))

        im = Image.open(os.path.join(directory, f))
        im = np.asarray(im.convert('RGB').getdata()).astype(np.float)
        mins = np.min(im)
        maxs = np.max(im)
        im = (im - mins)/(maxs - mins)*2 - 1

        # reshape image
        im = im.transpose((1,0))
        im = im.reshape(n_chan, dims, dims)

        im_data.append(im)

    return(im_data)


def load_data(name, n, directory, batch_size, dimentions):
    """
    im_data is a list of n x 4096 x 3
    name.npy is n x 3 x 64 x 64
    """
    if not os.path.isfile(name):
        im_data = data_celeb(dimentions, n=n)
        np.save(name, im_data)
    else:
        im_data = np.load(name)

    im_data = torch.Tensor(im_data)
    im_data = torch.utils.data.DataLoader(im_data, batch_size=batch_size, num_workers=2)

    return(im_data)

