import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from easydict import EasyDict
import argparse
import os
import matplotlib.pyplot as plt
from _main_base import *

#---
# config
#---
cfg = EasyDict()

# class
cfg.CLASS_LABEL = ['akahara', 'madara'] # list, dict('label' : '[B, G, R]')
cfg.CLASS_NUM = len(cfg.CLASS_LABEL)

# model
cfg.INPUT_Z_DIM = 100
cfg.INPUT_MODE = None

cfg.OUTPUT_HEIGHT = 64
cfg.OUTPUT_WIDTH = 64
cfg.OUTPUT_CHANNEL = 3
cfg.OUTPUT_MODE = 'RGB'  # RGB, GRAY, EDGE, CLASS_LABEL

cfg.G_DIM = 128
cfg.D_DIM = 256

cfg.GPU = False
cfg.DEVICE = torch.device("cuda" if cfg.GPU and torch.cuda.is_available() else "cpu")

# train
cfg.TRAIN = EasyDict()
cfg.TRAIN.DISPAY_ITERATION_INTERVAL = 50

cfg.TRAIN.MODEL_G_SAVE_PATH = 'models/GAN_G_{}.pt'
cfg.TRAIN.MODEL_D_SAVE_PATH = 'models/GAN_D_{}.pt'
cfg.TRAIN.MODEL_SAVE_INTERVAL = 200
cfg.TRAIN.ITERATION = 5000
cfg.TRAIN.MINIBATCH = 32
cfg.TRAIN.OPTIMIZER = torch.optim.SGD
cfg.TRAIN.LEARNING_RATE = 0.01
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.LOSS_FUNCTION = loss_fn = torch.nn.BCELoss()

cfg.TRAIN.DATA_PATH = '../Dataset/train/images/'
cfg.TRAIN.DATA_HORIZONTAL_FLIP = True
cfg.TRAIN.DATA_VERTICAL_FLIP = True
cfg.TRAIN.DATA_ROTATION = False

# test
cfg.TEST = EasyDict()
cfg.TEST.MODEL_G_PATH = cfg.TRAIN.MODEL_G_SAVE_PATH.format('final')
cfg.TEST.DATA_PATH = '../Dataset/test/images/'
cfg.TEST.MINIBATCH = 10
cfg.TEST.ITERATION = 2

# random seed
torch.manual_seed(0)


# make model save directory
def make_dir(path):
    if '/' in path:
        model_save_dir = '/'.join(path.split('/')[:-1])
        os.makedirs(model_save_dir, exist_ok=True)

make_dir(cfg.TRAIN.MODEL_G_SAVE_PATH)
make_dir(cfg.TRAIN.MODEL_D_SAVE_PATH)
    

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.module = torch.nn.Sequential(OrderedDict({
            'linear1' : torch.nn.Linear(cfg.INPUT_Z_DIM, cfg.G_DIM),
            'linear1_bn' : torch.nn.BatchNorm1d(cfg.G_DIM),
            'linear1_leakyReLU' : torch.nn.LeakyReLU(0.2),
            'linear2' : torch.nn.Linear(cfg.G_DIM, cfg.G_DIM * 2),
            'linear2_bn' : torch.nn.BatchNorm1d(cfg.G_DIM * 2),
            'linear2_leakyReLU' : torch.nn.LeakyReLU(0.2),
            'linear3' : torch.nn.Linear(cfg.G_DIM * 2, cfg.G_DIM * 4),
            'linear3_bn' : torch.nn.BatchNorm1d(cfg.G_DIM * 4),
            'linear3_leakyReLU' : torch.nn.LeakyReLU(0.2),
            'linear_out' : torch.nn.Linear(cfg.G_DIM * 4, cfg.OUTPUT_HEIGHT * cfg.OUTPUT_WIDTH * cfg.OUTPUT_CHANNEL),
            'linear_out_tanh' : torch.nn.Tanh()
        }))

    def forward(self, x):
        x = self.module(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.module = torch.nn.Sequential(OrderedDict({
            'linear1' : torch.nn.Linear(cfg.OUTPUT_HEIGHT * cfg.OUTPUT_WIDTH * cfg.OUTPUT_CHANNEL, cfg.D_DIM * 2),
            'linear1_leakyReLU' : torch.nn.LeakyReLU(0.2),
            'linear2' : torch.nn.Linear(cfg.D_DIM * 2, cfg.D_DIM),
            'linear2_leakyReLU' : torch.nn.LeakyReLU(0.2),
            'linear_out' : torch.nn.Linear(cfg.D_DIM, 1),
            'linear_out_sigmoid' : torch.nn.Sigmoid()
        }))

    def forward(self, x):
        x = self.module(x)
        return x


# train
def train():
    # model
    G = Generator().to(cfg.DEVICE)
    D = Discriminator().to(cfg.DEVICE)

    opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)
    opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)

    path_dict = data_load(cfg)
    paths = path_dict['paths']
    paths_gt = path_dict['paths_gt']

    # training
    mbi = 0
    train_N = len(paths)
    train_ind = np.arange(train_N)
    np.random.seed(0)
    np.random.shuffle(train_ind)

    ones = torch.tensor([1] * cfg.TRAIN.MINIBATCH, dtype=torch.float).to(cfg.DEVICE)
    zeros = ones * 0

    print('training start')
    progres_bar = ''
    
    for i in range(cfg.TRAIN.ITERATION):
        if mbi + cfg.TRAIN.MINIBATCH > train_N:
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[ : (cfg.TRAIN.MINIBATCH - (train_N - mbi))]))
            mbi = cfg.TRAIN.MINIBATCH - (train_N - mbi)
        else:
            mb_ind = train_ind[mbi : mbi + cfg.TRAIN.MINIBATCH]
            mbi += cfg.TRAIN.MINIBATCH

        opt_D.zero_grad()
        opt_G.zero_grad()

        # sample X
        Xs = torch.tensor(get_image(paths[mb_ind], cfg, cfg.INPUT_MODE), dtype=torch.float).to(cfg.DEVICE)
        Xs = torch.reshape(Xs, [cfg.TRAIN.MINIBATCH, -1])

        # sample x
        z = np.random.uniform(-1, 1, size=(cfg.TRAIN.MINIBATCH, cfg.INPUT_Z_DIM))
        z = torch.tensor(z, dtype=torch.float).to(cfg.DEVICE)

        # forward
        Gz = G(z)
        #Gz = torch.reshape(Gz, [cfg.TRAIN.MINIBATCH, cfg.OUTPUT_CHANNEL, cfg.OUTPUT_HEIGHT, cfg.OUTPUT_WIDTH])

        D_real = D(Xs)[..., 0]
        D_fake = D(Gz)[..., 0]

        # update G
        loss_G = cfg.TRAIN.LOSS_FUNCTION(D_fake, ones)
        loss_G.backward(retain_graph=True)
        opt_G.step()

        # update D
        loss_D_real = cfg.TRAIN.LOSS_FUNCTION(D_real, ones)
        loss_D_fake = cfg.TRAIN.LOSS_FUNCTION(D_fake, zeros)
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        opt_D.step()

        progres_bar += '|'
        print('\r' + progres_bar, end='')

        if (i + 1) % 10 == 0:
            progres_bar += str(i + 1)
            print('\r' + progres_bar, end='')

        # display training state
        if (i + 1) % cfg.TRAIN.DISPAY_ITERATION_INTERVAL == 0:
            print('\r' + ' ' * len(progres_bar), end='')
            print('\rIter : {} , Loss G (Fake : {:.4f}) , Loss D : {:.4f} (Real : {:.4f} , Fake : {:.4f})'.format(
                i + 1, loss_G.item(), loss_D.item(), loss_D_real.item(), loss_D_fake.item()))
            progres_bar = ''

        # save parameters
        if (cfg.TRAIN.MODEL_SAVE_INTERVAL != False) and ((i + 1) % cfg.TRAIN.MODEL_SAVE_INTERVAL == 0):
            G_save_path = cfg.TRAIN.MODEL_G_SAVE_PATH.format('iter{}'.format(i + 1))
            D_save_path = cfg.TRAIN.MODEL_D_SAVE_PATH.format('iter{}'.format(i + 1))
            torch.save(G.state_dict(), G_save_path)
            torch.save(D.state_dict(), D_save_path)
            print('save G >> {}, D >> {}'.format(G_save_path, D_save_path))

    G_save_path = cfg.TRAIN.MODEL_G_SAVE_PATH.format('final')
    D_save_path = cfg.TRAIN.MODEL_D_SAVE_PATH.format('final')
    torch.save(G.state_dict(), G_save_path)
    torch.save(D.state_dict(), D_save_path)
    print('save G >> {}, D >> {}'.format(G_save_path, D_save_path))


# test
def test():
    print('-' * 20)
    print('test function')
    print('-' * 20)
    G = Generator().to(cfg.DEVICE)
    G.load_state_dict(torch.load(cfg.TEST.MODEL_G_PATH, map_location=torch.device(cfg.DEVICE)))
    G.eval()

    np.random.seed(0)
    
    for i in range(cfg.TEST.ITERATION):
        z = np.random.uniform(-1, 1, size=(cfg.TEST.MINIBATCH, cfg.INPUT_Z_DIM))
        z = torch.tensor(z, dtype=torch.float).to(cfg.DEVICE)

        Gz = G(z).detach().cpu().numpy()

        Gz = (Gz * 127.5 + 127.5).astype(np.uint8)
        Gz = Gz.reshape([cfg.TEST.MINIBATCH, cfg.OUTPUT_CHANNEL, cfg.OUTPUT_HEIGHT, cfg.OUTPUT_WIDTH])
        Gz = Gz.transpose(0,2,3,1)

        for i in range(cfg.TEST.MINIBATCH):
            _G = Gz[i]
            plt.subplot(1, cfg.TEST.MINIBATCH, i + 1)
            plt.imshow(_G)
            plt.axis('off')

        plt.show()


def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()
    return args

# main
if __name__ == '__main__':
    args = arg_parse()

    if args.train:
        train()
    if args.test:
        test()

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")
