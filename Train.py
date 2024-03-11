import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd

import glob

from collections import OrderedDict
import torch
import joblib
import torch.backends.cudnn as cudnn

import torch.optim as optim

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from Networks.networks import MODEL as net

from losses import CharbonnierLoss_IR,CharbonnierLoss_VI, tv_vi,tv_ir

device = torch.device('cuda:0')
use_gpu = torch.cuda.is_available()

if use_gpu:
    print('GPU Mode Acitavted')
else:
    print('CPU Mode Acitavted')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='...', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--weight', default=[0.03,1000,10,100], type=float)

    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--alpha', default=300, type=int,
                        help='number of new channel increases per depth (default: 300)')
    args = parser.parse_args()

    return args


def rotate(image, s):
    if s == 0:
        image = image
    if s == 1:
        HF = transforms.RandomHorizontalFlip(p=1)
        image = HF(image)
    if s == 2:
        VF = transforms.RandomVerticalFlip(p=1)
        image = VF(image)
    return image


def color2gray(image, s):
    if s == 0:
        image = image
    if s ==1:
        l = image.convert('L')
        n = np.array(l)
        image = np.expand_dims(n, axis=2)
        image = np.concatenate((image, image, image), axis=-1)
        image = Image.fromarray(image).convert('RGB')
    return image


class GetDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):


        ir =...
        vi = ...

        if self.transform is not None:
            tran = transforms.ToTensor()
            ir = tran(ir)

            vi = tran(vi)


            return ir,vi

    def __len__(self):
        return len(self.imageFolderDataset)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader_ir,train_loader_vi, model, criterion_CharbonnierLoss_IR,  criterion_CharbonnierLoss_VI, criterion_tv_ir, criterion_tv_vi,optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    losses_CharbonnierLoss_IR = AverageMeter()
    losses_CharbonnierLoss_VI = AverageMeter()
    losses_tv_ir= AverageMeter()
    losses_tv_vi = AverageMeter()
    weight = args.weight
    model.train()

    for i, (ir,vi)  in tqdm(enumerate(train_loader_ir), total=len(train_loader_ir)):

        if use_gpu:

            ir = ir.cuda()
            vi = vi.cuda()

        else:

            ir = ir
            vi = vi

        out = model(ir,vi)

        CharbonnierLoss_IR = weight[0] *  criterion_CharbonnierLoss_IR(out, ir)
        CharbonnierLoss_VI = weight[1] *  criterion_CharbonnierLoss_VI(out, vi)
        loss_tv_ir = weight[2] * criterion_tv_ir(out, ir)
        loss_tv_vi = weight[3] * criterion_tv_vi(out, vi)
        loss = CharbonnierLoss_IR +CharbonnierLoss_VI + loss_tv_ir + loss_tv_vi

        losses.update(loss.item(), ir.size(0))
        losses_CharbonnierLoss_IR.update(CharbonnierLoss_IR.item(), ir.size(0))
        losses_CharbonnierLoss_VI.update(CharbonnierLoss_VI.item(), ir.size(0))
        losses_tv_ir.update(loss_tv_ir.item(), ir.size(0))
        losses_tv_vi.update(loss_tv_vi.item(), ir.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('CharbonnierLoss_IR', losses_CharbonnierLoss_IR.avg),
        ('CharbonnierLoss_VI', losses_CharbonnierLoss_VI.avg),
        ('loss_tv_ir', losses_tv_ir.avg),
        ('loss_tv_vi', losses_tv_vi.avg),
    ])
    return log


def main():
    args = parse_args()

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    cudnn.benchmark = True

    training_dir_ir = ...
    folder_dataset_train_ir = glob.glob(training_dir_ir + "*.bmp")
    training_dir_vi =...

    folder_dataset_train_vi = glob.glob(training_dir_vi + "*.bmp")

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))
                                          ])

    dataset_train_ir = GetDataset(imageFolderDataset=folder_dataset_train_ir,
                                                  transform=transform_train)
    dataset_train_vi = GetDataset(imageFolderDataset=folder_dataset_train_vi,
                                  transform=transform_train)

    train_loader_ir = DataLoader(dataset_train_ir,
                              shuffle=True,
                              batch_size=args.batch_size)
    train_loader_vi = DataLoader(dataset_train_vi,
                                 shuffle=True,
                                 batch_size=args.batch_size)
    model = net()
    if use_gpu:
        model = model.cuda()
        model.cuda()

    else:
        model = model
    criterion_CharbonnierLoss_IR = CharbonnierLoss_IR
    criterion_CharbonnierLoss_VI = CharbonnierLoss_VI
    criterion_tv_ir = tv_ir
    criterion_tv_vi = tv_vi
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps)

    log = pd.DataFrame(index=[],
                       columns=['epoch',

                                'loss',
                                'CharbonnierLoss_IR',
                                'CharbonnierLoss_VI',
                                'loss_tv_ir',
                                'loss_tv_vi',
                                ])

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch + 1, args.epochs))

        train_log = train(args, train_loader_ir, train_loader_vi, model,  criterion_CharbonnierLoss_IR, criterion_CharbonnierLoss_VI,
                          criterion_tv_ir, criterion_tv_vi, optimizer, epoch)  # 训练集


        print('loss: %.4f - CharbonnierLoss_IR: %.4f -CharbonnierLoss_VI: %.4f - loss_tv_ir: %.4f - loss_tv_vi: %.4f '
              % (train_log['loss'],
                 train_log['CharbonnierLoss_IR'],
                 train_log['CharbonnierLoss_VI'],
                 train_log['loss_tv_ir'],
                 train_log['loss_tv_vi'],

                 ))

        tmp = pd.Series([
            epoch + 1,

            train_log['loss'],
            train_log['CharbonnierLoss_IR'],
            train_log['CharbonnierLoss_VI'],
            train_log['loss_tv_ir'],
            train_log['loss_tv_vi'],

        ], index=['epoch', 'loss', 'CharbonnierLoss_IR', 'CharbonnierLoss_VI', 'loss_tv_ir', 'loss_tv_vi'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)


        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), 'models/%s/model_{}.pth'.format(epoch+1) %args.name)


if __name__ == '__main__':
    main()


