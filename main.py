#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.nn as nn
from data import ModelNet40, Kitti2015Reg, SceneFlow
from model import DCP, DCFlow, UnsupervisedDCFlow
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm


# Part of the code is referred from: https://github.com/floodsung/LearningToCompare_FSL

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def main():
    parser = argparse.ArgumentParser(description='3D flow')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dcp', metavar='N',
                        choices=['dcp', 'dcflow', 'unsupervised_dcflow'],
                        help='Model to use, [dcp, dcflow]')
    parser.add_argument('--emb_nn', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding nn to use, [pointnet, dgcnn]')
    parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
                        choices=['identity', 'transformer'],
                        help='Attention-based pointer generator to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd', metavar='N',
                        choices=['mlp', 'svd', 'pointnet'],
                        help='Head to use, [mlp, svd, pointnet]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40', 'kitti2015reg', 'kitti2015flow', 'flyingthings3dflow'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--display_scene_flow', action='store_true', default=True,
                        help='view the scene flow at testing')
    parser.add_argument('--onlytrain', action='store_true', default=False,
                        help='Only performs training when --eval is not passed')    
    parser.add_argument('--resume_training', action='store_true', default=False,
                        help='Resume training from model_path or best checkpoint') 
    parser.add_argument('--eval_full', action='store_true', default=False,
                        help='Eval on entire pointcloud') 


    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    if args.dataset == 'modelnet40':
        train_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='train', gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='test', gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'kitti2015reg':
        train_loader = DataLoader(
            Kitti2015Reg(num_points=args.num_points, partition='train', gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            Kitti2015Reg(num_points=args.num_points, partition='test', gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)    
    elif args.dataset == 'kitti2015flow' or args.dataset == 'flyingthings3dflow':
        train_loader = DataLoader(
            SceneFlow(dataset_name=args.dataset, num_points=args.num_points, partition='train', gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        if args.onlytrain:
            test_loader = None
        else:
            if args.eval_full:
                partition = 'full'
            else:
                partition = 'test'
            test_loader = DataLoader(
                SceneFlow(dataset_name=args.dataset, num_points=args.num_points, partition=partition, gaussian_noise=args.gaussian_noise,
                        unseen=args.unseen, factor=args.factor),
                batch_size=args.test_batch_size, shuffle=False, drop_last=False) 
    else:
        raise Exception("not implemented")

    if args.model == 'dcp' and args.dataset != 'kitti2015flow':
        net = DCP(args).cuda()
        if args.eval:
            if args.model_path is '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            net.load_state_dict(torch.load(model_path), strict=False)
    elif args.model == 'dcflow':
        net = DCFlow(args).cuda()
        if args.eval or args.resume_training:
            if args.model_path is '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained/checkpoint model")
                return
            net.load_state_dict(torch.load(model_path), strict=False)
    elif args.model == 'unsupervised_dcflow':
        net = UnsupervisedDCFlow(args).cuda()
        if args.eval or args.resume_training:
            if args.model_path is '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained/checkpoint model")
                return
            net.load_state_dict(torch.load(model_path), strict=False)        
    else:
        raise Exception('Not implemented')

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.eval:
        if args.model == 'dcp':
            from registration import test
            test(args, net, test_loader, boardio, textio)
        elif args.model == 'dcflow':
            from scene_flow import test_flow
            test_flow(args, net, test_loader, boardio, textio)
        elif args.model == 'unsupervised_dcflow':
            from unsupervised_scene_flow import test_flow
            test_flow(args, net, test_loader, boardio, textio)    
    else:
        if args.model == 'dcp':
            from registration import train
            train(args, net, train_loader, test_loader, boardio, textio)
        elif args.model == 'dcflow':
            from scene_flow import train_flow
            train_flow(args, net, train_loader, test_loader, boardio, textio)
        elif args.model == 'unsupervised_dcflow':
            from unsupervised_scene_flow import train_flow
            train_flow(args, net, train_loader, test_loader, boardio, textio)    


    print('FINISH')
    boardio.close()


if __name__ == '__main__':
    main()
