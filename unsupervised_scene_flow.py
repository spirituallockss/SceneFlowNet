from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from tqdm import tqdm
from util import visualize_transformed, transform_point_cloud


def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

def test_one_epoch(args, net, test_loader):
    net.eval()

    total_loss = 0
    num_examples = 0

    if args.eval_full:
        target_full = None
        pred_transformed = None
        src_full = None

    for src, target, gt_flow in tqdm(test_loader):
        src = src.cuda()
        target = target.cuda()

        batch_size = src.size(0)
        num_examples += batch_size
        rotation_ab_pred, translation_ab_pred = net(src, target)

        ###########################
        transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)
        loss =  EPE(transformed_src, target)    

        total_loss += loss.item() * batch_size

        if args.eval_full:
            if target_full is None:
                target_full = target.cpu().detach().numpy()
                pred_transformed = transformed_src.cpu().detach().numpy()
                src_full = src.cpu().detach().numpy()
            else:
                target_full = np.concatenate([target_full, target.cpu().detach().numpy()], axis=2)
                pred_transformed = np.concatenate([pred_transformed, transformed_src.cpu().detach().numpy()], axis=2)
                src_full = np.concatenate([src_full, src.cpu().detach().numpy()], axis=2)


    if args.display_scene_flow and args.eval and args.eval_full:
        visualize_transformed(src_full.squeeze(), target_full.squeeze(), pred_transformed.squeeze())
    elif args.display_scene_flow and args.eval:
        visualize_transformed(src.cpu().detach().numpy(), target.cpu().detach().numpy(), transformed_src.cpu().detach().numpy())



    return total_loss * 1.0 / num_examples


def train_one_epoch(args, net, train_loader, opt):
    net.train()

    total_loss = 0
    num_examples = 0

    for src, target, gt_flow in tqdm(train_loader):
        src = src.cuda()
        target = target.cuda()

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size
        rotation_ab_pred, translation_ab_pred = net(src, target)

        ###########################
        transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)
        loss =  EPE(transformed_src, target)

        loss.backward()
        opt.step()
        total_loss += loss.item() * batch_size


    return total_loss * 1.0 / num_examples


def test_flow(args, net, test_loader, boardio, textio):

    test_loss = test_one_epoch(args, net, test_loader)


    textio.cprint('==FINAL TEST==')
    textio.cprint('EPOCH:: %d, Loss: %f'% (-1, test_loss))


def train_flow(args, net, train_loader, test_loader, boardio, textio):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)

    if args.onlytrain:
        best_test_loss = None
        test_loss = None
    else:
        best_test_loss = np.inf
    

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(args, net, train_loader, opt)
        scheduler.step()
        if not args.onlytrain:
            test_loss = test_one_epoch(args, net, test_loader)

            if best_test_loss >= test_loss:
                best_test_loss = test_loss
                if torch.cuda.device_count() > 1:
                    torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
                else:
                    torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)

        textio.cprint('==TRAIN==')
        textio.cprint('EPOCH:: %d, Loss: %f'% (epoch, train_loss))

        if not args.onlytrain:
            textio.cprint('==TEST==')
            textio.cprint('EPOCH:: %d, Loss: %f'% (epoch, test_loss))

            textio.cprint('==BEST TEST==')
            textio.cprint('EPOCH:: %d, Loss: %f'% (epoch, best_test_loss))

            boardio.add_scalar('A->B/test/loss', test_loss, epoch)
            boardio.add_scalar('A->B/test/best_loss', best_test_loss, epoch)
        
        boardio.add_scalar('A->B/train/loss', train_loss, epoch)
            
        if epoch%20==0:
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
            gc.collect()