#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'modelnet40_data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR,
                          'modelnet40_ply_hdf5_2048')):
        www = \
            'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % zipfile)


def kitti_download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'kitti_data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'kitti_data.zip')):
        www = \
            'https://drive.google.com/file/d/190WbiZSEFzuW_hfxDBZh3wW2TnNcz9AD/view?usp=sharing'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % zipfile)


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'modelnet40_data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR,
                             'modelnet40_ply_hdf5_2048',
                             'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name, 'r+')
        data = (f['data'])[:].astype('float32')
        label = (f['label'])[:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return (all_data, all_label)


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1),
                                   xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    (N, C) = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip,
                          clip)
    return pointcloud


class ModelNet40(Dataset):

    def __init__(
        self,
        num_points,
        partition='train',
        gaussian_noise=False,
        unseen=False,
        factor=4,
        ):
        (self.data, self.label) = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.factor = factor
        if self.unseen:

            # #simulate testing on first 20 categories while training on last 20 categories

            if self.partition == 'test':
                self.data = self.data[self.label >= 20]
                self.label = self.label[self.label >= 20]
            elif self.partition == 'train':
                self.data = self.data[self.label < 20]
                self.label = self.label[self.label < 20]

    def __getitem__(self, item):
        pointcloud = (self.data[item])[:self.num_points]

        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5),
                                  np.random.uniform(-0.5, 0.5),
                                  np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley,
                anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T \
            + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        return (
            pointcloud1.astype('float32'),
            pointcloud2.astype('float32'),
            R_ab.astype('float32'),
            translation_ab.astype('float32'),
            R_ba.astype('float32'),
            translation_ba.astype('float32'),
            euler_ab.astype('float32'),
            euler_ba.astype('float32'),
            )

    def __len__(self):
        return self.data.shape[0]


def load_kitti_reg_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'kitti_data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR,
                             'kitti_registration_%s*.h5' % partition)):
        f = h5py.File(h5_name, 'r+')
        data = (f['data'])[:].astype('float32')
        label = (f['label'])[:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return (all_data, all_label)


class Kitti2015Reg(Dataset):

    def __init__(
        self,
        num_points,
        partition='train',
        gaussian_noise=False,
        unseen=False,
        factor=4,
        ):
        print ('----- Loading KITTI 2015 DATASET -----')
        (self.data, self.label) = load_kitti_reg_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.factor = factor
        if self.unseen:

            # ####### simulate testing on first 20 categories while training on last 20 categories

            if self.partition == 'test':
                self.data = self.data[self.label >= 20]
                self.label = self.label[self.label >= 20]
            elif self.partition == 'train':
                self.data = self.data[self.label < 20]
                self.label = self.label[self.label < 20]

    def __getitem__(self, item):
        pointcloud = (self.data[item])[:self.num_points]
        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5),
                                  np.random.uniform(-0.5, 0.5),
                                  np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley,
                anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T \
            + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        return (
            pointcloud1.astype('float32'),
            pointcloud2.astype('float32'),
            R_ab.astype('float32'),
            translation_ab.astype('float32'),
            R_ba.astype('float32'),
            translation_ba.astype('float32'),
            euler_ab.astype('float32'),
            euler_ba.astype('float32'),
            )

    def __len__(self):
        return self.data.shape[0]


# pc1, pc2, gt

def load_scene_flow_data(dataset_name, partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, dataset_name + '_data')
    all_pc1 = []
    all_pc2 = []
    all_gt = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, dataset_name
                             + '_scene_flow_%s*.h5' % partition)):
        f = h5py.File(h5_name, 'r+')
        pc1 = (f['pc1'])[:].astype('float32')
        pc2 = (f['pc2'])[:].astype('float32')
        gt = (f['gt'])[:].astype('float32')
        f.close()
        all_pc1.append(pc1)
        all_pc2.append(pc2)
        all_gt.append(gt)
    all_pc1 = np.concatenate(all_pc1, axis=0)
    all_pc2 = np.concatenate(all_pc2, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)

    # ####
    # len_data = {'train':100, 'test':50}
    # dt_pts = len_data[partition]
    # all_pc1 = all_pc1[:dt_pts,:,:]
    # all_pc2 = all_pc2[:dt_pts,:,:]
    # all_gt = all_gt[:dt_pts,:,:]

    return (all_pc1, all_pc2, all_gt)


class SceneFlow(Dataset):

    def __init__(
        self,
        dataset_name,
        num_points,
        partition='train',
        gaussian_noise=False,
        unseen=False,
        factor=4,
        ):
        if dataset_name == 'flyingthings3dflow':
            dataset_name = 'flyingthings3d'
        elif dataset_name == 'kitti2015flow':
            dataset_name = 'kitti'

        print ('----- Loading %s Scene Flow Dataset -----' \
            % dataset_name.upper())
        (self.pc1, self.pc2, self.gt) = \
            load_scene_flow_data(dataset_name, partition)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.factor = factor
        if self.unseen:

            # ####### simulate testing on first 20 categories while training on last 20 categories

            if self.partition == 'test':
                self.data = self.data[self.label >= 20]
                self.label = self.label[self.label >= 20]
            elif self.partition == 'train':
                self.data = self.data[self.label < 20]
                self.label = self.label[self.label < 20]

    def __getitem__(self, item):
        pointcloud1 = (self.pc1[item])[:self.num_points]
        pointcloud2 = (self.pc2[item])[:self.num_points]
        gt_flow = (self.gt[item])[:self.num_points]
        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)
        if self.partition != 'train':
            np.random.seed(item)

        pointcloud1 = pointcloud1.T
        pointcloud2 = pointcloud2.T
        gt_flow = gt_flow.T

        return (pointcloud1.astype('float32'),
                pointcloud2.astype('float32'), gt_flow.astype('float32'
                ))

    def __len__(self):
        return self.pc1.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data in train:
        print (len(data))
        break