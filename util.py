#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation
from math import cos, sin, radians
import sys
import mayavi.mlab as mlab
import os.path as osp
import pickle

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')

SCALE_FACTOR = 0.05
MODE = 'sphere'
DRAW_LINE = True   
def visualize_scene(pc1, pc2, sf, output):

	if pc1.shape[1] != 3:
		pc1 = pc1.T
		pc2 = pc2.T
		sf = sf.T
		output = output.T
	
	gt = pc1 + sf
	pred = pc1 + output
	
	print('pc1, pc2, gt, pred', pc1.shape, pc2.shape, gt.shape, pred.shape)


	fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=(1,1,1), engine=None, size=(1600, 1000))
	
	if False: #len(sys.argv) >= 4 and sys.argv[3] == 'pc1':
		mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], color=(0,0,1), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # blue
	
	if False:
		mlab.points3d(pc2[:, 0], pc2[:, 1], pc2[:, 2], color=(0,1,1), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # cyan

	mlab.points3d(gt[:, 0], gt[:, 1], gt[:, 2], color=(1,0,0), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # red
	mlab.points3d(pred[:, 0], pred[:,1], pred[:,2], color=(0,1,0), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # green
	
	
	# DRAW LINE
	if True:
		N = 2
		x = list()
		y = list()
		z = list()
		connections = list()

		inner_index = 0
		for i in range(gt.shape[0]):
			x.append(gt[i, 0])
			x.append(pred[i, 0])
			y.append(gt[i, 1])
			y.append(pred[i, 1])
			z.append(gt[i, 2])
			z.append(pred[i, 2])

			connections.append(np.vstack(
				[np.arange(inner_index,   inner_index + N - 1.5),
				np.arange(inner_index + 1,inner_index + N - 0.5)]
			).T)
			inner_index += N

		x = np.hstack(x)
		y = np.hstack(y)
		z = np.hstack(z)

		connections = np.vstack(connections)

		src = mlab.pipeline.scalar_scatter(x, y, z)

		src.mlab_source.dataset.lines = connections
		src.update()
		
		lines= mlab.pipeline.tube(src, tube_radius=0.005, tube_sides=6)
		mlab.pipeline.surface(lines, line_width=2, opacity=.4, color=(1,1,0))
	# DRAW LINE END

	
	mlab.view(90, # azimuth
	         150, # elevation
			 50, # distance
			 [0, -1.4, 18], # focalpoint
			 roll=0)

	mlab.orientation_axes()

	mlab.show()



def visualize_transformed(pc1, pc2, output):

	if pc1.shape[1] != 3:
		pc1 = pc1.T
		pc2 = pc2.T
		output = output.T
	
	pred = pc1 + output
	
	print('pc1, pc2, pred', pc1.shape, pc2.shape, pred.shape)


	fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=(1,1,1), engine=None, size=(1600, 1000))
	
	if False: #len(sys.argv) >= 4 and sys.argv[3] == 'pc1':
		mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], color=(0,0,1), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # blue
	
	if False:
		mlab.points3d(pc2[:, 0], pc2[:, 1], pc2[:, 2], color=(0,1,1), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # cyan

	mlab.points3d(pred[:, 0], pred[:,1], pred[:,2], color=(0,1,0), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # green
	
	
	mlab.view(90, # azimuth
	         150, # elevation
			 50, # distance
			 [0, -1.4, 18], # focalpoint
			 roll=0)

	mlab.orientation_axes()

	mlab.show()