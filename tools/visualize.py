
# Author: Francois Grondin
# Date: October 19, 2020
# Affiliation: Universite de Sherbrooke
# Contact: francois.grondin2@usherbrooke.ca

import argparse
import matplotlib.pyplot as plt
import os,sys,inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from bird import BIRD

parser = argparse.ArgumentParser()

parser.add_argument('--root', default='', type=str, help='Root to save the datasets')
parser.add_argument('--folds', default=[1,2,3,4,5,6,7,8,9,10], type=int, nargs='+', help='List of folds (between 1 and 100)')
parser.add_argument('--item', default=0, type=int, help='Sample index in the dataset')
parser.add_argument('--view', default='room', type=str, choices=['room', 'rir', 'meta'])

args = parser.parse_args()

rir = BIRD(root=args.root, folder_in_archive='Bird', folds=args.folds)

hs, meta = rir[args.item]

if args.view == 'room':

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	ax.set_box_aspect([meta['L'][0],meta['L'][1],meta['L'][2]])
	ax.set_xlim3d(0,meta['L'][0])
	ax.set_ylim3d(0,meta['L'][1])
	ax.set_zlim3d(0,meta['L'][2])
	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')
	ax.set_zlabel('$z$')

	ax.scatter(meta['mics'][0][0],meta['mics'][0][1],meta['mics'][0][2],c='k')
	ax.scatter(meta['mics'][1][0],meta['mics'][1][1],meta['mics'][1][2],c='k')
	ax.plot([meta['mics'][0][0],meta['mics'][1][0]],[meta['mics'][0][1],meta['mics'][1][1]],[meta['mics'][0][2],meta['mics'][1][2]], c='k')

	ax.scatter(meta['srcs'][0][0],meta['srcs'][0][1],meta['srcs'][0][2],c='b')
	ax.scatter(meta['srcs'][1][0],meta['srcs'][1][1],meta['srcs'][1][2],c='g')
	ax.scatter(meta['srcs'][2][0],meta['srcs'][2][1],meta['srcs'][2][2],c='r')
	ax.scatter(meta['srcs'][3][0],meta['srcs'][3][1],meta['srcs'][3][2],c='c')

	plt.show()

if args.view == 'rir':

	plt.subplot(4,2,1)
	plt.plot(hs[0,:].numpy())
	plt.title('$h_{1,1}$', y=1.08)
	plt.subplot(4,2,2)
	plt.plot(hs[1,:].numpy())
	plt.title('$h_{1,2}$', y=1.08)
	plt.subplot(4,2,3)
	plt.plot(hs[2,:].numpy())
	plt.title('$h_{2,1}$', y=1.08)
	plt.subplot(4,2,4)
	plt.plot(hs[3,:].numpy())
	plt.title('$h_{2,2}$', y=1.08)
	plt.subplot(4,2,5)
	plt.plot(hs[4,:].numpy())
	plt.title('$h_{3,1}$', y=1.08)
	plt.subplot(4,2,6)
	plt.plot(hs[5,:].numpy())
	plt.title('$h_{3,2}$', y=1.08)	
	plt.subplot(4,2,7)
	plt.plot(hs[6,:].numpy())
	plt.title('$h_{4,1}$', y=1.08)
	plt.subplot(4,2,8)
	plt.plot(hs[7,:].numpy())
	plt.title('$h_{4,2}$', y=1.08)	
	plt.tight_layout()

	plt.show()

if args.view == 'meta':

	print(meta)

