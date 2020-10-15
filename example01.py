
import argparse
import matplotlib.pyplot as plt

from bird import BIRD

parser = argparse.ArgumentParser()

parser.add_argument('--root', default='', type=str, help='Root to save the datasets')
parser.add_argument('--folds', default=[1,2,3,4,5,6,7,8,9,10], type=int, nargs='+', help='List of folds')
parser.add_argument('--item', default=0, type=int, help='Sample index in the dataset')

args = parser.parse_args()

rir = BIRD(root=args.root, folder_in_archive='Bird', folds=args.folds)

hs, meta = rir[args.item]

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