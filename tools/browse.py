
# Author: Francois Grondin
# Date: November 3, 2020
# Affiliation: Universite de Sherbrooke
# Contact: francois.grondin2@usherbrooke.ca

import argparse
import os,sys,inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from bird import BIRD

parser = argparse.ArgumentParser()

parser.add_argument('--root', default='', type=str, help='Root to save the datasets')
parser.add_argument('--folds', default=[1,2,3,4,5,6,7,8,9,10], type=int, nargs='+', help='List of folds')
parser.add_argument('--room', default=[5.0, 15.0, 5.0, 15.0, 3.0, 4.0], type=float, nargs='+', help='Room dimensions')
parser.add_argument('--alpha', default=[0.2, 0.8], type=float, nargs='+', help='Alpha range')
parser.add_argument('--c', default=[335.0, 355.0], type=float, nargs='+', help='Speed of sound range')
parser.add_argument('--d', default=[0.01, 0.30], type=float, nargs='+', help='Microphone spacing range')
parser.add_argument('--r', default=[0.0, 22.0, 0.0, 22.0, 0.0, 22.0, 0.0, 22.0], type=float, nargs='+', help='Distance between source and microphones')

args = parser.parse_args()

rir = BIRD(root=args.root, folder_in_archive='Bird', folds=args.folds, room=args.room, alpha=args.alpha, c=args.c, d=args.d, r=args.r)

print(rir._df)
