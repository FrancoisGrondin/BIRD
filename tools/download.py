import argparse
import os,sys,inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from bird import BIRD

parser = argparse.ArgumentParser()

parser.add_argument('--root', default='', type=str, help='Root to save the datasets')
parser.add_argument('--folds', default=[1,2,3,4,5,6,7,8,9,10], type=int, nargs='+', help='List of folds')

args = parser.parse_args()

rir = BIRD(root=args.root, folder_in_archive='Bird', folds=args.folds)
