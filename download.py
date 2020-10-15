import argparse

from bird import BIRD

parser = argparse.ArgumentParser()

parser.add_argument('--root', default='', type=str, help='Root to save the datasets')
parser.add_argument('--folds', default=[1,2,3,4,5,6,7,8,9,10], type=int, nargs='+', help='List of folds')

rir = BIRD(root=args.root, folder_in_archive='Bird', folds=args.folds)
