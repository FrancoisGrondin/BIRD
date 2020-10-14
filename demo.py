
<<<<<<< HEAD
import argparse
from bird import BIRD
from torchaudio.datasets import LIBRISPEECH

parser = argparse.ArgumentParser()

parser.add_argument('--root', default='', type=str, help='Root to save the datasets')

args = parser.parse_args()

rir = BIRD(root=args.root, folder_in_archive='Bird')
speech = LIBRISPEECH(root=args.root, folder_in_archive='LibriSpeech', url='train-clean-100', download=True)
