# Author: Francois Grondin
# Date: October 19, 2020
# Affiliation: Universite de Sherbrooke
# Contact: francois.grondin2@usherbrooke.ca

import argparse
import os,sys,inspect
import matplotlib.pyplot as plt
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from bird import BIRD
from torchaudio.datasets import LIBRISPEECH
from augment import SSL

parser = argparse.ArgumentParser()

parser.add_argument('--root', default='', type=str, help='Root to save the datasets')
parser.add_argument('--folds_train', default=[1,2,3,4,5,6,7], type=int, nargs='+', help='List of BIRD folds for training')
parser.add_argument('--folds_eval', default=[8,9], type=int, nargs='+', help='List of BIRD folds for validation')
parser.add_argument('--folds_test', default=[10], type=int, nargs='+', help='List of BIRD folds for test')

args = parser.parse_args()

# This holds the RIR dataset for training, validation and testing

folder_in_archive_rir = 'Bird'
rir_train = BIRD(root=args.root, folder_in_archive=folder_in_archive_rir, folds=args.folds_train)
rir_eval = BIRD(root=args.root, folder_in_archive=folder_in_archive_rir, folds=args.folds_eval)
rir_test = BIRD(root=args.root, folder_in_archive=folder_in_archive_rir, folds=args.folds_test)

# This holds the speech dataset for training, validation and testing

folder_in_archive_speech = 'LibriSpeech'
speech_train = LIBRISPEECH(root=args.root, folder_in_archive=folder_in_archive_speech, url='train-clean-100', download=True)
speech_eval = LIBRISPEECH(root=args.root, folder_in_archive=folder_in_archive_speech, url='dev-clean', download=True)
speech_test = LIBRISPEECH(root=args.root, folder_in_archive=folder_in_archive_speech, url='test-clean', download=True)

# We can simply create augmented data with this training dataset

augmented_train = SSL(rir=rir_train, speech=speech_train, samples_count=10000)

Ys, taus = augmented_train[1]

Y1 = torch.squeeze(Ys[0,:,:,:], dim=0)
Y2 = torch.squeeze(Ys[1,:,:,:], dim=0)

print(taus)

plt.subplot(2,1,1)
plt.imshow(np.log(Y1[:,:,0]**2 + Y1[:,:,1]**2 + 1E-10), aspect='auto')
plt.gca().invert_yaxis()
plt.title(r'$10 \log |Y_1|^2$')
plt.xlabel('Frame index')
plt.ylabel('Frequency index')
plt.subplot(2,1,2)
plt.imshow(np.log(Y2[:,:,0]**2 + Y2[:,:,1]**2 + 1E-10), aspect='auto')
plt.gca().invert_yaxis()
plt.title(r'$10 \log |Y_2|^2$')
plt.xlabel('Frame index')
plt.ylabel('Frequency index')
plt.tight_layout()
plt.show()

