
import argparse
from bird import BIRD
from torchaudio.datasets import LIBRISPEECH

import numpy as np
import matplotlib.pyplot as plt
import progressbar

parser = argparse.ArgumentParser()

parser.add_argument('--root', default='', type=str, help='Root to save the datasets')

args = parser.parse_args()

rir = BIRD(root=args.root, folder_in_archive='Bird')
#speech = LIBRISPEECH(root=args.root, folder_in_archive='LibriSpeech', url='train-clean-100', download=True)

rt60s = np.zeros(len(rir), dtype=np.float32)

i = 0
for h, meta in progressbar.progressbar(rir):

	rt60 = BIRD.getRT60(meta)
	rt60s[i] = rt60
	i += 1

rtn = plt.hist(rt60s, bins=np.linspace(0.0,2.0,201))
plt.show()

print(rtn[0])