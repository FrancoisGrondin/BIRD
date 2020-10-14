
import argparse
from bird import BIRD

import numpy as np
import matplotlib.pyplot as plt
import progressbar

parser = argparse.ArgumentParser()

parser.add_argument('--root', default='', type=str, help='Root to save the datasets')

args = parser.parse_args()

rir = BIRD(root=args.root, folder_in_archive='Bird', folds=[1])

rt60s = np.zeros(len(rir), dtype=np.float32)

i = 0
for h, meta in progressbar.progressbar(rir):

	rt60 = BIRD.getRT60(meta)
	rt60s[i] = rt60
	i += 1

counts, bins = plt.hist(rt60s, bins=np.linspace(0.0,2.0,201))
plt.show()

meanbins = 0.5 * (bins[1:] + bins[:-1])

print(counts)
print(meanbins)