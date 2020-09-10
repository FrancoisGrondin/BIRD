
import json
import os
import pandas
import torchaudio

from torch.utils.data import Dataset
from torchaudio.datasets.utils import (download_url, extract_archive, walk_files)

URLS = [ {'fold': 0,  'url': 'https://www.dropbox.com/s/guxqijuyspgevu0/fold00.zip', 'checksum': None},
         {'fold': 1,  'url': 'https://www.dropbox.com/s/o9wm1p1lf542knx/fold01.zip', 'checksum': None},
         {'fold': 2,  'url': 'https://www.dropbox.com/s/kfijdgbxaasutge/fold02.zip', 'checksum': None},
         {'fold': 3,  'url': 'https://www.dropbox.com/s/hmozp7u9cxwpz2a/fold03.zip', 'checksum': None},
         {'fold': 4,  'url': 'https://www.dropbox.com/s/vnnm4omswlhzakk/fold04.zip', 'checksum': None},
         {'fold': 5,  'url': 'https://www.dropbox.com/s/yufpa38tl1lzs8t/fold05.zip', 'checksum': None},
         {'fold': 6,  'url': 'https://www.dropbox.com/s/0xkyrub3m2udy5y/fold06.zip', 'checksum': None},
         {'fold': 7,  'url': 'https://www.dropbox.com/s/qlsgj1cz9p2jlb9/fold07.zip', 'checksum': None},
         {'fold': 8,  'url': 'https://www.dropbox.com/s/dwl836vyarmobv6/fold08.zip', 'checksum': None},
         {'fold': 9,  'url': 'https://www.dropbox.com/s/5ut4whnw3c5bnqw/fold09.zip', 'checksum': None},
         {'fold': 10, 'url': 'https://www.dropbox.com/s/g51x0f6hjpu9f4i/fold10.zip', 'checksum': None},
         {'fold': 11, 'url': 'https://www.dropbox.com/s/x023092a9zujruh/fold11.zip', 'checksum': None},
         {'fold': 12, 'url': 'https://www.dropbox.com/s/fs4izb5ncohyng8/fold12.zip', 'checksum': None},
         {'fold': 13, 'url': 'https://www.dropbox.com/s/o99uobxebz6a6p9/fold13.zip', 'checksum': None},
         {'fold': 14, 'url': 'https://www.dropbox.com/s/u6cxp2gezd4t0se/fold14.zip', 'checksum': None},
         {'fold': 15, 'url': 'https://www.dropbox.com/s/p8igu18hs0o504c/fold15.zip', 'checksum': None},
         {'fold': 16, 'url': 'https://www.dropbox.com/s/tmbbswj7he9md52/fold16.zip', 'checksum': None},
         {'fold': 17, 'url': 'https://www.dropbox.com/s/r6ynn6lkz7zyezr/fold17.zip', 'checksum': None},
         {'fold': 18, 'url': 'https://www.dropbox.com/s/2xec5nrohb4xtvx/fold18.zip', 'checksum': None},
         {'fold': 19, 'url': 'https://www.dropbox.com/s/hbx3x0lj8od56sl/fold19.zip', 'checksum': None} ]

class BIRD(Dataset):

	def __init__(
		self, 
		root=None, 
		folder_in_archive='BIRD',
		ext_audio='.flac', 
		download = False, 
		room = [5.0, 15.0, 5.0, 15.0, 3.0, 4.0], 
		beta = [0.2, 0.8],
		c = [340.0, 355.0],
		d = [0.04, 0.20],
        folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
	):

		self._path = os.path.join(root, folder_in_archive)
		self._ext_audio = ext_audio
		self._csv_file = os.path.join(root, folder_in_archive, 'bird.csv')

		df = pandas.read_csv(self._csv_file)

		dist = ((df['m1x'] - df['m2x']) ** 2 + (df['m1y'] - df['m2y']) ** 2 + (df['m1z'] - df['m2z']) ** 2) ** 0.5

		filter_room = (df['Lx'] >= room[0]) & (df['Lx'] <= room[1]) & (df['Ly'] >= room[2]) & (df['Ly'] <= room[3]) & (df['Lz'] >= room[4]) & (df['Lz'] <= room[5])
		filter_c = (df['c'] >= c[0]) & (df['c'] <= c[1])
		filter_beta = (df['beta'] >= beta[0]) & (df['beta'] <= beta[1])
		filter_d = (dist >= d[0]) & (dist <= d[1])

		filter_all = filter_room & filter_c & filter_beta & filter_d

		self._df = df[filter_all]

	def __len__(self):

		return len(self._df)

	def __getitem__(self, idx):

		item = self._df.iloc[idx]

		key = item['id']
		path = os.path.join(self._path, key[0], key[1], key[2], key+self._ext_audio)
		x, _ = torchaudio.load(path)

		meta = {}
		meta['L'] = [item['Lx'], item['Ly'], item['Lz']]
		meta['beta'] = item['beta']
		meta['c'] = item['c']
		meta['mics'] = [[item['m1x'], item['m1y'], item['m1z']], [item['m2x'], item['m2y'], item['m2z']]]
		meta['srcs'] = [[item['s1x'], item['s1y'], item['s1z']], [item['s2x'], item['s2y'], item['s2z']], [item['s3x'], item['s3y'], item['s3z']], [item['s4x'], item['s4y'], item['s4z']]]

		return x, meta
