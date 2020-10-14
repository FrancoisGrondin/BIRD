
import json
import os
import pandas
import torchaudio

from torch.utils.data import Dataset
from torchaudio.datasets.utils import (download_url, extract_archive, walk_files)

URLS = [ {'url': 'https://www.dropbox.com/s/6dvlgpo1eo7me4v/fold01.zip?dl=0', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/37nt7g20sjwkwcr/fold02.zip?dl=0', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/ioba7ied7p4xpmr/fold03.zip?dl=0', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/k5az3t6zis018z4/fold04.zip?dl=0', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/1dzi6o6ktcfebj8/fold05.zip?dl=0', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/ujxw8sq83vnio1l/fold06.zip?dl=0', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/2avtbjf4gadrf31/fold07.zip?dl=0', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/ekte8957bemelrb/fold08.zip?dl=0', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/j6at686fnkfhuf5/fold09.zip?dl=0', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/wspip9kyed1jjmr/fold10.zip?dl=0', 'checksum': None} ]

class BIRD(Dataset):

    def __init__(
        self, 
        root=None, 
        folder_in_archive='BIRD',
        ext_audio='.flac', 
        room = [5.0, 15.0, 5.0, 15.0, 3.0, 4.0], 
        alpha = [0.2, 0.8],
        c = [335.0, 355.0],
        d = [0.01, 0.30],
        folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ):

        self._path = os.path.join(root, folder_in_archive)
        self._ext_audio = ext_audio
        self._df = None

        eps = 1E-2

        if not os.path.isdir(self._path):
            os.mkdir(self._path)

        for fold in folds:

            directory = os.path.join(self._path, 'fold%02u' % fold)
            archive = os.path.join(self._path, 'fold%02u.zip' % fold)

            if not os.path.isdir(directory):
                if not os.path.isfile(archive):
                    print('Downloading fold %u...' % fold)
                    download_url(URLS[fold]['url'], self._path)
                print('Extracting fold %u...' % fold)
                extract_archive(archive)

            csv_file = os.path.join(directory, 'fold%02u.csv' % fold)

            df = pandas.read_csv(csv_file)
            df.insert(0, 'fold', fold)

            dist = ((df['m1x'] - df['m2x']) ** 2 + (df['m1y'] - df['m2y']) ** 2 + (df['m1z'] - df['m2z']) ** 2) ** 0.5

            filter_room = (df['Lx'] >= room[0]) & (df['Lx'] <= room[1]) & (df['Ly'] >= room[2]) & (df['Ly'] <= room[3]) & (df['Lz'] >= room[4]) & (df['Lz'] <= room[5])
            filter_c = (df['c'] >= c[0]) & (df['c'] <= c[1])
            filter_alpha = (df['alpha'] >= alpha[0]) & (df['alpha'] <= alpha[1])
            filter_d = (dist >= (d[0]-eps)) & (dist <= (d[1]+eps))

            filter_all = filter_room & filter_c & filter_beta & filter_d

            self._df = pandas.concat([self._df, df[filter_all]])

    def __len__(self):

        return len(self._df)

    def __getitem__(self, idx):

        item = self._df.iloc[idx]

        key = item['id']
        fold = item['fold']
        path = os.path.join(self._path, 'fold%02u' % fold, key[0], key[1], key + self._ext_audio)

        x, _ = torchaudio.load(path)

        meta = {}
        meta['L'] = [item['Lx'], item['Ly'], item['Lz']]
        meta['alpha'] = item['alpha']
        meta['c'] = item['c']
        meta['mics'] = [[item['m1x'], item['m1y'], item['m1z']], [item['m2x'], item['m2y'], item['m2z']]]
        meta['srcs'] = [[item['s1x'], item['s1y'], item['s1z']], [item['s2x'], item['s2y'], item['s2z']], [item['s3x'], item['s3y'], item['s3z']], [item['s4x'], item['s4y'], item['s4z']]]

        return x, meta
