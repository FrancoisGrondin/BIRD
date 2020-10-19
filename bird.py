
# Author: Francois Grondin
# Date: October 19, 2020
# Affiliation: Universite de Sherbrooke
# Contact: francois.grondin2@usherbrooke.ca

import json
import os
import pandas
import torchaudio

from torch.utils.data import Dataset
from torchaudio.datasets.utils import (download_url, extract_archive, walk_files)

URLS = [ {},
         {'url': 'https://www.dropbox.com/s/6dvlgpo1eo7me4v/fold01.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/37nt7g20sjwkwcr/fold02.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/ioba7ied7p4xpmr/fold03.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/k5az3t6zis018z4/fold04.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/1dzi6o6ktcfebj8/fold05.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/ujxw8sq83vnio1l/fold06.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/2avtbjf4gadrf31/fold07.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/ekte8957bemelrb/fold08.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/j6at686fnkfhuf5/fold09.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/wspip9kyed1jjmr/fold10.zip?dl=1', 'checksum': None} ]

class BIRD(Dataset):

    # Initialize the BIRD dataset. It is possible to provide a custom range for the simulation 
    # parameters and load a subset of the samples.
    #
    # root                  String that points to the root folder that contains the dataset.
    # folder_in_archive     This is the name of the folder created that will store the dataset.
    # ext_audio             Audio extension for the files containing the RIRs.
    # room                  Room dimension: [L_x,min, L_x,max, L_y,min, L_y,max, L_z,min, L_z,max].
    # alpha                 Absorption coefficient: [alpha_min, alpha_max].
    # c                     Speed of sound: [c_min, c_max].
    # d                     Microphone spacing: [d_min, d_max].
    # folds                 Folds to load.

    def __init__(
        self, 
        root, 
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

        # Create directory if needed

        if not os.path.isdir(self._path):
            os.mkdir(self._path)

        # Load each fold

        for fold in folds:

            directory = os.path.join(self._path, 'fold%02u' % fold)
            archive = os.path.join(self._path, 'fold%02u.zip' % fold)

            # Download and extract fold

            if not os.path.isdir(directory):
                if not os.path.isfile(archive):
                    print('Downloading fold %u...' % fold)
                    download_url(URLS[fold]['url'], self._path)
                print('Extracting fold %u...' % fold)
                extract_archive(archive)
                os.remove(archive)

            # Load meta data in CSV

            csv_file = os.path.join(directory, 'fold%02u.csv' % fold)

            # Create dataframe and add the fold index

            df = pandas.read_csv(csv_file)
            df.insert(0, 'fold', fold)

            # Compute distance between both mics

            dist = ((df['m1x'] - df['m2x']) ** 2 + (df['m1y'] - df['m2y']) ** 2 + (df['m1z'] - df['m2z']) ** 2) ** 0.5

            # Select subset given the intervals provided by the user

            filter_room = (df['Lx'] >= room[0]) & (df['Lx'] <= room[1]) & (df['Ly'] >= room[2]) & (df['Ly'] <= room[3]) & (df['Lz'] >= room[4]) & (df['Lz'] <= room[5])
            filter_c = (df['c'] >= c[0]) & (df['c'] <= c[1])
            filter_alpha = (df['alpha'] >= alpha[0]) & (df['alpha'] <= alpha[1])
            filter_d = (dist >= (d[0]-eps)) & (dist <= (d[1]+eps))
            filter_all = filter_room & filter_c & filter_alpha & filter_d

            # Append to dataframe

            self._df = pandas.concat([self._df, df[filter_all]])

    # Return the number of samples.

    def __len__(self):

        return len(self._df)

    # Return the item at index idx. This returns the RIRs in a tensor 8x16000, and the
    # meta data in a dict.

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

    # Compute the reverberation time RT60 from the meta data.

    @staticmethod
    def getRT60(meta):

        c = meta['c']
        alpha = meta['alpha']
        Lx = meta['L'][0]
        Ly = meta['L'][1]
        Lz = meta['L'][2]

        rt60 = (27.6310 / (alpha * c)) * (Lx*Ly*Lz/(Lx*Ly + Ly*Lz + Lz*Lx))

        return rt60

    # Compute the time difference of arrival (TDOA) for each source from the meta data.

    @staticmethod
    def getTDOA(meta):

        eps = 1E-10

        fs = 16000.0
        c = meta['c']

        m1x = meta['mics'][0][0]
        m1y = meta['mics'][0][1]
        m1z = meta['mics'][0][2]

        m2x = meta['mics'][1][0]
        m2y = meta['mics'][1][1]
        m2z = meta['mics'][1][2]

        dmx = m1x - m2x
        dmy = m1y - m2y
        dmz = m1z - m2z

        mmx = 0.5 * (m1x + m2x)
        mmy = 0.5 * (m1y + m2y)
        mmz = 0.5 * (m1z + m2z)

        fx = (fs/c) * dmx
        fy = (fs/c) * dmy
        fz = (fs/c) * dmz

        I = len(meta['srcs'])

        tdoas = []

        for i in range(0,I):

            six = meta['srcs'][i][0]
            siy = meta['srcs'][i][1]
            siz = meta['srcs'][i][2]

            diffx = six - mmx
            diffy = siy - mmy
            diffz = siz - mmz

            norm = (diffx ** 2 + diffy ** 2 + diffz ** 2) ** 0.5

            nx = diffx / (norm + eps)
            ny = diffy / (norm + eps)
            nz = diffz / (norm + eps)

            tau = fx * nx + fy * ny + fz * nz

            tdoas.append(tau)

        return tdoas