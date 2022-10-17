
# Author: Francois Grondin
# Date: October 19, 2020
# Affiliation: Universite de Sherbrooke
# Contact: francois.grondin2@usherbrooke.ca

import json
import os
import pandas
import torchaudio

from torch.utils.data import Dataset
from torchaudio.datasets.utils import (download_url, extract_archive)

URLS = [ {},
         {'url': 'https://www.dropbox.com/s/g4vliv6k8tkc45h/fold001.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/2fjpby9203n9nfz/fold002.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/6goqj1i2bw0s7ek/fold003.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/x2iw8znhjbbnoab/fold004.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/c2v39f3kbcspcz5/fold005.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/l8cgieevvso7466/fold006.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/n7r3avo6o0igbz3/fold007.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/yl07phjmcr2aqnl/fold008.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/98bc76abtns6mkn/fold009.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/2us6s1n2neacae7/fold010.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/0y6fs4wb0j2r3ca/fold011.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/fqbkrz2uia9hk9s/fold012.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/ioz5vvoa571fpeh/fold013.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/mlnrk2ppfd61xq6/fold014.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/u961w6gzlqhwr43/fold015.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/dg5ykbb4i3gfde0/fold016.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/cbcoz257g7wxsw1/fold017.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/8v745e1xux647iy/fold018.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/zuomkob2qlee01h/fold019.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/4h0quhjg8dapjf0/fold020.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/e2b4jecn54blu52/fold021.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/5jooo8onew29r87/fold022.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/ht1ds46mi8jpqhf/fold023.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/vcgdh6afogjp91g/fold024.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/azqn0zaj9lna34y/fold025.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/v0xbn43xfqlbnvd/fold026.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/b45ncmlg2zluk32/fold027.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/u7o0ysehwsmavnl/fold028.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/otlud756dskupmj/fold029.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/iabgyth2w609dor/fold030.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/zk8qkkxab8saclv/fold031.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/q39ft675zwm8mge/fold032.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/mc85otewbzg2s0g/fold033.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/ukj8xfrk32df0b2/fold034.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/j9lcckvzjb3jszp/fold035.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/ctzkeq53pvs060o/fold036.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/d92gnyoe3c7soej/fold037.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/wg27k0bci2t1lcq/fold038.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/gbmvvcyw4m1aihn/fold039.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/ktp9cgg3zwc3gyy/fold040.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/n8c0m024x7fe5dj/fold041.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/lgb5nsa3v6emnz8/fold042.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/xbgjgw9zhi8j3b2/fold043.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/kfpc8l39yehzgew/fold044.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/fmz8jjyn5ewez2k/fold045.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/8w5q6cm1vt3npdu/fold046.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/th614up6pr10zmy/fold047.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/nv7yjw28jc8zpol/fold048.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/fyb0hz7va16eqh4/fold049.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/5d1u7yaaj5i0gib/fold050.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/eisvrhvfaowivhw/fold051.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/q1nktnfvn5h67yy/fold052.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/9xbfbcku9vrnqlj/fold053.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/h1chwxjn75q6udt/fold054.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/oopu03v2l77j470/fold055.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/cc7nh9ouleggu49/fold056.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/iftsgqrbxfw41zt/fold057.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/vfmveb55agvnhql/fold058.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/noeno23l5ils04x/fold059.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/a2qsd3u6sk798ux/fold060.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/2qfkpmi12u8hdaq/fold061.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/7tpyhya4ymmxkjb/fold062.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/ds5u1msjp1vdz06/fold063.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/rw1i93kldhp5rvx/fold064.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/2u19a0ixtwq9gvy/fold065.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/cskp9xb9iqgeunf/fold066.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/adecv2mri6r8ahp/fold067.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/26sjoukbx2dbmtt/fold068.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/v6tn30pqwjksryb/fold069.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/9vontx0uwod8qt7/fold070.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/s9wzsym7ndr443y/fold071.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/pzsoe0t0evf3gv2/fold072.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/cn2fgpe6bpfz5nc/fold073.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/yoya5or660mnl2w/fold074.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/hpfsfy3oou4uth1/fold075.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/7z1nk0txwjmz0w5/fold076.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/qwztni6yw2cxr9q/fold077.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/kc5x9slfoxyuf1m/fold078.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/krb1d8raxc606tj/fold079.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/158jjapbuzuqid0/fold080.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/sezteqauxeqcuvy/fold081.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/wfy0tw2lrxndiyl/fold082.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/fu6h6sj0gqkk85w/fold083.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/yv18btsx22zornh/fold084.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/sb36pvkr6ijpc8r/fold085.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/dl3j2lm2harobzl/fold086.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/lxubp43ablw1pxw/fold087.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/bthp5jhwze07g5j/fold088.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/lkqcddl91divq9x/fold089.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/c96xd1ue7zkyjas/fold090.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/l392ugi809lleg8/fold091.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/inwrvxd1jfluhni/fold092.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/xbacsn4fcpit94a/fold093.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/5k3y4q9nhsggs9j/fold094.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/mhplxyo4mwfr1l1/fold095.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/3z5nre4eopnrjyl/fold096.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/oy90h8q0do4d6n4/fold097.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/sdjtjkv7ollkn0h/fold098.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/fxs6rkxbqweop4q/fold099.zip?dl=1', 'checksum': None},
         {'url': 'https://www.dropbox.com/s/lag6dyl3515n7ra/fold100.zip?dl=1', 'checksum': None} ]

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
    # folds                 Folds to load (between 1 and 100).

    def __init__(
        self, 
        root, 
        folder_in_archive='BIRD',
        ext_audio='.flac', 
        room = [5.0, 15.0, 5.0, 15.0, 3.0, 4.0], 
        alpha = [0.2, 0.8],
        c = [335.0, 355.0],
        d = [0.01, 0.30],
        r = [0.0, 22.0, 0.0, 22.0, 0.0, 22.0, 0.0, 22.0],
        folds = list(range(0,101)),
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

            directory = os.path.join(self._path, 'fold%03u' % fold)
            archive = os.path.join(self._path, 'fold%03u.zip' % fold)

            # Download and extract fold

            if not os.path.isdir(directory):
                if not os.path.isfile(archive):
                    print('Downloading fold %u...' % fold)
                    download_url(URLS[fold]['url'], self._path)
                print('Extracting fold %u...' % fold)
                extract_archive(archive)
                os.remove(archive)

            # Load meta data in CSV

            csv_file = os.path.join(directory, 'fold%03u.csv' % fold)

            # Create dataframe and add the fold index

            df = pandas.read_csv(csv_file)
            df.insert(0, 'fold', fold)

            # Compute distance between both mics

            dist = ((df['m1x'] - df['m2x']) ** 2 + (df['m1y'] - df['m2y']) ** 2 + (df['m1z'] - df['m2z']) ** 2) ** 0.5

            # Compute center of mass of mics

            m0x = (df['m1x'] + df['m2x']) / 2.0
            m0y = (df['m1y'] + df['m2y']) / 2.0
            m0z = (df['m1z'] + df['m2z']) / 2.0
            
            # Distance between mics and srcs

            r1 = ((df['s1x'] - m0x) ** 2 + (df['s1y'] - m0y) ** 2 + (df['s1z'] - m0z) ** 2) ** 0.5
            r2 = ((df['s2x'] - m0x) ** 2 + (df['s2y'] - m0y) ** 2 + (df['s2z'] - m0z) ** 2) ** 0.5
            r3 = ((df['s3x'] - m0x) ** 2 + (df['s3y'] - m0y) ** 2 + (df['s3z'] - m0z) ** 2) ** 0.5
            r4 = ((df['s4x'] - m0x) ** 2 + (df['s4y'] - m0y) ** 2 + (df['s4z'] - m0z) ** 2) ** 0.5

            # Select subset given the intervals provided by the user

            filter_room = (df['Lx'] >= room[0]) & (df['Lx'] <= room[1]) & (df['Ly'] >= room[2]) & (df['Ly'] <= room[3]) & (df['Lz'] >= room[4]) & (df['Lz'] <= room[5])
            filter_c = (df['c'] >= c[0]) & (df['c'] <= c[1])
            filter_alpha = (df['alpha'] >= alpha[0]) & (df['alpha'] <= alpha[1])
            filter_d = (dist >= (d[0]-eps)) & (dist <= (d[1]+eps))
            filter_r1 = (r1 >= r[0]) & (r1 <= r[1])
            filter_r2 = (r2 >= r[2]) & (r2 <= r[3])
            filter_r3 = (r3 >= r[4]) & (r3 <= r[5])
            filter_r4 = (r4 >= r[6]) & (r4 <= r[7])
            filter_r = filter_r1 & filter_r2 & filter_r3 & filter_r4            
            
            filter_all = filter_room & filter_c & filter_alpha & filter_d & filter_r

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
        path = os.path.join(self._path, 'fold%03u' % fold, key[0], key[1], key + self._ext_audio)

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

    
