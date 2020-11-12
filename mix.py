import numpy as np
import pandas as pd
import random as rnd
import torch

class MIX:

    # This dataset uses a room impulse response and a speech datasets, to create augmented
    # data based on different mixing parameters.
    #
    # rir                   Room impulse response dataset (e.g. BIRD).
    # speech                Speech dataset (e.g. LibriSpeech).
    # snr                   SNR (dB): [SNR_min, SNR_max].
    # volume                Volume (linear): [volume_min, volume_max].
    # count                 Number of active sources: [count_min, count_max] (inclusive).
    # duration              Segment duration (in samples).
    # samples_count         Number of samples.

    def __init__(self, rir, speech, snr=[-5,+5], volume=[0.01,0.99], count=[1,4], duration=16000, samples_count=10000):

        self.rir = rir
        self.speech = speech

        np.random.seed(0)

        self.rirs = np.random.randint(0, len(rir), samples_count)
        self.speeches = np.random.randint(0, len(speech), (samples_count,4))
        self.counts = np.random.randint(count[0], count[1]+1, samples_count)
        self.snrs = np.random.uniform(snr[0], snr[1], (samples_count,4))
        self.shifts = np.random.uniform(0.0, 1.0, (samples_count,4))
        self.volumes = np.random.uniform(volume[0], volume[1], samples_count)
        self.duration = duration

    # Return the number of samples.

    def __len__(self):

        return len(self.rirs)

    # Return the mixed segments. The segments are of dimensions (duration x 2).
    # It also returns the meta data for the rir, the number of active sources
    # and the segments for each source and microphone (number of sources x duration x 2).

    def __getitem__(self, idx):

        hs, meta = self.rir[self.rirs[idx]]
        count = self.counts[idx]

        ys = np.zeros((self.duration,2), dtype=np.float32)

        xs = np.zeros((count,self.duration,2), dtype=np.float32)

        # For each active source

        for i in range(0, count):

            h1 = hs[i*2+0,:].numpy()
            h2 = hs[i*2+1,:].numpy()

            x = np.squeeze(self.speech[self.speeches[idx,i]][0].numpy());

            # Perform convolution between RIR and source

            y1 = np.convolve(x, h1)
            y2 = np.convolve(x, h2)

            # Shift randomly in time

            y1 = np.roll(y1, int(self.shifts[idx,i] * y1.shape[0]))
            y1 = y1[0:self.duration]
            y2 = np.roll(y2, int(self.shifts[idx,i] * y2.shape[0]))
            y2 = y2[0:self.duration]

            # Normalize power

            E1 = np.sum(y1 ** 2)
            E2 = np.sum(y2 ** 2)
            E = 0.5 * (E1 + E2)

            y1 /= (E ** 0.5 + 1E-10)
            y2 /= (E ** 0.5 + 1E-10)

            # Compute gain and apply

            g = 10 ** (self.snrs[idx,i] / 10.0)

            xs[i,:,0] = g * y1
            xs[i,:,1] = g * y2

            ys[:,0] += g * y1
            ys[:,1] += g * y2

        # Apply volume

        v = self.volumes[idx] / (np.max(np.abs(ys)) + 1e-10)

        xs *= v
        ys *= v

        return ys, meta, count, xs

