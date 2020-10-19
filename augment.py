
# Author: Francois Grondin
# Date: October 19, 2020
# Affiliation: Universite de Sherbrooke
# Contact: francois.grondin2@usherbrooke.ca

from mix import MIX
from bird import BIRD
import torchaudio
import torch

class SSL:

    # This dataset uses a room impulse response and a speech datasets, to create augmented
    # data to perform sound source localization.
    #
    # rir                   Room impulse response dataset (e.g. BIRD).
    # speech                Speech dataset (e.g. LibriSpeech).
    # samples_count         Number of augmented segments to generate.

	def __init__(self, rir, speech, samples_count):

		self.mix = MIX(rir=rir, speech=speech, count=[2,2], duration=80000, samples_count=samples_count)

    # Return the number of samples.

	def __len__(self):

		return len(self.mix)

    # Return the item at index idx. This returns the STFTs of microphones 1 and 2, and the 
    # TDOAs of sources 1 and 2

	def __getitem__(self, idx):

		ys, meta, _, _ = self.mix[idx]

		ys = torch.from_numpy(ys)

		Ys = torchaudio.functional.spectrogram(waveform=torch.transpose(ys, 0, 1),
											   pad=0,
											   window=torch.hann_window(400),
											   n_fft=512, 
											   hop_length=256,
											   win_length=400, 
											   power=None,
											   normalized=False)

		taus = BIRD.getTDOA(meta)
		taus = torch.tensor(taus[0:2])

		return Ys, taus

class RT60:

    # This dataset uses a room impulse response and a speech datasets, to create augmented
    # data to estimate the reverberation time RT60.
    #
    # rir                   Room impulse response dataset (e.g. BIRD).
    # speech                Speech dataset (e.g. LibriSpeech).
    # samples_count         Number of augmented segments to generate.

	def __init__(self, rir, speech, samples_count):

		self.mix = MIX(rir=rir, speech=speech, count=[1,1], duration=80000, samples_count=samples_count)

    # Return the number of samples.

	def __len__(self):

		return len(self.mix)

    # Return the item at index idx. This returns the STFTs of microphones 1 and 2, and the 
    # RT60 value for the room

	def __getitem__(self, idx):

		ys, meta, _, _ = self.mix[idx]

		ys = torch.from_numpy(ys)

		Ys = torchaudio.functional.spectrogram(waveform=torch.transpose(ys, 0, 1),
											   pad=0,
											   window=torch.hann_window(400),
											   n_fft=512, 
											   hop_length=256,
											   win_length=400, 
											   power=None,
											   normalized=False)

		rt60 = BIRD.getRT60(meta)

		return Ys, rt60

class CNT:

    # This dataset uses a room impulse response and a speech datasets, to create augmented
    # data to count the number of sources (between 1 and 4).
    #
    # rir                   Room impulse response dataset (e.g. BIRD)
    # speech                Speech dataset (e.g. LibriSpeech)
    # samples_count         Number of augmented segments to generate

	def __init__(self, rir, speech, samples_count):

		self.mix = MIX(rir=rir, speech=speech, count=[1,4], duration=80000, samples_count=samples_count)

    # Return the number of samples.

	def __len__(self):

		return len(self.mix)

    # Return the item at index idx. This returns the STFTs of microphones 1 and 2, and the 
    # number of active sources (between 1 and 4)

	def __getitem__(self, idx):

		ys, _, count, _ = self.mix[idx]

		ys = torch.from_numpy(ys)

		Ys = torchaudio.functional.spectrogram(waveform=torch.transpose(ys, 0, 1),
											   pad=0,
											   window=torch.hann_window(400),
											   n_fft=512, 
											   hop_length=256,
											   win_length=400, 
											   power=None,
											   normalized=False)

		return Ys, count

class IRM:

    # This dataset uses a room impulse response and a speech datasets, to create augmented
    # data to estimate an ideal ratio mask for the target sound source.
    #
    # rir                   Room impulse response dataset (e.g. BIRD)
    # speech                Speech dataset (e.g. LibriSpeech)
    # samples_count         Number of augmented segments to generate

	def __init__(self, rir, speech, samples_count):

		self.mix = MIX(rir=rir, speech=speech, count=[2,2], duration=80000, samples_count=samples_count)

    # Return the number of samples.

	def __len__(self):

		return len(self.mix)

    # Return the item at index idx. This returns the STFTs of microphones 1 and 2, the 
    # ideal ratio masks for microphones 1 and 2, and the TDOA of source 1 (the target source)

	def __getitem__(self, idx):

		ys, meta, _, xs = self.mix[idx]

		xs = torch.from_numpy(xs)
		ys = torch.from_numpy(ys)

		X1s = torchaudio.functional.spectrogram(waveform=torch.transpose(xs[0,:,:], 0, 1),
											    pad=0,
											    window=torch.hann_window(400),
											    n_fft=512, 
											    hop_length=256,
											    win_length=400, 
											    power=None,
											    normalized=False)

		X2s = torchaudio.functional.spectrogram(waveform=torch.transpose(xs[1,:,:], 0, 1),
											    pad=0,
											    window=torch.hann_window(400),
											    n_fft=512, 
											    hop_length=256,
											    win_length=400, 
											    power=None,
											    normalized=False)

		Ys = torchaudio.functional.spectrogram(waveform=torch.transpose(ys, 0, 1),
											   pad=0,
											   window=torch.hann_window(400),
											   n_fft=512, 
											   hop_length=256,
											   win_length=400, 
											   power=None,
											   normalized=False)

		M1s = (X1s[0,:,:,0] ** 2 + X1s[0,:,:,1] ** 2) / (X1s[0,:,:,0] ** 2 + X1s[0,:,:,1] ** 2 + X2s[0,:,:,0] ** 2 + X2s[0,:,:,1] ** 2)
		M2s = (X1s[1,:,:,0] ** 2 + X1s[1,:,:,1] ** 2) / (X1s[1,:,:,0] ** 2 + X1s[1,:,:,1] ** 2 + X2s[1,:,:,0] ** 2 + X2s[1,:,:,1] ** 2)
		Ms = torch.cat((torch.unsqueeze(M1s, dim=0), torch.unsqueeze(M2s, dim=0)), 0)

        tau = BIRD.getTDOA(meta)[0]

		return Ys, Ms, tau

