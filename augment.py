
from mix import MIX
from bird import BIRD
import torchaudio
import torch

class SSL:

	def __init__(self, rir, speech, samples_count):

		self.mix = MIX(rir=rir, speech=speech, count=[2,2], duration=80000, samples_count=samples_count)

	def __len__(self):

		return len(self.mix)

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

	def __init__(self, rir, speech, samples_count):

		self.mix = MIX(rir=rir, speech=speech, count=[1,1], duration=80000, samples_count=samples_count)

	def __len__(self):

		return len(self.mix)

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

	def __init__(self, rir, speech, samples_count):

		self.mix = MIX(rir=rir, speech=speech, count=[1,4], duration=80000, samples_count=samples_count)

	def __len__(self):

		return len(self.mix)

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

	def __init__(self, rir, speech, samples_count):

		self.mix = MIX(rir=rir, speech=speech, count=[2,2], duration=80000, samples_count=samples_count)

	def __len__(self):

		return len(self.mix)

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
		Ms = M1s * M2s

		return Ys, Ms

