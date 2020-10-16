# BIRD: Big Impulse Response Dataset

BIRD is an open dataset that consists of 100,000 multichannel room impulse responses generated using the image method.
This makes it the **largest multichannel open dataset currently available**.
We provide some Python code that shows how to download and use this dataset to perform online data augmentation.
The code is compatible with the PyTorch dataset class, which eases integration in existing deep learning projects based on this framework.

## Paper

If you use BIRD in your research, please consider citing the [official paper](http://arxiv.org):

```
@inproceedings{grondin2021bird,
  title={BIRD: Big Impulse Response Dataset},
  author={Grondin, Fran{\c{c}}ois and Lauzon, Jean-Samuel and Michaud, Simon and Ravanelli, Mirco and Michaud, Fran{\c}ois},
  booktitle={Arxiv link},
  year={2021}
}
```

## Download the dataset

Here's how to quickly install and use the dataset:

1. Install required libraries:

```
pip3 install -r requirements.txt
```

2. Download the complete dataset and store it on your hardware (11.3 GB):

```
python3 tools/download.py
```

## Visualize the dataset

We can visualize a sample from the dataset. Suppose we want to look at sample 4, we would use the following commands:

### Show the room configuration

Launch the following script:

```
python3 tools/visualize.py --view room --item 4
```

Which returns the following plot:

![Room configuration](/images/room.png)

### Show the room impulse responses

Launch the following script:

```
python3 tools/visualize.py --view rir --item 4
```

Which returns the following plot:

![Room Impulse Responses](/images/rir.png)

### Display the META data

Launch the following script:

```
python3 tools/visualize.py --view meta --item 4
```

Which prints the following string in the terminal:

```
{
    'L': [14.83, 11.49, 3.01], 
    'alpha': 0.36,
    'c': 350.5,
    'mics': [[14.141, 2.934, 1.895], 
             [14.224, 3.010, 2.161]],
    'srcs': [[0.811, 5.702, 1.547], 
             [6.658, 4.000, 2.582], 
             [5.340, 9.433, 1.775], 
             [12.164, 8.109, 2.161]]
}
```

## Examples

Here are a few examples that show how BIRD can be used to perform data augmentation.

### Sound Source Localization

Launch the following script:

```
python3 examples/sound_source_localization.py
```

Which outputs the following spectrograms:

![SSL Spectrogram](/images/ssl_spex.png)

And generates the following TDOAs:

```
tensor([-2.3261,  2.0721], dtype=torch.float64)
```

### Reverberation Time Estimation

Launch the following script:

```
python3 examples/reverberation_time_estimation.py
```

Which outputs the following spectrograms:

![RT60 Spectrogram](/images/rt60_spex.png)

And generates the following RT60:

```
0.416
```

### Counting Speech Sources

Launch the following script:

```
python3 examples/counting_speech_sources.py
```

Which outputs the following spectrograms:

![CNT Spectrogram](/images/cnt_spex.png)

And generates the following source count:

```
3
```

### Ideal Ratio Mask Estimation

Launch the following script:

```
python3 examples/ideal_ratio_mask_estimation.py
```

Which outputs the following spectrograms:

![IRM Spectrogram](/images/irm_spex.png)

And the following ideal ratio mask:

![IRM Mask](/images/irm_mask.png)