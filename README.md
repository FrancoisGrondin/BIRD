# BIRD: Big Impulse Response Dataset

BIRD is an open dataset that consists of 100,000 multichannel room impulse responses generated using the image method.
We provide some Python code that shows how to download and use this dataset to perform online data augmentation.

## Download the dataset

Here's how to quickly install and use the dataset:

1. Install required libraries:

```
pip3 install -r requirements.txt
```

2. Download the complete dataset and store it on your hardware (11.3 GB):

```
python3 download.py
```

## Visualize the dataset

To visualize a sample from the dataset, use the following commands:

### Show the room configuration

```
python3 visualize.py --view room --item 4
```

![Room configuration](/room.png)

## Show the room impulse responses

```
python3 visualize.py --view rir --item 4
```

![Room Impulse Responses](/rir.png)