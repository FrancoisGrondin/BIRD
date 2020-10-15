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

We can visualize a sample from the dataset. Suppose we want to look at sample 4, we would use the following commands:

### Show the room configuration

```
python3 visualize.py --view room --item 4
```

![Room configuration](/room.png)

### Show the room impulse responses

```
python3 visualize.py --view rir --item 4
```

![Room Impulse Responses](/rir.png)

### Display the META data

```
python3 visualize.py --view meta --item 4
```

```
{'L': [14.83, 11.49, 3.01], 'alpha': 0.36, 'c': 350.5, 'mics': [[14.140999999999998, 2.9339999999999997, 1.895], [14.224, 3.01, 2.161]], 'srcs': [[0.8109999999999999, 5.702000000000001, 1.547], [6.6579999999999995, 4.0, 2.582], [5.34, 9.433, 1.775], [12.164000000000001, 8.109, 2.161]]}
```