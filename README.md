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