# BIRD
Big Impulse Response Dataset

BIRD is an open dataset that consists of 100,000 multichannel room impulse responses generated using the image method.
We provide some Python code that shows how to download and use this dataset to perform online data augmentation.

## Download the dataset

Here's how to quickly install and use the dataset:

1. Create and activate a virtual environment (optional):

```
python3 -m venv bird
source bird/bin/activate
```

2. Install required libraries:

```
pip3 install -r requirements.txt
```

3. Download the complete dataset and store it on your hardware (11.3 GB):

```
python3 download.py
```