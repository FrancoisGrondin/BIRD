
from dataset import BIRD
import matplotlib.pyplot as plt

bird = BIRD(root='/Users/grof2802/Downloads/', folds=[0, 1])

x, meta = bird[0]

print(len(bird))

