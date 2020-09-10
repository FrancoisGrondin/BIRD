
from dataset import BIRD

bird = BIRD(root='/media/fgrondin/Scratch/tmp/', d=[0.049, 0.051], folds=[0, 1])

x, meta = bird[600]

print(x.shape)
print(meta)