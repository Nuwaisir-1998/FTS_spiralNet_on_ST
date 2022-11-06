import numpy as np
import pandas as pd

v = np.load('/content/drive/MyDrive/Nuwaisir/FTS/spiralnet/checkpoints_3/SliceGAN/dlpfc_test/result_slice/metrics.npz')
print(v['PSNR'])