import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec
from keras.models import load_model

G = load_model("models/generator_10000.h5")
z = np.random.uniform(-1, 1, size=(100, 100))
samples = G.predict(z)

num_rows = 10
num_cols = 10

fig = plt.figure()
gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.0)

ax = [plt.subplot(gs[i]) for i in range(num_rows * num_cols)]
gs.update(hspace=0)
#gs.tight_layout(fig, h_pad=0,w_pad=0)

for i, im in enumerate(samples):
    ax[i].imshow(np.squeeze(im), cmap='gray')
    ax[i].axis('off')

plt.show()
