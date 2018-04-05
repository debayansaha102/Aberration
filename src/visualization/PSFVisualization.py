import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class PSFVisualization(object):

    def __init__(self, file=""):
        if file:
            self.psf = tifffile.imread(file)
        self.X = np.arange(0, self.psf.shape[0], 1);
        self.Y = np.arange(0,self.psf.shape[1], 1);


    def _3D(self):
        fig = plt.figure();
        ax = fig.gca(projection='3d');
        X, Y = np.meshgrid(self.X, self.Y)
        surf = ax.plot_surface(X, Y, self.psf, cmap=cm.coolwarm)
        fig.colorbar(surf)

    def _2D(self):
        plt.imshow(self.psf)


PSFVisualization('/Users/dsaha/Desktop/ExperimentData/BScope/Flat1-Smooth.tif')._3D()
plt.show()

