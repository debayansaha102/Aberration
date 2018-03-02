## Phase Retreival using Deep Net
# coding: utf-8

# In[1]:

from phasediv import PhaseDiv3
from spimagine import volshow
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
import sys
sys.path.append("/Users/dsaha/Python/alpao_calibration")
from calibration import Calibration
get_ipython().magic('matplotlib inline')
import json
from scipy.spatial.distance import cdist
mirror = Calibration()
import datetime
import os
from random import random


# In[2]:

p = PhaseDiv3(dshape = (128,128,128),   # shape of the output stack (Nz,Ny,Nx)
              units = (.1,)*3,          # pixelsize (dz,dy,dx) in micron
              lam = 0.5,                # wavelength in micron
              NA1=0.8,                 # Numerical Aperture of the detection obcejtive
                NA2 = 0,
              n = 1.33                  # refractive index of medium
              )


# In[82]:

#Training for two modes of aberration"zernike(4 and 5)" with 1000 weights from "0,2" 1000 data points
datapoints = 1000
number_of_modes = 1
lowerlimit=0
upperlimit=1
weights = np.zeros((datapoints,number_of_modes))
dataset = np.zeros((datapoints,128*128))


for i in range(0,datapoints):
    for j in range(0,number_of_modes):
        #if j==3 or j==4:
        weights[i,j] = np.random.uniform(lowerlimit,upperlimit)

for i in range(0,datapoints):
    wavefront = 0.0*p.zernike(1)
    for j in range(0,number_of_modes):
        if j==0:
            wavefront = wavefront + weights[i,j]*p.zernike(5)
        else:
            pass
        
        
    psf = np.fft.fftshift(p._psf_incoherent(wavefront))
    focal_psf = psf[p.Nz//2]
    dataset[i,:]= focal_psf.flatten()
weights.reshape((datapoints,number_of_modes));
#print(weights)


# In[83]:

print(dataset.shape)
print(focal_psf.flatten().shape)
print(weights.shape)


# In[85]:

inp = Input(shape=(16384,))

lay = Dense(128, activation="tanh")(inp)

#for _ in range(3):
#    lay = Dense(128, activation="sigmoid")(lay)
    
oup = Dense(number_of_modes)(lay)
#oup = Dense(10)(inp)
m = Model(inp,oup)


# In[86]:

#print(m.summary())
m.compile(Adam(lr=0.0004),'mse')
#m.compile(loss='mean_squared_logarithmic_error',optimizer=Adam(0.0004))
hist = m.fit(dataset,weights,batch_size=10,epochs=200, validation_split=.1)


# In[84]:

plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.gca().set_yscale("log")
plt.gca().set_xscale("log")


# In[93]:

# define the wavefront as combination of some zernikes, e.g. astigmatism (z_5) and spherical (z_11)
plt.figure(figsize=(7,7))
unknown_wavefront = 0.3*p.zernike(5)#+0.1*p.zernike(5)
plt.subplot(1,2,1)
plt.imshow(np.fft.fftshift(unknown_wavefront))
unknown_psf = np.fft.fftshift(p._psf_incoherent(unknown_wavefront))
unknown_focal_psf = unknown_psf[p.Nz//2]
plt.subplot(1,2,2)
plt.imshow(unknown_focal_psf)


# In[94]:

predicted_zern_weights = m.predict(unknown_focal_psf.reshape(1,16384))
print(predicted_zern_weights)


# In[70]:

predicted_wavefront = np.zeros((128,128))
for i in range(1,number_of_modes):
    predicted_wavefront = predicted_wavefront+ predicted_zern_weights[0,i]*p.zernike(i)
    #print(predicted_zern_weights[0,i])
    plt.imshow(np.fft.fftshift(predicted_wavefront))


# In[ ]:



