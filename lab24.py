import numpy as NP
import numpy.linalg as LA
import imageio as IO
import matplotlib.pyplot as PLT
import scipy.misc

F = scipy.misc.face(gray=True)

U,S,VT = LA.svd(F)

s=200

Ftest = U[:,:s].dot(NP.diag(S[:s])).dot(VT[:s])

# Show the compressed image in the plt plot
PLT.imshow(Ftest, cmap='gray')
# save the plot image as a png file
PLT.savefig('compressed_raccoon.png')

def ftest(s):
  ushape = U[:,:s].shape
  vtshape = VT[:s].shape
  sigma = len(S)
  image = F.shape
  
  mem = ushape[0]*ushape[1] + sigma + vtshape[0]*vtshape[1]
  
  perc = mem/(image[0]*image[1])
  
  return perc

for s in [500, 300, 200, 100, 20, 1]:
  print(ftest(s))