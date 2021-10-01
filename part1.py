import argparse

from time import time

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pImg

debug = False #True
GaussianKernel_1 = 1/ 16 * np.array([[1,2,1],
                            [2,4,2],
                            [1,2,1]], dtype = np.float32)

GaussianKernel_2 = 1/273 * np.array([[1,4,7,4,1],
                              [4,16,26,16,4], 
                              [7,26,41,26,7],
                              [4,16,26,16,4],
                              [1,4,7,4,1]], dtype = np.float32)

DoGKernel_1 = np.array([
                        [1,2,1],
                        [0, 0, 0],
                        [-1,-2,-1]], dtype =np.float32)
DoGKernel_2 = np.array([
                        [1,0,-1],
                        [2,0,-2],
                        [1,0,-1]], dtype =np.float32)

if debug: print(DoGKernel_1, DoGKernel_2, GaussianKernel_1, GaussianKernel_2)



conv = lambda img, kernel, len_i, len_j, offset :\
  np.array([[ 
  (img[i - offset : i + offset + 1, 
       j - offset : j + offset + 1]\
       * kernel).sum() \
    for j in range(offset, len_j+offset)]  \
      for i in range(offset, len_i+offset)])

Sobel = lambda X, shp, offset: np.sqrt(
      (conv(X, DoGKernel_1, shp[0], shp[1], offset) ** 2) 
      + (conv(X, DoGKernel_2, shp[0], shp[1], offset) ** 2))


kernels = {
  "GBlur_3x3" : GaussianKernel_1,
  "GBlur_5x5" : GaussianKernel_2,
  "DoG_X" : DoGKernel_2,
  "DoG_Y": DoGKernel_1}


def pad(img : np.ndarray, padding : int):
  """take input image and use its shape info to give initial dimensions
  for padded version. Then, offset the image indexing by padding 
  amount to write contents
  """
  padded = np.zeros((img.shape[0] + padding * 2, img.shape[1] + padding * 2))
  padded[padding:-padding, padding:-padding] = img[:,:]
  return padded

def apply(img : np.ndarray, kernel : np.ndarray, original : (int, int), debug = False):
  """ img is padded image, kernel is (3,3) or (5,5) isometric gaussian discretized,
      original is original image dimensions, scalar is scaling float to normalize convolution"""
  write_to = np.zeros(original, dtype= np.float32)
  offset = kernel.shape[0]//2  
  print(kernel)
  for i in range(offset, write_to.shape[0]):
    for j in range(offset, write_to.shape[1] ):
      sliced = img[i - offset : i + offset + 1, 
                   j - offset : j + offset + 1]
      try:
        write_to[i-offset,j-offset] =   \
          (img[ i - offset : i + offset +1,\
                j - offset : j + offset +1 ]\
                * kernel).sum() #hadamard product and sum, scale by unweight avg
        if debug: 
          intermediate = img[ i - offset : i + offset + 1, j - offset: j + offset + 1] * kernel
          _sum = intermediate.sum()
          print(kernel, intermediate, _sum, sliced)
          input(write_to[i - offset, j-offset])

      except ValueError as ve:
        print(img, img.shape, img[i - offset : i + offset + 1, j - offset : j + offset + 1])
  return write_to
  


def process(img : np.ndarray, padding : int):
  if len(img.shape) == 3:
    img_T = img.transpose(2,0,1)
    bfr = np.zeros(
        (3, img.shape[0] + padding * 2, 
        img.shape[1] + padding * 2), 
      dtype = np.float32)
    for i in range(3):
      bfr[i] = pad(img_T[i], padding)
    return bfr.transpose(1,2,0); #return image to original dimensions


  return pad(img, padding) #not necessary for this return since only rank 2 tensor

def load_img(path):
  img = pImg.open(path)
  np_img = np.array(img, dtype = np.uint8)
  img.close()
  return np_img


def apply(img, kernel, shape):
    return conv(img, kernel, img.shape[0], img.shape[1], kernel.shape[0] // 2);
if __name__ == "__main__":
  parser = argparse.ArgumentParser("load img data for p1")
  parser.add_argument("img", type = str,
          help = "file path (relative or absolute) for PIL load")
  parser.add_argument("--padding", type = int,
          help = "amount of padding, default is 0", default = 0)
  parser.add_argument("--kernel", type = int,
          help = "kernel numbers - 0: gaussian (3,3), 1: gaussian (5,5),  2: DoG, 3: DoG Y, 4: Sobel", default = 0)
  parser.add_argument("--sequence", type = str,
          help = "sequence is a set of integers that will correspond to order of operations of kernels applied. a good one is \"2, 5\", written exactly like this, for 5,5 gaussian followed by the sobel. default will be None, this overrides kernel",default= None)
  
  
  kernels = [GaussianKernel_1, GaussianKernel_2, DoGKernel_1, DoGKernel_2, Sobel]
  args = parser.parse_args()
  np_img = load_img(args.img)
  



  if len(np_img.shape) == 3:
    np_img = np_img.transpose(2,0,1)[0]

  p = process(img = np_img, padding = args.padding)
  time1 = time()

  if args.kernel == 4:
    offset = 1
    out = Sobel(p, np_img.shape, offset)
  elif args.kernel < 4 and args.kernel >= 0:
    kernel = kernels[args.kernel]
    out = conv(p, kernels[args.kernel], np_img.shape[0], np_img.shape[1], kernels[args.kernel].shape[0] // 2);
  
  time2 = time()
  diff1 = time2 - time1
  print("TIME FOR LIST COMPREHENSION", time2 - time1)
  fig = plt.figure(figsize = (15,15))  
  ax = fig.add_subplot(111)
  ax.imshow(out, cmap = "Greys")
  plt.show()
