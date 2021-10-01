import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from time import time

diff_sq = lambda x1, x2,dim: np.power( x1 - x2, 2)
l2norm  = lambda x1, x2, dim: np.sqrt( diff_sq(x1, x2, dim).sum(axis = dim))
arg_min = lambda x, dim: x.argmin(axis = dim)
get_min = lambda x, dim: x[arg_min(x, dim)]

#hopefully faster as a single lined list comprehension
conv = lambda img, kernel, len_i, len_j, offset :\
  np.array([[ (img[i - offset : i + offset + 1, j - offset : j + offset + 1] * kernel).sum() \
    for j in range(len_j)] for i in range(len_i)])

def file_read(img_path: str):
  return cv.imread(img_path, 0)
def classify(img1,img2, thresh: float):
  if isinstance(img1, str):
    img1 = file_read(img1)
  if isinstance(img2, str):
    img2 = file_read(img2)
  
  sift = cv.xfeatures2d.SIFT_create()
  keypoint1, desc1 = sift.detectAndCompute(img1,None)
  keypoint2, desc2 = sift.detectAndCompute(img2,None)
  return (keypoint1, keypoint2), (desc1, desc2)
drawkeys = lambda img, x: cv.drawKeypoints(img, x,outImage=None)




import pickle as pkl
def nnmatch(kps, descs, write: str = None):
  #min_l2
  outcomes = {}

  t1 = time()
  print(len(descs[0]))
  print(len(descs[1]))
  for i, desc in enumerate(descs[0]):
    outcomes[i] = arg_min(l2norm(descs[0][i], descs[1], 1),0)

    print(f"iteration {i}")
    print(outcomes[i])
  t2 = time()
  print("took ", t2 - t1)
  if write:
    with open(write, "wb") as f:
      pkl.dump(pkl.dumps(outcomes),f)
      #pkl.dump(pkl.dumps(kps),f)
      #pkl.dump(pkl.dumps(descs),f)
    
  return outcomes
def datagather():
  """utility for quick terminal interpreter debugging"""
  f1, f2 = file_read("SIFT1_img.jpg"),  file_read("SIFT2_img.jpg")
  return f1,f2
def read_pkl(fil: str):
  with open(fil, "rb") as f:
    out = pkl.loads(pkl.load(f))
  return out

def write_pkl(fil: str, *args):
  with open(fil, "wb") as f:
    for arg in args:
      pkl.dump(pkl.dumps(arg),f)


if __name__ == "__main__":
  import argparse 
  parser = argparse.ArgumentParser()
  """parser.add_argument("--wpkl", type = str, help = "store intermediate results in pickle at location provided (as file location for write), default = None", default = None)
  parser.add_argument("--rpkl", type = str, help = "read intermediate results from pickle at location provided (as file location for read), default = None", default = None)"""
  parser.add_argument("-t","--threshold", type = float, help = "threshold for keypoint similarity. higher is more similar, and exists in a probability metric space, i.e. is bounded below by 0 inclusive, above by 1 inclusive, default = 0.9", default = 0.9)
  args = parser.parse_args()

  f1, f2 = file_read("SIFT1_img.jpg"),  file_read("SIFT2_img.jpg")
  
  outcomes = kps = descs = None
  kps, descs= classify(f1,f2,0.75)
  
  """
  if args["rpkl"]: 
      outcomes = read_pkl(args["rpkl"])
  else:
    if args["wpkl"] is not None:
      outcomes = nnmatch(kps, descs, args["wpkl"])
    else:
      outcomes = nnmatch(kps, descs)
  """
#the above was implemented properly but did not play well with cv2 keypoint matching visuals, so is not included in final runtime
  #img2 = cv.polylines(f2,[np.int32(dst)],True,255,3, cv.LINE_AA)
  matchesMask = None

  cv_brute_force_matcher = cv.BFMatcher()
  matches = cv_brute_force_matcher.knnMatch(descs[0], descs[1], k = 2)
  confident_matches = \
    [[m] for m,n in matches if m.distance < args.threshold * n.distance]


  back = int(0.1*len(matches))
  dists = np.array([m.distance for m,n in matches])
  indices = dists.argsort()[:back][::-1]
  confident_matches = np.array(confident_matches)[indices]

  #draw_params = dict(matchColor = (0,255,0), # draw matches in green color
  img = cv.drawMatchesKnn(f1,kps[0], f2, kps[1], confident_matches, flags = 2, outImg = None)
  # last arg required to handle errorneous autogened python signature (was used to write to image matrix)

  """singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
  matches = []
  print(outcomes)
  for k, v in outcomes.items():
    matches.append([descs[1][v]])
  
  """;
  plt.imshow(img, "Greys"), plt.show()
  plt.imshow(drawkeys(f1,kps[0]))
  plt.show()
