import argparse

try:
  #Case GUI
  import matplotlib
  import matplotlib.pyplot as plt
  import matplotlib.colors as mcolors
except Exception  as e:
  #Case no GUI
  print(e)
  print("continuing without graphics...")

import numpy as np
import pickle as pkl
import copy
sanitycheck = False

l1_norm = lambda X, axis : X.sum(axis = axis)
l2_norm = lambda X, axis : np.sqrt(np.power(X,2).sum(axis = 1))

gen_mu  = lambda X, K : np.random.uniform(
                                   l2_norm(X, 0).min(), 
                                   l2_norm(X, 0).max(), 
                                   size = (K, *X.shape))



def xfetch(X, els):
  """A tad more clean form of retrieval for readability"""
  return np.array(X[els])

def vis(X, mu, classes, **kwargs):
  """
  Visual scatter handler, kwargs include
  size_k : size of centroid marker, 
  size : size of data points,
  iter : iteration corresponding to frame
  err  : iteration loss

  """

  if len(mu) > 10:
    colors = list(mcolors.CSS4_COLORS.values())[35:]
  else:
    colors = list(mcolors.TABLEAU_COLORS.values())
  
  for i in range(len(mu)):
    members = xfetch(X, classes[i]) 
    if "iter" in kwargs.keys():
      plt.title("Iteration %d" % (kwargs["iter"] + 1))

    if "err" in kwargs.keys():
      plt.annotate("objective: {:20.16f}".format(kwargs["err"]),
                    xycoords = "figure pixels", 
                    xy = (5,5))

    plt.scatter(members[:,0], members[:,1],
                color= colors[i],
                marker = "o", 
                s = [kwargs["size"]] * len(members))


    plt.scatter(mu[i,0], mu[i,1], 
                color = colors[i],
                marker = "P",
                s = kwargs["size_k"])

  plt.show()


def kmeans(K, X, visuals: bool = False, all_frames: bool = False,
          size_k : float = 0.5, size : float = 0.1, verbose: bool = False):
  #all frames copies data from each iteration and stores 
  #size_k and size are visual parameters, which adjust render size of centroid and points

  mu = np.random.uniform(X[:,].min(), X[:,].max(), size=(K, *X[0].shape)) # centroid of class k
  frames = []
  
  #Note, frames will be empty unless all frames is requested
  if visuals and not all_frames:
    print("entering visuals not all frames: locals are " + str(locals()))
    #quikly assign classes for visual purposes
    group = [[] for k in range(K)]
    
    for i in range(len(X)):
      eucSqDists = np.power(X[i] - mu, 2).sum(axis = 1)
      mn_idx = eucSqDists.argmin(axis = 0); 
      group[mn_idx].append(i) #assignment to grouping k
    classes = list(map(np.array, group))

    vis(X, mu, classes, size_k = size_k, size = size)


  elif all_frames:
    prev_err = 0
    err = 1
    
    #mimic stopping conditions outside EM loop call
    while err != prev_err:
      prev_err = err
      classes, mu, err = EM(K,X,mu,iter_cap = 1, verbose = verbose)
      frames.append((classes.copy(), mu.copy(),copy.deepcopy(err)))

      if verbose: print(err)
    if visuals:
      for i, frame in enumerate(frames):
        vis(X, frame[1], frame[0], size_k = size_k, size = size, err = err, iter = i)

  else:
    classes, mu, err = EM(K, X, mu, verbose = verbose)
    print(err)
  
  if visuals:
    vis(X, mu, classes, size_k = size_k, size = size)
  return classes, mu, err, frames

def optimal_clustering(K, X, iters: int, all_frames: bool = False, verbose: bool = False):
  outcomes = []
  for r in range(iters):
    try:
      outcomes.append(kmeans(K,X, all_frames))
    except :
      r -= 1 # account for odd div by zero error , theres something going wrong with class assignment here
  
  min_loss_idx = np.array([outcome[2] for outcome in outcomes])

  if verbose:
    print(outcomes[min_loss_idx.argmin()])

  else:
    print("min loss: " , outcomes[min_loss_idx.argmin()][2])

  return outcomes[min_loss_idx.argmin()]

def EM(K, X, mu, iter_cap = 9999, classes = [], verbose: bool = False):
  #K is number of centroids
  #X is dataset
  #mu is centroid matrix
    
  #iter_cap is number of iterations before breaking loop (useful in per update anim, etc)
  #classes is assignments to different cluster groups, none by default since that is what is being discovered, but useful if re-entering
  global sanitycheck # i.e. are we in debug mode?

  X_len   = X.shape[0]
  mu_prev = np.zeros((2,2)) #declare for loop condition
  
  group      = classes
  group_prev = None
  
  cnt       = 0
  loss      = 0
  prev_loss = 1


  while loss != prev_loss:
    prev_loss = loss
    loss = 0
    group = [[] for k in range(K)]
    
    #expectation
    for i in range(X_len):
      eucSqDists = np.power(X[i] - mu, 2).sum(axis = 1)
      mn_idx = eucSqDists.argmin(axis = 0); 
      if sanitycheck: 
        print(mn_idx)
        print(eucSqDists[mn_idx], mn_idx)
      try:
        group[mn_idx].append(i) #assignment to grouping k
      

      except Exception as e:
        print(e)
    
    mu_prev = mu.copy()
    group_prev = copy.copy(group)

    #maximization
    group_npy = list(map(np.array, group))
    for k in range(K):
      if len(group_npy[k]) == 0: continue
      mu[k] = 1/len(group_npy[k]) * X[group_npy[k]].sum(axis = 0)
      l = np.power(X[group_npy[k]] - mu[k], 2).sum(axis = 1)
      loss += l.sum()

    if verbose:
      print(f"error for iter {cnt} is {loss}")
    
    cnt+=1
    if cnt == iter_cap:
      break
  
  return group_npy, mu, loss # assignments and centroids

def loaddata(path):
  data = []
  
  colors = ["b", "g", "r", "c", "y", "p"]
  with open(path, "r") as f:
    for line in f.readlines() :
      data.append(
          list(filter(lambda x: len(x) > 0, line.strip().split(" "))),
        )
  X = np.array(data, dtype = np.float32)
  return X
  
gen_random_colors = lambda i : np.random.uniform(0,1, (i,3))

def render_assignments(classes, centroids, img_data, verbose : bool = False):
    """Takes in pixels for clustering class"""
    colors = gen_random_colors(len(centroids))
    pixel_class = np.zeros_like(img_data).reshape(-1,3).astype(np.float32)
    scalar = 255 // len(centroids)
    for i in range(len(centroids)):
      try:
        members = xfetch(img_data, classes[i])
        pixel_class[classes[i]] = colors[i]
        if verbose: print(pixel_class[classes[i]]," \n\n", colors[i])
      except Exception as e:
        print(e)
        print(i, classes[i])
    return pixel_class


def proc_img__cluster(K, np_img, verbose: bool = False):
  origshp = np_img.shape
  if len(np_img.shape) == 3:
    np_img = np_img.reshape(-1,3)
  classes, centroids, err, frames = kmeans(K, np_img, verbose = verbose)
  
  pixels      = render_assignments(classes, centroids, np_img, verbose = verbose) #pixel class assigned in here
  pixel_class = pixels.reshape(*origshp)
  plt.imshow(pixel_class)

def xfetch(X, els):
  return np.array(X[els])



if __name__ == "__main__":
  
  from part1 import load_img

  parser = argparse.ArgumentParser()
  parser.add_argument("src", type = str, help = "enter relative or asolute path for loading datafile")
  parser.add_argument("-s","--size", type = float, help = "enter size of plotting for visuals, per point, default = 0.4", default = 0.4)
  parser.add_argument("-k", type = int, help = "number of centroids, default is 5", default = 5)
  parser.add_argument("-v","--verbose", action = "store_true", help = "verbosity displays far more runtime data")
  parser.add_argument("--size-k", type = float, help = "enter size of plotting centroid for visuals, default = 0.4", default = 1.4)
  parser.add_argument("--iters", type = int, help = "number of iterations of reseeding for compare towards optimal clutsering, default = 1", default = 1)
  parser.add_argument("--all-frames",action = "store_true", help = "store all frames, i.e. classification and centroid loc per iteration")
  parser.add_argument("--visuals",action = "store_true", help = "set this flag for visuals, currently only for non 3channel data")
  
  args = parser.parse_args()

  if ".jpg" in args.src  : 
    X = load_img(args.src); args.iters = None
    proc_img__cluster(args.k ,X, verbose = args.verbose)
    plt.show()

  else:

    X = loaddata(args.src)
    try:
      if args.iters != 1:
        classes, centroids, err, frames = optimal_clustering(args.k, X, args.iters) #returns optimal clustering
      elif args.all_frames and args.visuals:
        classes, centroids, err, frames = kmeans(args.k, X, size_k = args.size_k, size= args.size, all_frames = args.all_frames)
        for frame in frames:
          vis(X, frame[1], frame[0], size_k = args.size_k, size = args.size, err = frame[2])
      else:
        classes, centroids, err, frames = kmeans(args.k, X, visuals = args.visuals, size_k = args.size_k, size= args.size, all_frames = args.all_frames)
    except IndexError as ie:
      raise IndexError("Failure in centroid assignment due to algorithmic instability (try selecting a lower k value)")

  """    plt.scatter(centroids[i,0], centroids[i,1], color = colors[i], label = "+", s = [args.s])
    members = xfetch(X, classes[i])
    plt.scatter(members[:,0], members[:,1], color =colors[i], label = "o", s = [args.s] * members.shape[0])"""
