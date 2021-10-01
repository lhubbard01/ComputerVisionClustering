#GUI IMPORTS
import mpl_toolkits.mplot3d
import matplotlib
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk


















class Window:
  def __init__(self,**kwargs): # title:str = None, legend: bool = False, d3 : bool = False):
    #print("loadin graphical instance")
    self.params = {}
    
    if "figsize" in kwargs.keys():
      #print("figsize")
      figsize = kwargs["figsize"]
    self.fig = plt.figure(figsize = figsize)
    subplot_kws = None
    if "subplots" in kwargs.keys():
      #print("subplots: ",str(kwargs["subplots"]))
      subplot_kws = kwargs["subplots"]

    self.canvas = self.fig.canvas
    self.canvas_p  = self.fig.canvas.get_parent()
    self.win = self.fig.canvas.get_parent().get_parent()
    self.axes = []

    if subplot_kws:
      for loc in subplot_kws.keys():
        exec("self.axes.append(self.fig.add_subplot(int(loc), **subplot_kws[loc]))", locals(), globals())
      
  
  def show(self):
    plt.show()
if __name__ == "__main__":
  win = Window(figsize = (15,15), subplots = {"121" : {"projection": "3d", "title": "3d proj"}, "122":{"title":"2d proj"}})
  win.show()
