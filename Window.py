#GUI IMPORTS
import mpl_toolkits.mplot3d
import matplotlib
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk

#DATA IMPORTS
import numpy as np



#ASSIGNMENT 1 IMPORTS
from part1 import * 
from part2 import *




from part3 import *












gen_random_colors = lambda i : np.random.uniform(0,1, (i,3))
conn = lambda obj, cb, event: exec("obj.connect(event, cb)", locals(), globals())

class Window:
  def __init__(self,**kwargs): # title:str = None, legend: bool = False, d3 : bool = False):

    print("loadin graphical instance")
    self.params = {}
    self.k = 5
    self.kentry = None 

    self.cluster_3d = False
    choices = Gtk.ListStore(str)
    choices.append(["GAUSS"])
    choices.append(["IMAGE CLUST"])
    choices.append(["SIFT"])
    choices.append(["RAND DATA"])

    self.img_data = self.pixel_class = None
    self.fig = plt.figure(figsize=(15,15))
    self.canvas = self.fig.canvas
    self.canvas_p  = self.fig.canvas.get_parent()
    self.win = self.fig.canvas.get_parent().get_parent()
    self.win.remove(self.canvas_p)
    self.grid = Gtk.Grid()
    self.grid.set_column_spacing(5)
    self.win.add(self.grid) # set the layout to grid, cleans up gui
  
    
    #this retains the properties of the initial canvas widget
    self.grid.attach(self.canvas_p, 1, 1, 4,5,)
    self.grid.get_children()[-1].set_hexpand(True)
    self.grid.get_children()[-1].set_vexpand(True)

    toolbar = self.canvas.manager.toolbar
    self.widgets = Gtk.Grid( margin = 5, vexpand = True)
    self.widgets.show()
    self.grid.attach(self.widgets,5,1,5,1)
    
    self.win.show_all()
    #self.canvas_p.set_orientation(Gtk.Orientation.
    vbox = self.fig.canvas.manager.vbox
    self.params["title"] = kwargs["title"]
    self.params["legend"] = kwargs["legend"]
    self.params["3d" ] = kwargs["d3"]
    self.canvas_p.reorder_child(self.canvas, -1)
    self.ax_00 = self.fig.add_subplot(221, projection = "3d")
    self.ax_00.set_title("3d cluster render")

    self.ax_01 = self.fig.add_subplot(222)
    self.ax_01.set_title("clustering anim")

    self.ax_10 = self.fig.add_subplot(223)
    self.ax_10.set_title("Image display")






    self.ax_11 = self.fig.add_subplot(224)
    self.ax_11.set_title("Image KMeans assignments")
    btn = Gtk.Button("Load")
    
    img_path = Gtk.Entry()
    self.p2 =  Gtk.Entry()

    colors = gen_random_colors(5)
    self.X = X = loaddata("510_cluster_dataset.txt")
    classes, centroids, err, frames = kmeans(5, X)
    for i in range(5): 
      self.ax_01.scatter(centroids[i,0], centroids[i,1], color = colors[i], label = "o", s = [0.75])
      members = xfetch(X, classes[i])
      self.ax_01.scatter(members[:,0], members[:,1], color = colors[i], label = "o", s = [0.75] * members.shape[0])
    #to_matrix_from_vector = lambda el, s1, s2: el // s1, el % s2
    def update_img_process(self, parent):
      iter_loc = self.get_active_iter()
      if iter_loc is not None:
        model = self.get_model()
        
        if model[iter_loc][0] == "SIFT":
          if not parent.p2.get_visible():
            parent.p2.set_visible(True)

        parent.proc = model[iter_loc][0]
      print(parent.proc)

    self.combo = Gtk.ComboBox.new_with_model(choices)
    cell = Gtk.CellRendererText()
    self.combo.pack_start(cell,True )
    self.combo.connect("changed", update_img_process, self)
    self.combo.add_attribute(cell, "text",0)
    self.widgets.add(self.combo)
    









    




    def img_path_loadimg(self, parent):
      path_ = img_path.get_text()
      current = parent.proc
      np_img = load_img(path_)
      parent.img_data = np_img.copy()
      if current == "IMAGE CLUST":
        parent.proc_img__cluster(parent.k,np_img)
      elif current == "GAUSS":
        pass
      elif current == "SIFT":
        img1 = file_read(path_)
        img2 = file_read(parent.p2.get_text())
        classify(img1, img2, thresh = 0.75)
    
    img_path.connect("activate", img_path_loadimg, self)
    img_path.show()
     


    self.widgets.attach_next_to(img_path, self.combo,Gtk.PositionType.BOTTOM,2,1)# False, False, 0)
    self.widgets.attach_next_to(self.p2, img_path,Gtk.PositionType.BOTTOM,3,1)# False, False, 0)

    self.w_prev = img_path
    placehold = Gtk.Grid()
    self.grid.attach(placehold, 1,1,1,1)
    def set_image(self, parent, img_path):
      tmp = img_path.get_text()
      fdialog = Gtk.FileChooserDialog(title = "select image",
                                      parent = parent.win,
                                      action = Gtk.FileChooserAction.OPEN)

      response = fdialog.run()
      if response == Gtk.ResponseType.OK:
          print("success")
      fdialog.destroy()
        
    btn.connect("clicked", set_image, self, img_path)
    btn.show()
    self.widgets.attach_next_to(btn, img_path, Gtk.PositionType.RIGHT,1,1)# True,False, 0)
   
    self.w_prev = btn
    if self.params["legend"]:
      self.fig.legend()
    
    if self.params["title"]:
      try: self.fig.title(title)
      except Exception as e: print(e)
    
    cbtn = Gtk.CheckButton(label = "render 3d cluster")
    cbtn.set_active(False)
    def cbtn_oncl(self, parent):
      parent.cluster_3d = not parent.cluster_3d
    cbtn.connect("clicked", cbtn_oncl, self)
    cbtn.show()
    
    self.widgets.attach_next_to(cbtn, self.combo, Gtk.PositionType.RIGHT,1,1)# False, False, 0)
    


    def clear_ax00(self, parent):
      parent.ax_00.cla()
      parent.fig.canvas.draw()
    clear_ax00_btn = Gtk.Button(label="Clear 3d")
    clear_ax00_btn.show()
    
    self.widgets.attach_next_to(clear_ax00_btn, cbtn, Gtk.PositionType.RIGHT,1,1)
    menubar = Gtk.MenuBar()
    menubar.show()
    self.grid.attach(menubar, 1, 1, 1,5)
    item1 = Gtk.MenuItem.new_with_label("XYZ")
    azimuth_slider = Gtk.Scale.new_with_range(
      Gtk.Orientation(0), min = 0, max = 360, step = 1)
    azimuth_slider.show()

    vbox.pack_start(azimuth_slider, False, False, 0)
    vbox.reorder_child(toolbar, -1) #set toolbar to first pos
    
    elev_slider = Gtk.Scale.new_with_range(
        Gtk.Orientation(0), min = -90, max = 90, step = 1)
    elev_slider.show()

    vbox.pack_start(elev_slider, False, False, 0)
    vbox.reorder_child(toolbar, -1) #set toolbar to first pos


    def update_az(self, other, parent):
        #callback for updating 3d slider paramd renders
        #almost identical to update_elev
        val = self.get_value()
        parent.ax_00.view_init(azim = other.get_value(), 
                           elev =  int(val))
        parent.fig.canvas.draw()
    azimuth_slider.connect("value_changed", update_az, elev_slider, self)
    

    def update_elev(self, other, parent):
        #could just hanlde this with adding fields to class dynamically
        #through exec and handle on per declaration basis?
        val = self.get_value()
        parent.ax_00.view_init(azim = other.get_value(), 
                           elev =   int(val))
    
        parent.fig.canvas.draw()

    elev_slider.connect("value-changed", update_elev, azimuth_slider, self)
    
    def gen_new_clustering(self, parent, centroid_E):
      parent.k = new_K =int(centroid_E.get_text())
      current = parent.proc
      if current == "GAUSS":pass
      elif current == "IMAGE CLUST":
        print("clustering launched")
        parent.proc_img__cluster(parent.k, parent.img_data.copy())
      elif current == "RAND DATA":
        classes, centroids = kmeans(parent.k, parent.X)
        parent.display_clusters(classes, centroids, parent.X)
      else: 
        classes, centroids = kmeans(new_K, parent.X)
        parent.display_clusters(classes, centroids, parent.X)
    #parent
    centroid_E = Gtk.Entry()
    centroid_btn = Gtk.Button(label = "Update K")
    centroid_btn.show()
    centroid_E.show()
    self.kentry = centroid_E
    centroid_btn.connect("clicked", gen_new_clustering, self, centroid_E)
    self.widgets.attach(centroid_E, 1,4, 1,1)
    self.widgets.attach_next_to(centroid_btn, centroid_E, Gtk.PositionType.RIGHT,1,1)
    btn_redraw = Gtk.Button(label = "redraw")
    redraw = lambda s, parent: parent.fig.canvas.draw()
    btn_redraw.connect("clicked", redraw, self)
    self.widgets.attach_next_to(btn_redraw, centroid_btn, Gtk.PositionType.BOTTOM,1,1,)

    self.win.show_all()
    print("complete")

  def display_clusters(self, classes, centroids, data):
    colors = gen_random_colors(len(centroids))
    self.ax_01.cla()
    for i in range(len(centroids)):
      self.ax_01.scatter(centroids[i,0], centroids[i,1], color = colors[i], label = "o", s = [0.75])
      members = xfetch(data, classes[i])
      self.ax_01.scatter(members[:,0], members[:,1], color = colors[i], label = "o", s = [0.75] * members.shape[0])
    
    self.fig.canvas.draw()
    print("new clsutering generated")
  
  print("nice")

  def show(self):
    plt.show()



  
  def display_clusters_3d(self, classes, centroids, img_data):
    colors = gen_random_colors(len(centroids))
    self.ax_00.cla()
    for i in range(len(centroids)):
      self.ax_00.scatter(centroids[i,0], centroids[i,1], centroids[i,2], color = colors[i], label = "o", s = [0.75])
      members = xfetch(img_data, classes[i])
      self.ax_00.scatter(members[:,0], members[:,1], members[:,2], color = colors[i], label = "o", s = [0.75] * members.shape[0])
    
    self.fig.canvas.draw()

    print("new clsutering generated")
 
  def render_assignments(self, classes, centroids, img_data):
    colors = gen_random_colors(len(centroids))
    pixel_class = np.zeros_like(self.img_data).reshape(-1,3)
    scalar = 255 // len(centroids)
    



    for i in range(len(centroids)):
      members = xfetch(img_data, classes[i])
      pixel_class[classes[i]] = colors[i]





      """
      if i % 3 == 0:
        pixel_class[classes[i]] = np.array([0, (i//3 + 1) * scalar,0],dtype = np.uint8)
      if i % 3 == 2:
        pixel_class[classes[i]] = np.array([0,0,(i//3 + 1) * scalar],dtype = np.uint8)"""
    

    return pixel_class

  def proc_img__cluster(self, K, np_img): 
        if len(np_img.shape) == 3:
          np_img = np_img.reshape(-1,3)
        classes, centroids, err, frames = kmeans(K, np_img, verbose = True)
        #to_matrix_from_vector)c, parent.img_data.shape[0], , parent.img_data.shape[1])
        if self.cluster_3d: 
          self.display_clusters_3d(classes, centroids, np_img)
        pixels=self.render_assignments(classes, centroids, np_img) #pixel class assigned in here
        self.pixel_class = pixels.reshape(*self.img_data.shape)
        self.ax_10.imshow(self.img_data)
        self.ax_11.imshow(self.pixel_class)
        self.fig.canvas.draw()

if __name__ == "__main__":
  win = Window(title = "demo", legend = True, d3 = True)
  win.show()
#sift
#animation buffer
