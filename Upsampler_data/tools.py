#%%

#from collections import defaultdict
#import pandas as pd
#from time import time_ns
#import os
#from matplotlib.colors  import LinearSegmentedColormap
#from glob import glob as crappy_glob


#def glob(*args, **kwargs):
#    return [x.replace("\\", "/") for x in crappy_glob(*args, **kwargs)]


#def mkdir(path):
#    if not os.path.exists(path): os.makedirs(path)
#    return path

class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self


