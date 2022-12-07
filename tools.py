from collections import defaultdict
import pandas as pd
from time import time_ns
import os
from matplotlib.colors  import LinearSegmentedColormap
from glob import glob as crappy_glob


def glob(*args, **kwargs):
    return [x.replace("\\", "/") for x in crappy_glob(*args, **kwargs)]

class Timer:
    def __init__(self, active=True):
        self.data = defaultdict(lambda: defaultdict(float))
        self.current = []
        self.total_running_time = 0
        self.active = active
     
    def evaluate(self, filename = None):
        
        output = defaultdict(list)
        total = self.total_running_time*1e-9
        for task, data in self.data.items():
            data = dict(data)
            N = data["executions"]
            T = data["total time"]*1e-9
            T_sq = data["total time squared"]*1e-18
            output["Task"].append(task)
            output["Exections"].append(N)
            output["Total time"].append(T)
            output["Ratio time"].append(T/total)
            output["Average time"].append(T/N)
            output["Standard div time"].append((T_sq/N + T**2/N**2)**0.5)

        output = pd.DataFrame(output)
        if filename is not None: output.to_csv(filename.replace(".csv", "") + ("_unfinished_tasks" if len(self.current) else ""))
        return output, total

    def activate(self): self.active = True
    def deactivate(self): self.active = False

    def start(self, task, stop_previous=True):
        if self.active:
            if stop_previous and len(self.current) > 0: self.stop()
            task = task if len(self.current) == 0 else f"{self.current[-1][0]}/{task}"
            self.current.append((task, time_ns()))
    
    def stop(self):
        if self.active:
            assert len(self.current) > 0, RuntimeError('Timer was stopped with no running tasks')
            stop = time_ns()
            task, start = self.current.pop(-1)
            if len(self.current) == 0:
                self.total_running_time += stop-start
            self.data[task]["total time"] += stop-start
            self.data[task]["total time squared"] += (stop-start)**2
            self.data[task]["executions"] += 1
    
    def __call__(self, label=None, stop_previous=True):
        if label is None: self.stop()
        else: self.start(label, stop_previous)


def mkdir(path: str, strip=True):
    path_ = path.strip("/")
    idx = [i for i, c in enumerate(path_) if c == "/"] + [len(path_)]
    add = 0
    for i, pos in reversed(list(enumerate(idx))):
        if os.path.exists(path_[:pos]):
            add = int(os.path.exists(path_[:pos]))
            break
    for j in idx[i+add:]:
        os.mkdir(path_[:j])
    return path
        

def str_replaces(original_string: str, replace_tuples):
    for old, new in replace_tuples:
        original_string = original_string.replace(old, new)
    return original_string


def get_cmap(colors=None):
    # colors = [(153/255, 0, 0), (1, 1, 1), (142/255, 143/255, 147/255)][::-1]  # R -> G -> B
    colors = [(207/255, 49/255, 51/255), (1, 1, 1), (109/255, 111/255, 114/255)][::-1] if colors is None else colors  # R -> G -> B
    n_bin = 255  # Discretizes the interpolation into bins
    cmap_name = 'my_list'

    # Create the colormap
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)




if __name__ == "__main__":
    timer = Timer()
    timer("hej")
    timer.evaluate("timer1")
    timer()
    timer.evaluate("timer2")
