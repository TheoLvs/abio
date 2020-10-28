
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from ipywidgets import interact,widgets
from IPython.display import HTML,display
from PIL import Image
import cv2
import ffmpeg
import imageio
import datetime

def to_bw_rgb(x):
    """Converts 2D Black & White array to 3D B&W array
    Values in the array are between 0 and 1
    """
    return (np.repeat((1-x)[:,:,None],3,2) * 255).astype(np.uint8)


def to_rgb_cmap(x,cmap = "viridis",n_colors = None):
    """Converts 2D colored array to 3D array applying a colormap scale
    Values in the 2D array are between 0 and N 
    """
    if n_colors is None:
        n_colors = np.max(x) + 1
    return plt.cm.get_cmap(cmap,lut = n_colors)(np.uint8(x),bytes = True)[:,:,:3]


def img_resize(x,size,method="nearest"):
    # https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
    assert method in ["nearest","cubic"]
    if method == "nearest":
        method = cv2.INTER_NEAREST
    elif method == "cubic":
        method = cv2.INTER_CUBIC

    return cv2.resize(x,dsize = size,interpolation = method)


def to_colored(states,only_alive = True):
    
    values = np.sum(states,axis = 0)
    if only_alive:
        values *= states[-1]
        
    return values




class Animation:
    def __init__(self,states,name = "Animation"):

        self._raw = states
        self.states = states
        self.name = name


    def _apply_cmap(self,x,cmap = "viridis",resize = None,method = "nearest"):
        
        n_colors = np.max(x)
        if n_colors == 1:
            if resize is None:
                x = [to_bw_rgb(y) for y in x]
                return x
            else:
                x = [img_resize(to_bw_rgb(y),size = resize,method = method) for y in x]
                return x

        else:
            if resize is None:
                x = [to_rgb_cmap(y,cmap = cmap,n_colors = n_colors) for y in x]
                return x
            else:
                x = [img_resize(to_rgb_cmap(y,cmap = cmap,n_colors = n_colors),size = resize,method = method) for y in x]
                return x

    def transform(self,persistence = 10,only_alive = True,cmap = "viridis",resize = None,method = "nearest"):
        self.states = [
            to_colored(self._raw[max(0,i-persistence+1):i+1],only_alive = only_alive)
            for i in range(len(self._raw))
        ]

        self.states = self._apply_cmap(self.states,cmap = cmap,resize = resize,method = method)




    def show_in_notebook(self,fps = 5,figsize = (8,8),cmap = "viridis"):
        # Prepare widgets
        play = widgets.Play(
            value=0,
            min=0,
            max=len(self.states) - 1,
            step=1,
            interval=int(1000/fps),
            description="Press play",
            disabled=False
        )

        slider = widgets.IntSlider(min = 0,value = 0,max = len(self.states) - 1,step = 1)
        widgets.jslink((play, 'value'), (slider, 'value'))


        # Visualize frames and widgets
        @interact(i = play)
        def show(i):
            plt.figure(figsize = figsize)
            plt.axis('off')
            plt.imshow(self.states[i],cmap = cmap)

        # Display on the notebook
        display(slider)


    def _make_filepath(self,fps,extension=".gif"):
        return f"{str(datetime.datetime.now())[:19].replace(' ','_').replace(':','-')}_fps{fps}_{self.name}{extension}"


    def save_video(self,filepath = None,fps=30, vcodec='libx264',cmap = "viridis",resize = None,method = "nearest"):
        if filepath is None:
            filepath = self._make_filepath(fps,".mp4")
        print(f"... Saving video at {filepath}")
        # From https://github.com/kkroening/ffmpeg-python/issues/246
        images = np.asarray(self.states)
        n,height,width,channels = images.shape
        process = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
                .output(filepath, pix_fmt='yuv420p', vcodec=vcodec, r=fps)
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )
        for frame in images:
            process.stdin.write(
                frame
                    .astype(np.uint8)
                    .tobytes()
            )
        process.stdin.close()
        process.wait()
        return filepath


    def save_gif(self,filepath = None,fps = 30,cmap = "viridis",resize = None,method = "nearest"):
        if filepath is None:
            filepath = self._make_filepath(fps,".gif")
        print(f"... Saving gif at {filepath}")
        imageio.mimsave(filepath,self.states,fps=fps)
        return filepath
