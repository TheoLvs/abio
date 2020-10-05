
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from ipywidgets import interact,widgets
from IPython.display import HTML,display
from PIL import Image
import ffmpeg
import imageio

def to_bw_rgb(x):
    return (np.repeat((1-x)[:,:,None],3,2) * 255).astype(np.uint8)


def to_colored(states,only_alive = True):
    
    values = np.sum(states,axis = 0)
    if only_alive:
        values *= states[-1]
        
    return values




class Animation:
    def __init__(self,states):

        self._raw = states
        self.states = states


    def to_bw_rgb(self):
        self.states = [to_bw_rgb(x) for x in self._raw]

    def to_colored(self,persistence = 10,only_alive = True):
        self.states = [
            to_colored(self._raw[max(0,i-persistence+1):i+1],only_alive = only_alive)
            for i in range(len(self._raw))
        ]

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


    def save_video(self,filepath,fps=30, vcodec='libx264'):
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


    def save_gif(self,filepath,fps = 30):
        imageio.mimsave(filepath,self.states,fps=fps)
