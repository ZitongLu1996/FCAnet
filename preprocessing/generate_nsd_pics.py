import pandas as pd
import numpy as np
from os.path import join as pjoin
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

basedir = '/Users/yilewang/Desktop/nsd'
stimuli_dir = pjoin(basedir, 'nsddata_stimuli')
stimuli_file = pjoin(stimuli_dir, 'nsd_stimuli.hdf5')

def main():
    # Get data key
    f = h5py.File(stimuli_file, 'r')
    data_key = list(f.keys())[0]
    num_images = f[data_key].shape[0]
    f.close()

    # Add progress bar with tqdm
    for i in tqdm(range(num_images)):
        # Re-open file, read image data, then close
        f = h5py.File(stimuli_file, 'r')
        single_img = f[data_key][i, :, :, :]
        f.close()

        fig = plt.figure(figsize=(4.25, 4.25), dpi=100)
        # hide axis
        # Remove white space
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Remove the ticks and their labels
        plt.xticks([]), plt.yticks([])

        # Remove the axes
        [axi.set_axis_off() for axi in plt.gcf().axes]

        plt.imshow(single_img)
        # save image with index as name
        plt.savefig(pjoin(stimuli_dir, 'pics', str(i)+'.png'), bbox_inches="tight", pad_inches=0)

        # clear the current figure after saving it
        plt.clf()
        plt.close()

if __name__ == '__main__':
    main()

