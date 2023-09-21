import pandas as pd
import numpy as np
from os.path import join as pjoin
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

basedir = '/mnt/c/Users/Wayne/Desktop/nsd'
stimuli_file = pjoin(basedir, 'nsd_stimuli.hdf5')

# read hdf5 file
with h5py.File(stimuli_file, 'r') as f:
    # get data key
    data_key = list(f.keys())[0]
    dataset = f[data_key]

    def main():
        # Create an empty numpy array to hold the image data
        single_img = np.empty((dataset.shape[1], dataset.shape[2], dataset.shape[3]), dtype=dataset.dtype)

        # Add progress bar with tqdm
        for i in tqdm(range(dataset.shape[0])):
            # Read the image data directly into the numpy array
            dataset.read_direct(single_img, np.s_[i, :, :, :])

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
            plt.savefig(pjoin(basedir, 'pics', str(i)+'.png'), bbox_inches="tight", pad_inches=0)
            
            # clear the current figure after saving it
            plt.clf()
            plt.close()

    if __name__ == '__main__':
        main()

