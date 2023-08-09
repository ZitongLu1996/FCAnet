import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from os.path import join as pjoin
import csv
import itertools
import argparse
import time
from tqdm import tqdm
import multiprocessing

class NSD_Processor:
    def __init__(self, subject="sub_01"):
        self.basedir = '/mnt/c/Users/Wayne/Desktop/nsd'
        self.subject = subject
        self.betas_dir = pjoin(self.basedir, f'{self.subject}/nsd_betas')
        self.design_file = pjoin(self.basedir, f'{self.subject}/nsd_design')
        self.roi_dir = pjoin(self.basedir, f'{self.subject}/nsd_roi')
        self.session_num = 37
        self.run_num_less = 12
        self.run_num_more = 14
        self.visualrois_num = 8
        self.kastner_num = 4

    # define all functions
    def separate_tab(self, file_path):
        """
        A function to create a dictionary from a tab separated file
        
        """
        label_dict = {}
        with open(file_path) as ctab:
            reader = csv.reader(ctab, delimiter='\t')
            for row in reader:
                # do something with row
                label_dict[row[0].split()[0]] = row[0].split()[1]
        return label_dict

    def count_roi_voxels_num(self, roi, label):
        return np.where(roi == label)[0].shape[0]

    def generate_list_voxel_3d(self, roi, label):
        _axis_list = []
        all_3d = np.where(roi == label)
        for i in range(len(all_3d[0])):
            _axis_list.append([all_3d[0][i], all_3d[1][i], all_3d[2][i]])
        return _axis_list

    def decompose_3d_to_voxel_id(self, combined_dict):
        # create an empty panda dataframe
        _df = pd.DataFrame(columns=['x', 'y', 'z', 'voxel_id', 'label'])
        counter = 0
        keys_dict = list(combined_dict.keys())
        for i in range(len(keys_dict)):
            for z in range(len(list(combined_dict[keys_dict[i]]))):
                # add a new row to the empty dataframe
                # use pd concat to add a new row
                single_row = pd.DataFrame([{'x': combined_dict[keys_dict[i]][z][0], 'y': combined_dict[keys_dict[i]][z][1], 'z': combined_dict[keys_dict[i]][z][2], 'voxel_id': counter, 'label': keys_dict[i]}])
                _df = pd.concat([_df, single_row], axis=0, ignore_index=True)
                counter += 1
        return _df

    def concat_all_designs(self,design_file):
        # create an empty array
        _1d_list = []
        ## create strings to read data
        for i in range(self.session_num):
            for z in range(self.run_num_more):
                try:
                    _filename = pjoin(design_file, 'design_session' + str(i+1).zfill(2)+'_run'+str(z+1).zfill(2)+'.tsv')
                    ## read tsv data
                    _data = pd.read_csv(_filename, sep='\t', header=None).values
                    
                    # get num larger than 0
                    _data_nonzero = _data[_data > 0]
                    _1d_list.append(_data_nonzero)
                except:
                    continue
        return np.array(list(itertools.chain(*_1d_list)))
    def concat_single_session_designs(self,design_file, session_num):
        # create an empty array
        _1d_list = []
        ## create strings to read data
        for z in range(self.run_num_more):
            try:
                _filename = pjoin(design_file, 'design_session' + str(session_num).zfill(2)+'_run'+str(z+1).zfill(2)+'.tsv')
                ## read tsv data
                _data = pd.read_csv(_filename, sep='\t', header=None).values
                
                # get num larger than 0
                _data_nonzero = _data[_data > 0]
                _1d_list.append(_data_nonzero)
            except:
                continue
        return np.array(list(itertools.chain(*_1d_list)))


    def get_betas_table(self, session_num):
        pd_sub_table_train = pd.DataFrame()
        pd_sub_table_test = pd.DataFrame()
        # get the design
        session_design = self.concat_single_session_designs(self.design_file, session_num)
        _filename = pjoin(self.betas_dir, 'betas_session' + str(session_num).zfill(2)+'.hdf5')
        with h5py.File(_filename, 'r') as f:
            # get data key
            data_key = list(f.keys())[0]
            dataset = f[data_key]
            for j in range(dataset.shape[0]):
                single_stimuli_voxels_array = np.zeros((len(self.voxel_locator.index), 1))
                for index, (x, y, z) in enumerate(zip(self.voxel_locator['z'], self.voxel_locator['y'], self.voxel_locator['x'])):
                    single_stimuli_voxels_array[index] = dataset[j, x, y, z]
                _one_stimuli = pd.DataFrame(single_stimuli_voxels_array[:,0]/300, columns=[session_design[j]])
                if session_design[j] not in self.shared_1000.values:
                    pd_sub_table_train = pd.concat([pd_sub_table_train, _one_stimuli], axis=1)
                else:
                    pd_sub_table_test = pd.concat([pd_sub_table_test, _one_stimuli], axis=1)
        pd_sub_table_train.to_csv(pjoin(self.basedir, f'{self.subject}/train_betas_session{session_num}.csv'))
        pd_sub_table_test.to_csv(pjoin(self.basedir, f'{self.subject}/test_betas_session{session_num}.csv'))

    def concate_all_betas_tables(self):
        for i in range(self.session_num):
            pd_sub_table_train = pd.read_csv(pjoin(self.basedir, f'{self.subject}/train_betas_session{i+1}.csv'))
            pd_sub_table_test = pd.read_csv(pjoin(self.basedir, f'{self.subject}/test_betas_session{i+1}.csv'))
            if i == 0:
                pd_table_train = pd_sub_table_train
                pd_table_test = pd_sub_table_test
            else:
                pd_table_train = pd.concat([pd_table_train, pd_sub_table_train], axis=1)
                pd_table_test = pd.concat([pd_table_test, pd_sub_table_test], axis=1)
        # add labels to the table
        pd_table_train = pd.concat([pd_table_train, self.voxel_locator['label']])
        pd_table_test = pd.concat([pd_table_test, self.voxel_locator['label']])
        # write to csv
        pd_table_train.to_csv(pjoin(self.basedir, f'{self.subject}/train_betas.csv'))
        pd_table_test.to_csv(pjoin(self.basedir, f'{self.subject}/test_betas.csv'))

        
    def main(self):
        ############################################

        # pre step, prepare the data

        ## prf rois
        lh_prf_visual_file = pjoin(self.roi_dir, 'lh.prf-visualrois.nii.gz')
        rh_prf_visual_file = pjoin(self.roi_dir, 'rh.prf-visualrois.nii.gz')
        prf_visual_labels_file = pjoin(self.basedir, 'prf-visualrois.mgz.ctab')
        prf_visualrois_lables = self.separate_tab(prf_visual_labels_file)

        ## kastner rois
        kastner_file = pjoin(self.roi_dir, 'Kastner2015.nii.gz')
        kastner_labels_file = pjoin(self.basedir, 'Kastner2015.mgz.ctab')
        kastner_labels = self.separate_tab(kastner_labels_file)

        lh_prf_img = nib.load(lh_prf_visual_file)
        rh_prf_img = nib.load(rh_prf_visual_file)
        kastner_img = nib.load(kastner_file)

        lh_prf_data = lh_prf_img.get_fdata()
        rh_prf_data = rh_prf_img.get_fdata()
        kastner_data = kastner_img.get_fdata()
        ################################################

        # Main Steps

        # step 1, label all voxles
        dict4combined_axis = {}
        _tmp_dict = {}
        for i in range(1,self.visualrois_num):
            _tmp_dict["left_"+prf_visualrois_lables[str(i)]] = self.generate_list_voxel_3d(lh_prf_data, i)
            _tmp_dict["right_"+prf_visualrois_lables[str(i)]] = self.generate_list_voxel_3d(rh_prf_data, i)
            # integrate left and right
            left_right = [i for i in _tmp_dict["left_"+prf_visualrois_lables[str(i)]]]
            left_right.extend([i for i in _tmp_dict["right_"+prf_visualrois_lables[str(i)]]])
            dict4combined_axis[prf_visualrois_lables[str(i)]] = left_right


        for i in range(self.visualrois_num,self.visualrois_num+self.kastner_num):
            dict4combined_axis[kastner_labels[str(i)]] = self.generate_list_voxel_3d(kastner_data, i)


        self.voxel_locator = self.decompose_3d_to_voxel_id(dict4combined_axis)

        # step 2 loading shared pics and classify train and test:
        self.shared_1000 = pd.read_csv(pjoin(self.basedir, 'shared1000.tsv'), header=None)

        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)

        tasks = [i for i in range(1, self.session_num+1)]
        pool.map(self.get_betas_table, tasks)
        
        self.concate_all_betas_tables(self)

        


    # def main(self):
    #     ############################################

    #     # pre step, prepare the data

    #     ## prf rois
    #     lh_prf_visual_file = pjoin(self.roi_dir, 'lh.prf-visualrois.nii.gz')
    #     rh_prf_visual_file = pjoin(self.roi_dir, 'rh.prf-visualrois.nii.gz')
    #     prf_visual_labels_file = pjoin(self.basedir, 'prf-visualrois.mgz.ctab')
    #     prf_visualrois_lables = self.separate_tab(prf_visual_labels_file)

    #     ## kastner rois
    #     kastner_file = pjoin(self.roi_dir, 'Kastner2015.nii.gz')
    #     kastner_labels_file = pjoin(self.basedir, 'Kastner2015.mgz.ctab')
    #     kastner_labels = self.separate_tab(kastner_labels_file)

    #     lh_prf_img = nib.load(lh_prf_visual_file)
    #     rh_prf_img = nib.load(rh_prf_visual_file)
    #     kastner_img = nib.load(kastner_file)

    #     lh_prf_data = lh_prf_img.get_fdata()
    #     rh_prf_data = rh_prf_img.get_fdata()
    #     kastner_data = kastner_img.get_fdata()
    #     ################################################

    #     # Main Steps

    #     # step 1, label all voxles
    #     dict4combined_axis = {}
    #     _tmp_dict = {}
    #     for i in range(1,self.visualrois_num):
    #         _tmp_dict["left_"+prf_visualrois_lables[str(i)]] = self.generate_list_voxel_3d(lh_prf_data, i)
    #         _tmp_dict["right_"+prf_visualrois_lables[str(i)]] = self.generate_list_voxel_3d(rh_prf_data, i)
    #         # integrate left and right
    #         left_right = [i for i in _tmp_dict["left_"+prf_visualrois_lables[str(i)]]]
    #         left_right.extend([i for i in _tmp_dict["right_"+prf_visualrois_lables[str(i)]]])
    #         dict4combined_axis[prf_visualrois_lables[str(i)]] = left_right


    #     for i in range(self.visualrois_num,self.visualrois_num+self.kastner_num):
    #         dict4combined_axis[kastner_labels[str(i)]] = self.generate_list_voxel_3d(kastner_data, i)


    #     voxel_locator = self.decompose_3d_to_voxel_id(dict4combined_axis)

    #     # step 2 loading shared pics and classify train and test:
    #     shared_1000 = pd.read_csv(pjoin(self.basedir, 'shared1000.tsv'), header=None)

    #     # step 3, use the labels to generate betas tables
    #     # concat all design files
    #     all_design_image = self.concat_all_designs(self.design_file)


    #     # step 4, generate betas tables
    #     pd_table_train = pd.DataFrame()
    #     pd_table_test = pd.DataFrame()


    #     stimuli_index = 0
    #     for i in tqdm(range(self.session_num)):
    #         _filename = pjoin(self.betas_dir, 'betas_session' + str(i+1).zfill(2)+'.hdf5')
    #         with h5py.File(_filename, 'r') as f:
    #             # get data key
    #             data_key = list(f.keys())[0]
    #             dataset = f[data_key]
    #             for j in range(dataset.shape[0]):
    #                 single_stimuli_voxels_array = np.zeros((len(voxel_locator.index), 1))
    #                 for index, (x, y, z) in enumerate(zip(voxel_locator['z'], voxel_locator['y'], voxel_locator['x'])):
    #                     single_stimuli_voxels_array[index] = dataset[j, x, y, z]
    #                 _one_stimuli = pd.DataFrame(single_stimuli_voxels_array[:,0]/300, columns=[all_design_image[stimuli_index]])
    #                 if all_design_image[stimuli_index] not in shared_1000.values:
    #                     pd_table_train = pd.concat([pd_table_train, _one_stimuli], axis=1)
    #                 else:
    #                     pd_table_test = pd.concat([pd_table_test, _one_stimuli], axis=1)
    #                 stimuli_index += 1

    #     # add a label column to pd_table_train
    #     pd_table_train = pd.concat([pd_table_train, voxel_locator['label']])
    #     # add a label column to pd_table_test
    #     pd_table_test = pd.concat([pd_table_test, voxel_locator['label']])

    #     pd_table_train.to_csv(pjoin(self.basedir, 'train_betas.csv'))
    #     pd_table_test.to_csv(pjoin(self.basedir, 'test_betas.csv'))

if __name__ == "__main__":
    # add arguments
    parser = argparse.ArgumentParser(description='define subject')
    parser.add_argument('-s','--subject', type=str, default='sub_01', help='define subject')
    args = parser.parse_args()

    processor = NSD_Processor(args.subject)
    processor.main()