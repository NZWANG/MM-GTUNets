import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils.tools import (get_ids, get_timeseries, get_networks, get_upper_triangle_networks, get_subject_score,
                         ordinal_encoding, subject_connectivity, move_files_to_main_directory, replace_non_floats)
from nilearn import datasets
from opt import OptInit
import shutil
import glob

class MyDataloader:
    def __init__(self, args, features=None, labels=None, ph_dict=None, ph_data=None):
        self.args = args
        self.data_folder = self.args.data_folder
        self.tensor_folder = os.path.join(self.args.data_folder, 'save_tensor')
        self.seed = args.seed
        self.score_list = args.scores
        self.features = features
        self.labels = labels
        self.ph_dict = ph_dict
        self.ph_data = ph_data
        try:
            self.ids = self.get_subject_IDs()
        except FileNotFoundError:
            self.ids = None

    def load_data(self, save=False):
        print("\nLoading features....")
        self.fcs = self.get_networks(self.ids, norm=True, kind='correlation')

        f_name = 'up_triangles.pt'
        f_load_path = os.path.join(self.tensor_folder, f_name)
        if os.path.exists(f_load_path) and (not save):
            self.features = self.load_tensor(f_load_path)
        else:
            self.features = self.get_upper_triangle_networks(self.ids)

        print("Loading labels....")
        l_name = 'labels' + '.pt'
        l_load_path = os.path.join(self.tensor_folder, l_name)
        if os.path.exists(l_load_path) and (not save):
            self.labels = self.load_tensor(l_load_path)
        else:
            self.labels = self.get_labels(self.ids, binary=True)

        print("Loading phenotypic data....")
        dict_name = 'ph_dict_' + str('_'.join(self.score_list)) + '.pt'
        data_name = 'ph_data_' + str('_'.join(self.score_list)) + '.pt'
        dict_load_path = os.path.join(self.tensor_folder, dict_name)
        data_load_path = os.path.join(self.tensor_folder, data_name)
        if os.path.exists(dict_load_path) and os.path.exists(data_load_path) and (not save):
            self.ph_dict = self.load_tensor(dict_load_path)
            self.ph_data = self.load_tensor(data_load_path)
        else:
            self.ph_dict, self.ph_data = self.get_phenotypic_data(self.ids, self.score_list)

        print("Data has loaded!\n")

        if save:
            self.save_tensor(self.features, f_name, self.tensor_folder)
            self.save_tensor(self.labels, l_name, self.tensor_folder)
            self.save_tensor(self.ph_dict, dict_name, self.tensor_folder)
            self.save_tensor(self.ph_data, data_name, self.tensor_folder)
            print("Data has saved!")
        return self.features, self.labels, self.ph_dict, self.ph_data

    def get_subject_IDs(self, id_file="id.txt", num_subjects=None):
        subject_IDs = get_ids(self.args, id_file, num_subjects)
        return subject_IDs

    def get_timeseries(self, subject_list):
        timeseries = get_timeseries(subject_list, self.args)
        return timeseries

    def calculate_connectivity(self, timeseries, subject, kind='correlation', save=True):
        connectivity = subject_connectivity(timeseries, subject, self.args, kind=kind, save=save)
        return connectivity

    def get_networks(self, subject_list, norm=True, kind='correlation'):
        networks = get_networks(subject_list, self.args, norm, kind)
        return networks

    def get_upper_triangle_networks(self, subject_list, norm=True):
        networks = get_upper_triangle_networks(subject_list, self.args, norm)
        return networks

    def get_sub_scores(self, subject_list, score, encode=True):
        args = self.args
        if encode:
            scores = ordinal_encoding(get_subject_score(subject_list, score, args))
        else:
            scores = get_subject_score(subject_list, score, args)
        return scores

    def get_labels(self, subject_list, binary=True):
        information = self.get_sub_scores(subject_list, self.args.labels)
        values = np.array(list(information.values()), dtype=np.int32)
        if binary:
            # Healthy controls are encoded as 0, and all subtype patients are encoded as 1.
            values[values > 1] = 1
        labels = values
        return labels

    def get_phenotypic_data(self, subject_list, score_list):
        args = self.args
        ph_data = []
        ph_dict = {}

        for score in score_list:
            if score == args.ages:
                information = self.get_sub_scores(subject_list, score, encode=False)
                values = np.array(list(information.values()), dtype=np.float32)
                ph_data.append(values)
                ph_dict[score] = values
            else:
                information = self.get_sub_scores(subject_list, score, encode=True)
                values = np.array(list(information.values()), dtype=np.int32)
                ph_data.append(values)
                ph_dict[score] = values

        ph_data = np.array(ph_data).astype(np.float32)
        ph_data = np.swapaxes(ph_data, 0, 1)
        return ph_dict, ph_data

    def fetch_abide(self, derivatives):
        data_folder = self.args.data_folder
        num_subjects = self.args.num_subjects

        datasets.fetch_abide_pcp(data_dir=data_folder, n_subjects=num_subjects, global_signal_regression=False,
                                 pipeline='cpac', band_pass_filtering=True, derivatives=derivatives)
        move_files_to_main_directory(data_folder)

    def process_abide(self, kind='correlation'):
        self.ids = self.get_subject_IDs()
        time_series = self.get_timeseries(self.ids)
        connectivities = []
        for i in range(len(self.ids)):
            connectivities.append(self.calculate_connectivity(time_series[i], self.ids[i], kind=kind))
            print(f"subject_{self.ids[i]}'s {kind} connectivity has been calculated!")
        print("all finished!")
        connectivities = np.array(connectivities, dtype=np.float32)
        return connectivities

    def process_adhd200(self, kind='correlation', filter=True):
        data_folder = self.args.data_folder
        altas = self.args.atlas
        if filter:
            prefix = 'sfnwmrda'
        else:
            prefix = 'snwmrda'

        self.ids = self.get_subject_IDs()

        for id_ in self.ids:
            file_pattern = os.path.join(data_folder, f'*/{prefix}{id_}*rest_1_{altas}_TCs.1D')
            files = glob.glob(file_pattern)

            for file_name in files:
                replace_non_floats(file_name)
                shutil.move(file_name, os.path.join(data_folder, f'{id_}_rois_{altas}.1D'))
                print(f"{file_name} has processed!")

                subdir_name = os.path.dirname(file_name)
                shutil.rmtree(subdir_name)

        time_series = self.get_timeseries(self.ids)
        connectivities = []
        for i in range(len(self.ids)):
            connectivities.append(self.calculate_connectivity(time_series[i], self.ids[i], kind=kind))

        connectivities = np.array(connectivities, dtype=np.float32)
        return connectivities

    def data_split(self, n_folds, val_ratio=0):
        skf = StratifiedKFold(n_splits=n_folds, random_state=self.seed, shuffle=True)
        if val_ratio == 0:
            cv_splits = list(skf.split(self.features, self.labels))
        else:
            cv_splits = []
            for train_index, test_index in skf.split(self.features, self.labels):
                train_index, val_index = train_test_split(train_index, test_size=val_ratio, random_state=self.seed,
                                                          stratify=self.labels[train_index])
                cv_splits.append((train_index, val_index, test_index))
        return cv_splits

    def save_tensor(self, tensor, name, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, name)
        torch.save(tensor, save_path)

    def load_tensor(self, load_path):
        tensor = torch.load(load_path)
        return tensor


if __name__ == "__main__":
    settings = OptInit(dataset="ABIDE", atlas="aal")
    settings.args.data_folder = rf"../data/{settings.args.dataset}_{settings.args.atlas}/"
    opt = settings.initialize()
    dl = MyDataloader(opt)
    y = dl.get_labels(dl.ids)
    ts = dl.get_timeseries(dl.ids)

