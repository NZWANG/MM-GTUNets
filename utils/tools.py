import numpy as np
import scipy.io as sio
import torch
from nilearn import connectome
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
import shutil
import csv
import os
import re
from scipy.spatial import distance
import warnings

# Ignore the specific FutureWarning from Nilearn
warnings.filterwarnings("ignore", category=FutureWarning, module="nilearn.connectome.connectivity_matrices")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', save=False, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.save = save
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.save:
                self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif score >= self.best_score + self.delta:
            self.best_score = score
            if self.save:
                self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def get_ids(args, id_file="id.txt", num_subjects=None):
    """Obtain the ID of subjects."""
    data_folder = args.data_folder
    id_path = os.path.join(data_folder, id_file)

    if os.path.exists(id_path):
        subject_IDs = np.genfromtxt(id_path, dtype=str)
    else:
        create_id_file(args)
        subject_IDs = np.genfromtxt(id_path, dtype=str)
    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs


def create_id_file(args, phenotypic_file="Phenotypic_V1_0b_preprocessed1.csv"):
    """Create the ID file."""
    data_folder = args.data_folder
    scores_dict = {}

    phenotype = os.path.join(data_folder, phenotypic_file)
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            scores_dict[row[args.key]] = row[args.key]
    l = []
    for v in scores_dict.keys():
        l.append(v)
    fl = os.path.join(data_folder, "id.txt")
    with open(fl, "w") as f:
        for item in l:
            f.write("%s\n" % item)


def subject_connectivity(timeseries, subject, args, kind='correlation', save=True):
    atlas_name = args.atlas
    save_path = args.data_folder
    # print("Estimating %s matrix for subject %s" % (kind, subject))

    # calculate the function matrix according to the type of connection to be calculated
    if kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform([timeseries])[0]

    if save:
        subject_file = os.path.join(save_path, subject + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
        sio.savemat(subject_file, {'connectivity': connectivity})

    return connectivity


def get_timeseries(subject_list, args):
    # stores a list of time series, the shape of each subject's time series is (timepoints x regions)
    timeseries = []

    data_folder = args.data_folder
    dataset = args.dataset
    atlas_name = args.atlas

    for i in range(len(subject_list)):
        print("\nStart reading timeseries files.\n")
        if dataset == "ABIDE":
            ro_file = [f for f in os.listdir(data_folder) if
                       f.endswith(subject_list[i] + '_rois_' + atlas_name + '.1D')]
        elif dataset == "ADHD":
            ro_file = [f for f in os.listdir(data_folder) if
                       f.endswith(subject_list[i] + '_rois_' + atlas_name + '.1D')]
        else:
            raise ValueError("No such dataset!")
        fl = os.path.join(data_folder, ro_file[0])
        print("\nReading timeseries file %s\n" % fl)
        timeseries.append(np.loadtxt(fl, skiprows=0))
    print("Reading timeseries files finished!")

    return timeseries


def get_fc(file_name, variable, norm):
    matrix = sio.loadmat(file_name)[variable]
    if norm:
        with np.errstate(divide='ignore', invalid='ignore'):
            # Fisher-Z normalization
            norm_matrix = np.arctanh(matrix)
            norm_matrix[norm_matrix == float('inf')] = 0
        return norm_matrix
    else:
        return matrix


def get_networks(subject_list, args, norm=True, kind='correlation'):
    data_folder = args.data_folder
    atlas_name = args.atlas
    variable = args.variable
    dataset = args.dataset

    graphs = []

    # get the fc matrix
    for subject in subject_list:
        if dataset == 'ABIDE':
            fl = os.path.join(data_folder, subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + ".mat")
        elif dataset == 'ADHD':
            fl = os.path.join(data_folder, subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + ".mat")
        else:
            raise ValueError("No such dataset!")
        if norm:
            norm_matrix = get_fc(fl, variable, norm)
            graphs.append(norm_matrix)
        else:
            fc_matrix = get_fc(fl, variable, norm)
            graphs.append(fc_matrix)

    graphs = np.array(graphs, dtype=np.float32)

    return graphs


def get_upper_triangle_networks(subject_list, args, norm=True, kind='correlation'):
    data_folder = args.data_folder
    atlas_name = args.atlas
    variable = args.variable
    dataset = args.dataset

    all_networks = []

    for subject in subject_list:
        if dataset == 'ABIDE':
            fl = os.path.join(data_folder, subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + ".mat")
        elif dataset == 'ADHD':
            fl = os.path.join(data_folder, subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + ".mat")
        else:
            raise ValueError("No such dataset!")
        norm_matrix = get_fc(fl, variable, norm=norm)
        all_networks.append(norm_matrix)

    all_networks = np.array(all_networks)

    idx = np.triu_indices_from(all_networks[0], 1)
    vec_networks = [mat[idx] for mat in all_networks]
    matrix = np.vstack(vec_networks)

    return matrix


def get_subject_score(subject_list, score, args, phenotypic_file="Phenotypic_V1_0b_preprocessed1.csv"):
    """Obtain phenotypic information of subjects."""
    data_folder = args.data_folder
    scores_dict = {}

    phenotype = os.path.join(data_folder, phenotypic_file)
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            subject_id = row[args.key]
            if subject_id in subject_list:
                subject_score = row[score]
                # filling missing values with -1
                if subject_score == '' or subject_score == 'N/A' or subject_score == 'nan':
                    scores_dict[subject_id] = '-1'
                else:
                    scores_dict[subject_id] = subject_score
    return scores_dict


def cal_feature_sim(features, tensor=True, self_loop=True):
    if torch.is_tensor(features):
        # calculate the correlation coefficient distance
        dist = torch.cdist(features, features, p=2)
        # calculate sigma
        sigma = torch.mean(dist)
        # calculate feature similarity
        feature_sim = torch.exp(- dist ** 2 / (2 * sigma ** 2))
        if not self_loop:
            feature_sim_no_self_loop = feature_sim.clone()
            feature_sim_no_self_loop.fill_diagonal_(0)
            feature_sim = feature_sim_no_self_loop
        if not tensor:
            feature_sim = feature_sim.numpy()
        return feature_sim
    else:
        distv = distance.pdist(features, metric='correlation')
        dist = distance.squareform(distv)
        sigma = np.mean(dist)
        feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))
        if not self_loop:
            np.fill_diagonal(feature_sim, 0)
        if tensor:
            feature_sim = torch.tensor(feature_sim, dtype=torch.float32)
        return feature_sim


def ordinal_encoding(data_dict):
    """
    Map string data to integer, filling missing values with -1.
    """
    value_to_int = {}
    int_list = []
    curr_int = 0

    for v in data_dict.values():
        if v is None or v == '':
            int_list.append(-1)
        elif v not in value_to_int:
            value_to_int[v] = curr_int
            curr_int += 1
            int_list.append(value_to_int[v])
        else:
            int_list.append(value_to_int[v])

    int_dict = {}
    for k, v in data_dict.items():
        if v is None or v == '':
            int_dict[k] = -1
        else:
            int_dict[k] = value_to_int[v]

    return int_dict


def feature_selection(matrix, labels, train_ind, fnum, tensor=True):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    estimator = RidgeClassifier()
    selector = RFE(estimator=estimator, n_features_to_select=fnum, verbose=0, step=100)
    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)
    if tensor:
        x_data = torch.tensor(x_data, dtype=torch.float32)

    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])

    return x_data


def move_files_to_main_directory(main_directory):
    # Traverse all contents in the main directory
    for root, dirs, files in os.walk(main_directory, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            new_location = os.path.join(main_directory, name)
            # Move files to the main directory
            shutil.move(file_path, new_location)
        for name in dirs:
            dir_path = os.path.join(root, name)
            # Attempt to remove empty directories
            try:
                os.rmdir(dir_path)
            except OSError:
                print(f"Directory is not empty or an error occurred: {dir_path}")

def replace_non_floats(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        words = lines[i].split()
        for j in range(len(words)):
            try:
                float(words[j])
            except ValueError:
                words[j] = re.sub(r'\S', ' ', words[j])

        lines[i] = ' '.join(words) + '\n'

    with open(file_name, 'w') as f:
        f.writelines(lines)


def print_result(opt, n_folds, accs, sens, spes, aucs, f1):
    print("=> Average test accuracy in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(accs), np.var(accs)))
    print("=> Average test sensitivity in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(sens), np.var(sens)))
    print("=> Average test specificity in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(spes), np.var(spes)))
    print("=> Average test AUC in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(aucs), np.var(aucs)))
    print("=> Average test F1-score in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(f1), np.var(f1)))
    print("{} Saved model to:{}".format("\u2714", opt.ckpt_path))


def save_result(opt, n_folds, accs, sens, spes, aucs, f1):
    result_path = f'./result/{opt.model}_{opt.dataset}_{opt.atlas}'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    save_path = os.path.join(result_path, 'result.txt')
    with open(save_path, 'a') as f:
        print("========================================================", file=f)
        print("=> Average test accuracy in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(accs), np.var(accs)), file=f)
        print("=> Average test sensitivity in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(sens), np.var(sens)), file=f)
        print("=> Average test specificity in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(spes), np.var(spes)), file=f)
        print("=> Average test AUC in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(aucs), np.var(aucs)), file=f)
        print("=> Average test F1-score in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(f1), np.var(f1)), file=f)
        print("========================================================", file=f)






