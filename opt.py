import datetime
import argparse
import random
import numpy as np
import torch


def choose_dataset(dataset):
    data_dict = {}
    if dataset == "ABIDE":
        data_dict['num_subjects'] = 871
        data_dict['num_classes'] = 2
        data_dict['key'] = 'SUB_ID'
        data_dict['labels'] = 'DX_GROUP'
        data_dict['ages'] = 'AGE_AT_SCAN'
        data_dict['genders'] = 'SEX'
        data_dict['sites'] = 'SITE_ID'
        data_dict['variable'] = 'connectivity'
        data_dict['scores'] = [data_dict['sites'], data_dict['genders'], data_dict['ages']]
    elif dataset == "ADHD":
        data_dict['num_subjects'] = 582
        data_dict['num_classes'] = 2
        data_dict['key'] = 'ScanDir ID'
        data_dict['labels'] = 'DX'
        data_dict['ages'] = 'Age'
        data_dict['genders'] = 'Gender'
        data_dict['sites'] = 'Site'
        data_dict['variable'] = 'connectivity'
        data_dict['scores'] = [data_dict['sites'], data_dict['genders'], data_dict['ages']]
    else:
        print("plase input correct content!")
        raise ValueError
    return data_dict


def choose_atlas(atlas):
    if atlas == 'aal':
        num_rois = 116
    elif atlas == 'ho':
        num_rois = 111
    return atlas, num_rois


class OptInit:
    def __init__(self, model=None, dataset="ABIDE", atlas="aal"):
        data_dict = choose_dataset(dataset)
        atlas, num_rois = choose_atlas(atlas)
        self.parser = argparse.ArgumentParser()
        # data
        self.parser.add_argument('--data_folder', default=rf'data/{dataset}_{atlas}', type=str, help='data folder')
        self.parser.add_argument('--atlas', type=str, default=atlas,
                                 help='atlas for network construction (node definition)')
        self.parser.add_argument('--num_rois', type=int, default=num_rois, help='number of brain regions')
        self.parser.add_argument('--num_classes', type=int, default=data_dict['num_classes'], help='number of classes')
        self.parser.add_argument('--num_subjects', type=int, default=data_dict['num_subjects'],
                                 help='number of subjects')
        self.parser.add_argument('--key', type=str, default=data_dict['key'], help='key values for image data')
        self.parser.add_argument('--labels', type=str, default=data_dict['labels'], help='the title of labels')
        self.parser.add_argument('--ages', type=str, default=data_dict['ages'], help='the title of ages')
        self.parser.add_argument('--genders', type=str, default=data_dict['genders'], help='the title of genders')
        self.parser.add_argument('--sites', type=str, default=data_dict['sites'], help='the title of sites')
        self.parser.add_argument('--scores', default=data_dict['scores'], type=list, help='phenotypic scores')
        self.parser.add_argument('--variable', type=str, default=data_dict['variable'],
                                 help='variable name of .mat file')
        self.parser.add_argument('--dataset', default=dataset, type=str, help='name of dataset')

        # hyper parameters
        self.parser.add_argument('--model', default=model, type=str, help='name of model')
        self.parser.add_argument('--node_dim', type=int, default=500, help='dimension of node features after rfe')
        self.parser.add_argument('--img_depth', default=2, type=int, help='depth of the img_unet')
        self.parser.add_argument('--ph_depth', default=3, type=int, help='depth of the ph_unet')
        self.parser.add_argument('--hidden', type=int, default=128, help='hidden channels of the unet')
        self.parser.add_argument('--out', type=int, default=16, help='out channels of the unet')
        self.parser.add_argument('--dropout', default=0.3, type=float, help='ratio of dropout')
        self.parser.add_argument('--edge_drop', default=0.3, type=float, help='ratio of edge dropout')
        self.parser.add_argument('--pool_ratios', default=0.8, type=float,
                                 help='pooling ratio to be used in the Graph_Unet')
        self.parser.add_argument('--smh', type=float, default=1, help='graph_loss_smooth')
        self.parser.add_argument('--deg', type=float, default=1e-4, help='graph_loss_degree')
        self.parser.add_argument('--val', type=float, default=1e-2, help='graph_loss_value')

        # train parameter
        self.parser.add_argument('--use_cpu', action='store_true', help='use cpu?')
        self.parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')
        self.parser.add_argument('--train', default=True, type=bool, help='train(default) or test')
        self.parser.add_argument('--seed', type=int, default=911, help='random state')
        self.parser.add_argument('--early_stop', type=int, default=100, help='early stop patience')
        self.parser.add_argument('--lr', default=1e-4, type=float, help='initial model learning rate')
        self.parser.add_argument('--vae_lr', default=1e-3, type=float, help='initial vae learning rate')
        self.parser.add_argument('--wd', default=5e-4, type=float, help='initial weight decay')
        self.parser.add_argument('--epoch', default=500, type=int, help='number of epochs for training')
        self.parser.add_argument('--folds', default=10, type=int, help='cross validation folds')

        # train setting
        self.parser.add_argument('--log_save', type=bool, default=False, help='save log or not')
        self.parser.add_argument('--model_save', type=bool, default=True, help='save model or not')
        self.parser.add_argument('--result_save', type=bool, default=True, help='save result or not')
        self.parser.add_argument('--print_freq', default=5, type=int, help='print frequency')
        self.parser.add_argument('--ckpt_path', type=str, default=rf'./save_model/{model}_{dataset}_{atlas}/',
                                 help='checkpoint path to save trained models')

        args = self.parser.parse_args()
        args.time = datetime.datetime.now().strftime("%y%m%d")

        if args.use_cpu:
            args.device = torch.device('cpu')
        else:
            args.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

        self.args = args

    def print_args(self):
        # self.args.printer args
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("==========     CONFIG END    =============")
        print("\n")
        phase = 'train' if self.args.train == 1 else 'eval'
        print('===> Phase is {}.'.format(phase))

    def initialize(self):
        self.set_seed(self.args.seed)
        return self.args

    def set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
