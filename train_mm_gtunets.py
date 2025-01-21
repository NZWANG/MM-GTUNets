import torch
import numpy as np
import os
from opt import OptInit
from utils.tools import (EarlyStopping, feature_selection, print_result, save_result)
from utils.metrics import accuracy, auc, metrics
from utils.mydataloader import MyDataloader
from model.mm_gtunets import MM_GTUNets
from model.rp_graph import create_reward_penalty_graph
from tensorboardX import SummaryWriter


def train():
    print("  Number of training samples %d" % len(train_ind))
    print("  Number of validation samples %d" % len(val_ind))
    print("  Start training...")
    acc = 0
    correct = 0
    best_loss = 1e10
    best_epo = 0
    for epoch in range(opt.epoch):
        model.train()
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs, graph_loss = model(x, ph_features, affinity_graphs)
        outputs_train = outputs[train_ind]
        loss = loss_fn(outputs_train, labels[train_ind]) + graph_loss
        if opt.log_save:
            writer.add_scalar("train\tloss", loss.item(), epoch)

        loss.backward()
        optimizer.step()

        correct_train, acc_train = accuracy(outputs[train_ind].detach().cpu().numpy(), y[train_ind])
        if opt.log_save:
            writer.add_scalar("train\tacc", acc_train, epoch)

        model.eval()
        with torch.set_grad_enabled(False):
            outputs, graph_loss = model(x, ph_features, affinity_graphs)
        loss_val = loss_fn(outputs[val_ind], labels[val_ind]) + graph_loss
        if opt.log_save:
            writer.add_scalar("val\tloss", loss_val.item(), epoch)
        outputs_val = outputs[val_ind].detach().cpu().numpy()
        correct_val, acc_val = accuracy(outputs_val, y[val_ind])
        if opt.log_save:
            writer.add_scalar("val\tacc", acc_val, epoch)

        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        val_sen, val_spe, val_f1 = metrics(outputs_val, y[val_ind])
        val_auc = auc(outputs_val, y[val_ind])
        if epoch % opt.print_freq == 0:
            print(
                "Epoch: {},\tlr: {:.5f},\ttrain loss: {:.5f},\ttrain acc: {:.5f},\teval loss: {:.5f},\teval acc: {:.5f} ,"
                "\teval spe: {:.5f}\teval_sen: {:.5f}".format(epoch, lr, loss.item(), acc_train.item(),
                                                              loss_val.item(), acc_val.item(), val_spe, val_sen))
        if best_loss > loss_val:
            best_loss = loss_val
            best_epo = epoch
            acc = acc_val
            correct = correct_val
            aucs[fold] = val_auc
            sens[fold] = val_sen
            spes[fold] = val_spe
            f1[fold] = val_f1
            if (opt.ckpt_path != '') and opt.model_save:
                if not os.path.exists(opt.ckpt_path):
                    os.makedirs(opt.ckpt_path)
                torch.save(model.state_dict(), fold_model_path)
                print("Epoch:{} {} Saved model to:{}".format(epoch, "\u2714", fold_model_path))

        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    accs[fold] = acc
    corrects[fold] = correct

    print("\r\n => Fold {} best val_loss {:.5f}, val_acc {:.5f}, epoch {}\n".format(fold, best_loss, acc, best_epo))


def evaluate():
    print("  Number of testing samples %d" % len(test_ind))
    print("  Start testing...")
    model.load_state_dict(torch.load(fold_model_path), strict=False)
    model.eval()
    outputs, _ = model(x, ph_features, affinity_graphs)
    outputs_test = outputs[test_ind].detach().cpu().numpy()
    corrects[fold], accs[fold] = accuracy(outputs_test, y[test_ind])
    sens[fold], spes[fold], f1[fold] = metrics(outputs_test, y[test_ind])
    aucs[fold] = auc(outputs_test, y[test_ind])
    print("  Fold {} test accuracy {:.5f}, AUC {:.5f}".format(fold, accs[fold], aucs[fold]))


if __name__ == "__main__":
    # model parameter settings
    settings = OptInit(model="MM_GTUNets", dataset="ABIDE", atlas="aal")
    # modify existing parameters
    # settings.args.train = False
    # settings.args.img_depth = 2
    # settings.args.ph_depth = 3
    # settings.args.node_dim = 2500
    # settings.args.pool_ratios = 0.1
    # settings.args.scores = [settings.args.sites]
    # settings.args.smh = 1e-1
    settings.args.device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    # initialize parser
    opt = settings.initialize()
    settings.print_args()

    dl = MyDataloader(opt)

    x, y, ph_dict, ph_data = dl.load_data(save=False)

    labels = torch.tensor(y, dtype=torch.long).to(opt.device)
    ph_features = torch.from_numpy(ph_data).float()

    # k-fold cross validation
    n_folds = opt.folds
    cv_splits = dl.data_split(n_folds, val_ratio=0.1)
    corrects = np.zeros(n_folds, dtype=np.int32)
    accs = np.zeros(n_folds, dtype=np.float32)
    sens = np.zeros(n_folds, dtype=np.float32)
    spes = np.zeros(n_folds, dtype=np.float32)
    f1 = np.zeros(n_folds, dtype=np.float32)
    aucs = np.zeros(n_folds, dtype=np.float32)
    times = np.zeros(n_folds, dtype=np.float32)

    for fold in range(n_folds):
        print("\r\n========================== Fold {} ==========================".format(fold))
        train_ind = cv_splits[fold][0]
        val_ind = cv_splits[fold][1]
        test_ind = cv_splits[fold][2]

        # feature_selection
        x = feature_selection(x, y, train_ind, opt.node_dim)

        affinity_graphs = torch.from_numpy(
            create_reward_penalty_graph(ph_dict, y, train_ind, val_ind, test_ind, opt)).float()
        model = MM_GTUNets(opt, fold).to(opt.device)
        print(model)

        # record training time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)

        early_stopping = EarlyStopping(patience=opt.early_stop, verbose=True)
        fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)
        if opt.log_save:
            writer = SummaryWriter(f"./log/{opt.model}_{opt.dataset}_{opt.atlas}_log/{fold}")
        if opt.train == 1:
            # Recording start time
            start.record()
            # pretraining vae
            model.train_vae(ph_features)
            # training model
            train()
            # Recording end time
            end.record()
            # Synchronizing GPU and CPU
            torch.cuda.synchronize()
            time = start.elapsed_time(end) / 1000
            print(f"Training time of {fold}-fold data: {time} s\n")
            times[fold] = time
            # evaluating model
            evaluate()
        elif opt.train == 0:
            evaluate()
            print(f"img_weight: {model.img_weight}")
            print(f"ph_weight: {model.ph_weight}")
            print(f"scores_weight: {opt.scores}:{model.rp_attention.weights.data}")
    print("\r\n========================== Finish ==========================")
    if opt.train == 1:
        print("=> Average training time in {}-fold CV: {:.5f} s".format(n_folds, np.mean(times)))
    print_result(opt, n_folds, accs, sens, spes, aucs, f1)
    save_result(opt, n_folds, accs, sens, spes, aucs, f1)
