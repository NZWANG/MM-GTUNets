import torch
from math import log10
import numpy as np 
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from scipy.special import softmax
import warnings



def PSNR(mse, peak=1.):
	return 10 * log10((peak ** 2) / mse)


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(preds, labels):
    """Accuracy, auc with masking.Acc of the masked samples"""
    correct_prediction = np.equal(np.argmax(preds, 1), labels).astype(np.float32)
    return np.sum(correct_prediction), np.mean(correct_prediction)


def prf(preds, labels):
    ''' input: logits, labels  '''
    warnings.filterwarnings("ignore")
    pred_lab = np.argmax(preds, 1)
    p, r, f, s = precision_recall_fscore_support(labels, pred_lab, average='binary')
    return [p, r, f]


def auc(preds, labels, is_logit=True):
    ''' input: logits, labels  ''' 
    if is_logit:
        pos_probs = softmax(preds, axis=1)[:, 1]
    else:
        pos_probs = preds[:, 1]
    try:
        auc_out = roc_auc_score(labels, pos_probs)
    except:
        auc_out = 0
    return auc_out


def numeric_score(pred, gt):
    FP = np.float64(np.sum((pred == 1) & (gt == 0)))
    FN = np.float64(np.sum((pred == 0) & (gt == 1)))
    TP = np.float64(np.sum((pred == 1) & (gt == 1)))
    TN = np.float64(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN


def metrics(preds, labels):
    preds = np.argmax(preds, 1) 
    FP, FN, TP, TN = numeric_score(preds, labels)
    sensitivity = TP / (TP + FN + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    recall = sensitivity
    F1 = 2 * precision * recall / (precision + recall + 1e-10)
    return sensitivity, specificity, F1






