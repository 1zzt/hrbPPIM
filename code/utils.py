import torch
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, \
        precision_score, recall_score, \
        f1_score, confusion_matrix, accuracy_score, matthews_corrcoef



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

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg
    


def performance_evaluation(output, labels):
    output =  torch.from_numpy(output)
    pred_scores = output
    roc_auc = roc_auc_score(labels, pred_scores)
    prec, reca, _ = precision_recall_curve(labels, pred_scores)
    aupr = auc(reca, prec)

    best_threshold = 0.5
    pred_labels = output > best_threshold
    precision = precision_score(labels, pred_labels)
    accuracy = accuracy_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    (tn, fp, fn, tp) = confusion_matrix(labels, pred_labels).ravel()
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(labels, pred_labels)

    return roc_auc, aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels
