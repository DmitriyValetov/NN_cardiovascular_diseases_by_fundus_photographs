from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, roc_curve, auc, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from .configs import ModelType



def get_metrics(reals, predicted, model_type, labels=None):
    if model_type == ModelType.regr:
        return {
            "mse": mean_squared_error(reals, predicted),
            "mae": mean_absolute_error(reals, predicted),
            "r": pearsonr(predicted, reals)[0],
            "r2": r2_score(reals, predicted),
        }
    assert not labels is None
    cm = confusion_matrix(reals, predicted, labels=labels)
    return {
        "acc": accuracy_score(reals, predicted),
        "recall": recall_score(reals, predicted),
        "f1": f1_score(reals, predicted),
    }

def calc_accuracy(reals, predicted):
    acc = accuracy_score(predicted, reals)
    return acc

def calc_fpr_tpr_auc(reals, predicted):
    fpr, tpr, threshold = roc_curve(reals, predicted)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def calc_auc_score(reals, predicted):
    return calc_fpr_tpr_auc(reals, predicted)[2]

def calc_errors(submission):
    errors = submission['predicted'] - submission['real']
    return errors