import seaborn as sns
import matplotlib.pyplot as plt

# from pycm import ConfusionMatrix
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from .configs import ModelType
from .metrics import calc_fpr_tpr_auc, calc_accuracy, calc_errors


def plot_train_process(tgt_name, train_losses, test_losses, results_path):
    plt.plot(list(range(len(train_losses))), train_losses, label="train")
    plt.plot(list(range(len(test_losses))), test_losses, label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    plt.title(f"{tgt_name}, losses")
    plt.savefig(Path(results_path)/"losses.png", dpi=100)
    plt.close()

def plot_scatter(reals, predicted, title, save_path=None, show=False):
    plt.scatter(reals, predicted)
    plt.title(title)
    plt.xlabel('Real')
    plt.ylabel('Predicted')
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_cm(cm, title, save_path=None, show=False):
# cm = ConfusionMatrix(y_actu, y_pred)
    cm.plot()
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def display_results(reals, predicted, model_name, model_type, mode, results_path, labels=None):
    if model_type == ModelType.regr:
        plt.scatter(reals, predicted)
        plt.plot(list(range(0, 100)), list(range(0, 100)), 'k')
        mae = mean_absolute_error(reals, predicted)
        r, p = pearsonr(predicted, reals)
        r2 = r2_score(reals, predicted)
        plt.title('Predictions for ' + model_name + ', MAE = ' + str(round(mae, 2)) + ', Pearson r = ' + str(round(r, 2)) + "R2 = " + str(round(r2, 2)))
        plt.xlabel('True age')
        plt.ylabel('Predicted age')
        plt.grid()
        # plt.show()
        plt.savefig(Path(results_path)/f"{mode}_scatter.png")
        plt.close()

    if model_type == ModelType.clss:
        assert not labels is None
        _, ax = plt.subplots()
        acc = accuracy_score(reals, predicted)
        cm = confusion_matrix(reals, predicted, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        ax.set_title(model_name + ', accuracy = ' + str(round(acc, 2)))
        disp.plot(ax=ax, cmap='BuPu')
        # plt.show()
        plt.savefig(Path(results_path)/f"{mode}_cm.png")
        plt.close()


def plot_roc_curve(reals, predicted, model_name):
    fpr, tpr, roc_auc = calc_fpr_tpr_auc(reals, predicted)
    plt.plot(fpr, tpr, label=model_name)


def display_submission(reals, predicted, model_name, model_type):
    if model_type == ModelType.regr:
        plt.plot(list(range(0, 100)), list(range(0, 100)), 'k')
        plt.scatter(reals, predicted)
        mae = mean_absolute_error(predicted, reals)
        r, p = pearsonr(predicted, reals)
        plt.title('Predictions for ' + model_name + ', MAE = ' + str(round(mae, 2)) + ', Pearson r = ' + str(round(r, 2)))
        plt.xlabel('True age')
        plt.ylabel('Predicted age')
        plt.grid()
        plt.show()

    if model_type == ModelType.clss:
        fig, ax = plt.subplots()
        acc = calc_accuracy(reals, predicted)
        cm = confusion_matrix(reals, predicted, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        ax.set_title('Confusion matrix for ' + model_name + ', accuracy = ' + str(round(acc, 2)))
        disp.plot(ax=ax, cmap='BuPu')
        plt.show()


def display_hist(reals, predicted, model_name, model_type):
    if model_type == ModelType.Age:
        errors = calc_errors(reals, predicted)
        sns.distplot(errors)

        m = np.mean(errors)
        v = np.std(errors)

        plt.title('Histogram of errors for test data' + ', mean = ' + str(round(m, 2)) + ', variance = ' + str(round(v, 2)))
        plt.xlabel('Errors')
        plt.xlim([-100, 100])
        plt.grid()
        plt.show()
    
    if model_type == ModelType.Gender:
        fpr, tpr, roc_auc = calc_fpr_tpr_auc(reals, predicted)
        plt.title('ROC-AUC curve for test data' + ', ROC-AUC score = ' + str(round(roc_auc, 2)))
        plt.plot(fpr, tpr, 'b')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.grid()
        plt.show()

