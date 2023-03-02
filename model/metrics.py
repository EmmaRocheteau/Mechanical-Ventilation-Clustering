import numpy as np
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    c_matrix = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix)


def sigmoid(x):
     return 1 / (1 + np.exp(-x))


def compute_binary_metrics(y_true, y_pred, verbose=False):
    y_pred = sigmoid(y_pred)  # need to add a sigmoid in here because we are using BCEWithLogitsLoss and we aren't applying the sigmoid activation in the model
    pred_labels = np.where(y_pred > 0.5, 1.0, 0.0)

    if verbose:
        cf = confusion_matrix(y_true, pred_labels, labels=range(2))
        print('confusion matrix:')
        print(cf)

    if len(set(y_true)) != 1:
        auroc = metrics.roc_auc_score(y_true, y_pred)
        auprc = metrics.average_precision_score(y_true, y_pred)
    else:
        auroc = np.nan
        auprc = np.nan

    results = {'auroc': auroc,
               'auprc': auprc}
    results = {key: float(results[key]) for key in results}
    if verbose:
        for key in results:
            print('{}: {:.4f}'.format(key, results[key]))

    return results

def compute_categorical_metrics(y_true, y_pred, verbose=False):
    pred_class = np.argmax(y_pred, axis=-1)
    results = {'accuracy': accuracy_score(y_true, pred_class)}
    results['balanced_accuracy'] = metrics.balanced_accuracy_score(y_true, pred_class)
    results['log_loss'] = metrics.log_loss(y_true, y_pred)
    if verbose:
        cf = metrics.confusion_matrix(y_true, pred_class)
        cr = classification_report(y_true, pred_class)
        print('Confusion matrix:')
        print(cf)
        print('Classification report:')
        print(cr)

    return results


class CustomBins:
    inf = 1e18
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)

def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0]
        b = CustomBins.bins[i][1]
        if a <= x < b:
            if one_hot:
                onehot = np.zeros((CustomBins.nbins,))
                onehot[i] = 1
                return onehot
            return i
    return

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(4/24, y_true))) * 100  # this stops the mape being a stupidly large value when y_true happens to be very small

def mean_squared_logarithmic_error(y_true, y_pred):
    return np.mean(np.square(np.log((y_true + 1)/(y_pred + 1))))

def regression_metrics(y_true, y_pred, verbose=False):
    y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_pred]
    if verbose:
        cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
        print('Custom bins confusion matrix:')
        print(cf)

    kappa = metrics.cohen_kappa_score(y_true_bins, prediction_bins, weights='linear')
    mad = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    msle = mean_squared_logarithmic_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    return mad, mse, mape, msle, r2, kappa

def compute_duration_metrics(y_true, y_pred, verbose=False):
    mad, mse, mape, msle, r2, kappa = regression_metrics(y_true, y_pred, verbose)
    results = {'mad': mad,
               'mse': mse,
               'mape': mape,
               'msle': msle,
               'r2': r2,
               'kappa': kappa}
    results = {key: float(results[key]) for key in results}
    if verbose:
        for key in results:
            print('{}: {:.4f}'.format(key, results[key]))
    return results

def compute_online_metrics(y_true, y_pred, verbose=False):
    mse = metrics.mean_squared_error(y_true, y_pred)
    results = {'mse': float(mse)}
    if verbose:
        print('mse: {:.4f}'.format(mse))
    return results