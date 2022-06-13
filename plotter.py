import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.metrics import r2_score, auc

def plot_pred_act(act, pred, feature_name, model, reg_line=True, label=''):
    #xy_max = np.max([np.max(act), np.max(pred)])
    xy_max = 8.5
    plot = plt.figure(figsize=(6,6))
    plt.plot(act, pred, 'o', ms=9, mec='k', mfc='silver', alpha=0.4)
    plt.plot([0, xy_max], [0, xy_max], 'k--', label='ideal')
    if reg_line:
        polyfit = np.polyfit(act, pred, deg=1)
        reg_ys = np.poly1d(polyfit)(np.unique(act))
        plt.plot(np.unique(act), reg_ys, alpha=0.8, label='linear fit')
    plt.axis('scaled')
    plt.xlabel(f'Actual {label}')
    plt.ylabel(f'Predicted {label}')
    plt.title(f'{model}, {feature_name}, r2: {r2_score(act, pred):0.3f}')
    plt.legend(loc='upper left')
    
    return plot

def remove_anomalies(actual, predict):
    anomaly_idx = np.where(predict <= 0)
    #since actual is a series class
    actual = actual.to_numpy()
    if len(anomaly_idx[0]) == 0:
        return actual, predict, 0
    ac, pr = np.delete(actual, anomaly_idx), np.delete(predict, anomaly_idx)
    return ac, pr, len(anomaly_idx[0])

def chemeq(x):
    '''
    python chemical equation label for plot
    '''
    x = x.replace(" ", "")
    split = re.split('(\d+|[A-Za-z]+)', x)
    for idx, c in enumerate(split):
        try: 
            int(c)
            split[idx] = f'_{{{c}}}'
        except:
            pass
    split.insert(0, '$\mathregular{')
    split.append('}$')
    return ''.join(split)

def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='darkorange', label="ROC curve (area = %0.2f)" % auc(fper, tper))
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    #
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()

from shapely.geometry import  LineString
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(8, 8))

    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    
    line_1 = LineString(np.column_stack((thresholds, precisions[:-1])))
    line_2 = LineString(np.column_stack((thresholds, recalls[:-1])))
    intersection = line_1.intersection(line_2)

    plt.plot(*intersection.xy, 'ro', label=f'threshold = {intersection.x:.2f}')

    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    print(f'coordination is {intersection.coords}')