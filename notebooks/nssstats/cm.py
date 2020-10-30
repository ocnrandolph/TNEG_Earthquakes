import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def cm_analysis(y_true, y_pred, labels, filename = None, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.

    Modified from https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7#file-plot_confusion_matrix-py
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(object)
    annot_kws = {'fontsize': 12, 'fontweight' : 'bold'}
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j == 0:
                s = cm_sum[i]
                annot[i, j] = 'True Negatives:\n %.1f%%\n%d/%d' % (p, c, s) 
            elif i == j == 1:
                s = cm_sum[i]
                annot[i, j] = 'True Positives:\n %.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            elif i == 0 and j == 1:
                s = cm_sum[i]
                annot[i, j] = 'False Positives:\n %.1f%%\n%d/%d' % (p, c, s)
            else:
                s = cm_sum[i]
                annot[i, j] = 'False Negatives:\n %.1f%%\n%d/%d' % (p, c, s)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, linewidths = 2, linecolor = 'black', annot_kws = annot_kws,
               cmap = 'Blues')
