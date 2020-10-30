import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def calc_tpr(y_true, y_prob, threshold = 0.5):
    tp = len(y_prob[(y_true == 1) & (y_prob >= threshold)])
    tpr = tp / np.sum(y_true == 1)
    
    return tpr

def calc_fpr(y_true, y_prob, threshold = 0.5):
    fp = len(y_prob[(y_true == 0) & (y_prob >= threshold)])
    fpr = fp / np.sum(y_true == 0)
    
    return fpr

def tpr_fpr(y_true, y_prob):
    thresholds = np.linspace(start = 0, stop = 1, num = 100)
    tpr = [calc_tpr(y_true, y_prob, threshold) for threshold in thresholds]
    fpr = [calc_fpr(y_true, y_prob, threshold) for threshold in thresholds]
    return tpr, fpr

def roc_curve(y_true, y_prob, area = False):
    tpr, fpr = tpr_fpr(y_true, y_prob)
    
    roc_plot = plt.plot(fpr, tpr)
    if area:
        plt.fill_between(fpr, tpr, color = 'lightblue')
    plt.plot([0,1], [0,1], color = 'black')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    return roc_plot;

def roc_interact(threshold, y_true, y_prob, tpr, fpr, alpha = 0.6):
    fig = plt.figure(figsize=(7, 7)) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3]) 
    
    ax0 = plt.subplot(gs[0])
    
    ax0.scatter(y_prob[y_true==0], (np.zeros_like(y_true).astype(float) - 0.05 + 0.1*(y_true == 1))[y_true==0], 
            c = 'blue', label = 'non-carrier', 
            alpha = alpha, edgecolor = 'black')
    ax0.scatter(y_prob[y_true==1], (np.zeros_like(y_true).astype(float) - 0.05 + 0.1*(y_true == 1))[y_true==1], 
            c = 'red', label = 'carrier', 
            alpha = alpha, edgecolor = 'black')
    ax0.plot([-.0,1.], [0,0], linewidth = 3, color = 'black')
    for i in [0,1]:
        ax0.plot([i,i], [0.1, -0.1], linewidth = 3, color = 'black')
        ax0.annotate(s = str(i), xy = (i, -0.125), ha = 'center', va = 'top', fontsize = 12, fontweight = 'bold')
        
    ax0.plot([threshold, threshold], [-0.4, 0.4], color = 'red', linestyle = '--', linewidth = 3)
    
    ax0.title.set_text("Predicted Probabilities")
    plt.sca(ax0)
    plt.yticks([])
    plt.legend(loc = 'upper right')
    ax0.set_ylim(-0.5, 0.75); 
    
    ax1 = plt.subplot(gs[1])
    
    ax1.plot(fpr, tpr, linewidth = 1.5)
    ax1.plot([0,1], [0,1], color = 'black')
    
    plt.sca(ax1)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    ax1.scatter([calc_fpr(y_true, y_prob, threshold)], [calc_tpr(y_true, y_prob, threshold)], color = 'black', zorder = 500);
