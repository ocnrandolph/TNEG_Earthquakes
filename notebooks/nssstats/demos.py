import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, binom, poisson
from scipy.stats import skewnorm

def binom_normal_plot(n, p):
    x_binom = np.arange(start = 0, stop = n+1)
    y_binom = binom.pmf(k = x_binom, n = n, p = p) 
    
    x_norm = np.linspace(-1, n+1, num = 200)
    y_norm = norm.pdf(x_norm, loc = n*p, scale = np.sqrt(n*p*(1-p)))
    
    fig, ax = plt.subplots(figsize = (8,6))
    
    plt.bar(x_binom, y_binom, label = 'Binomial Distribution')
    xmin, xmax = plt.xlim()
    plt.plot(x_norm, y_norm, linewidth = 3, color = 'red', label = 'Normal Distribution')
    plt.legend()
    plt.xlim(xmin, xmax);

def poisson_pmf_plot(rate = 8):
    fig, ax = plt.subplots(figsize = (10,5))
    x = np.arange(0, 25)
    y = poisson.pmf(x, mu = rate)
    
    plt.bar(x, y)
    plt.ylim(0, 0.3)
    plt.title('Poisson Distribution, Rate = {}'.format(rate))
    plt.ylabel('density')
    plt.xticks(x)

def binom_poisson(n, show_probabilities = False):
    successes = 8 
    
    x_binom = np.arange(0, min(n + 1, 25))
    y_binom = binom.pmf(x_binom, p = successes/n, n = n)
    
    x_poisson = np.arange(0, 25)
    y_poisson = poisson.pmf(x_poisson, mu = successes)

    fig, ax = plt.subplots(nrows = 2, ncols=1, figsize = (10,6), sharey = True, sharex = True)
    ax[0].bar(x_binom, y_binom)
    ax[0].set_title('Binomial Distribution, {} Trials, p = {} / {} = {}'.format(n, successes, n, round(successes/n, 4)))
    
    
    ax[1].bar(x_poisson, y_poisson)
    ax[1].set_title('Poisson Distribution, rate = {}'.format(successes))
    
    if show_probabilities:
        for x, y in zip(x_binom, y_binom):
            ax[0].annotate(s = str(round(y,4)), xy = (x,0.01), rotation = 90, ha = 'center', va = 'bottom')

        for x, y in zip(x_poisson, y_poisson):
            ax[1].annotate(s = str(round(y,4)), xy = (x,0.01), rotation = 90, ha = 'center', va = 'bottom');

def confidence_interval_plot(area = 0.95, sample_mean = 0):
    x = np.linspace(norm.ppf(0.001), norm.ppf(0.999), 100)
    fig, ax = plt.subplots(1, 1, figsize = (14,4))
    ax.plot(x, norm.pdf(x),'r-', lw=3, alpha=1, label='norm pdf', color = 'black')
    x_min = norm.ppf((1-area) / 2)
    x_max = norm.ppf(1 - (1-area) / 2)
    if x_min < x_max:
        section = np.arange(x_min, x_max, 0.01)
        plt.fill_between(section,norm.pdf(section), color = 'slateblue')
    else:
        section_1 = np.arange(-3, x_max, 0.01)
        plt.fill_between(section_1, norm.pdf(section_1), color = 'slateblue')
        section_2 = np.arange(x_min, 3, 0.01)
        plt.fill_between(section_2, norm.pdf(section_2), color = 'slateblue')
    if x_min != -3:
        plt.vlines(x = x_min, ymin = 0, ymax = norm.pdf(x_min), lw = 3, color = 'black')
        #plt.xticks(ticks = [x_min], labels = [f'z = {x_min}'], fontsize = 14)
        plt.annotate(s = f'{round(x_min,2)}', xy = (x_min, -0.01), fontsize = 14, fontweight = 'bold', 
                     va = 'top', ha = 'center', color = 'orangered')
    if x_max != 3:
        plt.vlines(x = x_max, ymin = 0, ymax = norm.pdf(x_max), lw = 3, color = 'black')
        #plt.xticks(ticks = [x_max], labels = [f'z = {x_max}'], fontsize = 14)
        plt.annotate(s = f'{round(x_max,2)}', xy = (x_max, -0.01), fontsize = 14, fontweight = 'bold', 
                     va = 'top', ha = 'center', color = 'orangered')
        
    plt.annotate(s = f'Area = {area}', xy = (0, 0.01), fontsize = 14,
                va = 'bottom', ha = 'center', fontweight = 'bold', color = 'yellow')
    if True:
        plt.plot([sample_mean + x_min, sample_mean - 0.2], [-0.1, -0.1], lw = 3, color = 'red')
        plt.plot([sample_mean + 0.2, sample_mean + x_max], [-0.1, -0.1], lw = 3, color = 'red')
        plt.plot([sample_mean + x_min, sample_mean + x_min], [-0.09, -0.11], lw = 3, color = 'red')
        plt.plot([sample_mean + x_max, sample_mean + x_max], [-0.09, -0.11], lw = 3, color = 'red')
        plt.plot([sample_mean, sample_mean], [0, -0.06], lw = 3, linestyle = '-', color = 'black')
        plt.annotate(s = '$\\overline{x}$', xy = (sample_mean, -0.1), 
                     va = 'center', ha = 'center',
                     fontsize = 14, fontweight = 'bold')
        plt.annotate(s = r'0', xy = (0, -0.01), fontsize = 14, fontweight = 'bold', 
                     va = 'top', ha = 'center', color = 'orangered')
        
    plt.plot([0, 0], [-0.15, 0.4], linestyle = '--', color = 'black')
    plt.annotate(s = '$\mu$', xy = (0, -0.15), fontsize = 14, ha = 'center', va = 'top')
        
    plt.hlines(y = 0, xmin = -3, xmax = 3)
    plt.yticks([])
    plt.xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.ylim(plt.ylim()[0] - 0.05, plt.ylim()[1])
    plt.xlim(-5.5, 5.5);


def bootstrap_confidence_interval_plot(area = 0.95, sample_mean = 0, a = 5):
    x = np.linspace(skewnorm.ppf(0.00001, a = a), skewnorm.ppf(0.999, a = a), 100)
    y = skewnorm.pdf(x, a = a)
    
    mode = x[np.argmax(y)]
    
    sample_mean += mode
    
    fig, ax = plt.subplots(1, 1, figsize = (14,4))
    ax.plot(x, skewnorm.pdf(x, a = a),'r-', lw=3, alpha=1, label='skewnorm pdf', color = 'black')
    
    x_min = skewnorm.ppf((1-area) / 2, a = a)
    x_max = skewnorm.ppf(1 - (1-area) / 2, a = a)
    
    section = np.arange(x_min, x_max, 0.01)
    plt.fill_between(section,skewnorm.pdf(section, a = a), color = 'slateblue')
    if x_min != -3:
        plt.vlines(x = x_min, ymin = 0, ymax = skewnorm.pdf(x_min, a = a), lw = 3, color = 'black')
        plt.annotate(s = 'a', xy = (x_min, -0.01), fontsize = 14, fontweight = 'bold', 
                     va = 'top', ha = 'center', color = 'orangered')
    if x_max != 3:
        plt.vlines(x = x_max, ymin = 0, ymax = skewnorm.pdf(x_max, a = a), lw = 3, color = 'black')
        plt.annotate(s = 'b', xy = (x_max, -0.01), fontsize = 14, fontweight = 'bold', 
                     va = 'top', ha = 'center', color = 'orangered')
        
    plt.annotate(s = f'Area = {area}', xy = (mode + 0.5, 0.01), fontsize = 14,
                va = 'bottom', ha = 'center', fontweight = 'bold', color = 'yellow')
    
    plt.plot([sample_mean - (x_max - mode), sample_mean - 0.1], [-0.1, -0.1], lw = 3, color = 'red')
    plt.plot([sample_mean + 0.1, sample_mean + (mode - x_min)], [-0.1, -0.1], lw = 3, color = 'red')
    plt.plot([sample_mean - (x_max - mode),sample_mean - (x_max - mode)], [-0.09, -0.11], lw = 3, color = 'red')
    plt.plot([sample_mean + (mode - x_min),sample_mean + (mode - x_min)], [-0.09, -0.11], lw = 3, color = 'red')
    plt.plot([sample_mean, sample_mean], [0, -0.06], lw = 3, linestyle = '-', color = 'black')
    plt.annotate(s = 's', xy = (sample_mean, -0.1), 
                     va = 'center', ha = 'center',
                     fontsize = 14, fontweight = 'bold')
    #plt.annotate(s = r'0', xy = (0, -0.01), fontsize = 14, fontweight = 'bold', 
    #                 va = 'top', ha = 'center', color = 'orangered')
        
    plt.plot([mode, mode], [-0.15, skewnorm.pdf(mode, a = a)], linestyle = '--', color = 'black')
    plt.annotate(s = 'P', xy = (mode, -0.15), fontsize = 14, ha = 'center', va = 'top', fontweight = 'bold')
        
    plt.hlines(y = 0, xmin = -3, xmax = skewnorm.ppf(0.999, a = a))
    plt.yticks([])
    plt.xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.ylim(plt.ylim()[0] - 0.05, plt.ylim()[1])
    plt.xlim(-1, 4.5);
