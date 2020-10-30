import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import t
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
from matplotlib.patches import Rectangle

def range_plot(x, **kwargs):
    center = np.mean([np.min(x), np.max(x)])
    
    y = plt.hist(x, **kwargs)[0].max()
    y_range = plt.ylim()[1] - plt.ylim()[0]
    x_range = plt.xlim()[1] - plt.xlim()[0]
    
    plt.hlines(y = 0, xmin = np.min(x) - 0.01*x_range, xmax = np.max(x) + 0.01*x_range)
    plt.annotate(s = 'Range', ha = 'center', va = 'bottom', fontweight = 'bold',
                 xy = (center, y + 0.05*y_range), fontsize = 14)
    plt.plot([np.min(x), np.max(x)], [y + 0.05*y_range, y + 0.05*y_range], linewidth = 3, color = 'black')
    plt.plot([np.min(x), np.min(x)], [y+.02*y_range, y + .08*y_range], linewidth = 3, color = 'black')
    plt.plot([np.max(x), np.max(x)], [y+.02*y_range, y + .08*y_range], linewidth = 3, color = 'black')

    plt.ylim(-.1*y_range, y + 0.15*y_range);

def std_plot(x, **kwargs):
    mu = np.mean(x)
    sigma = np.std(x)

    y = plt.hist(x, **kwargs)[0].max()
    y_range = plt.ylim()[1] - plt.ylim()[0]
    x_range = plt.xlim()[1] - plt.xlim()[0]

    plt.hlines(y = 0, xmin = np.min(x) - 0.01*x_range, xmax = np.max(x) + 0.01*x_range)
    plt.annotate(s = 'Mean', ha = 'center', va = 'top', fontweight = 'bold',
                 xy = (np.mean(x), -.01*y_range), xytext = (np.mean(x), -.15*y_range), arrowprops=dict(width = 8, headwidth = 20, facecolor = 'red'))

    plt.annotate(s = '$\sigma$', ha = 'center', va = 'bottom', fontweight = 'bold',
                  xy = (mu - sigma / 2, y + 0.05*y_range), fontsize = 14)
    plt.annotate(s = '$\sigma$', ha = 'center', va = 'bottom', fontweight = 'bold',
                  xy = (mu + sigma / 2, y + 0.05*y_range), fontsize = 14)
    plt.plot([mu - sigma, mu], [y + 0.05*y_range, y + 0.05*y_range], linewidth = 3, color = 'black')
    plt.plot([mu, mu +sigma], [y + 0.05*y_range, y + 0.05*y_range], linewidth = 3, color = 'black')

    plt.plot([mu - sigma, mu - sigma], [y+.02*y_range, y+.08*y_range], linewidth = 3, color = 'black')
    plt.plot([mu, mu], [y+.02*y_range, y+.08*y_range], linewidth = 3, color = 'black')
    plt.plot([mu + sigma, mu + sigma], [y+.02*y_range, y+.08*y_range], linewidth = 3, color = 'black')

    plt.ylim(-.22*y_range, y + 0.15*y_range);


def iqr_plot(x, **kwargs):
    lq = np.quantile(x, 0.25)
    uq = np.quantile(x, 0.75)
    med = np.quantile(x, 0.5)

    y = plt.hist(x, **kwargs)[0].max()
    y_range = plt.ylim()[1] - plt.ylim()[0]
    x_range = plt.xlim()[1] - plt.xlim()[0]

    plt.hlines(y = 0, xmin = np.min(x) - 0.01*x_range, xmax = np.max(x) + 0.01*x_range)
    plt.annotate(s = 'Median', ha = 'center', va = 'top', fontweight = 'bold',
                 xy = (med, -.01*y_range), xytext = (med, -.15*y_range), arrowprops=dict(width = 8, headwidth = 20, facecolor = 'blue'))

    plt.annotate(s = 'IQR', ha = 'center', va = 'bottom', fontweight = 'bold',
                  xy = ((lq + uq) / 2, y + 0.05*y_range), fontsize = 14)
    
    plt.plot([lq, uq], [y + 0.05*y_range, y + 0.05*y_range], linewidth = 3, color = 'black')
    
    plt.plot([lq, lq], [y+.02*y_range, y+.08*y_range], linewidth = 3, color = 'black')
    plt.plot([uq, uq], [y+.02*y_range, y+.08*y_range], linewidth = 3, color = 'black')

    plt.ylim(-.22*y_range, y + 0.15*y_range);

def qq_plot(data):
    mu = np.mean(data)
    sigma = np.std(data)
    
    sm.qqplot(data, line='45', loc = mu, scale = sigma);

def hypot_plot_mean(data, popmean, type = 'both', area = True):
    
    df = len(data) - 1
    
    test_stat, p = ttest_1samp(data, popmean = popmean)
    
    if type != 'both':
        p = p/2
    
    p = round(p,4)
    
    x_min = min(-np.abs(test_stat) - 0.25, -3)
    x_max = max(np.abs(test_stat) + 0.25, 3)
    
    
    x = np.linspace(x_min, x_max, 100)
    
    fig, ax = plt.subplots(1, 1, figsize = (8,4))
    ax.plot(x, t.pdf(x, df = df),'r-', lw=3, alpha=1, label='norm pdf', color = 'black')
    
    if type == 'both':
        left_section = np.linspace(x_min, -np.abs(test_stat), 100)
        right_section = np.linspace(np.abs(test_stat), x_max, 100)
        sections = [left_section, right_section]
        edges = [-np.abs(test_stat), np.abs(test_stat)]
        
    elif type == 'left':
        left_section = np.linspace(x_min, test_stat, 100)
        sections = [left_section]
        edges = [test_stat]

    elif type == 'right':
        right_section = np.linspace(test_stat, x_max, 100)
        sections = [right_section]
        edges = [test_stat]
    
    if area:
        for section in sections:
            plt.fill_between(section, t.pdf(section, df = df), color = 'red')
    for edge in edges:
        plt.vlines(x = edge, ymin =0, ymax = t.pdf(edge, df = df), lw = 3, color = 'black')
        plt.annotate(s = np.round(edge,4), xy = (edge, -0.01), fontsize = 14, fontweight = 'bold', 
                     va = 'top', ha = 'center')
        
    plt.annotate(s = 'Test\n Statistic', ha = 'center', va = 'top', fontweight = 'bold',
             xy = (test_stat, -.05), xytext = (test_stat, -.1), arrowprops=dict(width = 8, headwidth = 20, facecolor = 'red'))
    
    if area:
        area = p
        if type == 'both':
            #area = round(2*t.cdf(-np.abs(test_stat), df = n-1), 4)

            test_stat = np.abs(test_stat)

            plt.annotate(s = 'Area = {}'.format(area), ha = 'center', fontweight = 'bold', fontsize = 14,
                    xy = ((test_stat + x_max) / 2, t.pdf((test_stat + x_max)/2, df = df) + 0.01),
                    xytext = ((test_stat + x_max) / 2, t.pdf((test_stat + x_max)/2, df = df) + 0.2),
                    arrowprops=dict(width = 4, headwidth = 8, facecolor = 'black'))

            plt.annotate(s = 'Area = {}'.format(area), ha = 'center', fontweight = 'bold', fontsize = 14,
                    xy = (-test_stat + 0.1, t.pdf(test_stat, df = df)/2),
                    xytext = ((test_stat + x_max) / 2, t.pdf((test_stat + x_max)/2, df = df) + 0.2),
                    arrowprops=dict(width = 4, headwidth = 8, facecolor = 'black'))

        if type == 'right':
            #area = round(t.cdf(-test_stat, df = n-1), 4)

            plt.annotate(s = 'Area = {}'.format(area), ha = 'center', fontweight = 'bold', fontsize = 14,
                    xy = ((test_stat + x_max) / 2, t.pdf((test_stat + x_max)/2, df = df) + 0.01),
                    xytext = ((test_stat + x_max) / 2, t.pdf((test_stat + x_max)/2, df = df) + 0.2),
                    arrowprops=dict(width = 4, headwidth = 8, facecolor = 'black'))

        if type == 'left':
            #area = round(t.cdf(test_stat, df = n-1), 4)

            plt.annotate(s = 'Area = {}'.format(area), ha = 'center', fontweight = 'bold', fontsize = 14,
                    xy = ((test_stat + x_min) / 2, t.pdf((test_stat + x_min)/2, df = df) + 0.01),
                    xytext = ((test_stat + x_min) / 2, t.pdf((test_stat + x_min)/2, df = df) + 0.2),
                    arrowprops=dict(width = 4, headwidth = 8, facecolor = 'black'))
    
    
    plt.hlines(y = 0, xmin = min(-3, -np.abs(test_stat + 0.25)), xmax = max(3, np.abs(test_stat + .25)))
    plt.yticks([])
    plt.xticks([])
    plt.annotate(s = '0', xy = (0, -0.01), fontsize = 14, fontweight = 'bold', 
                 va = 'top', ha = 'center')  
       
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.ylim(plt.ylim()[0] - .05, plt.ylim()[1])
    plt.tight_layout()
    plt.title('Sampling Distribution of the $\\frac{\\bar{x} - \\mu}{s / \\sqrt{n}}$, Assuming $H_0$', fontsize = 14);

def hypot_plot_mean_2sample(data1, data2, type = 'both', area = True):
    
    df = min(len(data1) - 1, len(data2) -1)
    
    test_stat, p = ttest_ind(data1, data2, equal_var = False)
    
    if type != 'both':
        p = p/2
    
    p = round(p,4)
    
    x_min = min(-np.abs(test_stat) - 0.25, -3)
    x_max = max(np.abs(test_stat) + 0.25, 3)
    
    
    x = np.linspace(x_min, x_max, 100)
    
    fig, ax = plt.subplots(1, 1, figsize = (8,4))
    ax.plot(x, t.pdf(x, df = df),'r-', lw=3, alpha=1, label='norm pdf', color = 'black')
    
    if type == 'both':
        left_section = np.linspace(x_min, -np.abs(test_stat), 100)
        right_section = np.linspace(np.abs(test_stat), x_max, 100)
        sections = [left_section, right_section]
        edges = [-np.abs(test_stat), np.abs(test_stat)]
        
    elif type == 'left':
        left_section = np.linspace(x_min, test_stat, 100)
        sections = [left_section]
        edges = [test_stat]

    elif type == 'right':
        right_section = np.linspace(test_stat, x_max, 100)
        sections = [right_section]
        edges = [test_stat]
    
    if area:
        for section in sections:
            plt.fill_between(section, t.pdf(section, df = df), color = 'red')
    for edge in edges:
        plt.vlines(x = edge, ymin =0, ymax = t.pdf(edge, df = df), lw = 3, color = 'black')
        plt.annotate(s = np.round(edge,4), xy = (edge, -0.01), fontsize = 14, fontweight = 'bold', 
                     va = 'top', ha = 'center')
        
    plt.annotate(s = 'Test\n Statistic', ha = 'center', va = 'top', fontweight = 'bold',
             xy = (test_stat, -.05), xytext = (test_stat, -.1), arrowprops=dict(width = 8, headwidth = 20, facecolor = 'red'))
    
    if area:
        area = p
        if type == 'both':
            #area = round(2*t.cdf(-np.abs(test_stat), df = n-1), 4)

            test_stat = np.abs(test_stat)

            plt.annotate(s = 'Area = {}'.format(area), ha = 'center', fontweight = 'bold', fontsize = 14,
                    xy = ((test_stat + x_max) / 2, t.pdf((test_stat + x_max)/2, df = df) + 0.01),
                    xytext = ((test_stat + x_max) / 2, t.pdf((test_stat + x_max)/2, df = df) + 0.2),
                    arrowprops=dict(width = 4, headwidth = 8, facecolor = 'black'))

            plt.annotate(s = 'Area = {}'.format(area), ha = 'center', fontweight = 'bold', fontsize = 14,
                    xy = (-test_stat + 0.1, t.pdf(test_stat, df = df)/2),
                    xytext = ((test_stat + x_max) / 2, t.pdf((test_stat + x_max)/2, df = df) + 0.2),
                    arrowprops=dict(width = 4, headwidth = 8, facecolor = 'black'))

        if type == 'right':
            #area = round(t.cdf(-test_stat, df = n-1), 4)

            plt.annotate(s = 'Area = {}'.format(area), ha = 'center', fontweight = 'bold', fontsize = 14,
                    xy = ((test_stat + x_max) / 2, t.pdf((test_stat + x_max)/2, df = df) + 0.01),
                    xytext = ((test_stat + x_max) / 2, t.pdf((test_stat + x_max)/2, df = df) + 0.2),
                    arrowprops=dict(width = 4, headwidth = 8, facecolor = 'black'))

        if type == 'left':
            #area = round(t.cdf(test_stat, df = n-1), 4)

            plt.annotate(s = 'Area = {}'.format(area), ha = 'center', fontweight = 'bold', fontsize = 14,
                    xy = ((test_stat + x_min) / 2, t.pdf((test_stat + x_min)/2, df = df) + 0.01),
                    xytext = ((test_stat + x_min) / 2, t.pdf((test_stat + x_min)/2, df = df) + 0.2),
                    arrowprops=dict(width = 4, headwidth = 8, facecolor = 'black'))
    
    
    plt.hlines(y = 0, xmin = min(-3, -np.abs(test_stat + 0.25)), xmax = max(3, np.abs(test_stat + .25)))
    plt.yticks([])
    plt.xticks([])
    plt.annotate(s = '0', xy = (0, -0.01), fontsize = 14, fontweight = 'bold', 
                 va = 'top', ha = 'center')  
       
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.ylim(plt.ylim()[0] - .05, plt.ylim()[1])
    plt.tight_layout()
    plt.title('Sampling Distribution of the $\\frac{\\bar{x}_1 - \\bar{x}_2}{s}$, Assuming $H_0$', fontsize = 14);


def hypot_plot_proportion_2sample(counts, nobs, alternative = 'two-sided', area = True):
    
    if alternative == 'two-sided':
        type = 'both'
    elif alternative == 'smaller':
        type = 'left'
    elif alternative == 'larger':
        type = 'right'
    
    test_stat, p = proportions_ztest(counts, nobs, alternative = alternative)
    
    p = round(p,4)
    
    x_min = min(-np.abs(test_stat) - 0.25, -3)
    x_max = max(np.abs(test_stat) + 0.25, 3)
    
    x = np.linspace(x_min, x_max, 100)
    
    fig, ax = plt.subplots(1, 1, figsize = (8,4))
    ax.plot(x, norm.pdf(x),'r-', lw=3, alpha=1, label='norm pdf', color = 'black')
    
    if type == 'both':
        left_section = np.linspace(x_min, -np.abs(test_stat), 100)
        right_section = np.linspace(np.abs(test_stat), x_max, 100)
        sections = [left_section, right_section]
        edges = [-np.abs(test_stat), np.abs(test_stat)]
        
    elif type == 'left':
        left_section = np.linspace(x_min, test_stat, 100)
        sections = [left_section]
        edges = [test_stat]

    elif type == 'right':
        right_section = np.linspace(test_stat, x_max, 100)
        sections = [right_section]
        edges = [test_stat]
    
    if area:
        for section in sections:
            plt.fill_between(section, norm.pdf(section), color = 'red')
    for edge in edges:
        plt.vlines(x = edge, ymin =0, ymax = norm.pdf(edge), lw = 3, color = 'black')
        plt.annotate(s = np.round(edge,4), xy = (edge, -0.01), fontsize = 14, fontweight = 'bold', 
                     va = 'top', ha = 'center')
        
    plt.annotate(s = 'Test\n Statistic', ha = 'center', va = 'top', fontweight = 'bold',
             xy = (test_stat, -.05), xytext = (test_stat, -.1), arrowprops=dict(width = 8, headwidth = 20, facecolor = 'red'))
    
    if area:
        area = p
        if type == 'both':
            #area = round(2*t.cdf(-np.abs(test_stat), df = n-1), 4)

            test_stat = np.abs(test_stat)

            plt.annotate(s = 'Area = {}'.format(area), ha = 'center', fontweight = 'bold', fontsize = 14,
                    xy = ((test_stat + x_max) / 2, norm.pdf((test_stat + x_max)/2) + 0.01),
                    xytext = ((test_stat + x_max) / 2, norm.pdf((test_stat + x_max)/2) + 0.2),
                    arrowprops=dict(width = 4, headwidth = 8, facecolor = 'black'))

            plt.annotate(s = 'Area = {}'.format(area), ha = 'center', fontweight = 'bold', fontsize = 14,
                    xy = (-test_stat + 0.1, norm.pdf(test_stat)/2),
                    xytext = ((test_stat + x_max) / 2, norm.pdf((test_stat + x_max)/2) + 0.2),
                    arrowprops=dict(width = 4, headwidth = 8, facecolor = 'black'))

        if type == 'right':
            #area = round(t.cdf(-test_stat, df = n-1), 4)

            plt.annotate(s = 'Area = {}'.format(area), ha = 'center', fontweight = 'bold', fontsize = 14,
                    xy = ((test_stat + x_max) / 2, norm.pdf((test_stat + x_max)/2) + 0.01),
                    xytext = ((test_stat + x_max) / 2, norm.pdf((test_stat + x_max)/2) + 0.2),
                    arrowprops=dict(width = 4, headwidth = 8, facecolor = 'black'))

        if type == 'left':
            #area = round(t.cdf(test_stat, df = n-1), 4)

            plt.annotate(s = 'Area = {}'.format(area), ha = 'center', fontweight = 'bold', fontsize = 14,
                    xy = ((test_stat + x_min) / 2, norm.pdf((test_stat + x_min)/2) + 0.01),
                    xytext = ((test_stat + x_min) / 2, norm.pdf((test_stat + x_min)/2) + 0.2),
                    arrowprops=dict(width = 4, headwidth = 8, facecolor = 'black'))
    
    
    plt.hlines(y = 0, xmin = min(-3, -np.abs(test_stat + 0.25)), xmax = max(3, np.abs(test_stat + .25)))
    plt.yticks([])
    plt.xticks([])
    plt.annotate(s = '0', xy = (0, -0.01), fontsize = 14, fontweight = 'bold', 
                 va = 'top', ha = 'center')  
       
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.ylim(plt.ylim()[0] - .05, plt.ylim()[1])
    plt.tight_layout()
    plt.title('Sampling Distribution of the $\\frac{\\bar{p}_1 - \\bar{p}_2}{s_p}$, Assuming $H_0$', fontsize = 14);

def deviation_plot(Player, df):
    
    i = df.player.tolist().index(Player)
    
    salary = df.loc[i,'salary']
    mean = df['salary'].mean()

    print('Player: {}'.format(df.loc[i, 'player']))
    print('Salary: ${:,}'.format(df.loc[i, 'salary']))
    print('Mean Salary: ${:,}'.format(int(mean)))
    
    fig, ax = plt.subplots(figsize = (10,3))
    plt.scatter(df['salary'], [0 + 0.015]*len(df['salary']), edgecolor = 'black', s = 100)
    xmin, xmax = plt.xlim()
    
    plt.scatter([salary], [0 + 0.015], edgecolor = 'black', color = 'orange', s = 100)
    plt.annotate(s = 'Mean ($\mu$) \n \${:,}'.format(int(mean)), ha = 'center', va = 'top', fontweight = 'bold', fontsize = 12,
                 xy = (df['salary'].mean(), -.01), xytext = (df['salary'].mean(), -.06),
                 arrowprops=dict(width = 8, headwidth = 20, facecolor = 'red'))

    lineheight = 0.1
    
    plt.plot([mean, salary], [lineheight,lineheight], color = 'black', linewidth = 3)
    for x in [mean, salary]:
        plt.plot([x,x], [lineheight - 0.01, lineheight + 0.01], color = 'black', linewidth = 3)

    plt.annotate(s = '$x_i$ = \${:,}'.format(salary), xy = (salary, 0.035), fontsize = 14, fontweight = 'bold',
                ha = 'center', va = 'bottom')
    
    if mean > salary: color = 'red'
    else: color = 'blue'    
    plt.annotate(s = '$x_i - \mu =$ \${:,}'.format(int(salary - mean)),
                 xy = ((mean + salary) / 2, lineheight + 0.025),
                 fontsize = 14, fontweight = 'bold', ha = 'center', va = 'bottom', color = color)

    xmin -= 4000000
    xmax += 4000000
    plt.hlines(y = 0, xmin = xmin, xmax = xmax)
    plt.xlim(xmin, xmax)
    plt.ylim(-0.15, 0.2)
    plt.yticks([]);

def calibration_curve(y_true, y_prob, n_bins=5,
                      strategy='quantile', alpha = 0.05,
                     figsize = (6,4), original_data = True):
    """Compute true and predicted probabilities for a calibration curve.
    The method assumes the inputs come from a binary classifier.
    Calibration curves may also be referred to as reliability diagrams.
    Read more in the :ref:`User Guide <calibration>`.
    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.
    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.
    normalize : bool, optional, default=False
        Whether y_prob needs to be normalized into the bin [0, 1], i.e. is not
        a proper probability. If True, the smallest value in y_prob is mapped
        onto 0 and the largest one onto 1.
    n_bins : int
        Number of bins. A bigger number requires more data. Bins with no data
        points (i.e. without corresponding values in y_prob) will not be
        returned, thus there may be fewer than n_bins in the return value.
    strategy : {'uniform', 'quantile'}, (default='quantile')
        Strategy used to define the widths of the bins.
        uniform
            All bins have identical widths.
        quantile
            All bins have the same number of points.
    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).
    """
    if strategy == 'quantile':  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif strategy == 'uniform':
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    else:
        raise ValueError("Invalid entry to 'strategy' input. Strategy "
                         "must be either 'quantile' or 'uniform'.")

    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    lower_bound = np.zeros_like(bin_total).astype('float')
    upper_bound = np.zeros_like(bin_total).astype('float')
    
    for i, (count, nobs) in enumerate(zip(bin_true, bin_total)):
        if nobs != 0:
            lower_bound[i], upper_bound[i] = proportion_confint(count = count, nobs = nobs, alpha = alpha)
        else:
            pass
        

    nonzero = bin_total != 0
    prob_true = (bin_true[nonzero] / bin_total[nonzero])
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])
    
    lower_bound = lower_bound[nonzero]
    upper_bound = upper_bound[nonzero]
    
    fig, ax = plt.subplots(figsize = figsize)
    
    plt.scatter(prob_pred, prob_true, edgecolor = 'black')
    # Add the confidence intervals:
    for lb, ub, x in zip(lower_bound, upper_bound, prob_pred):
        plt.plot([x, x], [lb, ub], color = 'black')
    
    # Include the original data
    if original_data:
        plt.scatter(y_prob, 1.1*y_true - 0.05, alpha = 0.5, color = 'red', zorder = -5)
    
    plt.plot([-0.05, 1.05], [-0.05, 1.05], color = 'black', linestyle = '--')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Truth')
    plt.title('Calibration Curve')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

def predicted_probability_plot(y_true, y_proba):
    alpha = 0.6

    fig, ax = plt.subplots(figsize = (6,2))
    plt.scatter(y_proba[y_true==0], (np.zeros_like(y_true) - 0.05 + 0.1*(y_true == 1))[y_true==0], 
                c = 'blue', label = 'non-carrier', 
                alpha = alpha, edgecolor = 'black')
    plt.scatter(y_proba[y_true==1], (np.zeros_like(y_true) - 0.05 + 0.1*(y_true == 1))[y_true==1], 
                c = 'red', label = 'carrier', 
                alpha = alpha, edgecolor = 'black')

    plt.plot([-.0,1.], [0,0], linewidth = 3, color = 'black')


    for i in [0,1]:
        plt.plot([i,i], [0.1, -0.1], linewidth = 3, color = 'black')
        plt.annotate(s = str(i), xy = (i, -0.125), ha = 'center', va = 'top', fontsize = 12, fontweight = 'bold')

    plt.title("Predicted Probabilities")

    plt.yticks([])

    plt.legend()
    plt.ylim(-0.5, 0.5);

def quadrant_plot(x, y, quadrant = None, figsize = (8,6), labels = None):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    fig, ax = plt.subplots(figsize = figsize)
    plt.scatter(x = x, y = y, zorder = 500, color = 'black', alpha = 0.7)
    plt.axvline(x = x_mean, color = 'black')
    plt.axhline(y = y_mean, color = 'black')
    if labels:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    
    if quadrant in [1,2,3,4]:
        color = 'blue'
        if quadrant == 1:
           r1 = Rectangle((x_mean, y_mean), plt.xlim()[1] - x_mean, plt.ylim()[1] - y_mean, color=color, alpha = 0.5) 
        elif quadrant == 2:
           r1 = Rectangle((x_mean, plt.ylim()[0]), plt.xlim()[1] - x_mean, y_mean - plt.ylim()[0], color=color, alpha = 0.5) 
        elif quadrant == 3:
           r1 = Rectangle((plt.xlim()[0], plt.ylim()[0]), x_mean - plt.xlim()[0], y_mean - plt.ylim()[0], color=color, alpha = 0.5) 
        elif quadrant == 4:
            r1 = Rectangle((plt.xlim()[0], y_mean), x_mean - plt.xlim()[0], plt.ylim()[1] - y_mean, color=color, alpha = 0.5)
        ax.add_artist(r1);

def half_plot(x, y, half = None, figsize = (8,6), labels = None):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    fig, ax = plt.subplots(figsize = figsize)
    plt.scatter(x = x, y = y, zorder = 500, color = 'black', alpha = 0.7)
    plt.axvline(x = x_mean, color = 'black')
    plt.axhline(y = y_mean, color = 'black')
    if labels:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    
    if half in ['left', 'right']:
        if half == 'left':
            r1 = Rectangle((plt.xlim()[0], plt.ylim()[0]), x_mean - plt.xlim()[0], plt.ylim()[1] - plt.ylim()[0], color='lightblue', alpha = 0.5)
        elif half == 'right':
            r1 = Rectangle((x_mean, plt.ylim()[0]), plt.xlim()[1] - x_mean, plt.ylim()[1] - plt.ylim()[0], color='lightblue', alpha = 0.5) 
        ax.add_artist(r1); 
