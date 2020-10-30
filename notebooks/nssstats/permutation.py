import numpy as np
import matplotlib.pyplot as plt

def generate_permutations(values, label_count, num_permutations = 10000, statistic = np.mean, **kwargs):
    permutation_differences = [None] * num_permutations
    for i in range(num_permutations):
        np.random.shuffle(values)
        permutation_differences[i] = statistic(values[:label_count], **kwargs) - statistic(values[label_count:], **kwargs)
        
    return permutation_differences

def generate_permutations_correlation(A, B, num_permutations = 10000):
    permutation_differences = [None] * num_permutations
    for i in range(num_permutations):
        np.random.shuffle(B)
        permutation_differences[i] = np.corrcoef(A, B)[0,1]
        
    return permutation_differences

def permutation_test_p(permutation_differences, observed_difference, alternative = 'two-sided'):
    if alternative == 'larger':
        return len([x for x in permutation_differences if x >= observed_difference]) / len(permutation_differences)
        
    if alternative == 'smaller':
        return len([x for x in permutation_differences if x <= observed_difference]) / len(permutation_differences)
    
    if alternative == 'two-sided':
        return len([x for x in permutation_differences if x <= -np.abs(observed_difference) or x >= np.abs(observed_difference)]) / len(permutation_differences)

def permutation_test_plot(permutation_differences, observed_difference, alternative = 'two-sided'):
    N, bins, patches = plt.hist(permutation_differences, bins = 40)
    xmin, xmax = plt.xlim()
    ymax = plt.ylim()[1]
    
    if alternative == 'larger':
        for i, bin in enumerate(bins[:-1]):
            if bin >= observed_difference - 0.5*(bins[1] - bins[0]):
                patches[i].set_facecolor('red')
                
    if alternative == 'smaller':
        for i, bin in enumerate(bins[:-1]):
            if bin <= observed_difference - 0.5*(bins[1] - bins[0]):
                patches[i].set_facecolor('red')

    if alternative == 'two-sided':
        for i, bin in enumerate(bins[:-1]):
            if bin <= -np.abs(observed_difference) - 0.5*(bins[1] - bins[0]):
                patches[i].set_facecolor('red')
            if bin >= np.abs(observed_difference) - 0.5*(bins[1] - bins[0]):
                patches[i].set_facecolor('red')
        plt.vlines(x = -observed_difference, color = 'black', ymin = 0, ymax = ymax)
                        
    plt.vlines(x = observed_difference, color = 'black', ymin = 0, ymax = ymax)
    plt.hlines(y=0, xmin = xmin, xmax = xmax)

    plt.annotate(s = 'Observed\n Difference', xy = (observed_difference, 0), 
                 xytext = (observed_difference,-0.12*ymax), ha = 'center', va = 'top',
                 arrowprops=dict(width = 8, headwidth = 20, facecolor = 'red'))
    plt.yticks([x for x in plt.yticks()[0] if x >= 0])
    plt.ylim(-0.25*ymax, ymax)
    plt.title('Distribution of Permutation Statistics');
