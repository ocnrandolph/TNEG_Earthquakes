import numpy as np

def bootstrap_ci(values, conf_level = 0.95, num_resamples = 10000, statistic = np.mean, **kwargs):
	point_estimate = statistic(values, **kwargs)    
	margin = (1 - conf_level) / 2
	lower_index = int(num_resamples * margin)
	upper_index = int(num_resamples * (1-margin))

	resample_statistics = []
	for _ in range(num_resamples):
		resample = np.random.choice(values, len(values))
		resample_statistics.append(statistic(resample, **kwargs))

	top_quantile = np.quantile(resample_statistics, q = 1 - margin)
	bottom_quantile = np.quantile(resample_statistics, q = margin)

	return point_estimate - (top_quantile - point_estimate), point_estimate + (point_estimate - bottom_quantile) 
