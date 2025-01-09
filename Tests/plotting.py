import matplotlib.pyplot as plt
import numpy as np


def plot_percentage_below_thresholds(rotation_errors, translation_errors, thresholds, save_path=None):
    """
    Plot the percentage of errors below different thresholds.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Percentage of errors below different thresholds')

    for i, (errors, title) in enumerate(zip([rotation_errors, translation_errors], ['Rotation', 'Translation'])):
        ax[i].set_title(f'{title} errors')
        ax[i].set_xlabel('Threshold')
        ax[i].set_ylabel('Percentage of errors')
        ax[i].set_xticks(thresholds)
        ax[i].set_ylim(0, 100)

        for threshold in thresholds:
            percentage = np.sum(errors < threshold) / len(errors) * 100
            ax[i].bar(threshold, percentage, width=0.1, color='blue')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()