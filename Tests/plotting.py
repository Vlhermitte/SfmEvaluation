import matplotlib.pyplot as plt
import numpy as np


def plot_error_distributions(results, save_path=None):
    """
    Plot histograms of rotation and translation errors.

    Args:
        results (dict): Dictionary containing lists of errors
            - relative_rotation_error: List of rotation errors in degrees
            - relative_translation_error: List of normalized translation errors
    """
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot rotation error histogram
    rotation_bins = np.linspace(0, max(results['relative_rotation_error']), 20)
    ax1.hist(results['relative_rotation_error'], bins=rotation_bins, color='blue', alpha=0.7)
    ax1.set_title('Distribution of Rotation Errors')
    ax1.set_xlabel('Rotation Error (degrees)')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3)

    # Add mean and median lines for rotation
    rotation_mean = np.mean(results['relative_rotation_error'])
    rotation_median = np.median(results['relative_rotation_error'])
    ax1.axvline(rotation_mean, color='red', linestyle='--', label=f'Mean: {rotation_mean:.2f}°')
    ax1.axvline(rotation_median, color='green', linestyle='--', label=f'Median: {rotation_median:.2f}°')
    ax1.legend()

    # Plot translation error histogram
    translation_bins = np.linspace(0, max(results['relative_translation_error']), 20)
    ax2.hist(results['relative_translation_error'], bins=translation_bins, color='orange', alpha=0.7)
    ax2.set_title('Distribution of Translation Errors')
    ax2.set_xlabel('Translation Error (normalized)')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)

    # Add mean and median lines for translation
    translation_mean = np.mean(results['relative_translation_error'])
    translation_median = np.median(results['relative_translation_error'])
    ax2.axvline(translation_mean, color='red', linestyle='--', label=f'Mean: {translation_mean:.2f}')
    ax2.axvline(translation_median, color='green', linestyle='--', label=f'Median: {translation_median:.2f}')
    ax2.legend()

    # Adjust layout and display
    plt.tight_layout()

    # Print summary statistics
    print("\nSummary Statistics:")
    print("Rotation Error (degrees):")
    print(f"  Mean: {rotation_mean:.3f}")
    print(f"  Median: {rotation_median:.3f}")
    print(f"  Std Dev: {np.std(results['relative_rotation_error']):.3f}")

    print("\nTranslation Error:")
    print(f"  Mean: {translation_mean:.3f}")
    print(f"  Median: {translation_median:.3f}")
    print(f"  Std Dev: {np.std(results['relative_translation_error']):.3f}")

    # Save the plot
    if save_path is not None:
        plt.savefig(f'{save_path}_error_distributions')
    plt.close()


def plot_cumulative_errors(results, save_path=None):
    """
    Plot cumulative distribution of rotation and translation errors.

    Args:
        results (dict): Dictionary containing lists of errors
            - relative_rotation_error: List of rotation errors in degrees
            - relative_translation_error: List of normalized translation errors
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Sort errors for cumulative plot
    rotation_sorted = np.sort(results['relative_rotation_error'])
    translation_sorted = np.sort(results['relative_translation_error'])
    y = np.arange(1, len(rotation_sorted) + 1) / len(rotation_sorted)

    # Plot cumulative rotation errors
    ax1.plot(rotation_sorted, y * 100, 'b-', linewidth=2)
    ax1.set_title('Cumulative Distribution of Rotation Errors')
    ax1.set_xlabel('Rotation Error (degrees)')
    ax1.set_ylabel('Percentage of Images (%)')
    ax1.grid(True, alpha=0.3)

    # Plot cumulative translation errors
    ax2.plot(translation_sorted, y, 'orange', linewidth=2)
    ax2.set_title('Cumulative Distribution of Translation Errors')
    ax2.set_xlabel('Translation Error (normalized)')
    ax2.set_ylabel('Percentage of Images (%)')
    ax2.grid(True, alpha=0.3)

    # Adjust layout and display
    plt.tight_layout()

    # Save the plot
    if save_path is not None:
        plt.savefig(f'{save_path}_cumulative_errors')
    plt.close()