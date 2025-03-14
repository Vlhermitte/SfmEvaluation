import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def plot_vram_usage(df, save_path=None):
    # Plot the memory usage using elapsed seconds for the x-axis
    plt.figure(figsize=(10, 5))
    plt.plot(df['elapsed_seconds'], df['memory.used [MiB]'], label='Used')

    # Set the y-axis limit with evenly spaced ticks
    max_value = df['memory.total [MiB]'].max()
    ticks = np.linspace(0, max_value, 10)  # 10 evenly spaced ticks
    plt.yticks(ticks)
    plt.ylim(0, max_value)

    plt.xlabel('Elapsed Time (seconds)')
    plt.ylabel('Memory (MB)')
    plt.title('Memory Usage')
    plt.legend()
    if save_path:
        plt.savefig(save_path / 'vram_usage.png')
    else:
        plt.show()

def read_log(path: Path):
    df = pd.read_csv(path, sep=', ')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S.%f')
    # remove MiB from the columns and convert to float
    df['memory.total [MiB]'] = df['memory.total [MiB]'].str.replace('MiB', '').astype(float)
    df['memory.used [MiB]'] = df['memory.used [MiB]'].str.replace('MiB', '').astype(float)
    df['memory.free [MiB]'] = df['memory.free [MiB]'].str.replace('MiB', '').astype(float)

    # Create a new column for elapsed time in seconds from the first timestamp
    start_time = df['timestamp'].iloc[0]
    df['elapsed_seconds'] = (df['timestamp'] - start_time).dt.total_seconds().astype(int)

    return df

if __name__ == '__main__':
    # Read the vram_usage.log file
    path = Path('../../data/results/acezero/ETH3D/courtyard/')
    df = read_log(path / 'vram_usage.log')
    plot_vram_usage(df, save_path=path)
