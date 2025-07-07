import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self, **kwargs):
        self.plot_style(**kwargs)

    def plot_style(self, **kwargs):
        plt.rcParams['figure.figsize'] = kwargs.get('figsize', [10, 5])
        plt.rcParams['figure.dpi'] = kwargs.get('dpi', 100)
        plt.rcParams['font.size'] = kwargs.get('font_size', 14)
        plt.rcParams['font.family'] = kwargs.get('font_family', 'Arial')
        plt.rcParams['axes.grid'] = kwargs.get('grid', True)
        plt.rcParams['axes.labelsize'] = kwargs.get('label_size', 14)
        plt.rcParams['axes.titlesize'] = kwargs.get('title_size', 16)
        plt.rcParams['axes.titleweight'] = kwargs.get('title_weight', 'bold')
        plt.rcParams['axes.titlepad'] = kwargs.get('title_pad', 10)
        plt.rcParams['axes.titlecolor'] = kwargs.get('title_color', 'black')
        plt.rcParams['lines.linewidth'] = kwargs.get('line_width', 2)
        plt.rcParams['axes.linewidth'] = kwargs.get('axes_width', 3)
        plt.rcParams['xtick.major.width'] = kwargs.get('xtick_width', 3)
        plt.rcParams['ytick.major.width'] = kwargs.get('ytick_width', 3)
        plt.rcParams['xtick.major.size'] = kwargs.get('xtick_size', 10)
        plt.rcParams['ytick.major.size'] = kwargs.get('ytick_size', 10)
        plt.rcParams['xtick.minor.width'] = kwargs.get('xtick_minor_width', 1)
        plt.rcParams['ytick.minor.width'] = kwargs.get('ytick_minor_width', 1)
        plt.rcParams['xtick.minor.size'] = kwargs.get('xtick_minor_size', 5)
        plt.rcParams['ytick.minor.size'] = kwargs.get('ytick_minor_size', 5)
        plt.rcParams['xtick.major.pad'] = kwargs.get('xtick_major_pad', 10)
        plt.rcParams['ytick.major.pad'] = kwargs.get('ytick_major_pad', 10)
        plt.rcParams['xtick.minor.pad'] = kwargs.get('xtick_minor_pad', 5)
        plt.rcParams['ytick.minor.pad'] = kwargs.get('ytick_minor_pad', 5)

    def plot_data(self, y: np.ndarray, x: np.ndarray = None, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots()

        if x is None:
            x = np.arange(y.shape[0])
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        ax.plot(x, y)
        ax.grid(True)
        if kwargs.get('xlabel', None) is not None:
            ax.set_xlabel(kwargs.get('xlabel', 'Time'))
        if kwargs.get('ylabel', None) is not None:
            ax.set_ylabel(kwargs.get('ylabel', 'Value'))
        if kwargs.get('title', None) is not None:
            ax.set_title(kwargs.get('title', 'Data'))
        if kwargs.get('legend', None) is not None:
            ax.legend(kwargs.get('legend', []))
        if kwargs.get('xlim', None) is not None:
            ax.set_xlim(kwargs.get('xlim', None))
        if kwargs.get('ylim', None) is not None:
            ax.set_ylim(kwargs.get('ylim', None))
        if kwargs.get('xticklabels', None) is not None:
            ax.set_xticklabels(kwargs.get('xticklabels', None))
        if kwargs.get('yticklabels', None) is not None:
            ax.set_yticklabels(kwargs.get('yticklabels', None))
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        if kwargs.get('savefig', None) is not None:
            fig.savefig(kwargs.get('savefig', None))
        plt.show()
        return fig, ax
    
    def data_plot(self, data: dict, wavelength_idx: int, **kwargs):
        for folder_name in data:
            min_length = min(len(data[folder_name][channel]) for channel in data[folder_name])
            temp = np.empty((min_length,8))
            for channel in data[folder_name]:
                temp[:,channel-1] = data[folder_name][channel][:min_length,wavelength_idx]
            legend = np.arange(1,9)
            self.plot_data(temp, legend=legend, title=folder_name, **kwargs)
    
    def plot_ratio(self, data: dict, wavelength_idx_1: int, wavelength_idx_2: int, **kwargs):
        for folder_name in data:
            min_length = min(len(data[folder_name][channel]) for channel in data[folder_name])
            temp = np.empty((min_length,8))
            for channel in data[folder_name]:
                temp[:,channel-1] = data[folder_name][channel][:min_length,wavelength_idx_1] / data[folder_name][channel][:min_length,wavelength_idx_2]
            legend = np.arange(1,9)
            self.plot_data(temp, legend=legend, title=folder_name, **kwargs)