from matplotlib import pyplot as plt

def plot_signal(signal_data, title = None):
    """
    Plot the generated signal.
    
    Parameters:
    signal_data (xarray.DataArray): The generated signal to plot.
    title (str): Title for the plot.
    """

    plt.figure(figsize=(10, 6))
    if signal_data is not None:
        plt.plot(signal_data.time, signal_data.values, label='Generated Signal')
        plt.xlabel('Time ({})'.format(signal_data.attrs['unit']))
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()