import numpy as np
import os
import matplotlib.pyplot as plt

def read_and_plot_txt(file_path):
    """
    Reads a text file containing numerical data, converts it to a numpy array,
    and plots the data as a vector.

    Parameters:
        file_path (str): Path to the text file.

    Returns:
        None
    """
    try:
        # Read the text file and convert to numpy array
        data = np.loadtxt(file_path, delimiter=',')
        kernel = file_path.split('_kernel')[-1].split('_')[0]
        images = file_path.split('_img')[-1].split('_')[0]
        # Plot each column as a separate graph
        plt.figure(figsize=(15, 10))
        # for i in range(1):
        #     k = data.shape[1]
        #     plot_data = data[int(i*k):int((i+1)*k), :]
        for j in range(data.shape[0]//1000):
            # plt.subplot(2, 1, i + 1)
            plt.plot(data[j,:], marker='o')
            plt.title('Correlation for {} images and {}x{} kernel'.format(images, kernel, kernel))
            plt.xlabel("Index dz")
            plt.ylabel(f"Correlation")
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(file_path.replace('.txt', '.png'), dpi=300)
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

def read_and_plot_correlation_data(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    for n_img, kernel_data in data.items():
        if len(kernel_data) == 1:
            axes = [axes]  # Ensure axes is iterable for a single subplot
        for ax, (kernel, corr_data) in zip(axes, kernel_data.items()):
            for i in range(len(corr_data)):
                x_points = np.linspace(190, 250, len(corr_data[i]))
                ax.plot(x_points, corr_data[i] * 100, marker='o', label='{} imgs'.format(n_img))
            ax.set_title(f'Correlation for {kernel}x{kernel} kernel', pad=20)
            ax.set_xlabel("Tested Z points [mm]")
            ax.set_ylabel("Correlation [%]")
        # if ax is axes[0]:
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(file_path.replace('correl.npy', '{}.png').format(file_path.split('correl')[0].split('/')[-1]), dpi=300)
    plt.show()


def read_and_plot_correlation_data_filter(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    filter_n_img = [1,2]  # exemplo: coloque aqui os valores desejados de n_img

    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    for n_img, kernel_data in data.items():
        if int(n_img) in filter_n_img:
            continue
        if len(kernel_data) == 1:
            axes = [axes]  # Garante iterabilidade para um único subplot
        for ax, (kernel, corr_data) in zip(axes, kernel_data.items()):
            for i in range(len(corr_data)):
                x_points = np.linspace(190, 250, len(corr_data[i]))
                ax.plot(x_points, corr_data[i] * 100, marker='o', label='{} imgs'.format(n_img))
            ax.set_title(f'Correlation for {kernel}x{kernel} kernel', pad=20)
            ax.set_xlabel("Tested Z points [mm]")
            ax.set_ylabel("Correlation [%]")
            ax.grid(True)
    ax.legend()
    plt.tight_layout()
    out_name = file_path.split('correl')[0].split('/')[-1]
    plt.savefig(file_path.replace('correl.npy', '{}.png').format(out_na), dpi=300)
    plt.show()

def read_and_plot_correlation_data_per_kernel(file_path):
    """
    Plots correlation data from a .npy file, saving one figure per kernel size.
    """
    data = np.load(file_path, allow_pickle=True).item()
    filter_n_img = []  # exemplo: coloque aqui os valores desejados de n_img

    # Descobrir todos os kernels disponíveis
    all_kernels = set()
    for kernel_data in data.values():
        all_kernels.update(kernel_data.keys())

    for kernel in all_kernels:
        plt.figure(figsize=(10, 6))
        for n_img, kernel_data in data.items():
            if int(n_img) in filter_n_img:
                continue
            if kernel not in kernel_data:
                continue
            corr_data = kernel_data[kernel]
            for i in range(len(corr_data)):
                x_points = np.linspace(190, 250, len(corr_data[i]))
                plt.plot(x_points, corr_data[i] * 100, marker='o', label=f'{n_img} imgs' if i == 0 else "")
        plt.title(f'Correlation for {kernel}x{kernel} kernel')
        plt.xlabel("Tested Z points [mm]")
        plt.ylabel("Correlation [%]")
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.tight_layout()
        out_name = file_path.split('correl')[0].split('/')[-1]
        plt.savefig(file_path.replace('correl.npy', f'{out_name}_kernel{kernel}.png'), dpi=300)
        plt.show()



path = '/home/daniel/Documents/stereo_zscan/20250515-20250513_step10_plano_d2-correl'
# files = sorted(os.listdir('/home/daniel/Documents/stereo_zscan/teste'))
# for file in files:
#     print(file)
#     read_and_plot_txt(os.path.join('/home/daniel/Documents/stereo_zscan/teste', file))
# read_and_plot_txt("path_to_your_file.txt")
# read_and_plot_correlation_data(os.path.join(path, 'correl.npy'))
# read_and_plot_correlation_data_filter(os.path.join(path, 'correl.npy'))
read_and_plot_correlation_data_per_kernel(os.path.join(path, 'correl.npy'))
# read_and_plot_txt('teste/correlation_img5_kernel3_.txt')