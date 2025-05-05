import numpy as np
import matplotlib.pyplot as plt
import os

def process_and_plot_txt(files):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for i, file in enumerate(files):
        # Read the file and convert to numpy array
        data = np.loadtxt(file, delimiter='\t')
        
        # Plot data
        axes[i].imshow(data)
        axes[i].set_title(f'Graph {i+1}: {file}')
        
        # Save back to txt with tab separator
        # output_file = file.replace('.txt', '_processed.txt')
        # np.savetxt(output_file, data, delimiter='\t', fmt='%.6f')
    
    plt.tight_layout()
    plt.show()

# Example usage
path = '/home/daniel/Documents/Luan/Exemplo medição - Bancada Luan/300x300'
files = os.listdir(path)
files = [os.path.join(path, file) for file in files if file.endswith('.txt')]
process_and_plot_txt(files)
