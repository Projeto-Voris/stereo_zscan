import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

class PlotClass:
    def __init__(self):
        pass

    def to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor
    
    def plot_Z_surface_3d(self, x,y,z, title="Z surface 3D"):
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.to_numpy(x), self.to_numpy(y), self.to_numpy(z),    cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
        ax.set_title(title)
        plt.show()

    def plot_Z_surface(self, X, Y, Z_surface, title="Z surface"):
        """
        X, Y: meshgrid (Nx,Ny)
        Z_surface: (Nx,Ny)
        """
        plt.figure(figsize=(8,6))
        cp = plt.pcolormesh(X.cpu(), Y.cpu(), Z_surface.cpu(), cmap='viridis', shading='auto')
        plt.colorbar(cp, label="Z [mm]")
        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.title(title)
        plt.axis("equal")
        plt.show()

    def plot_3d_points(self, x, y, z, color=None, title='Plot 3D'):

        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.title.set_text(title)

        scatter = ax.scatter(self.to_numpy(x), self.to_numpy(y), self.to_numpy(z), c=self.to_numpy(color), cmap='viridis', marker='o')
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]'); ax.set_zlabel('Z [mm]')
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    def plot_surface_grid(self, grid):
        """
        Plots the candidate grid stored in self.grid as a 3D surface.
        Plots the middle Z slice for visualization.
        """


        Nx, Ny, Nz, _ = grid.shape
        start = Nz // 4
        end = 3 * Nz // 4

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for mid in range(start, end):
            X_grid = grid[:, :, mid, 0].cpu().numpy()
            Y_grid = grid[:, :, mid, 1].cpu().numpy()
            Z_grid = grid[:, :, mid, 2].cpu().numpy()
            ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', edgecolor='none', alpha=0.9)

        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
        ax.set_title("Candidate Grid (middle Z slice)")
        plt.show()