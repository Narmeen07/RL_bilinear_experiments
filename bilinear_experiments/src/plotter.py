import numpy as np
import matplotlib.pyplot as plt

class EigenvectorPlotter:
    def __init__(self, matrix, kernel_size,out_channel_index, in1_index=None, in2_index=None):
        # Make the matrix symmetric
        self.matrix = (matrix + matrix.T) / 2
        self.out_channel_index = out_channel_index
        self.in1_index = in1_index
        self.in2_index = in2_index
        self.kernel_size = kernel_size

    def non_zero_eigenvector_weights(self):
        self.eigvals, self.eigvecs = np.linalg.eigh(self.matrix)
        
        # Sort eigenvalues and eigenvectors by eigenvalue magnitude
        sorted_indices = np.argsort(np.abs(self.eigvals))[::-1]
        self.eigvals = self.eigvals[sorted_indices]
        self.eigvecs = self.eigvecs[:, sorted_indices]
        self.non_zero_eigenvecs = [self.eigvecs[:, i].reshape((self.kernel_size, self.kernel_size)) for i in range(2)]
        self.non_zero_eigenvals =  self.eigvals[:2] 

    def get_non_zero_eigenvec(self):
        return self.non_zero_eigenvecs
    
    def plot(self):
        self.non_zero_eigenvector_weights()
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 5, figsize=(16, 8))

        # Plot eigenvalues in the first subplot
        axes[0, 0].plot(np.abs(self.eigvals), 'o-', label='Eigenvalues')
        axes[0, 0].set_xlabel('Index')
        axes[0, 0].set_ylabel('Magnitude')
        if self.in1_index is not None and self.in2_index:
            axes[0, 0].set_title(f'Eigenvalues of out_channel: {self.out_channel_index}, in channel 1: {self.in1_index}, in channel 2: {self.in2_index}')
        else:
            axes[0, 0].set_title(f'Eigenvalues of out_channel: {self.out_channel_index} ')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Remove the unused subplot (axes[0, 1] and axes[0, 2])
        for i in range(1,5):
            fig.delaxes(axes[0, i])
 

        # Plot eigenvectors reshaped to 3x3 matrices with eigenvalue magnitudes in remaining subplots
        for i in range(5):
            eigvec_reshaped = self.eigvecs[:, i].reshape((self.kernel_size, self.kernel_size))  # Take the real part of the eigenvector
            ax = axes[(i // self.kernel_size) + 1, i % self.kernel_size]
            img = ax.imshow(eigvec_reshaped, cmap='RdBu')
            ax.set_title(f'Eigenvalue: {self.eigvals[i].real:.2f}')  # Show eigenvalue magnitude
            fig.colorbar(img, ax=ax)

        plt.tight_layout()
        plt.show()
