def plot_top_f1_scores(new_classification_data, categories, colors, top_n=15, title=None):
    import matplotlib.pyplot as plt
    import numpy as np

    # Parse f1-scores from the classification reports
    f1_scores = [report['0']['f1-score'] for report in new_classification_data]

    # Sort categories and scores by F1-score in descending order
    sorted_data = sorted(zip(categories, f1_scores, colors), key=lambda x: x[1], reverse=True)
    top_data = sorted_data[:top_n]  # Get top N scores

    # Unpack the sorted data
    top_categories, top_f1_scores, top_colors = zip(*top_data)

    # Print F1-scores for top N
    print(f"Top {top_n} F1-scores:")
    for category, score in zip(top_categories, top_f1_scores):
        print(f"{category}: {score:.4f}")

    # Create bar plot with zoomed-in y-axis for top N scores
    plt.figure(figsize=(12, 6))
    bars = plt.bar(top_categories, top_f1_scores, color=top_colors)
    if title:
        plt.title(title)
    plt.xlabel('Categories')
    plt.ylabel('F1-score')
    plt.ylim(max(0.5, min(top_f1_scores) - 0.05), 1.0)  # Dynamically set lower y-limit
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.show()


# weight visualizations a la Claude 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

def visualize_conv_weights(weights, figsize=(12, 8), cmap='RdBu', normalize=True):
    """
    Visualize convolutional layer weights
    
    Parameters:
    weights: numpy array of shape (num_filters, input_channels, height, width)
            or (num_filters, height, width) for single-channel inputs
    figsize: tuple of figure dimensions
    cmap: colormap to use (default: RdBu for diverging weights)
    normalize: whether to normalize weights to [-1, 1] range
    """
    # Add channel dimension if not present
    if len(weights.shape) == 3:
        weights = weights[:, np.newaxis, :, :]
    
    num_filters, num_channels, height, width = weights.shape
    
    # Calculate grid layout
    grid_size = int(np.ceil(np.sqrt(num_filters)))
    
    # Create figure and gridspec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(grid_size, grid_size, figure=fig)
    
    # Normalize weights if requested
    if normalize:
        weights = weights / np.max(np.abs(weights))
    
    # Plot each filter
    for i in range(num_filters):
        ax = fig.add_subplot(gs[i // grid_size, i % grid_size])
        
        if num_channels == 1:
            # Single channel visualization
            im = ax.imshow(weights[i, 0], cmap=cmap)
        else:
            # Multi-channel visualization - create RGB image
            # Normalize each channel separately
            rgb = np.zeros((height, width, 3))
            for c in range(min(num_channels, 3)):  # Only use first 3 channels for RGB
                channel_data = weights[i, c]
                channel_min, channel_max = channel_data.min(), channel_data.max()
                rgb[:, :, c] = (channel_data - channel_min) / (channel_max - channel_min)
            im = ax.imshow(rgb)
        
        ax.axis('off')
        ax.set_title(f'Filter {i+1}')
    
    plt.tight_layout()
    return fig

def visualize_feature_maps(feature_maps, figsize=(12, 8), cmap='viridis'):
    """
    Visualize feature maps (activations) from a convolutional layer
    
    Parameters:
    feature_maps: numpy array of shape (num_maps, height, width)
    figsize: tuple of figure dimensions
    cmap: colormap to use (default: viridis for activation values)
    """
    num_maps = feature_maps.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_maps)))
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(grid_size, grid_size, figure=fig)
    
    for i in range(num_maps):
        ax = fig.add_subplot(gs[i // grid_size, i % grid_size])
        im = ax.imshow(feature_maps[i], cmap=cmap)
        ax.axis('off')
        ax.set_title(f'Map {i+1}')
    
    plt.tight_layout()
    return fig

def visualize_weight_distribution(weights, figsize=(10, 6)):
    """
    Visualize the distribution of weights in a layer
    
    Parameters:
    weights: numpy array of weights
    figsize: tuple of figure dimensions
    """
    plt.figure(figsize=figsize)
    
    # Plot histogram with KDE
    sns.histplot(weights.ravel(), kde=True)
    plt.title('Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    
    # Add vertical line at zero
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    return plt.gcf()

def visualize_weight_stats(weights):
    """
    Print statistical information about weights
    
    Parameters:
    weights: numpy array of weights
    """
    stats = {
        'Mean': np.mean(weights),
        'Std': np.std(weights),
        'Min': np.min(weights),
        'Max': np.max(weights),
        'Sparsity': np.mean(weights == 0),
        'Abs Mean': np.mean(np.abs(weights))
    }
    
    for name, value in stats.items():
        print(f"{name}: {value:.4f}")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import cv2
from scipy.ndimage import zoom
import seaborn as sns

class ConvFilterAnalyzer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def get_filter_responses(self, image, layer_name, filter_indices=None):
        """
        Get activation maps for specific filters on an input image.
        
        Args:
            image: Input image (PIL Image or tensor)
            layer_name: Name of the convolutional layer to analyze
            filter_indices: List of filter indices to visualize (None for all)
        """
        if isinstance(image, Image.Image):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
            image = transform(image).unsqueeze(0)
        
        image = image.to(self.device)
        
        # Register hook to get intermediate activations
        activations = []
        def hook(module, input, output):
            activations.append(output)
        
        # Get the specified layer
        target_layer = dict([*self.model.named_modules()])[layer_name]
        handle = target_layer.register_forward_hook(hook)
        
        # Forward pass
        with torch.no_grad():
            self.model(image)
        
        handle.remove()
        
        # Get activation maps
        activation = activations[0]
        if filter_indices is not None:
            activation = activation[:, filter_indices]
        
        return activation.cpu().numpy()

    def visualize_filter_responses(self, image, layer_name, filter_indices=None, 
                                 figsize=(15, 10)):
        """
        Visualize how specific filters respond to an input image.
        """
        responses = self.get_filter_responses(image, layer_name, filter_indices)
        n_filters = responses.shape[1]
        
        # Calculate grid dimensions
        n_cols = min(8, n_filters)
        n_rows = (n_filters - 1) // n_cols + 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot original image
        plt.figure(figsize=(5, 5))
        plt.imshow(image if isinstance(image, Image.Image) else 
                  image.squeeze(0).permute(1, 2, 0).cpu().numpy())
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot activation maps
        for i in range(n_filters):
            ax = axes[i // n_cols, i % n_cols]
            activation = responses[0, i]
            
            # Normalize activation map
            activation = (activation - activation.min()) / \
                        (activation.max() - activation.min() + 1e-8)
            
            im = ax.imshow(activation, cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Filter {i}')
        
        plt.tight_layout()
        return fig

    def generate_filter_visualization(self, layer_name, filter_idx, 
                                    input_size=(224, 224), 
                                    n_iterations=30):
        """
        Generate an image that maximally activates a specific filter.
        """
        # Create a random image
        image = torch.randn(1, 3, *input_size).to(self.device)
        image.requires_grad_(True)
        
        # Get the target layer
        target_layer = dict([*self.model.named_modules()])[layer_name]
        
        optimizer = torch.optim.Adam([image], lr=0.1)
        
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            activations = []
            def hook(module, input, output):
                activations.append(output)
            
            handle = target_layer.register_forward_hook(hook)
            self.model(image)
            handle.remove()
            
            # Get activation of target filter
            activation = activations[0]
            
            # Calculate loss (negative activation to maximize it)
            loss = -activation[0, filter_idx].mean()
            
            # Add regularization for image clarity
            reg_loss = torch.norm(image) * 0.001
            total_loss = loss + reg_loss
            
            total_loss.backward()
            optimizer.step()
        
        # Normalize and convert to image
        generated_image = image.detach().cpu().squeeze(0)
        generated_image = (generated_image - generated_image.min()) / \
                         (generated_image.max() - generated_image.min())
        
        return generated_image.permute(1, 2, 0).numpy()

    def analyze_filter_patterns(self, layer_name, dataset_images, 
                              n_top_activations=5):
        """
        Analyze what patterns each filter responds to using a dataset of images.
        """
        all_activations = []
        image_indices = []
        
        # Get activations for all images
        for idx, image in enumerate(dataset_images):
            activations = self.get_filter_responses(image, layer_name)
            all_activations.append(activations)
            image_indices.extend([idx] * activations.shape[0])
        
        all_activations = np.concatenate(all_activations, axis=0)
        
        # For each filter, find images that cause highest activation
        n_filters = all_activations.shape[1]
        top_activations = {}
        
        for filter_idx in range(n_filters):
            filter_activations = all_activations[:, filter_idx]
            top_indices = np.argsort(filter_activations.max(axis=(1, 2)))[-n_top_activations:]
            top_activations[filter_idx] = {
                'image_indices': [image_indices[i] for i in top_indices],
                'activation_values': filter_activations[top_indices]
            }
        
        return top_activations

    def visualize_filter_evolution(self, image, layer_name, filter_idx):
        """
        Visualize how a filter's response changes across different scales.
        """
        original_response = self.get_filter_responses(image, layer_name, 
                                                    [filter_idx])[0, 0]
        
        scales = [0.5, 1.0, 1.5, 2.0]
        responses = []
        
        for scale in scales:
            # Resize image
            if isinstance(image, Image.Image):
                scaled_image = image.resize((int(image.size[0] * scale), 
                                          int(image.size[1] * scale)))
            else:
                scaled_image = F.interpolate(image.unsqueeze(0), 
                                          scale_factor=scale).squeeze(0)
            
            response = self.get_filter_responses(scaled_image, layer_name, 
                                               [filter_idx])[0, 0]
            responses.append(response)
        
        # Visualize
        fig, axes = plt.subplots(1, len(scales), figsize=(15, 3))
        for i, (response, scale) in enumerate(zip(responses, scales)):
            axes[i].imshow(response, cmap='viridis')
            axes[i].set_title(f'Scale {scale}x')
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig

# Usage example:
"""
# Initialize analyzer
analyzer = ConvFilterAnalyzer(model)

# 1. Visualize filter responses on a specific image
image = Image.open('example.jpg')
analyzer.visualize_filter_responses(image, 'conv1', filter_indices=[0,1,2,3])

# 2. Generate pattern that maximally activates a filter
pattern = analyzer.generate_filter_visualization('conv1', filter_idx=0)
plt.imshow(pattern)

# 3. Analyze patterns across dataset
dataset_images = [Image.open(f) for f in image_files]
patterns = analyzer.analyze_filter_patterns('conv1', dataset_images)

# 4. Study scale sensitivity
analyzer.visualize_filter_evolution(image, 'conv1', filter_idx=0)
"""