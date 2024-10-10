import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix

class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

def train_probe(probe, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        probe.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = probe(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        probe.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = probe(inputs)
                val_loss += criterion(outputs, targets).item()
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss/len(val_loader)}")

# 1. Neighboring Walls Probe
class NeighboringWallsProbe:
    """
    Probe to predict the presence of neighboring walls.
    
    This probe takes the activations from a specific layer of the model and predicts
    whether there are walls in the four cardinal directions (left, right, up, down)
    around the mouse's current position.
    
    The probe uses a linear layer followed by a sigmoid activation to output
    probabilities for each direction.
    """
    def __init__(self, input_dim, layer):
        self.probe = LinearProbe(input_dim, 4)
        self.layer = layer
        self.criterion = nn.BCEWithLogitsLoss()
    
    def prepare_data(self, dataset):
        X = np.array([d['activations'][self.layer].flatten() for d in dataset])
        y = np.array([[int('LEFT' in d['labels']['neighbouring_walls']),
                       int('RIGHT' in d['labels']['neighbouring_walls']),
                       int('UP' in d['labels']['neighbouring_walls']),
                       int('DOWN' in d['labels']['neighbouring_walls'])] for d in dataset])
        return train_test_split(X, y, test_size=0.2)
    
    def train(self, dataset):
        X_train, X_val, y_train, y_val = self.prepare_data(dataset)
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=32)
        optimizer = optim.Adam(self.probe.parameters())
        train_probe(self.probe, train_loader, val_loader, self.criterion, optimizer)

class NextNActionsProbe:
    """
    Probe to predict the next N actions.
    
    This probe takes the activations from a specific layer of the model and predicts
    the next N actions that the agent should take. N is configurable. Observations with
    fewer than N actions are removed from the dataset, and those with more are truncated to N.
    """
    def __init__(self, input_dim, n_action_classes, n_actions, layer):
        self.probe = LinearProbe(input_dim, n_actions * n_action_classes)
        self.layer = layer
        self.criterion = nn.CrossEntropyLoss()
        self.n_action_classes = n_action_classes
        self.n_actions = n_actions
    
    def prepare_data(self, dataset):
        X = []
        y = []
        
        original_size = len(dataset)
        for d in dataset:
            actions = d['labels']['next_n_actions']
            if len(actions) >= self.n_actions:
                X.append(d['activations'][self.layer].flatten())
                y.append(actions[:self.n_actions])  # Take only the first N actions
        
        filtered_size = len(X)
        print(f"Original dataset size: {original_size}")
        print(f"Filtered dataset size: {filtered_size}")
        print(f"Removed {original_size - filtered_size} observations with fewer than {self.n_actions} actions")

        X = np.array(X)
        y = np.array(y)
        
        return train_test_split(X, y, test_size=0.2)
    
    def train(self, dataset):
        X_train, X_val, y_train, y_val = self.prepare_data(dataset)
        
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=32)
        
        optimizer = optim.Adam(self.probe.parameters())
        
        for epoch in range(10):  # 10 epochs, adjust as needed
            self.probe.train()
            total_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.probe(inputs)
                outputs = outputs.view(-1, self.n_actions, self.n_action_classes)  # Reshape to (batch_size, n_actions, n_action_classes)
                targets = targets.view(-1, self.n_actions)  # Reshape to (batch_size, n_actions)
                loss = self.criterion(outputs.transpose(1, 2), targets)  # CrossEntropyLoss expects (N, C, d1, d2, ...)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            self.probe.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.probe(inputs)
                    outputs = outputs.view(-1, self.n_actions, self.n_action_classes)
                    targets = targets.view(-1, self.n_actions)
                    val_loss += self.criterion(outputs.transpose(1, 2), targets).item()
            
            print(f"Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}")


# 3. Cheese Presence Probe
class CheesePresenceProbe:
    """
    Probe to predict the presence of cheese in the maze.
    
    This probe takes the activations from a specific layer of the model and predicts
    whether cheese is present in each cell of the maze grid.
    
    The probe uses a linear layer followed by a sigmoid activation to output
    probabilities for cheese presence in each cell.
    """
    def __init__(self, input_dim, grid_size, layer):
        self.probe = LinearProbe(input_dim, grid_size * grid_size)
        self.layer = layer
        self.criterion = nn.BCEWithLogitsLoss()
    
    def prepare_data(self, dataset):
        X = np.array([d['activations'][self.layer].flatten() for d in dataset])
        y = np.array([d['labels']['cheese_presence'] for d in dataset])
        return train_test_split(X, y, test_size=0.2)
    
    def train(self, dataset):
        X_train, X_val, y_train, y_val = self.prepare_data(dataset)
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=32)
        optimizer = optim.Adam(self.probe.parameters())
        train_probe(self.probe, train_loader, val_loader, self.criterion, optimizer)

# 4. Mouse Location Probe
class MouseLocationProbe:
    """
    Probe to predict the location of the mouse in the maze.
    
    This probe takes the activations from a specific layer of the model and predicts
    the x and y coordinates of the mouse's location in the maze grid.
    
    The probe uses a linear layer to output two values representing the predicted
    x and y coordinates of the mouse.
    """
    def __init__(self, input_dim, layer):
        self.probe = LinearProbe(input_dim, 2)
        self.layer = layer
        self.criterion = nn.MSELoss()
    
    def prepare_data(self, dataset):
        X = np.array([d['activations'][self.layer].flatten() for d in dataset])
        y = np.array([d['labels']['mouse_location'] for d in dataset])
        return train_test_split(X, y, test_size=0.2)
    
    def train(self, dataset):
        X_train, X_val, y_train, y_val = self.prepare_data(dataset)
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=32)
        optimizer = optim.Adam(self.probe.parameters())
        train_probe(self.probe, train_loader, val_loader, self.criterion, optimizer)

def visualize_neighboring_walls_probe(probe, test_data):
    X_test = np.array([d['activations'][probe.layer].flatten() for d in test_data])
    y_test = np.array([[int('LEFT' in d['labels']['neighbouring_walls']),
                        int('RIGHT' in d['labels']['neighbouring_walls']),
                        int('UP' in d['labels']['neighbouring_walls']),
                        int('DOWN' in d['labels']['neighbouring_walls'])] for d in test_data])
    with torch.no_grad():
        y_pred = (probe.probe(torch.FloatTensor(X_test)).sigmoid().numpy() > 0.5).astype(int)
    
    accuracies = [accuracy_score(y_test[:, i], y_pred[:, i]) for i in range(4)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(['Left', 'Right', 'Up', 'Down'], accuracies)
    plt.title('Neighboring Walls Probe Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
    plt.show()

def visualize_next_n_actions_probe(probe, test_data):
    # Get all four values from prepare_data
    X_train, X_test, y_train, y_test = probe.prepare_data(test_data)
    
    # We'll use only X_test and y_test for visualization
    with torch.no_grad():
        outputs = probe.probe(torch.FloatTensor(X_test))
        outputs = outputs.view(-1, probe.n_actions, probe.n_action_classes)
        y_pred = outputs.argmax(dim=2).numpy()
    
    accuracies = [accuracy_score(y_test[:, i], y_pred[:, i]) for i in range(probe.n_actions)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, probe.n_actions + 1), accuracies, marker='o')
    plt.title('Next N Actions Probe Accuracy')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(range(1, probe.n_actions + 1))
    for i, v in enumerate(accuracies):
        plt.text(i + 1, v + 0.01, f'{v:.2f}', ha='center')
    plt.show()

def visualize_cheese_presence_probe(probe, test_data):
    X_train, X_test, y_train, y_test = probe.prepare_data(test_data)
    X_test, y_test = X_test[len(X_test)//2:], y_test[len(y_test)//2:]  # Use second half as test set

    with torch.no_grad():
        y_pred = (probe.probe(torch.FloatTensor(X_test)).sigmoid().numpy() > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test.flatten(), y_pred.flatten())
    
    cm = confusion_matrix(y_test.flatten(), y_pred.flatten())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Cheese Presence Probe Confusion Matrix\nAccuracy: {accuracy:.2f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def visualize_mouse_location_probe(probe, test_data):
    X_train, X_test, y_train, y_test = probe.prepare_data(test_data)
    X_test, y_test = X_test[len(X_test)//2:], y_test[len(y_test)//2:]  # Use second half as test set

    with torch.no_grad():
        y_pred = probe.probe(torch.FloatTensor(X_test)).detach().numpy()
    
    mse_x = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    mse_y = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test[:, 0], y_test[:, 1], c='blue', label='True', alpha=0.5)
    plt.scatter(y_pred[:, 0], y_pred[:, 1], c='red', label='Predicted', alpha=0.5)
    plt.title(f'Mouse Location Probe\nMSE X: {mse_x:.4f}, MSE Y: {mse_y:.4f}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.show()
