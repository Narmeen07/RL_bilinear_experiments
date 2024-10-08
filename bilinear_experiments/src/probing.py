import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

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
    def __init__(self, input_dim):
        self.probe = LinearProbe(input_dim, 4)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def prepare_data(self, dataset):
        X = np.array([d['activations']['conv_seqs_0_res_block0_conv1'].flatten() for d in dataset])
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
    def __init__(self, input_dim, n_action_classes, n_actions):
        self.probe = LinearProbe(input_dim, n_actions * n_action_classes)
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
                X.append(d['activations']['conv_seqs_0_res_block0_conv1'].flatten())
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
    def __init__(self, input_dim, grid_size):
        self.probe = LinearProbe(input_dim, grid_size * grid_size)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def prepare_data(self, dataset):
        X = np.array([d['activations']['conv_seqs_0_res_block0_conv1'].flatten() for d in dataset])
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
    def __init__(self, input_dim):
        self.probe = LinearProbe(input_dim, 2)
        self.criterion = nn.MSELoss()
    
    def prepare_data(self, dataset):
        X = np.array([d['activations']['conv_seqs_0_res_block0_conv1'].flatten() for d in dataset])
        y = np.array([d['labels']['mouse_location'] for d in dataset])
        return train_test_split(X, y, test_size=0.2)
    
    def train(self, dataset):
        X_train, X_val, y_train, y_val = self.prepare_data(dataset)
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=32)
        optimizer = optim.Adam(self.probe.parameters())
        train_probe(self.probe, train_loader, val_loader, self.criterion, optimizer)