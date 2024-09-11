import torch
import torch.nn as nn

class BilinearGatedFC(nn.Module):
    def __init__(self, in_features, out_features, bias = False):
        super(BilinearGatedFC, self).__init__()
        # Transform linear operation
        self.transform = nn.Linear(in_features, out_features, bias=bias)
        # Gating linear operation
        self.gate = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        transformed = self.transform(x)
        gating_factor = self.gate(x)
        return transformed * gating_factor

class BilinearGatedActivation2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,padding, bias = False):
        super(BilinearGatedActivation2D, self).__init__()
        # Convolution to transform the input
        self.transform = nn.Conv2d(in_channels, out_channels, kernel_size =kernel_size, padding=padding, bias=bias) 
        # Convolution to create a gating signal
        self.gate = nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size,padding = padding, bias=bias)

    def forward(self, x):

        transformed = self.transform(x)
        gating_factor = self.gate(x)
        return transformed * gating_factor  # Element-wise multiplication

class bilinearCNN(nn.Module):
    def __init__(self, obs_space, num_outputs):
        super(bilinearCNN, self).__init__()

        h, w, c = obs_space.shape
        self.num_outputs = num_outputs

        self.layer1a = BilinearGatedActivation2D(in_channels=c, out_channels=16, kernel_size=7, padding=3)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer2a = BilinearGatedActivation2D(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.layer2b = BilinearGatedActivation2D(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer3a = BilinearGatedActivation2D(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer4a = BilinearGatedActivation2D(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Compute the flattened dimension after convolutions and pooling
        self.flattened_dim = self._get_flattened_dim(h, w)

        self.gatedfc1 = BilinearGatedFC(in_features=self.flattened_dim, out_features=256)
        self.gatedfc2 = BilinearGatedFC(in_features=256, out_features=512)
        
        self.fc3 = nn.Linear(in_features=512, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=512, out_features=1)
        

    def _get_flattened_dim(self, h, w):
        x = torch.zeros(1, 3, h, w)  # Dummy input to compute the shape
        x = self.pool1(self.layer1a(x))
        x = self.pool2(self.layer2b(self.layer2a(x)))
        x = self.pool3(self.layer3a(x))
        x = self.pool4(self.layer4a(x))
        return x.numel()

    def forward(self, obs):
        assert obs.ndim == 4
        x = obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW

        x = self.layer1a(x)
        x = self.pool1(x)

        x = self.layer2a(x)
        x = self.layer2b(x)
        x = self.pool2(x)

        x = self.layer3a(x)
        x = self.pool3(x)

        x = self.layer4a(x)
        x = self.pool4(x)

        x = torch.flatten(x, start_dim=1)
        x = self.gatedfc1(x)
        x = self.gatedfc2(x)
        
        logits = self.fc3(x)
        dist = torch.distributions.Categorical(logits=logits)   
        value = self.value_fc(x)
        
        return dist, value

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path, device):
        self.load_state_dict(torch.load(model_path, map_location=device))

    def get_state_dict(self):
        return self.state_dict()
