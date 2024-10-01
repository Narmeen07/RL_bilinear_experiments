import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv0 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_size,
                               padding=padding, bias=False)
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_size,
                               padding=padding, bias=False)

    def gated_conv(self, x):
        A = self.conv0(x) 
        B = self.conv1(x) 
        return A * B
    
    def forward(self, x):
        inputs = x
        x = self.gated_conv(x)
        #conv at the end
        return x + inputs #  (conva(x)+ b1) * (convb(x) + b2)

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels, kernel_size):
        super(ConvSequence, self).__init__()
        #conv here
        self._input_shape = input_shape
        self._out_channels = out_channels
        padding = (kernel_size - 1) // 2
        #convb(conva(x)) * convc(conva(x))
        self.max_pool2d = nn.MaxPool2d(kernel_size=kernel_size,
                                       stride=2,
                                       padding=padding)
        self.res_block0 = ResidualBlock(self._out_channels, kernel_size)
        self.res_block1 = ResidualBlock(self._out_channels, kernel_size)

    def forward(self, x):
        x = self.max_pool2d(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2

class BimpalaCNN(nn.Module):
    def __init__(self, obs_space, num_outputs, kernel_size):
        super(BimpalaCNN, self).__init__()
        h, w, c = obs_space.shape
        shape = (c, h, w)

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels=c,
                              out_channels=32,
                              kernel_size=kernel_size,
                              padding=padding)
        
        conv_seqs = []
        for out_channels in [32, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels, kernel_size)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)

        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc1 = nn.Linear(in_features=shape[0] * shape[1] * shape[2],
                                   out_features=256, bias=False)
        self.hidden_fc2 = nn.Linear(in_features=shape[0] * shape[1] * shape[2],
                                   out_features=256, bias=False)

        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        nn.init.orthogonal_(self.logits_fc.weight, gain=0.01)
        nn.init.zeros_(self.logits_fc.bias)

    def gated_fc(self, x):
        A = self.hidden_fc1(x)
        B = self.hidden_fc2(x) 
        return A * B

    def forward(self, obs):
        assert obs.ndim == 4
        x = obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        x = self.conv(x)
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = self.gated_fc(x)
        logits = self.logits_fc(x)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value_fc(x)
        return dist, value

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path, device):
        self.load_state_dict(torch.load(model_path, map_location="cpu"))

    def get_state_dict(self):
        return self.state_dict

class TopKBimpalaCNN(nn.Module):
    def __init__(self, obs_space, num_outputs, kernel_size, B, topk):
        super(TopKBimpalaCNN, self).__init__()
        self.topk = topk
        h, w, c = obs_space.shape
        shape = (c, h, w)

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels=c,
                              out_channels=32,
                              kernel_size=kernel_size,
                              padding=padding)
        
        conv_seqs = []
        for out_channels in [32, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels, kernel_size)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)

        self.conv_seqs = nn.ModuleList(conv_seqs)
        
        self.B = B
        #initialise the hiddenfcs to calculate the values
        self.hidden_fc1 = nn.Linear(in_features=shape[0] * shape[1] * shape[2],
                                   out_features=256, bias=False)
        self.hidden_fc2 = nn.Linear(in_features=shape[0] * shape[1] * shape[2],
                                   out_features=256, bias=False)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)

        #the linear algebra
        eigvals, eigvecs = torch.linalg.eigh(self.B)
        """
        abs_eigenvalues = torch.abs(eigvals)
        _, indices = torch.sort(abs_eigenvalues, descending=True, dim=1)
        top_k_indices = indices[:, :self.topk]
        top_k_eigenvectors = torch.gather(eigvecs, 2, top_k_indices.unsqueeze(-1).expand(-1, -1, eigvecs.shape[-1]))
        self.top_k_eigenvectors = top_k_eigenvectors.transpose(1, 2)
        self.top_k_eigenvalues = torch.gather(eigvals, 1, top_k_indices)
        """
        self.top_k_eigenvectors = eigvecs
        self.top_k_eigenvalues = eigvals
        
        nn.init.orthogonal_(self.logits_fc.weight, gain=0.01)
        nn.init.zeros_(self.logits_fc.bias)

    def forward(self, obs):
        assert obs.ndim == 4
        x = obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        x = self.conv(x)
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)

        sims = torch.einsum("c f t, b f -> b c t", self.top_k_eigenvectors.to(x.device), x)
        logits = torch.einsum("c t, b c t -> b c", self.top_k_eigenvalues.to(x.device), sims**2)

        logits = logits + self.logits_fc.bias
        dist = torch.distributions.Categorical(logits=logits)
        x = self.hidden_fc1(x) * self.hidden_fc2(x)
        value = self.value_fc(x)
        return dist, value

    def transfer_params_from(self, other_model):
        """
        Transfer parameters from another model up to hidden_fc1 and hidden_fc2
        """
        # Transfer convolutional layers
        self.conv.load_state_dict(other_model.conv.state_dict())
        
        # Transfer conv sequences
        for self_seq, other_seq in zip(self.conv_seqs, other_model.conv_seqs):
            self_seq.load_state_dict(other_seq.state_dict())
        
        # Transfer and modify hidden_fc1 and hidden_fc2
        with torch.no_grad():
            self.hidden_fc1.weight.copy_(other_model.hidden_fc1.weight)
            self.hidden_fc2.weight.copy_(other_model.hidden_fc2.weight)
            self.logits_fc.weight.copy_(other_model.logits_fc.weight)
            self.logits_fc.bias.copy_(other_model.logits_fc.bias)
            self.value_fc.weight.copy_(other_model.value_fc.weight)
            
        
        print("Parameters transferred and modified successfully")

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path, device):
        self.load_state_dict(torch.load(model_path, map_location="cpu"))

    def get_state_dict(self):
        return self.state_dict()

