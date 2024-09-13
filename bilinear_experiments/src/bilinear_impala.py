import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, channels,kernel_size):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv0 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_size,
                               padding=padding)
        self.b_0 = nn.Parameter(torch.randn(channels))
        self.b_1 = nn.Parameter(torch.randn(channels))

    def gated_conv(self, x):
        A = self.conv0(x) + self.b_0.view(1, -1, 1, 1)  # Reshape bias to (1, channels, 1, 1)
        B = self.conv1(x) + self.b_1.view(1, -1, 1, 1)  # Reshape bias to (1, channels, 1, 1)
        return A * B
    
    def forward(self,x):
        inputs = x
        #here i'm skipping the gated(x) function
        x = self.gated_conv(x)
        x = self.conv2(x)
        return x + inputs

class ConvSequence(nn.Module):

    def __init__(self, input_shape, out_channels,kernel_size):
        super(ConvSequence, self).__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels=self._input_shape[0],
                              out_channels=self._out_channels,
                              kernel_size=kernel_size,
                              padding=padding)
        self.max_pool2d = nn.MaxPool2d(kernel_size=kernel_size,
                                       stride=2,
                                       padding=padding)
        self.res_block0 = ResidualBlock(self._out_channels, kernel_size)
        self.res_block1 = ResidualBlock(self._out_channels,kernel_size)


    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool2d(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2

class BimpalaCNN(nn.Module):
    """Network from IMPALA paper, to work with pfrl."""

    def __init__(self, obs_space, num_outputs, kernel_size):

        super(BimpalaCNN, self).__init__()

        h, w, c = obs_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels, kernel_size)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc1 = nn.Linear(in_features=shape[0] * shape[1] * shape[2],
                                   out_features=256)
        self.hidden_fc2 = nn.Linear(in_features=shape[0] * shape[1] * shape[2],
                                   out_features=256)
        self.bfc1 = nn.Parameter(torch.randn(256))
        self.bfc2 = nn.Parameter(torch.randn(256))
        #self.fc_dropout = nn.Dropout(p=fc_dropout_prob)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        # Initialize weights of logits_fc
        nn.init.orthogonal_(self.logits_fc.weight, gain=0.01)
        nn.init.zeros_(self.logits_fc.bias)
    def gated_fc(self,x):
        A = self.hidden_fc1(x) + self.bfc1
        B = self.hidden_fc2(x) + self.bfc2
        return A* B

    def forward(self, obs):
        assert obs.ndim == 4
        x = obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        #let's totally remove this relu function
        #x = torch.relu(x)
        x = self.gated_fc(x)
        #x = self.fc_dropout(x)  # Apply dropout after fully connected layer
        logits = self.logits_fc(x)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value_fc(x)
        return dist, value

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path, device):
        self.load_state_dict(torch.load(model_path, map_location="cpu"))

    def get_state_dict(self):
        return self.state_dict()
