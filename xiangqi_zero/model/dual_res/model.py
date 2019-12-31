import torch

from torch import nn

from xiangqi_zero.model.dual_res.config import ModelConfigDualRes
from xiangqi_zero.env.common import *


class DualResidualNetwork(nn.Module):
    """Implementation of the dual network from paper
    'Mastering the Game of Go without Human Knowledge' """

    def __init__(self, config):
        super(DualResidualNetwork, self).__init__()
        self.base_model = BaseInputModel(config)
        self.policy_layer = PolicyOutputLayer()
        self.value_layer = ValueOutputLayer()

    def forward(self, x):
        x = self.base_model()
        policy = self.policy_layer(x)
        value = self.value_layer(x)
        return policy, value


class ResidualBlock(nn.Module):

    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=256)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=256)

        # # Align the shape
        # if stride != 1 or in_channels != out_channels:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
        #         nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x
        # if self.shortcut is not None:
        #     residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.add(x, residual)
        x = self.relu(x)
        return x


class BaseInputModel(nn.Module):

    def __init__(self, config: ModelConfigDualRes):
        super(BaseInputModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=NUM_MODEL_INPUT, out_channels=256, kernel_size=3, stride=1)
        self.batch_norm = nn.BatchNorm2d(num_features=256)
        self.relu = nn.ReLU()
        self.res_blocks = nn.ModuleList([ResidualBlock() for _ in range(config.num_res_blocks)])

    def forward(self, x):  # N * C_in * H * W
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x)
        return x


class PolicyOutputLayer(nn.Module):

    def __init__(self):
        super(PolicyOutputLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(num_features=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=72, out_features=ACTION_SPACE)  # TODO

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.fc(x)
        return x


class ValueOutputLayer(nn.Module):

    def __init__(self):
        super(ValueOutputLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(num_features=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=72, out_features=256)  # TODO in / out size 8 * 9 = 72
        self.fc2 = nn.Linear(in_features=256, out_features=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x
