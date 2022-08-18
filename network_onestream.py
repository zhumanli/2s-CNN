import torch
from torch import nn
import torch.nn.functional as F
import dataset

# Network Module
class Network_OneStream(nn.Module):
    def __init__(self, in_channel, filters, num_classes):
        super(Network_OneStream, self).__init__()

        self.featuresJP = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, filters, kernel_size=(3, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )

        self.features3DRJDP = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, filters, kernel_size=(3, 19), stride=(1, 19), padding=(0, 0), bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):

        x = torch.transpose(x, 1, 3)
        if dataset.Model_Flag == "3DJP-CNN":
            out = self.featuresJP(x)
        else:
            out = self.features3DRJDP(x)

        out = F.adaptive_max_pool2d(out, 1)

        batch_size = out.size(0)
        out = out.view(batch_size, -1)
        out = self.fc(out)

        return F.log_softmax(out, dim=1)
