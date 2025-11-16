import torch.nn as nn


class AdapterBottleneck(nn.Module):
    """adapter:Parameter-Efficient Transfer Learning for NLP at https://arxiv.org/abs/1902.00751"""
    def __init__(self, in_feature, reduction=4, dropout=0.0):
        super(AdapterBottleneck, self).__init__()
        self.in_feature = in_feature
        layers = [
            nn.Linear(self.in_feature, self.in_feature // reduction, bias=True),
            nn.ReLU(inplace=False),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(self.in_feature // reduction, self.in_feature, bias=True))
        self.adapter = nn.Sequential(*layers)
        
    def forward(self, inputs):
        out = self.adapter(inputs)
        return out + inputs
    
class Conv2dAdapterBottleneck(nn.Module):
    """adapter:Parameter-Efficient Transfer Learning for NLP at https://arxiv.org/abs/1902.00751"""
    def __init__(self, inplanes, reduction=4, dropout=0.0) -> None:
        super(Conv2dAdapterBottleneck, self).__init__()
        layers = [
            nn.Conv2d(inplanes, inplanes // reduction, kernel_size=1, bias=False, stride=1),
            nn.ReLU(inplace=False),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(inplanes // reduction, inplanes, kernel_size=1, bias=False, stride=1))
        self.adapter = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.adapter(x) + x

