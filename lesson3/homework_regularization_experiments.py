import torch
import torch.nn as nn


# 3.2
# =============================================== Dropout с изменяющимся коэффициентом (Adaptive Dropout) ===============================================
class AdaptiveDropout(nn.Module):
    def __init__(self, p_start=0.5, p_end=0.1, total_epochs=10):
        super().__init__()
        self.p_start = p_start
        self.p_end = p_end
        self.total_epochs = total_epochs
        self.current_p = p_start

    def update_epoch(self, epoch):
        t = epoch / self.total_epochs
        self.current_p = self.p_start * (1 - t) + self.p_end * t

    def forward(self, x):
        if not self.training:
            return x
        return nn.functional.dropout(x, p=self.current_p, training=True)
     

# =============================================== BatchNorm с различными momentum ===============================================
class FlexibleBatchNorm(nn.BatchNorm1d):
    def __init__(self, num_features, momentum=0.1):
        super().__init__(num_features, momentum=momentum)

    def set_momentum(self, momentum):
        self.momentum = momentum

def momentum_schedule(epoch, max_epoch, m_start=0.1, m_end=0.9):
    return m_start + (m_end - m_start) * (epoch / max_epoch)


# =============================================== Комбинированная сеть (Dropout + BatchNorm + адаптация) ===============================================
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims

        self.dropouts = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            self.bns.append(nn.BatchNorm1d(dims[i+1]))
            self.dropouts.append(AdaptiveDropout())

        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x, epoch=None):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.bns[i](x)
            x = torch.relu(x)
            x = self.dropouts[i](x)

        return self.classifier(x)

    def update_adaptive_params(self, epoch, max_epoch):
        for i, do in enumerate(self.dropouts):
            do.update_epoch(epoch)

        momentum = momentum_schedule(epoch, max_epoch)
        for bn in self.bns:
            bn.momentum = momentum