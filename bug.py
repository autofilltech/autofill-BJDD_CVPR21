import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

net = nn.Linear(1,1)
x = torch.randn(1,1)

sw = tb.SummaryWriter("bug")
sw.add_graph(net, x)

