from torch import nn


l1_loss = nn.L1Loss(reduction='mean')
l2_loss = nn.MSELoss(reduction='mean')
