import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline_NN(nn.Module):

    def __init__(self, input_size=3, hidden_size= 64, output_size=1):

        super(Baseline_NN,self).__init__()

        self.l1 = nn.Linear(input_size,output_size)
        #self.l2 = nn.Linear(hidden_size,output_size)


    def forward(self,x):

        #x = F.relu(self.l1(x))
        x = self.l1(x)

        return x #.view(-1,1)
