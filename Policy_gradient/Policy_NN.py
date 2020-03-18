import torch
import torch.nn.functional as F
import torch.nn as nn



class Agent_nn(nn.Module):

    def __init__(self,Input_size=3, Hidden_size=64, output_size=1):

        super(Agent_nn,self).__init__()

        self.l1 = nn.Linear(Input_size, Hidden_size)
        self.l2 = nn.Linear(Hidden_size,output_size)
        #self.l3 = nn.Linear(Hidden_size,output_size)


    def forward(self,x):


        x = F.relu(self.l1(x))
        #x = F.relu(self.l2(x))
        x = self.l2(x)        #torch.tanh(self.l3(x)) * 2

        return x


    def REINFORCE(self,logp_a, advantage):



        Policy_cost = torch.dot(- logp_a.view(-1).double(), advantage.detach()) # need minus because need to perform gradient ascent


        return Policy_cost













