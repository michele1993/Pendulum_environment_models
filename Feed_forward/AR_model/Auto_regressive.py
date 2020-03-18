import torch
import torch.optim as opt
from torch.distributions import Normal
import torch.nn as nn

class AR_model(nn.Module):

    def __init__(self, t_steps, max_t_steps = 200, std = 0.3,ln_rate = 0.001):

        super().__init__()

        self.t_steps = t_steps

        self.max_t_steps = max_t_steps

        self.std = std

        params = torch.randn(t_steps)

        norm_param = params / sum(torch.abs(params))


        # if initialise from gaussian (0,1) then normalise by variance. so that variance of all = 1, and of each 1/d

        self.params = nn.Parameter(norm_param)

        self.bias = nn.Parameter(torch.rand(1))

        self.optimiser = opt.Adam(self.parameters(),ln_rate)



    def sample_actions(self):

        e = 0
        inputs = torch.zeros(self.t_steps)

    # store as a list of tensors
        for i in range(self.t_steps, self.max_t_steps + self.t_steps):


            inputs = torch.cat((inputs,torch.dot(inputs[torch.arange(i -1,i - self.t_steps -1 ,-1)], self.params) + self.bias))

            #inputs[0:i],self.params[0:i]) + self.bias


        d = Normal(inputs[self.t_steps:],self.std)

        actions = d.sample()

        self.log_p = d.log_prob(actions)

        return actions


    def update(self,rwds):

        loss = sum(torch.mul(-self.log_p, rwds))

        self.optimiser.zero_grad()

        loss.backward()

        self.optimiser.step()

        return loss

