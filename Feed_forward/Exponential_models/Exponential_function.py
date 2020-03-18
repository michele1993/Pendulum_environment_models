import numpy as np
import torch
from torch.distributions import Normal
import torch.optim as opt

import torch.nn as nn


class ExpF_agent(nn.Module):

    def __init__(self, n_exps, t_steps, std = 0.3, ln_rate = 0.001):

        super().__init__()

        miu = torch.rand((n_exps,))
        miu /= sum(miu)
        miu[::2] *= 1
        miu[1::2] *= -1  # start form 1 element select everything with step = 2 //i.e odd indexes


        self.miu = nn.Parameter(miu)

        self.decay_rate = nn.Parameter(torch.rand(n_exps).view(-1,n_exps)) # if use randn also get negative values, changing sign of the exponential

        self.n_exps = n_exps

        self.t_steps = torch.arange(t_steps, dtype=torch.float).view(t_steps,-1)

        self.t_steps /= torch.sum(self.t_steps)

        self.exp_f = lambda t,e: torch.exp(-t/e)

        self.std = std

        self.optimiser = opt.Adam(self.parameters(),ln_rate)



    def sample_actions(self):


        mean_value = torch.matmul(self.exp_f(self.t_steps,self.decay_rate), self.miu.view(self.n_exps, -1))


        d = Normal(mean_value, self.std)

        actions = d.sample()

        self.log_ps = d.log_prob(actions)

        return actions


    def update(self,rwds):


        loss = sum(-self.log_ps.view(-1) * rwds)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss




