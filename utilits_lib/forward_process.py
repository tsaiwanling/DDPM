from operator import itemgetter
import torch
import torch.nn.functional as F

class AddNoise:
    def __init__(self, TimeStep, device:str="cuda") -> None:
        self.betas = self.linear_schedule(timesteps=TimeStep).to(device)
    
    def gather(self, a, t, shape):
        '''
        To get the value in "a" with index "t", moreover, resize output as the "shape"
        '''
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(shape) - 1)))

    def linear_schedule(self, timesteps=100, start=1e-5, end=0.02):
        '''
        return a tensor of a linear schedule
        '''
        return torch.linspace(start, end, timesteps)
    
    def q(self, x_clear, t):
        '''
        x_clear: one image
        t: time index
        '''
        alpha_bar = torch.cumprod(1.0 - self.betas, dim=0)
        mean = self.gather(alpha_bar, t, x_clear.shape) ** 0.5 * x_clear
        var = 1.0 - self.gather(alpha_bar, t, x_clear.shape)
        noise = torch.randn_like(x_clear)
        x_noised = mean + (var ** 0.5) * noise
        return x_noised, noise
    
    def train(self, data, step):
        '''
        data: a batch of images
        step: a batch of time steps
        '''
        temp = []
        for d, t in zip(data, step):
            temp.append(self.q(d, t))        
        return torch.stack(list(map(itemgetter(0), temp))), torch.stack(list(map(itemgetter(1), temp)))