import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utilits_lib.forward_process import AddNoise
from utilits_lib.reverse_process import UNet
from torch.optim import Adam
from tqdm import tqdm

class diffusion_model():
    def __init__(self, T:int = 100, img_size:int = 64, batch_size:int = 8) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.T = T
        self.img_size = img_size
        self.batch_size = batch_size
        self.addNoise = AddNoise(TimeStep=T, device=self.device)
        self.model = UNet(time_dim=batch_size, img_size=img_size, device=self.device).to(self.device)

    def __get_loss(self, x_0, t):
        x_t, noise = self.addNoise.train(x_0, t)
        noise_prediction = self.model(x_t, t)
        return F.mse_loss(noise, noise_prediction)

    def train(self, dataset, lr:float = 0.01, epochs:int = 20):
        '''
        dataset: maybe a set/list of images
        lr: learning rate, default=0.01
        epochs: times to epochs, default=20
        '''
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        optimizer = Adam(self.model.parameters(), lr=lr)
        self.loss = []
        for epoch in tqdm(range(epochs)):
            self.model.train()
            if (epoch+1) % 1000 == 0:
                print(f"Epoch {epoch + 1} | Loss: {loss.item()} ")
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                t = torch.randint(0, self.T, (self.batch_size, 1), device=self.device).long()
                loss = self.__get_loss(batch[0], t)
                loss.backward()
                optimizer.step()
                self.loss.append(loss.item())
                torch.cuda.empty_cache()
                if loss.item() <1e-2:
                    optimizer = Adam(self.model.parameters(), lr=lr/10)

                if loss.item() < 1e-5:
                    break
        return self

    def gather(self, a, t, shape):
        '''
        To get the value in "a" with index "t", moreover, resize output as the "shape"
        '''
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(shape) - 1)))

    @torch.no_grad()  # stop gradient to avoid reverse gradient
    def sample_timestep(self, x, t_index:int):
        '''
        x: noise graph which want to denoise
        t_index: time index, step number
        '''
        # parameters
        t = torch.full((1,), t_index, device=self.device, dtype=torch.long)
        beta = self.addNoise.betas
        alpha = 1- beta
        alpha_t = self.gather(alpha, t, x.shape)
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_t = self.gather(alpha_bar, t, x.shape)
        eps_coef = (1 - alpha_t) / (1 - alpha_bar_t) ** 0.5
        # Calculate p(x_t-1|x_t)
        self.model.eval()
        noise = self.model(x, t.unsqueeze(0))
        mean = 1 / (alpha_t ** 0.5) * (x - eps_coef * noise) # Note minus sign
        var = self.gather(beta, t, x.shape)
        eps = torch.randn(x.shape, device=x.device)
        if t_index == 0:
            return mean
        else:
            return mean + (var ** 0.5) * eps 

    @torch.no_grad()
    def generator(self):
        '''
        Generating graph from the trained model.
        '''
        img = torch.randn((1, 3, self.img_size, self.img_size), device=self.device) # generate white noise
        for i in range(0, self.T)[::-1]: # repair from step T to step 1
            img = self.sample_timestep(img, i)
        img = 2* (img - img.min())/(img.max() - img.min()) - 1 # scale to -1 to 1
        return img