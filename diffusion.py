import os
import time
import tqdm
import numpy as np
import torch
import torch.utils.data as data

from utils import KMFlowTensorDataset, get_optimizer, loss_registry
from ema import EMAHelper
from unet import Model


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )

    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config

        # Load training and test datasets
        if os.path.exists(config.data.stat_path):
            print("Loading dataset statistics from {}".format(config.data.stat_path))
            train_data = KMFlowTensorDataset(config.data.data_dir, train_trajectory = config.data.train_trajectory, stat_path=config.data.stat_path)
        else:
            print("No dataset statistics found. Computing statistics...")
            train_data = KMFlowTensorDataset(config.data.data_dir, train_trajectory = config.data.train_trajectory)
            train_data.save_data_stats(config.data.stat_path)
        
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=config.training.batch_size,
                                                   shuffle=True,
                                                   num_workers=config.data.num_workers)
        print(f"train_loader: {train_loader}")
        model = Model(config)
        model = model.to(self.device)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0

        num_iter = 0
        print('Starting training...')
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            epoch_loss = []
            start_time = time.time()
            for i, x in enumerate(train_loader):
                # print(f"x: {x.shape}")
                # x = x.contiguous()
                
                n = x.size(0)
                model.train()
                step += 1

                x = x.to(self.device)  # size: [32, 3, 256, 256]
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)
                # print(loss)
                epoch_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                num_iter = num_iter + 1
            
            end_time = time.time()
            epoch_duration = end_time - start_time
            print("==========================================================")
            print("Epoch: {}/{}, Loss: {}, Duration: {}".format(epoch, self.config.training.n_epochs, np.mean(epoch_loss), epoch_duration))
        print("Finished training")

        torch.save(
            states,
            os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
        )
        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
        print("Model saved at: ", self.args.log_path + "ckpt_{}.pth".format(step))

  
    def test(self):
        pass

