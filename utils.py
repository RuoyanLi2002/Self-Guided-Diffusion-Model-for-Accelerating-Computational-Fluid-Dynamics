import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

from pytorch_wavelets import DWTForward

global_min_grad = 0.0
global_max_grad = 17.027891043097863
tau_1 = 0.05650500575453074

dwt = DWTForward(J=1, wave='db2', mode='zero').to(torch.device("cuda"))

class KMFlowTensorDataset(Dataset):
    def __init__(self, data_path, train_trajectory=40, test=False, stat_path=None, max_cache_len=4000):

        self.all_data = np.load(data_path)
        print('Data set shape: ', self.all_data.shape)
        idxs = np.arange(self.all_data.shape[0])

        self.train_idx_lst = idxs[15:]
        self.test_idx_lst = idxs[train_trajectory:]
        self.time_step_lst = np.arange(self.all_data.shape[1]-2)
        
        self.idx_lst = self.train_idx_lst
        print(f"self.idx_lst: {self.idx_lst.shape}")
        # if not test:
        #     self.idx_lst = self.train_idx_lst[:]
        # else:
        #     self.idx_lst = self.test_idx_lst[:]
        
        self.cache = {}
        self.max_cache_len = max_cache_len

        if stat_path is not None:
            self.stat_path = stat_path
            self.stat = np.load(stat_path)
        else:
            self.stat = {}
            self.prepare_data()

    def __len__(self):
        return len(self.idx_lst) * len(self.time_step_lst)

    def prepare_data(self):
        self.stat['mean'] = np.mean(self.all_data[self.train_idx_lst[:]].reshape(-1, 1))
        self.stat['scale'] = np.std(self.all_data[self.train_idx_lst[:]].reshape(-1, 1))
        data_mean = self.stat['mean']
        data_scale = self.stat['scale']
        print(f'Data statistics, mean: {data_mean}, scale: {data_scale}')


    def preprocess_data(self, data):
        data = (data - self.stat['mean']) / (self.stat['scale'])
        
        return data.astype(np.float32)

    def save_data_stats(self, out_dir):
        np.savez(out_dir, mean=self.stat['mean'], scale=self.stat['scale'])

    def __getitem__(self, idx):
        seed = self.idx_lst[idx // len(self.time_step_lst)]
        frame_idx = idx % len(self.time_step_lst)
        id = idx

        if id in self.cache.keys():
            return self.cache[id]
        else:
            frame0 = self.preprocess_data(self.all_data[seed, frame_idx])
            frame1 = self.preprocess_data(self.all_data[seed, frame_idx+1])
            frame2 = self.preprocess_data(self.all_data[seed, frame_idx+2])

            frame = np.concatenate((frame0[None, ...], frame1[None, ...], frame2[None, ...]), axis=0)
            self.cache[id] = frame

            if len(self.cache) > self.max_cache_len:
                self.cache.pop(list(self.cache.keys())[np.random.choice(len(self.cache))])

            return frame



def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    
    mean = 1.4131128469755236e-11
    scale = 4.768852160800451
    
    high = x0*scale + mean
    # print(f"high: {high.shape}")
    mask = create_wavelet_binary_mask(high)
    torch.set_printoptions(threshold=torch.inf)
    # print(f"mask: {mask} {mask.shape}")
    # print(f"x: {x.shape}")
    
    output = model(x, t.float())
    # print(f"output: {output.shape}")
    
    loss = (e - output).square()
    # print(f"loss: {loss.shape}")
    loss = mask * loss
    loss = loss.sum(dim=(1, 2, 3))
    # print(f"loss: {loss.shape}")
    # exit()
    
    if keepdim:
        return loss
    else:
        return loss.mean(dim=0)
    

def create_gradient_mask(vorticity_data, alpha = 2):
    grad_x = torch.abs(vorticity_data[:, :, 1:, :] - vorticity_data[:, :, :-1, :])
    grad_y = torch.abs(vorticity_data[:, :, :, 1:] - vorticity_data[:, :, :, :-1])

    grad_x = F.pad(grad_x, (0, 0, 1, 0))
    grad_y = F.pad(grad_y, (1, 0, 0, 0))

    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_norm = (grad_magnitude - global_min_grad) / (global_max_grad - global_min_grad)

    mask = torch.ones_like(grad_norm)
    mask = torch.where(grad_norm > tau_1, 1.25 + (alpha - 1) * (grad_norm - tau_1) / (1 - tau_1), mask)
    mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)

    return mask


def create_wavelet_binary_mask(vorticity_data, alpha=6.0):
    vorticity_data = vorticity_data.to(torch.float32)
    
    Yl, Yh = dwt(vorticity_data)
    
    LH = Yh[0][:, :, 0, :, :]
    HL = Yh[0][:, :, 1, :, :]
    HH = Yh[0][:, :, 2, :, :]

    high_freq_details = torch.abs(LH) + torch.abs(HL) + torch.abs(HH)
    
    max_value = high_freq_details.max()
    
    threshold = torch.quantile(high_freq_details.flatten(), 0.80)
    
    mask = torch.where(
        high_freq_details > threshold,
        1.25 + (alpha - 1.25) * (high_freq_details - threshold) / (max_value - threshold),
        torch.ones_like(high_freq_details)
    )
    
    upsampled_mask = F.interpolate(mask, size=(256, 256), mode='nearest')
    upsampled_mask = F.max_pool2d(upsampled_mask, kernel_size=3, stride=1, padding=1)

    return upsampled_mask

loss_registry = {
    'simple': noise_estimation_loss
}



