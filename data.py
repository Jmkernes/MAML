import torch
import numpy as np

class SinusoidGenerator:
    def __init__(self, K=10):
        self.K = K
        self.A = np.array(np.random.uniform(0.1, 5.0), dtype=np.float32)
        self.phi = np.array(np.random.uniform(0, np.pi), dtype=np.float32)
    def sample(self, K=None):
        K = self.K if K is None else K
        x = np.array(np.random.uniform(-5.0, 5.0, self.K), dtype=np.float32)
        return x, self(x)
    def __call__(self, x):
        return self.A * np.sin(x - self.phi)
    
class Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 512
    def __getitem__(self, idx):
        return SinusoidGenerator().sample()