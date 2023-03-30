import multiprocessing as mp
import torch
from joblib import Parallel, delayed
import os

def worker(model):
    print(f'hello world from {os.getpid()} and {model}')

if __name__ == '__main__':
    model = torch.load('./ressources/model_mnist.pth')

    Parallel(n_jobs=-1, backend='multiprocessing', prefer='processes')(delayed(worker)(model) for i in range(mp.cpu_count()))