import torch
import math

def pi(n):
    return math.pi * n

def gen_dataset(n_max, file_path='data/pi_dataset.pt'):
    data = []
    for n in range(1, n_max + 1):
        pn = pi(n)
        data.append((n, pn))
    torch.save(data, file_path)

n_max = 1000000
gen_dataset(n_max, 'data/pi_dataset.pt')