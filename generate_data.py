import torch
import math

FN_NAME = 'step'

def sin(n):
    return 2 * math.floor(n / 100)

def gen_dataset(n_max, file_path=f'data/{FN_NAME}_dataset.pt'):
    data = []
    for n in range(1, n_max + 1):
        fn = sin(n)
        data.append((n, fn))
    torch.save(data, file_path)

n_max = 1000000
gen_dataset(n_max, f'data/{FN_NAME}_dataset.pt')