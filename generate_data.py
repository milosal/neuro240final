import torch
import math

FN_NAME = 'rat'

def rat(n):

    return (pow(n, 3) / 2) / (pow(n, 2) - 1000001)

def gen_dataset(n_max, file_path=f'data/{FN_NAME}_dataset.pt'):
    data = []
    for n in range(1, n_max + 1):
        fn = rat(n)
        data.append((n, fn))
    torch.save(data, file_path)

n_max = 1000000
gen_dataset(n_max, f'data/{FN_NAME}_dataset.pt')