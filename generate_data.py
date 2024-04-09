import torch
import math

FN_NAME = 'pointil'

def pointil(n):
    if n % 240 == 0:
        return 1
    else:
        return 0

def gen_dataset(n_max, file_path=f'data/{FN_NAME}_dataset.pt'):
    data = []
    for n in range(1, n_max + 1):
        fn = pointil(n)
        data.append((n, fn))
    torch.save(data, file_path)

n_max = 1000000
gen_dataset(n_max, f'data/{FN_NAME}_dataset.pt')