import numpy as np
import matplotlib.pyplot as plt
import math

def dedekind_eta(tau):
    q = np.exp(2j * np.pi * tau)
    eta = q**(1/24) * np.product([1 - q**n for n in range(1, 100)])
    return eta.real

def log_gamma(x):
    return math.log(math.gamma(x))

def sin184(x):
    return math.sin((pow(x, 1.84)).real)

inputs = np.linspace(0.2, 10.0, 10000)
print(inputs)
values = np.array([log_gamma(input) for input in inputs])
print(values)
plt.figure(figsize=(10, 6))
plt.plot(inputs.real, values, label="log(gamma(x))")
plt.title("log(gamma(x))")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()