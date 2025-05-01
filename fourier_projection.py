import numpy as np
from scipy.fft import fft, ifft

def encode(x, k):
    fx = fft(x)
    return np.real(fx[:k])

def decode(z, d):
    fx_full = np.zeros(d, dtype=complex)
    fx_full[:len(z)] = z
    return np.real(ifft(fx_full))
