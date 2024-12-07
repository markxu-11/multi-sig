import numpy as np

# Vectorised
def rms(w):
    return np.sqrt(np.mean(np.square(w), axis=-1))

def mav(w):
    return np.mean(np.abs(w), axis=-1)

def zc(w):
    return np.count_nonzero(np.diff(np.sign(w), axis=-1), axis=-1)

def wl(w):
    return np.sum(np.abs(np.diff(w, axis=-1)), axis=-1)

def wamp_5(w):
    return np.sum(np.abs(np.diff(w, axis=-1)) > 5, axis=-1)

# Non-vectorised
def ssc(w):
    diff = np.diff(w)
    return np.sum((diff[:-1] * diff[1:]) < 0)
