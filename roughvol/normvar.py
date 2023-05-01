import numpy as np
from scipy.optimize import root_scalar, minimize
from numba import njit
from dataclasses import dataclass, field
from typing import Union

@dataclass
class MarketState:
    stock_price: Union[float, np.ndarray]
    interest_rate: Union[float, np.ndarray]

@dataclass
class HestonParameters:
    kappa:  Union[float, np.ndarray]
    gamma:  Union[float, np.ndarray]
    rho:  Union[float, np.ndarray]
    vbar:  Union[float, np.ndarray]
    v0:  Union[float, np.ndarray]

@njit(parallel=True)
def sample_variation(array: np.ndarray, 
                     p: float = 2):
    return np.sum(np.power(np.abs(np.diff(array)), p))

@njit(parallel=True)
def sample_normalized_variation(array:              np.ndarray,   # 
                                block_frequency:    np.ndarray,   # a strictly monotonic array, way less than the sampling_frequency
                                normalizing_array:  np.ndarray,   # 
                                sampling_frequency: np.ndarray,   # a strictly monotonic array
                                p:                  float = 1):

    array             = np.power(np.abs(np.diff(array)), p)
    normalizing_array = np.abs(np.diff(normalizing_array))
    sample_norm_var   = 0.0
    divprev           = 1.0
    for i in range(0, block_frequency.size-1):
        div = np.sum(np.power(normalizing_array[np.arange(np.where(sampling_frequency == block_frequency[i])[0], np.where(sampling_frequency == block_frequency[i+1])[0])], p))
        if np.isclose(div, 0.0):
            div = divprev
        else:
            divprev = div

        sample_norm_var += array[i] * (block_frequency[i+1] - block_frequency[i]) / div

    return sample_norm_var

def sample_roughness(array: np.ndarray, 
                     time:  np.ndarray) -> float:
    step = int(time.size*0.01)
    
    def W(p: float):
        return sample_normalized_variation(array[np.arange(0, time.size, step)], time[np.arange(0, time.size, step)], array, time, p) 
    
    def to_be_min_OLS(p: float):
        return (W(p) - time[time.size-1])**2
    
    return 1 / minimize(fun=to_be_min_OLS, x0=2.0, method='Powell').x[0]

