import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit, prange

@njit(parallel=True)
def log_returns(array: np.ndarray) -> np.ndarray:
    return np.diff(np.log(array))

@njit(parallel=True)
def rlz_vol_est(array: np.ndarray, 
                count: int
                ) -> np.ndarray:
    lr = log_returns(array)

    rlz_vol = np.zeros(int(lr.size/count))

    for i in prange(rlz_vol.size):
        rlz_vol[i] = np.sqrt(np.sum(np.power(np.diff(lr[i*count:(i+1)*count]), 2)))

    return rlz_vol

def custom_hist(array: np.ndarray, title: str, vline: float = None, save: bool = False):
    array = array[array>0]
    sns.kdeplot(array, label='KDE', color='r')
    sns.histplot(array, stat='density').set(title=title)
    if not vline == None:
        plt.axvline(vline, color='r', linestyle='--', label = 'Real Value')
    plt.axvline(np.mean(array), color='b', linestyle='-', label='$\mu$')
    plt.axvline(np.mean(array)+np.std(array), color='b', linestyle='-.', label='$\mu\pm\sigma$')
    plt.axvline(np.mean(array)-np.std(array), color='b', linestyle='-.')
    plt.legend()
    if save:
        plt.savefig(f'fig2/{title}.pdf')
    plt.show()
    
    print(f'mean: {np.mean(array)}, std: {np.std(array)}')