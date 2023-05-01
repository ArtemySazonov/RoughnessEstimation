import math
from typing import Union
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from scipy.stats import shapiro
from scipy.stats import normaltest
#import yfinance as yf
from copy import error
import math
from numba import njit, prange
from sklearn.metrics import mean_squared_error, r2_score

NUM_OF_Q      = 5
MINUTES_COUNT = 15
DAYS_COUNT    = 10
STEP_OF_Q     = 0.3
sD            = 1
pD            = 40


@njit(parallel=True)
def m(q, Delta, volatility_array):
    return np.mean(np.power(np.abs(np.diff(np.log(volatility_array[::Delta]))), q))

@njit(parallel=True)
def ACov(volatility_array, Delta):
    retval = 0
    N = volatility_array.size - Delta

    if N < 1:
        return 0
    
    retval = np.sum(volatility_array[:-Delta:]*volatility_array[Delta::]) / N
    return math.log(retval)

@njit(parallel=True)
def logCov(volatility_array, Delta):
    N   = volatility_array.size - Delta

    if N < 1:
        return 0
    
    volatility_array = np.log(volatility_array)

    IID = np.sum(volatility_array[:-Delta:]*volatility_array[Delta::]) / N
    I   = np.sum(volatility_array[:Delta:]) / N
    ID  = np.sum(volatility_array[Delta::]) / N

    return math.log(IID - I*ID)

@njit 
def skew(x1, x2, y1, y2):
    return (y2 - y1) / (x2 - x1)


def rlz_vol_est(df: pd.DataFrame, count: int, rolling_window: int=1) -> np.ndarray:
    rolling_window = int(rolling_window)
    returns = np.array(df["Mean"].to_numpy())
    log_returns = np.log(returns[1::rolling_window]) - np.log(returns[:-1:rolling_window])

    log_returns = log_returns[:int(log_returns.size/count)*count:]

    rlz_vol = np.sqrt(np.sum(np.power(np.diff(log_returns.reshape((int(log_returns.size/count), count)), axis=1), 2), axis=1))
    return rlz_vol

@njit
def excessed_kurtosis(X: np.ndarray) -> float:
    X = (X-np.mean(X))/np.std(X)
    return np.mean(X**4)-3

from . import utils

def analyse_volatility(name: str, mode: str='bb', rolling_window: int=1, show_pics = True, save_pics = False, smoothing: bool = False):
    if mode == 'yf':
        count = DAYS_COUNT
        #df = yf.download(name, '2000-01-01', '2019-01-01')
        #df["Mean"] = 0.5*(df["Open"]+df["Close"])
        #volatility_array = rlz_vol_est(df, count, rolling_window=rolling_window)
    elif mode == 'bb':
        count = MINUTES_COUNT
        df = pd.read_csv('data_bloomberg/'+name+'.csv', sep="\t")  
        df["Mean"] = 0.5*(df["High"]+df["Low"])
        volatility_array = utils.rlz_vol_est(df["Mean"].to_numpy(), count)
    elif mode == 'om':
        count = MINUTES_COUNT
        df = pd.read_csv('oxfordmanrealizedvolatilityindices.csv', sep=",")
        df = df[df["Symbol"] == name]
        volatility_array = np.sqrt(df["rv10"].to_numpy())
    else:
        error("Wrong mode")
        return -1

    if smoothing:
        show_pics = False
        save_pics = False
    
    if show_pics:
        print("Report on " + name)

    zetaq = np.zeros((2, NUM_OF_Q))

    if show_pics:
        figsize = (21, 29)
        fig, ax = plt.subplots(3, 3, figsize = figsize)
        if mode == 'bb':
            vol_time = np.arange(volatility_array.size, 0, -1)
        else:
            vol_time = np.arange(1, volatility_array.size+1, 1)
        ax[0, 0].plot(vol_time, volatility_array)
        ax[0, 0].set_xlabel("Time")
        ax[0, 0].set_ylabel("Realized volatility")

    for I in range(0, NUM_OF_Q):
        graph_data = np.zeros((2, pD-sD))
        q          = STEP_OF_Q*(1+I)
        line_start = math.log(sD)
        line_stop  = math.log(pD)
        
        for Delta in range(sD, pD):
            graph_data[0, Delta-sD] = math.log(Delta)
            graph_data[1, Delta-sD] = math.log(m(q, Delta, volatility_array))

        linear_model    = np.polyfit(graph_data[0],graph_data[1], 1)
        linear_model_fn = np.poly1d(linear_model)
        x_s             = np.arange(line_start, line_stop, 0.1)

        if show_pics:
            ax[0, 1].plot(x_s, linear_model_fn(x_s))
            ax[0, 1].scatter(graph_data[0], graph_data[1], label=str(round(q, 2)))

        skew_of_linear_model = skew(line_start, line_stop, linear_model_fn(line_start), linear_model_fn(line_stop))

        zetaq[0, I] = q
        zetaq[1, I] = skew_of_linear_model

    if show_pics:
        ax[0, 1].set_xlabel("$\log \Delta$")
        ax[0, 1].set_ylabel("$\log m$")
        ax[0, 1].legend()
    
    linear_model_H    = np.polyfit(zetaq[0], zetaq[1], 1)
    linear_model_H_fn = np.poly1d(linear_model_H)
    x_s               = np.arange(0, STEP_OF_Q*(NUM_OF_Q+1), STEP_OF_Q)

    if show_pics:
        ax[0, 2].plot(x_s, linear_model_H_fn(x_s), color="red")
        ax[0, 2].scatter(zetaq[0], zetaq[1])
        ax[0, 2].set_xlabel("$q$")
        ax[0, 2].set_ylabel("$\zeta_q$")

    H_est = skew(0, STEP_OF_Q*(NUM_OF_Q)+1, linear_model_H_fn(0), linear_model_H_fn(STEP_OF_Q*(NUM_OF_Q)+1))

    if smoothing:
        return H_est

    if show_pics:
        print("Estimated H parameter for " + name + " is equal to " + str(H_est))

    sz = 40
    graph_data = np.zeros((2, sz))

    for Delta in range(1, sz+1):
        graph_data[0, Delta-1] = Delta**(2*H_est)
        graph_data[1, Delta-1] = ACov(volatility_array, Delta)

    linear_model    = np.polyfit(graph_data[0],graph_data[1], 1)
    linear_model_fn = np.poly1d(linear_model)
    x_s             = np.arange(1, (sz+1)**(2*H_est), 0.1)

    if show_pics:
        ax[1, 0].plot(x_s, linear_model_fn(x_s), color="red")
        ax[1, 0].scatter(graph_data[0], graph_data[1])
        ax[1, 0].set_xlabel("$\Delta^{ 2H }$")
        ax[1, 0].set_ylabel("$\log E[\sigma_t \sigma_{t+\Delta}]$")

    for Delta in range(1, sz+1):
        graph_data[0, Delta-1] = math.log(Delta)
        graph_data[1, Delta-1] = logCov(volatility_array, Delta)

    linear_model    = np.polyfit(graph_data[0],graph_data[1], 1)
    linear_model_fn = np.poly1d(linear_model)
    x_s             = np.arange(0, math.log(sz+1), 0.1)

    if show_pics:
        if not save_pics:
            fig.suptitle(name)
        ax[1, 1].plot(x_s, linear_model_fn(x_s), color="red")
        ax[1, 1].scatter(graph_data[0], graph_data[1])
        ax[1, 1].set_xlabel("$\log \Delta$")
        ax[1, 1].set_ylabel("$\log Cov[\sigma_t, \sigma_{t+\Delta}]$")
        
    def lag_array(Delta):
        retarr = np.zeros(volatility_array.size - Delta) 
        if Delta >= 0:
            for i in prange(0, volatility_array.size-Delta):
                retarr[i] = np.log(volatility_array[i+Delta]) - np.log(volatility_array[i])
        else:
            for i in prange(0, volatility_array.size-math.abs(Delta)):
                retarr[i] = np.log(volatility_array[i]) - np.log(volatility_array[i-Delta])

        retarr = retarr/retarr.max()
        return retarr           

    if show_pics:
        bins_num  = 50
        X         = np.linspace(-2, 2, 500)
        indices1  = np.array([1, 2, 2, 2])
        indices2  = np.array([2, 0, 1, 2])
        Deltas    = np.array([1, 5, 10, 20])
        stdla1    = np.std(lag_array(1))
        for i in range(0, 4):
            la = lag_array(Deltas[i])
            ax[indices1[i], indices2[i]].hist(la, bins = bins_num, density = True, label='Emprirical density')
            sns.kdeplot(la, label='KDE', ax = ax[indices1[i], indices2[i]])
            ax[indices1[i], indices2[i]].set_title("Density of $\log \sigma_{ t+\Delta} - \log \sigma_{ t}$ with " + str(Deltas[i]) + " day lag, $\kappa = $" + str(round(excessed_kurtosis(la), 3)))
            ax[indices1[i], indices2[i]].plot(X, norm.pdf(X, 0, np.std(la)), color="red", label="Normal fit")
            ax[indices1[i], indices2[i]].plot(X, norm.pdf(X, 0, stdla1*Deltas[i]**H_est), color="green", label="Empirical fit")
            ax[indices1[i], indices2[i]].legend()

        for i in range(3):
            for j in range(3):
                ax[i, j].grid(True)
            
        if save_pics:
            plt.savefig("fig/" + name + ".pdf")
        
        plt.show()
        
    curt_range = 50
    curt_array = np.zeros(curt_range)

    return H_est

# Same as analyse_volatility method, only for multiple pictures output
def analyse_volatility_multipic(name: str, mode: str='yf', rolling_window: int=1, show_pics = True, save_pics = False):
    if mode == 'yf':
        count = DAYS_COUNT
        #df = yf.download(name, '2000-01-01', '2019-01-01')
        #df["Mean"] = 0.5*(df["Open"]+df["Close"])
        #volatility_array = rlz_vol_est(df, count, rolling_window=rolling_window)
    elif mode == 'bb':
        count = MINUTES_COUNT
        df = pd.read_csv('data_bloomberg/'+name+'.csv', sep="\t")  
        df["Mean"] = 0.5*(df["High"]+df["Low"])
        volatility_array = rlz_vol_est(df, count, rolling_window=rolling_window)
    elif mode == 'om':
        count = MINUTES_COUNT
        df = pd.read_csv('oxfordmanrealizedvolatilityindices.csv', sep=",")
        df = df[df["Symbol"] == name]
        volatility_array = np.sqrt(df["rv10"].to_numpy())

    if show_pics:
        print("Report on " + name)

    volatility_array = rlz_vol_est(df, count, rolling_window=rolling_window)
    volatility_array = volatility_array[~np.isnan(volatility_array)]
    zetaq            = np.zeros((2, NUM_OF_Q))

    if show_pics:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 7), sharex=True)
        if mode == 'yf':
            sto_time = np.arange(1, df["Mean"].size+1, 1)
            vol_time = np.arange(1, volatility_array.size+1, 1)
        elif mode == 'bb':
            sto_time = np.arange(df["Mean"].size, 0, -1)
            vol_time = np.arange(volatility_array.size, 0, -1)

        ax1.plot(sto_time, df["Mean"])
        ax1.set_ylabel("Price, RUB")

        ax2.plot(vol_time*15, volatility_array)
        ax2.set_xlabel("Time (minutes)")
        ax2.set_ylabel("RV")

        if save_pics:
            plt.savefig("fig/" + name + " RVol.pdf")

        plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))
    
    for I in range(0, NUM_OF_Q):
        graph_data = np.zeros((2, pD-sD))
        q          = STEP_OF_Q*(1+I)
        line_start = math.log(sD)
        line_stop  = math.log(pD)
        
        for Delta in range(sD, pD):
            graph_data[0, Delta-sD] = math.log(Delta)
            graph_data[1, Delta-sD] = math.log(m(q, Delta, volatility_array))

        linear_model    = np.polyfit(graph_data[0],graph_data[1], 1)
        linear_model_fn = np.poly1d(linear_model)
        x_s             = np.arange(line_start, line_stop, 0.1)

        if show_pics:
            ax1.plot(x_s, linear_model_fn(x_s))
            ax1.scatter(graph_data[0], graph_data[1], label=str(round(q, 2)))

        skew_of_linear_model = skew(line_start, line_stop, linear_model_fn(line_start), linear_model_fn(line_stop))

        zetaq[0, I] = q
        zetaq[1, I] = skew_of_linear_model


    if show_pics:
        ax1.set_xlabel("$\log \Delta$")
        ax1.set_ylabel("$\log m$")
        ax1.legend()
    
    linear_model_H    = np.polyfit(zetaq[0], zetaq[1], 1)
    linear_model_H_fn = np.poly1d(linear_model_H)
    x_s               = np.arange(0, STEP_OF_Q*(NUM_OF_Q+1), STEP_OF_Q)

    if show_pics:
        ax2.plot(x_s, linear_model_H_fn(x_s), color="red")
        ax2.scatter(zetaq[0], zetaq[1])
        ax2.set_xlabel("$q$")
        ax2.set_ylabel("$\zeta_q$")

        if save_pics:
            plt.savefig("fig/" + name + " Hurst Est.pdf")

        plt.show()

    H_est = skew(0, STEP_OF_Q*(NUM_OF_Q)+1, linear_model_H_fn(0), linear_model_H_fn(STEP_OF_Q*(NUM_OF_Q)+1))

    sz = 30
    graph_data = np.zeros((2, sz))

    for Delta in range(1, sz+1):
        graph_data[0, Delta-1] = Delta**(2*H_est)
        graph_data[1, Delta-1] = ACov(volatility_array, Delta)

    linear_model    = np.polyfit(graph_data[0],graph_data[1], 1)
    linear_model_fn = np.poly1d(linear_model)
    x_s             = np.arange(1, (sz+1)**(2*H_est), 0.1)

    if show_pics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

        ax1.plot(x_s, linear_model_fn(x_s), color="red")
        ax1.scatter(graph_data[0], graph_data[1])
        ax1.set_xlabel("$\Delta^{ 2H }$")
        ax1.set_ylabel("$\log E[\sigma_t \sigma_{t+\Delta}]$")

    for Delta in range(1, sz+1):
        graph_data[0, Delta-1] = math.log(Delta)
        graph_data[1, Delta-1] = logCov(volatility_array, Delta)

    linear_model    = np.polyfit(graph_data[0],graph_data[1], 1)
    linear_model_fn = np.poly1d(linear_model)
    x_s             = np.arange(0, math.log(sz+1), 0.1)

    if show_pics:
        ax2.plot(x_s, linear_model_fn(x_s), color="red")
        ax2.scatter(graph_data[0], graph_data[1])
        ax2.set_xlabel("$\log \Delta$")
        ax2.set_ylabel("$\log Cov[\sigma_t, \sigma_{t+\Delta}]$")

        if save_pics:
            plt.savefig("fig/" + name + " logE vs logD.pdf")
        plt.show()
        
    def lag_array(Delta):
        retarr = np.zeros(volatility_array.size - Delta) 
        if Delta >= 0:
            for i in range(0, volatility_array.size-Delta):
                retarr[i] = np.log(volatility_array[i+Delta]) - np.log(volatility_array[i])
        else:
            for i in range(0, volatility_array.size-math.abs(Delta)):
                retarr[i] = np.log(volatility_array[i]) - np.log(volatility_array[i-Delta])

        retarr = retarr/retarr.max()
        return retarr
    
    def normality_table_LaTeX():
        with open("tab/Normality tests " + name + " (cut).tex", "w") as f:
            f.write('\\begin{table}[h]\n')
            f.write('\t\\begin{tabular}{|c|c|c|c|c|c|}\n')
            f.write('\t\t\\hline\n')
            f.write('\t\t$\Delta$ &  Shapiro-Wilk (stat) & Shapiro-Wilk (p-value) & $K^2$ (stat) & $K^2$ (p-value) & Conclusion\\\\\\hline\n\t\t\\hline\n')
            
            stat_range = 30
            for i in range(1, stat_range+1):
                la = lag_array(i)

                statSW, pSW = shapiro(la)
                statK2, pK2 = normaltest(la)
                alpha = 0.05
                if pSW > alpha or pK2 > alpha:
                    conclusion = "Normal"
                else:
                    conclusion = "Not normal"
            
                f.write('\t\t' + str(i) + ' & ' + str(format(statSW,'.5E')) + ' & ' + str(format(pSW,'.3E')) + ' & ' + str(format(statK2,'.5E'))+ ' & ' + str(format(pK2,'.3E')) + ' & ' + conclusion +'\\\\\\hline\n')
            f.write('\t\\end{tabular}\n')
            f.write('\t\\caption{Normality tests for ' + name + '}\n')
            f.write('\t\\label{tab:normality_tests_' + name.split(" ")[0] + "_" + name.split(" ")[1] + '_cut}\n')
            f.write('\\end{table}\n')

    normality_table_LaTeX()

    fig, axs = plt.subplots(2, 2, figsize = (16, 10), sharex=True, sharey=True)

    if show_pics:
        bins_num = 50
        Deltas = np.array([[1, 5], [10, 20]])
        X = np.linspace(-2, 2, 500)
        if not save_pics:
            fig.suptitle("Density of $\log \sigma_{t+\Delta} - \log \sigma_{t}$ for " + name)
        
        stdla1 = np.std(lag_array(Deltas[0][0]))
        for i in range(2):
            for j in range(2):
                la = lag_array(Deltas[i][j])
                axs[i, j].hist(la, bins = bins_num, density = True, label='Empirical density')
                sns.kdeplot(la, label='KDE', ax = axs[i, j])
                axs[i, j].plot(X, norm.pdf(X, 0, np.std(la)), color="red", label="Normal fit")
                axs[i, j].plot(X, norm.pdf(X, 0, stdla1*Deltas[i, j]**H_est), color="green", label="Empirical fit")
                axs[i, j].set_title("$\Delta = " + str(Deltas[i, j]) + "$")
                axs[i, j].legend()
                axs[i, j].grid()

        if save_pics:
            fig.savefig("fig/" + name + " " + str(Delta) + " Lag Hists.pdf")
        plt.show()

    if show_pics:
        curt_array = np.zeros(60)
        for i in range(1, 61):
            la = lag_array(i)
            curt_array[i-1] = excessed_kurtosis(la)
        plt.figure(figsize=(12, 5))
        plt.plot(np.arange(1, 61), curt_array, '-x')
        plt.xlabel("Lag (Days)")
        plt.ylabel("$\kappa$")
        plt.grid(True)
        if save_pics:
            plt.savefig("fig/" + name + " Excessed Curtosis.pdf")
        plt.show()
    

    return H_est

def export_report_as_csv(name: Union[str, np.ndarray], mode='bb'):
    t = time.localtime()
    current_time = time.strftime("%H-%M-%S", t)
    with open("report_" + str(current_time) +".csv", "x") as f:
        f.write("equity,H_est\n")
        for equity in name:
            f.write(str(equity)+','+str(round(analyse_volatility(equity, mode=mode, show_pics=False), 7))+'\n')



######################
#  Smoothing effect  #
######################

def f(theta, H=0.5):
    return (1/((2*H+1)*(2*H+2)*theta**2)*((1+theta)**(2*H+2) - 2 - 2 * theta**(2*H+2) + (1-theta)**(2*H+2)))    

def smoothing_theoretical(delta: float, H: float):
    num_of_Deltas = 200
    plot = np.zeros((2, num_of_Deltas))

    Delta = np.arange(1, num_of_Deltas+1, 1)
    plot[0] = np.log(Delta)
    plot[1] = np.log(Delta**(2*H) * f(delta/Delta, H))

    linear_model    = np.polyfit(plot[0],plot[1], 1)
    linear_model_fn = np.poly1d(linear_model)
    x_s             = np.arange(0, 5, 0.1)

    print(skew(0, 1, linear_model_fn(0), linear_model_fn(1))*0.5)
    print(skew(0, 1, linear_model_fn(0), linear_model_fn(1))*0.5/H - 1)


    plt.scatter(plot[0], plot[1])
    plt.plot(x_s, linear_model_fn(x_s))
    plt.xlabel("$\log \Delta$")
    plt.ylabel("$\Delta^{ 2H } f(\delta/\Delta)$")
    plt.title("Theoretical Estimation of Smoothing Effect for YNDX RX Equity")
    plt.show()


def smoothing_empirical(name: str, count: int, show_pics: bool=True, save_pics: bool=False):
    num_of_wind = MINUTES_COUNT
    graph_data = np.zeros((2, num_of_wind))
    for i in range(1, num_of_wind+1):
        graph_data[0, i-1] = i
        graph_data[1, i-1] = analyse_volatility(name, mode='bb', rolling_window=i, smoothing = True)  

    if show_pics:
        linear_model    = np.polyfit(graph_data[0], graph_data[1], 1)
        linear_model_fn = np.poly1d(linear_model)
        x_s             = np.arange(0, count, 0.1)
        
        plt.figure(figsize=(12, 5))

        plt.plot(x_s, linear_model_fn(x_s), color="red", label="Skew = " + str(round((linear_model_fn(count)-linear_model_fn(0))/count, 6)))

        plt.scatter(graph_data[0], graph_data[1])
        plt.xlabel("Rolling Window")
        plt.ylabel("H Estimate") 
        plt.legend()
        plt.grid(True)
        mse = mean_squared_error(graph_data[1], linear_model_fn(graph_data[0])) 
        plt.title('RMSE = ' + str(round(np.sqrt(mse), 6)) + '$;  R^2$ = ' + str(round(r2_score(graph_data[1], linear_model_fn(graph_data[0])), 6)))

        if save_pics:
            plt.savefig("fig/" + name + " Smoothing Effect.pdf")
        plt.show()

