from matplotlib import pyplot as plt
import torch
import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import SymbolicAggregateApproximation

GLOBAL_SEED = 0
torch.manual_seed(GLOBAL_SEED)
torch.set_default_tensor_type(torch.DoubleTensor)

def generate_normal_time_series_set(p: int, 
                                    num_series: int, 
                                    noise_amp: float, 
                                    t_init: float, 
                                    t_end: float, 
                                    seed: int = GLOBAL_SEED) -> tuple:
    torch.manual_seed(seed)
    X = torch.normal(0.5, noise_amp, (num_series, p))
    T = torch.linspace(t_init, t_end, p)
    return X, T


def generate_anomalous_time_series_set(
    p: int,
    num_series: int,
    noise_amp: float,
    spike_amp: float,
    max_duration: int,
    t_init: float,
    t_end: float,
    seed: int = GLOBAL_SEED,
) -> tuple:
    """Generate an anomalous time series data set where the p elements of each sequence are
    from a normal distribution x_t ~ N(0, noise_amp). Then,
    anomalous spikes of random amplitudes and durations are inserted.
    """
    torch.manual_seed(seed)
    Y = torch.normal(0, noise_amp, (num_series, p))
    for y in Y:
        # 5–10 spikes allowed
        spike_num = torch.randint(low=5, high=10, size=())
        durations = torch.randint(low=1, high=max_duration, size=(spike_num,))
        spike_start_idxs = torch.randperm(p - max_duration)[:spike_num]
        for start_idx, duration in zip(spike_start_idxs, durations):
            y[start_idx : start_idx + duration] += torch.normal(0.0, spike_amp, (duration,))
    T = torch.linspace(t_init, t_end, p)
    return Y, T


def generate_synthetic_correlated_time_series(T, N):
    mu = np.array([-0.1, -0.2])
    sigma = np.array([[1, -0.5], [-0.5, 1]])
    dt = T / N
    M = len(mu)
    X0 = np.array([0, 0])
    dW = np.random.normal(0, np.sqrt(dt), (M, N))

    # Euler-Maruyama method
    X = np.zeros((M, N+1))
    X[:, 0] = X0

    for i in range(N):
        X[:, i+1] = X[:, i] + mu * dt + sigma @ dW[:, i]
    
    return X


def plot_series(T_norm, X_norm, T_anom = None, Y_anom = None):
    plt.figure()
    plt.plot(T_norm, X_norm[0], label="Normal")
    if T_anom and Y_anom:
        plt.plot(T_anom, Y_anom[1], label="Anomalous")
    plt.ylabel("$y(t)$")
    plt.xlabel("t")
    plt.grid()
    leg = plt.legend()
    
def convert_to_binary(time_series, width):
    if type(time_series[0]) == np.ndarray:
        result = []
        for ts in time_series:
            result.append(convert_to_binary(ts, width))
    else:
        result = np.array([np.binary_repr(x, width=width) for x in time_series])
    return result

def apply_sax(scaler, sax, time_series, n_segments, n_sax_symbols=4, inverse=False):
    if not inverse:
        # Normalize the time series
        time_series_scaled = scaler.fit_transform(time_series)
        n_paa_segments = n_segments  # Number of segments, it is not clear how to choose this parameter?
        n_sax_symbols = n_sax_symbols  # Alphabet size, e.g., 4 symbols: 0, 1, 2, 3
        time_series_sax = sax.fit_transform(time_series_scaled)
        binary_time_series = convert_to_binary(time_series_sax, n_sax_symbols)
        #print("Sax time series:", time_series_sax)
        return binary_time_series, time_series_sax
    else:
        time_series_inverse = sax.inverse_transform(sax.fit_transform(time_series))
        return time_series_inverse


# The following code takes a list such as
# [1,1,2,6,8,5,5,7,8,8,1,1,4,5,5,0,0,0,1,1,4,4,5,1,3,3,4,5,4,1,1]
# with states labeled as successive integers starting with 0
# and returns a transition matrix M,
# where M[i][j] is the probability of transitioning from i to j
# For m-dimensional time series X = (s_1, ..., s_l, ..., s_m)
# we use this function for each s_l
# This classical data is used in the training of the model
def classical_transition_matrix(transitions):
    n = 1 + max(transitions) #number of states
    # Initialize n x n dictionary of 0s
    M = [[0] * n for _ in range(n)]
    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1
    # Convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

def flatten_recursive(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten_recursive(item)
        elif isinstance(item, np.ndarray):
            yield from flatten_recursive(item)
        elif isinstance(item, torch.Tensor):
            yield from flatten_recursive(item)
        else:
            yield item