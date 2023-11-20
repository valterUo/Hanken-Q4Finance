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


def plot_series(T_norm, X_norm, T_anom = None, Y_anom = None):
    plt.figure()
    plt.plot(T_norm, X_norm[0], label="Normal")
    if T_anom and Y_anom:
        plt.plot(T_anom, Y_anom[1], label="Anomalous")
    plt.ylabel("$y(t)$")
    plt.xlabel("t")
    plt.grid()
    leg = plt.legend()
    

def convert_to_binary(symbols, bits):
    binary_representation = [''.join(format(symbol, f'0{bits}b') for symbol in row) for row in symbols]
    return np.array(binary_representation)


def apply_sax(time_series):
    # Normalize the time series
    scaler = TimeSeriesScalerMeanVariance()
    time_series_scaled = scaler.fit_transform(time_series)

    # Apply SAX with a larger alphabet size
    n_paa_segments = 4  # Number of segments
    n_sax_symbols = 4  # Alphabet size (4 symbols: 0, 1, 2, 3)
    sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
    time_series_sax = sax.fit_transform(time_series_scaled)

    # Convert SAX symbols to binary representation
    sax_symbols = sax.inverse_transform(time_series_sax)
    binary_sax_symbols = convert_to_binary(sax_symbols, bits=2)

    print("Original Time Series:", time_series)
    print("Normalized Time Series:", time_series_scaled)
    print("SAX Representation:", sax_symbols)
    print("Binary SAX Representation:", binary_sax_symbols)
    
    return binary_sax_symbols


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
    M = [[0]*n for _ in range(n)]
    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1
    # Convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M