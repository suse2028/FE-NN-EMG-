import numpy as np
import pandas as pd
import torch
import scipy
import scipy.signal as signal
from scipy.signal import butter, lfilter
from scipy.fft import fft, fftshift
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt

def get_data(file):
    data = pd.read_csv(file)


#applying bandpass filter
def bandpass_filter(signal, crit_freq=[20, 450], sampling_freq=125, plot=False, channel=0):
    order = 4

    b, a = scipy.signal.butter(order, crit_freq, btype='bandpass', fs=sampling_freq)
    processed_signal = scipy.signal.filtfilt(b, a, signal, 1)


    plt.figure()
    plt.xlabel('Time')
    plt.ylabel(f'Normalized amplitude of channel {channel}')
    plt.title(f'{crit_freq[0]}-{crit_freq[1]}Hz bandpass filter')
    signal_min = np.full((signal.shape[1], signal.shape[0]), np.min(signal, 1)).transpose()
    signal_max = np.full((signal.shape[1], signal.shape[0]), np.max(signal, 1)).transpose()
    normed_signal = (signal - signal_min) / (signal_max - signal_min)
    filtered_min = np.full((processed_signal.shape[1], processed_signal.shape[0]),
    np.min(processed_signal, 1)).transpose()
    filtered_max = np.full((processed_signal.shape[1], processed_signal.shape[0]),
                                   np.max(processed_signal, 1)).transpose()
    normed_filt = (processed_signal - filtered_min) / (filtered_max - filtered_min)
    plt.plot(np.arange(normed_signal[channel].size), normed_signal[channel], label='Input')
    plt.plot(np.arange(normed_filt[channel].size), normed_filt[channel], label='Transformed')
    plt.legend()

    return processed_signal

def notchElim(processed_signal):
    new_processed_signal = []
    w0 = [50,60]
    Q0 = 30
    new_processed_signal = processed_signal
    for f0 in w0:
        b, a = signal.iirnotch(f0, Q0)
        emg_filtered = signal.filtfilt(b, a, processed_signal)
    for data in processed_signal:
        new_processed_signal += np.absolute(data)

#EMG Rectification
def rectify(processed_signal):
    processed_signal_copy = processed_signal.copy()
    new_processed_signal = []
    for data in processed_signal_copy:
        new_processed_signal += np.absolute(data)

    #processed signal still contains positivity/negativity, will be important for setting up  boundary conditions
    return new_processed_signal

#when called, parameters of the following functions are going to the new_processed_signal

##FIX THIS
def FFTElim(processed_signal):
    new_processed_signal = scipy.fft(processed_signal)
    for i in new_processed_signal:
        if count(i) > 1 and not #periodic condition:

#Establish EMG windows
def segmentation(new_processed_signal, sampling_freq=125, window_size=1, window_shift=0.016):
  w_size = int(sampling_freq * window_size)
  w_shift = int(sampling_freq * window_shift)
  segments = []
  i = 0
  while i + w_size <= new_processed_signal.shape[1]:
    segments.append(new_processed_signal[:, i: i + w_size])
    i += w_shift
  return segments

def channel_rearrangment(new_processed_signal, channel_order):
    channel_order = [channel - 1 for channel in channel_order]
    reindexed = np.zeros_like(new_processed_signal)
    for i, ind in enumerate(channel_order):
        reindexed[i] = new_processed_signal[ind]
    return reindexed

#Channels available from the data
ordered_channels = [x for x in data[0]]

train_x, test_x, train_y, test_y = tts(new_processed_signal, labels, test_size = 0.25)
val_x, test_x = test_x[:len(test_x)//2], test_x[len(test_x)//2:]
val_y, test_y = test_y[:len(test_y)//2], test_y[len(test_y)//2:]

train_emg = []
train_labels = []
valid_emg = []
valid_labels = []
test_emg = []
test_labels = []
for sig, label in zip(train_x, train_y):
  if sig.shape[1] == 0: # excluding empty sample elements
    #print(name)
    continue
  reindexed_signal = channel_rearrangment(new_processed_signal, ordered_channels)
  filtered_sig = bandpass_filter(reindexed_signal, [5, 40], 125) # bandpass filter
  normed_sig = (filtered_sig - np.mean(filtered_sig, 1, keepdims=True)) / np.std(filtered_sig, 1, keepdims=True) # standard scaling
  if np.isnan(normed_sig).any(): # excluding sample elements with nans
    print("nan")
    continue
  signals = segmentation(normed_sig, 125, window_size = 1.5, window_shift = 0.0175) # segmentation
  labels = [label] * len(signals)
  train_emg.extend(signals)
  train_labels.extend(labels)

#Include more data separation for ideal pass here