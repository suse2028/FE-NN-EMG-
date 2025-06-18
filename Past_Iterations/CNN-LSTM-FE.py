import numpy as np
import pandas as pd
import torch
import scipy
import scipy.signal as signal
from scipy.signal import butter, lfilter
from scipy.fft import fft, ifft, fftfreq
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time


def download_uci_emg_dataset(save_dir='./data'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00481/EMG_data_for_gestures-master.zip"
    
    
    if not os.path.exists(os.path.join(save_dir, 'EMG_data_for_gestures-master')):
        print("Downloading UCI EMG dataset...")
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(save_dir)
        print("Dataset downloaded and extracted successfully!")
    else:
        print("UCI EMG dataset already exists.")
    
    return os.path.join(save_dir, 'EMG_data_for_gestures-master')

# Process UCI EMG dataset and extract walking data
def process_uci_emg_dataset(data_dir, output_file='emg_gait_data.csv'):
    """
    Process the UCI EMG dataset and extract walking-related data
    """
    emg_data = []
    labels = []
    walking_gesture_id = 5
    

    for participant_id in range(1, 37):  
        try:
            file_path = os.path.join(data_dir, f'subject{participant_id}_gesture{walking_gesture_id}_raw.csv')
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                
                X = df.iloc[:, 1:9].values  # EMG data
                y = np.full((len(X), 2), 0.0)  
                
                #Synthetic angle and position values based on time values in the array
                steps = np.arange(len(X)) / len(X) * 2 * np.pi
                y[:, 0] = np.sin(steps) * 0.5  # Angle (simulated)
                y[:, 1] = (np.cos(steps) + 1) * 0.5  # Position (simulated)
                
                emg_data.append(X)
                labels.append(y)
                
                print(f"Processed walking data from participant {participant_id}")
            
        except Exception as e:
            print(f"Error processing participant {participant_id}: {e}")
    
    if len(emg_data) > 0:
        # Combine all data
        X_combined = np.vstack(emg_data)
        y_combined = np.vstack(labels)
        
        # Create DataFrame with EMG channels and labels
        df_combined = pd.DataFrame(X_combined, columns=[f'EMG_{i+1}' for i in range(8)])
        df_combined['angle'] = y_combined[:, 0]
        df_combined['position'] = y_combined[:, 1]
        
        # Save to CSV
        df_combined.to_csv(output_file, index=False)
        print(f"Combined dataset saved to {output_file}")
        
        return output_file
    else:
        print("No walking data found!")
        return None

# Modified get_data function to handle the UCI dataset format
def get_data(file):
    data = pd.read_csv(file)
    
    # Extract EMG signals (first 8 columns)
    emg_columns = [col for col in data.columns if 'EMG_' in col]
    emg_signals = data[emg_columns].values.T
    
    # Extract labels (angle and position)
    labels = data[['angle', 'position']].values
    
    return emg_signals, labels

def bandpass_filter(signal_data, crit_freq=[20, 450], sampling_freq=125, plot=False, channel=0):
    order = 4
    b, a = scipy.signal.butter(order, crit_freq, btype='bandpass', fs=sampling_freq)
    processed_signal = scipy.signal.filtfilt(b, a, signal_data)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.xlabel('Time')
        plt.ylabel(f'Normalized amplitude of channel {channel}')
        plt.title(f'{crit_freq[0]}-{crit_freq[1]}Hz bandpass filter')

        signal_min = np.min(signal_data, axis=1, keepdims=True)
        signal_max = np.max(signal_data, axis=1, keepdims=True)
        normed_signal = (signal_data - signal_min) / (signal_max - signal_min)

        filtered_min = np.min(processed_signal, axis=1, keepdims=True)
        filtered_max = np.max(processed_signal, axis=1, keepdims=True)
        normed_filt = (processed_signal - filtered_min) / (filtered_max - filtered_min)

        plt.plot(np.arange(normed_signal[channel].size), normed_signal[channel], label='Input')
        plt.plot(np.arange(normed_filt[channel].size), normed_filt[channel], label='Transformed')
        plt.legend()
        plt.show()

    return processed_signal


def notch_filter(signal_data, notch_freqs=[50, 60], Q=30, sampling_freq=125):
    filtered_signal = signal_data.copy()

    for f0 in notch_freqs:
        b, a = signal.iirnotch(f0, Q, fs=sampling_freq)
        filtered_signal = signal.filtfilt(b, a, filtered_signal)

    return filtered_signal


def rectify(signal_data):
    return np.abs(signal_data)


def fft_analysis_and_filter(signal_data, sampling_freq=125, gait_freq_range=[0.5, 2.5], plot=False, channel=0):
    n_samples = signal_data.shape[1]
    n_channels = signal_data.shape[0]

    fft_result = fft(signal_data)
    freqs = fftfreq(n_samples, 1 / sampling_freq)

    mask = np.zeros((n_channels, n_samples), dtype=bool)
    for i in range(n_channels):
        mask[i] = (np.abs(freqs) < 0.1) | (
                    (np.abs(freqs) >= gait_freq_range[0]) & (np.abs(freqs) <= gait_freq_range[1]))

        for harmonic in range(2, 6):
            mask[i] |= ((np.abs(freqs) >= harmonic * gait_freq_range[0]) &
                        (np.abs(freqs) <= harmonic * gait_freq_range[1]))

    filtered_fft = fft_result.copy()
    for i in range(n_channels):
        filtered_fft[i, ~mask[i]] = 0

    filtered_signal = np.real(ifft(filtered_fft))

    if plot and channel < n_channels:
        plt.figure(figsize=(15, 10))

        plt.subplot(3, 1, 1)
        plt.title(f'Original EMG Signal - Channel {channel}')
        plt.plot(signal_data[channel])
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        plt.subplot(3, 1, 2)
        plt.title(f'Frequency Components - Channel {channel}')
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask][:int(n_samples / 2)]
        plt.plot(pos_freqs, 2.0 / n_samples * np.abs(fft_result[channel, pos_mask][:int(n_samples / 2)]))
        plt.axvspan(gait_freq_range[0], gait_freq_range[1], alpha=0.3, color='green', label='Gait Freq Range')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.title(f'Filtered EMG Signal - Channel {channel}')
        plt.plot(filtered_signal[channel])
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.show()

    return filtered_signal


def segmentation(signal_data, sampling_freq=125, window_size=1, window_shift=0.016):
    w_size = int(sampling_freq * window_size)
    w_shift = int(sampling_freq * window_shift)

    segments = []
    i = 0
    while i + w_size <= signal_data.shape[1]:
        segments.append(signal_data[:, i:i + w_size])
        i += w_shift

    return segments


def channel_rearrangement(signal_data, channel_order):
    channel_order = [channel - 1 for channel in channel_order]
    reindexed = np.zeros_like(signal_data)
    for i, ind in enumerate(channel_order):
        reindexed[i] = signal_data[ind]
    return reindexed


def normalize_signal(signal_data):
    mean = np.mean(signal_data, axis=1, keepdims=True)
    std = np.std(signal_data, axis=1, keepdims=True)
    normalized = (signal_data - mean) / std
    return normalized


def prepare_dataset(file_path, ordered_channels, test_size=0.25):
    emg_data, labels = get_data(file_path)

    if ordered_channels:
        emg_data = channel_rearrangement(emg_data, ordered_channels)

    filtered_emg = bandpass_filter(emg_data, [20, 450], 125)
    notched_emg = notch_filter(filtered_emg, [50, 60], 30, 125)
    rectified_emg = rectify(notched_emg)
    clean_emg = fft_analysis_and_filter(rectified_emg, 125, [0.5, 2.5])

    X_train, X_test, y_train, y_test = train_test_split(
        clean_emg.T,
        labels,
        test_size=test_size,
        random_state=42
    )

    X_val, X_test = X_test[:len(X_test) // 2], X_test[len(X_test) // 2:]
    y_val, y_test = y_test[:len(y_test) // 2], y_test[len(y_test) // 2:]

    X_train, X_val, X_test = X_train.T, X_val.T, X_test.T

    train_emg = []
    train_labels = []
    valid_emg = []
    valid_labels = []
    test_emg = []
    test_labels = []

    train_segments = segmentation(X_train, 125, window_size=1.5, window_shift=0.0175)
    train_emg.extend(train_segments)
    train_labels.extend([y_train[0]] * len(train_segments))

    val_segments = segmentation(X_val, 125, window_size=1.5, window_shift=0.0175)
    valid_emg.extend(val_segments)
    valid_labels.extend([y_val[0]] * len(val_segments))

    test_segments = segmentation(X_test, 125, window_size=1.5, window_shift=0.0175)
    test_emg.extend(test_segments)
    test_labels.extend([y_test[0]] * len(test_segments))

    return (
        np.array(train_emg), np.array(train_labels),
        np.array(valid_emg), np.array(valid_labels),
        np.array(test_emg), np.array(test_labels)
    )


class EMGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y


class CNNRegression(nn.Module):
    def __init__(self, input_channels, seq_length):
        super(CNNRegression, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout3 = nn.Dropout(0.3)

        self.flat_features = 16 * (seq_length // 8)

        self.fc1 = nn.Linear(self.flat_features, 64)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x, t=None):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = x.view(-1, self.flat_features)

        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        return output


class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, boundary_model=None):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.boundary_model = boundary_model

        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)

        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)

        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)

        self.W_g = nn.Linear(input_size, hidden_size)
        self.U_g = nn.Linear(hidden_size, hidden_size)

        self.W_boundary = nn.Linear(2, hidden_size)

    def forward(self, x, hx, boundary=None):
        h_prev, c_prev = hx

        i = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))

        f_base = self.W_f(x) + self.U_f(h_prev)

        if boundary is not None:
            boundary_influence = self.W_boundary(boundary)
            f = torch.sigmoid(f_base + boundary_influence)
        else:
            f = torch.sigmoid(f_base)

        o = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))
        g = torch.tanh(self.W_g(x) + self.U_g(h_prev))
        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1, boundary_model=None):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.boundary_model = boundary_model

        self.cells = nn.ModuleList([
            CustomLSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                boundary_model
            ) for i in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, boundary=None):
        batch_size, seq_len, _ = x.size()

        h_states = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c_states = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            for layer in range(self.num_layers):
                if layer == 0:
                    h_states[layer], c_states[layer] = self.cells[layer](
                        x_t, (h_states[layer], c_states[layer]), boundary
                    )
                else:
                    h_states[layer], c_states[layer] = self.cells[layer](
                        h_states[layer - 1], (h_states[layer], c_states[layer]), boundary
                    )

            outputs.append(h_states[-1].unsqueeze(1))

        output = torch.cat(outputs, dim=1)
        final_output = self.fc(output[:, -1, :])

        return final_output


def physics_loss(pred, x, t, w=4.0, z=3.0):
    dtheta_dt = torch.autograd.grad(
        outputs=pred,
        inputs=t,
        grad_outputs=torch.ones_like(pred),
        create_graph=True
    )[0]

    d2theta_dt2 = torch.autograd.grad(
        outputs=dtheta_dt,
        inputs=t,
        grad_outputs=torch.ones_like(dtheta_dt),
        create_graph=True
    )[0]

    residual = d2theta_dt2 + w * dtheta_dt + z * pred
    phys_loss = torch.mean(residual ** 2)

    return phys_loss


def train_cnn_model(model, train_loader, valid_loader, optimizer, num_epochs=100, device='cpu'):
    print("Starting CNN model training...")
    model.to(device)

    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        data_loss_total = 0.0
        phys_loss_total = 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            batch_size, channels, seq_len = x_batch.shape
            dt = 0.001
            t = torch.linspace(0, dt * (seq_len - 1), seq_len, device=device).view(1, -1, 1)
            t = t.repeat(batch_size, 1, 1).requires_grad_(True)

            x_batch.requires_grad_(True)

            optimizer.zero_grad()
            pred = model(x_batch, t)

            data_loss = F.mse_loss(pred, y_batch)
            phys_loss = physics_loss(pred, x_batch, t)
            total_loss = data_loss + 0.1 * phys_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            data_loss_total += data_loss.item()
            phys_loss_total += phys_loss.item()

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {total_loss.item():.4f}, "
                      f"Data Loss: {data_loss.item():.4f}, "
                      f"Physics Loss: {phys_loss.item():.4f}")

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                pred = model(x_batch)
                loss = F.mse_loss(pred, y_batch)
                valid_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_data_loss = data_loss_total / len(train_loader)
        avg_phys_loss = phys_loss_total / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)

        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f} seconds.")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Data Loss: {avg_data_loss:.4f}")
        print(f"  Physics Loss: {avg_phys_loss:.4f}")
        print(f"  Validation Loss: {avg_valid_loss:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


def train_lstm_model(model, boundary_model, train_loader, valid_loader, optimizer, num_epochs=100, device='cpu'):
    print("Starting LSTM model training...")
    model.to(device)
    boundary_model.to(device)
    boundary_model.eval()

    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            with torch.no_grad():
                boundary_cond = boundary_model(x_batch)

            optimizer.zero_grad()
            pred = model(x_batch, boundary_cond)

            loss = F.mse_loss(pred, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                boundary_cond = boundary_model(x_batch)
                pred = model(x_batch, boundary_cond)

                loss = F.mse_loss(pred, y_batch)
                valid_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)

        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f} seconds.")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_valid_loss:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


def evaluate_model(model, test_loader, device='cpu', boundary_model=None, is_lstm=False):
    print("Evaluating model on test data...")
    model.eval()
    if boundary_model:
        boundary_model.eval()

    test_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            if is_lstm and boundary_model:
                boundary_cond = boundary_model(x_batch)
                pred = model(x_batch, boundary_cond)
            else:
                pred = model(x_batch)

            loss = F.mse_loss(pred, y_batch)
            test_loss += loss.item()

            predictions.extend(pred.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    plt.figure(figsize=(12, 6))

    if predictions.shape[1] == 2:
        plt.subplot(1, 2, 1)
        plt.scatter(actuals[:, 0], predictions[:, 0], alpha=0.5)
        plt.plot([min(actuals[:, 0]), max(actuals[:, 0])], [min(actuals[:, 0]), max(actuals[:, 0])], 'r--')
        plt.xlabel('Actual Angle')
        plt.ylabel('Predicted Angle')
        plt.title('Angle Predictions')

        plt.subplot(1, 2, 2)
        plt.scatter(actuals[:, 1], predictions[:, 1], alpha=0.5)
        plt.plot([min(actuals[:, 1]), max(actuals[:, 1])], [min(actuals[:, 1]), max(actuals[:, 1])], 'r--')
        plt.xlabel('Actual Position')
        plt.ylabel('Predicted Position')
        plt.title('Position Predictions')
    else:
        plt.scatter(actuals, predictions, alpha=0.5)
        plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actuals')

    plt.tight_layout()
    plt.show()

    return avg_test_loss, predictions, actuals


def main(emg_data_file, ordered_channels=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_emg, train_labels, valid_emg, valid_labels, test_emg, test_labels = prepare_dataset(
        emg_data_file, ordered_channels
    )

    input_channels = train_emg.shape[1]
    seq_length = train_emg.shape[2]

    train_dataset = EMGDataset(train_emg, train_labels)
    valid_dataset = EMGDataset(valid_emg, valid_labels)
    test_dataset = EMGDataset(test_emg, test_labels)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    cnn_model = CNNRegression(input_channels, seq_length).to(device)
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

    trained_cnn = train_cnn_model(
        cnn_model,
        train_loader,
        valid_loader,
        cnn_optimizer,
        num_epochs=50,
        device=device
    )

    cnn_test_loss, cnn_predictions, cnn_actuals = evaluate_model(
        trained_cnn,
        test_loader,
        device=device
    )

    torch.save(trained_cnn.state_dict(), 'cnn_boundary_model.pth')

    input_size = train_emg.shape[1]
    hidden_size = 64
    output_size = train_labels.shape[1]

    lstm_model = CustomLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=2,
        output_size=output_size,
        boundary_model=trained_cnn
    ).to(device)

    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    trained_lstm = train_lstm_model(
        lstm_model,
        trained_cnn,
        train_loader,
        valid_loader,
        lstm_optimizer,
        num_epochs=50,
        device=device
    )

    lstm_test_loss, lstm_predictions, lstm_actuals = evaluate_model(
        trained_lstm,
        test_loader,
        device=device,
        boundary_model=trained_cnn,
        is_lstm=True
    )

    torch.save(trained_lstm.state_dict(), 'lstm_gait_model.pth')

    print("Training and evaluation completed!")
    print(f"CNN Model Test Loss: {cnn_test_loss:.4f}")
    print(f"LSTM Model Test Loss: {lstm_test_loss:.4f}")

    return trained_cnn, trained_lstm


if __name__ == "__main__":
    emg_data_file = "emg_gait_data.csv"
    ordered_channels = [1, 2, 3, 4, 5, 6, 7, 8]
    trained_cnn, trained_lstm = main(emg_data_file, ordered_channels)
