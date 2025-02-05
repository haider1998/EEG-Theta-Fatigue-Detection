import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump
import logging

# Set up logging for debugging and transparency
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and parameters
SAMPLING_RATE = 512  # EEG sampling frequency (Hz)
EPOCH_DURATION = 10  # Duration of each epoch in seconds
EPOCH_SAMPLES = SAMPLING_RATE * EPOCH_DURATION  # Total samples per epoch
THETA_BAND = (4, 7)  # Theta frequency band in Hz
DATA_DIR = 'data/'  # Directory where .mat files are stored
MODEL_OUTPUT_PATH = 'models/fatigue_detector.pkl'


# Bandpass filter design (using zero-phase filtering with filtfilt)
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, data)
    return filtered


# Compute theta power using Welch's method for a 1D signal (per channel)
def compute_theta_power(signal, fs):
    # Compute the power spectral density (PSD)
    f, Pxx = welch(signal, fs=fs, nperseg=fs * 2, scaling='density')
    # Identify frequency indices corresponding to theta band
    theta_idx = np.logical_and(f >= THETA_BAND[0], f <= THETA_BAND[1])
    theta_power = np.sum(Pxx[theta_idx])
    return theta_power


# Process a single .mat file to extract epochs and corresponding labels.
# We assume that the .mat file contains a matrix where:
#   - Column 0: timestamp (ignored here)
#   - Columns 1-16: EEG channels
#   - Column 17: trigger for condition 1 (eyes closed) → label 1 (fatigued)
#   - Column 18: trigger for condition 2 (eyes open)   → label 0 (alert)
def process_mat_file(filepath):
    logger.info(f"Processing file: {filepath}")
    mat = loadmat(filepath)
    # Assuming the main data matrix is the first variable; adjust as needed.
    data_key = [key for key in mat.keys() if not key.startswith('__')][0]
    data = mat[data_key]  # Shape: (num_samples, num_channels)

    # Columns:
    #   0: timestamp, 1-16: EEG channels, 17: trigger cond1, 18: trigger cond2.
    eeg_data = data[:, 1:17]  # shape: (n_samples, 16)
    trigger1 = data[:, 17].flatten()  # eyes closed → label 1
    trigger2 = data[:, 18].flatten()  # eyes open   → label 0

    features = []
    labels = []
    num_samples = data.shape[0]
    idx = 0
    while idx < num_samples:
        # Look for the start of an epoch: when either trigger1 or trigger2 is 1.
        if trigger1[idx] == 1 or trigger2[idx] == 1:
            # Ensure there are enough samples remaining for a full epoch
            if idx + EPOCH_SAMPLES <= num_samples:
                epoch = eeg_data[idx: idx + EPOCH_SAMPLES, :]  # shape: (EPOCH_SAMPLES, 16)
                # For each channel, first bandpass filter the signal to isolate theta,
                # then compute theta power.
                epoch_features = []
                for ch in range(epoch.shape[1]):
                    channel_signal = epoch[:, ch]
                    filtered = bandpass_filter(channel_signal, THETA_BAND[0], THETA_BAND[1], SAMPLING_RATE)
                    power = compute_theta_power(filtered, SAMPLING_RATE)
                    epoch_features.append(power)
                features.append(epoch_features)
                # Assign label based on which trigger is active at the epoch start
                label = 1 if trigger1[idx] == 1 else 0
                labels.append(label)
                # Skip to the end of the epoch to avoid overlapping windows (or adjust as needed)
                idx += EPOCH_SAMPLES
            else:
                break
        else:
            idx += 1
    return np.array(features), np.array(labels)


# Load and combine data from all .mat files in the directory
def load_all_data(data_dir):
    X_all = []
    y_all = []
    for file in os.listdir(data_dir):
        if file.endswith('.mat'):
            filepath = os.path.join(data_dir, file)
            X, y = process_mat_file(filepath)
            if X.size > 0:
                X_all.append(X)
                y_all.append(y)
    if X_all:
        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)
        logger.info(f"Loaded {X_all.shape[0]} epochs with {X_all.shape[1]} features each.")
        return X_all, y_all
    else:
        raise ValueError("No data loaded. Check your .mat files and directory.")


def main():
    # Load data
    X, y = load_all_data(DATA_DIR)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Data split into training and test sets.")

    # Hyperparameter tuning via GridSearchCV for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")

    # Evaluate with cross-validation on training set
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    logger.info(f"Cross-validation accuracy scores: {cv_scores}")
    logger.info(f"Mean CV accuracy: {np.mean(cv_scores):.3f}")

    # Evaluate on the test set
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test set accuracy: {acc:.3f}")
    logger.info("Classification Report:")
    report = classification_report(y_test, y_pred, target_names=["Alert", "Fatigued"])
    print(report)
    logger.info("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the model for later deployment (e.g., via FastAPI)
    dump(best_model, MODEL_OUTPUT_PATH)
    logger.info(f"Trained model saved to {MODEL_OUTPUT_PATH}")


if __name__ == '__main__':
    main()