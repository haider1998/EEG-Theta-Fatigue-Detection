import os
import glob
import numpy as np
import scipy.io
import scipy.signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from joblib import dump
import logging

# Set up logging for transparency
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and parameters
SAMPLING_RATE = 512  # Hz, EEG sampling frequency
EPOCH_DURATION = 10  # seconds per epoch
EPOCH_SAMPLES = SAMPLING_RATE * EPOCH_DURATION  # Samples per epoch
THETA_BAND = (4, 8)  # Theta frequency range in Hz
DATA_DIR = 'data/'  # Directory where .mat files are stored (update as needed)
MODEL_OUTPUT_PATH = 'models/fatigue_detector_theta.pkl'


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a zero-phase Butterworth bandpass filter.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    filtered = scipy.signal.filtfilt(b, a, data)
    return filtered


def compute_theta_power(signal, fs):
    """
    Compute theta power of a 1D signal using Welch's method.
    """
    freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=fs * 2, scaling='density')
    theta_mask = (freqs >= THETA_BAND[0]) & (freqs <= THETA_BAND[1])
    theta_power = np.sum(psd[theta_mask])
    return theta_power


def process_mat_file(filepath):
    """
    Process one .mat file to extract epochs and corresponding labels.

    Assumptions:
      - Column 0: Timestamp (ignored)
      - Columns 1-16: EEG channels
      - Column 17: Trigger for condition 1 (e.g., eyes closed => Fatigued, label 1)
      - Column 18: Trigger for condition 2 (e.g., eyes open   => Alert, label 0)
    """
    logger.info(f"Processing file: {filepath}")
    mat = scipy.io.loadmat(filepath)
    # Assume the main matrix is the first variable not starting with '__'
    data_key = [key for key in mat.keys() if not key.startswith('__')][0]
    data = mat[data_key]

    # Extract EEG channels and triggers
    eeg_data = data[:, 1:17]  # 16 channels of EEG data
    trigger_fatigued = data[:, 17].flatten()  # Trigger for fatigued state
    trigger_alert = data[:, 18].flatten()  # Trigger for alert state

    epochs = []
    labels = []
    num_samples = data.shape[0]
    idx = 0
    while idx < num_samples:
        if trigger_fatigued[idx] == 1 or trigger_alert[idx] == 1:
            if idx + EPOCH_SAMPLES <= num_samples:
                epoch = eeg_data[idx:idx + EPOCH_SAMPLES, :]  # shape: (EPOCH_SAMPLES, 16)
                # For each channel, filter to theta and compute power
                features = []
                for ch in range(epoch.shape[1]):
                    channel_signal = epoch[:, ch]
                    filtered_signal = bandpass_filter(channel_signal, THETA_BAND[0], THETA_BAND[1], SAMPLING_RATE)
                    power = compute_theta_power(filtered_signal, SAMPLING_RATE)
                    features.append(power)
                epochs.append(features)
                # Label assignment: label 1 for fatigued, 0 for alert
                label = 1 if trigger_fatigued[idx] == 1 else 0
                labels.append(label)
                idx += EPOCH_SAMPLES  # move to next epoch (non-overlapping)
            else:
                break
        else:
            idx += 1
    return np.array(epochs), np.array(labels)


def load_all_data(data_dir):
    """
    Load and combine data from all .mat files in the directory.
    Returns:
      X: Feature matrix (epochs x features)
      y: Labels
      groups: Subject indices (for leave-one-subject-out CV)
    """
    all_features = []
    all_labels = []
    groups = []
    mat_files = glob.glob(os.path.join(data_dir, 'subject_*.mat'))
    logger.info(f"Found {len(mat_files)} .mat files in {data_dir}.")
    for subj_idx, filepath in enumerate(mat_files):
        try:
            features, labels = process_mat_file(filepath)
            if features.size > 0:
                all_features.append(features)
                all_labels.append(labels)
                groups.extend([subj_idx] * len(labels))
                logger.info(f"Subject {subj_idx + 1}: {len(labels)} epochs extracted.")
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
    if all_features:
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        groups = np.array(groups)
        logger.info(f"Total epochs: {X.shape[0]} with {X.shape[1]} features per epoch.")
        return X, y, groups
    else:
        raise ValueError("No data loaded. Please check your .mat files and directory.")


def evaluate_model(X, y, groups):
    """
    Evaluate a RandomForest classifier using Leave-One-Group-Out (subject-wise) cross-validation.
    """
    logo = LeaveOneGroupOut()
    accuracies = []
    f1s = []
    conf_matrix_total = np.zeros((2, 2))

    for train_idx, test_idx in logo.split(X, y, groups):
        pipeline = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=100, random_state=42)
        )
        pipeline.fit(X[train_idx], y[train_idx])
        y_pred = pipeline.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], y_pred)
        f1 = f1_score(y[test_idx], y_pred)
        accuracies.append(acc)
        f1s.append(f1)
        conf_matrix_total += confusion_matrix(y[test_idx], y_pred)

    avg_acc = np.mean(accuracies)
    avg_f1 = np.mean(f1s)
    # Average confusion matrix across groups
    conf_matrix_avg = conf_matrix_total / len(np.unique(groups))

    logger.info(f"Average Accuracy: {avg_acc:.3f}")
    logger.info(f"Average F1 Score: {avg_f1:.3f}")
    logger.info("Confusion Matrix (averaged per subject):")
    logger.info(conf_matrix_avg)

    return avg_acc, avg_f1, conf_matrix_avg


def main():
    # Load data from the specified directory
    X, y, groups = load_all_data(DATA_DIR)

    # Evaluate using subject-level (leave-one-group-out) cross-validation
    avg_acc, avg_f1, conf_matrix = evaluate_model(X, y, groups)

    # Train final model on all data and save for future real-time deployment
    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    pipeline.fit(X, y)
    dump(pipeline, MODEL_OUTPUT_PATH)
    logger.info(f"Final model saved to {MODEL_OUTPUT_PATH}")

    # Output detailed classification report on the full dataset
    y_pred = pipeline.predict(X)
    print("Final Model Classification Report:")
    print(classification_report(y, y_pred, target_names=['Alert', 'Fatigued']))
    print("Final Confusion Matrix:")
    print(confusion_matrix(y, y_pred))


if __name__ == '__main__':
    main()