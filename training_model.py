import numpy as np
import scipy.io
import scipy.signal
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

# Configuration
DATA_PATH = 'data/'  # Path of .mat data files
MODEL_OUTPUT_PATH = 'models/fatigue_detector_eye.pkl'
FS = 512  # Sampling frequency from dataset
THETA_BAND = (4, 8)  # Theta frequency range
EPOCH_DURATION = 10  # Seconds per block
CHANNELS = ['FP1', 'FP2', 'FC5', 'FC6', 'FZ', 'T7', 'CZ', 'T8',
            'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'Oz', 'O2']  # 16 channels



def load_and_process(subject_path):
    """Load and process EEG data for one subject"""
    mat_data = scipy.io.loadmat(subject_path)['SIGNAL']

    # Extract components
    timestamps = mat_data[:, 0]  # Column 1
    eeg_data = mat_data[:, 1:17]  # Columns 2-17 (16 channels)
    eyes_closed_trig = mat_data[:, 17]  # Column 18 (eyes closed triggers)
    eyes_open_trig = mat_data[:, 18]  # Column 19 (eyes open triggers)

    return eeg_data, eyes_closed_trig, eyes_open_trig


def extract_epochs(eeg_data, trigger, label, fs=FS):
    """Extract labeled epochs based on trigger channel"""
    samples_per_epoch = int(EPOCH_DURATION * fs)
    trigger_starts = np.where(trigger == 1)[0]
    epochs = []

    for start in trigger_starts:
        end = start + samples_per_epoch
        if end <= len(eeg_data):
            epoch = eeg_data[start:end]
            epochs.append((epoch, label))

    return epochs


def preprocess_epoch(epoch):
    """Preprocess and extract theta features from an epoch"""
    # Bandpass filter for theta band
    b, a = scipy.signal.butter(4, [THETA_BAND[0] / (FS / 2), THETA_BAND[1] / (FS / 2)], 'bandpass')
    filtered = scipy.signal.filtfilt(b, a, epoch, axis=0)

    return filtered


def extract_features(epoch):
    """Extract theta-band features from preprocessed epoch"""
    features = []
    for ch in range(16):  # All 16 channels
        # theta power using Welch's method
        freqs, psd = scipy.signal.welch(epoch[:, ch], fs=FS, nperseg=256)
        theta_mask = (freqs >= THETA_BAND[0]) & (freqs <= THETA_BAND[1])
        theta_power = np.log(np.sum(psd[theta_mask]))

        # theta/theta ratio
        theta_mask = (freqs >= 4) & (freqs < 8)
        theta_power = np.sum(psd[theta_mask])
        theta_theta_ratio = theta_power / theta_power if theta_power > 0 else 0

        features.extend([theta_power, theta_theta_ratio])

    # Add occipital theta power (channels 13-15: O1, Oz, O2)
    occipital_power = np.mean([features[13 * 2], features[14 * 2], features[15 * 2]])
    features.append(occipital_power)

    return np.array(features)


def evaluate_model(model, X, y, groups):
    """Evaluate model using leave-one-subject-out cross-validation"""
    logo = LeaveOneGroupOut()
    accuracies, f1_scores = [], []
    cm = np.zeros((2, 2))

    for train_idx, test_idx in logo.split(X, y, groups):
        pipeline = make_pipeline(
            StandardScaler(),
            SelectKBest(f_classif, k=15),
            model
        )

        pipeline.fit(X[train_idx], y[train_idx])
        y_pred = pipeline.predict(X[test_idx])

        accuracies.append(accuracy_score(y[test_idx], y_pred))
        f1_scores.append(f1_score(y[test_idx], y_pred))
        cm += confusion_matrix(y[test_idx], y_pred)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return np.mean(accuracies), np.mean(f1_scores), cm


def main():
    mat_files = glob.glob(os.path.join(DATA_PATH, 'subject_*.mat'))
    print(f"Found {len(mat_files)} subject files")

    all_features, all_labels, groups = [], [], []

    for subj_id, mat_file in enumerate(mat_files):
        try:
            eeg_data, ec_trig, eo_trig = load_and_process(mat_file)

            # Extract epochs for both conditions
            ec_epochs = extract_epochs(eeg_data, ec_trig, label=0)
            eo_epochs = extract_epochs(eeg_data, eo_trig, label=1)

            # Process all epochs
            for epoch, label in ec_epochs + eo_epochs:
                processed = preprocess_epoch(epoch)
                features = extract_features(processed)

                all_features.append(features)
                all_labels.append(label)
                groups.append(subj_id)

            print(f"Processed subject {subj_id + 1}: {len(ec_epochs) + len(eo_epochs)} epochs")

        except Exception as e:
            print(f"Error processing {mat_file}: {str(e)}")

    X = np.array(all_features)
    y = np.array(all_labels)
    groups = np.array(groups)

    print(f"\nFinal dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y) / len(y)}")

    # Initialize models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200,
                                                class_weight='balanced',
                                                n_jobs=-1),
        "SVM": SVC(kernel='rbf', C=1.0, class_weight='balanced')
    }

    # Evaluate models
    results = {}
    best_model = None
    best_acc = 0.0
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        acc, f1, cm = evaluate_model(model, X, y, groups)

        if best_acc < acc:
            best_acc = acc
            best_model = model

        print(f"{name} Results:")
        print(f"Accuracy: {acc:.3f}")
        print(f"F1-score: {f1:.3f}")
        print("Confusion Matrix:")
        print(cm)

        # Plot confusion matrix
        plt.figure()
        sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues',
                    xticklabels=['Eyes Closed', 'Eyes Open'],
                    yticklabels=['Eyes Closed', 'Eyes Open'])
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        # Save the model for later deployment (e.g., via FastAPI)
        dump(best_model, MODEL_OUTPUT_PATH)
        print(f"Trained model saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()