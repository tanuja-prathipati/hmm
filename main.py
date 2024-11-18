# main.py

from audio_processing import load_data_from_directories
from model import train_hmm, recognize_speech

# Path to your dataset folder
DATASET_PATH = 'ds'

def main():
    print("Loading data...")
    # Load data and labels from your dataset
    data, labels = load_data_from_directories(DATASET_PATH)
    print(f"Loaded {len(data)} samples.")

    print("Training HMM model...")
    # Train the HMM model with the MFCC data
    model, le = train_hmm(data, labels)
    print("Model trained successfully!")

    # Test the model on a new audio file (replace with your own path)
    test_audio_file = 'ds/bird/9.wav'  # Update with a path to your test audio file
    print(f"Recognizing speech...")
    recognized_word = recognize_speech(model, le, test_audio_file)
    print(f"Predicted word: {recognized_word}")

if __name__ == "__main__":
    main()
