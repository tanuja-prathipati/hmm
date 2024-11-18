# model.py

import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder

# Function to train the Hidden Markov Model (HMM)
def train_hmm(data, labels, n_states=3):
    # Flatten the data and labels
    X = np.concatenate(data, axis=0)  # Combine all MFCC data
    le = LabelEncoder()  # Label encoder to transform labels to numeric values
    y = le.fit_transform(labels)

    # Initialize HMM model
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)

    # Fit the HMM model to the data (MFCC features)
    model.fit(X)
    
    return model, le

# Function to recognize speech using a trained HMM model
def recognize_speech(model, le, audio_file):
    from audio_processing import extract_mfcc

    mfcc = extract_mfcc(audio_file)
    # Use the trained model to predict the labels (phonemes)
    predicted_states = model.predict(mfcc)
    
    # Decode the labels back to the original words (bed, bird, cat)
    predicted_label = le.inverse_transform([predicted_states[0]])[0]
    return predicted_label
