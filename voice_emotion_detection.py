import numpy as np
import librosa
import librosa.display
import sounddevice as sd
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tkinter import Tk, Button, Label, filedialog, messagebox
import os
from glob import glob

# Load and preprocess audio data
def load_and_preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)  # Trim silence
    mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)  # Extract MFCC features
    return np.mean(mfccs.T, axis=0)  # Return mean of MFCCs

# Extract emotion label from RAVDESS file name
def extract_emotion_label(file_path):
    # RAVDESS file name format: 03-01-06-01-02-01-12.wav
    # Emotion is encoded in the third number (01 = neutral, 02 = calm, 03 = happy, etc.)
    file_name = os.path.basename(file_path)
    emotion_code = int(file_name.split("-")[2])
    emotion_map = {
        1: "neutral", 2: "calm", 3: "happy", 4: "sad",
        5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
    }
    return emotion_map.get(emotion_code, "unknown")

# Gender detection (placeholder function)
def detect_gender(file_path):
    # RAVDESS file name format: 03-01-06-01-02-01-12.wav
    # Gender is encoded in the fourth number (01 = male, 02 = female)
    file_name = os.path.basename(file_path)
    gender_code = int(file_name.split("-")[3])
    gender_map = {1: "male", 2: "female"}
    return gender_map.get(gender_code, "unknown")

# Emotion detection model training
def train_emotion_model(audio_files, labels):
    # Extract features from audio files
    features = [load_and_preprocess_audio(file) for file in audio_files]
    X = np.array(features)
    y = np.array(labels)

    # Encode emotion labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train SVM model
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    return model, le

# GUI for uploading and recording voice
class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection through Voice")
        self.model = None
        self.le = None

        Label(root, text="Upload or Record a Female Voice Note").pack()
        Button(root, text="Upload Voice Note", command=self.upload_voice_note).pack()
        Button(root, text="Record Voice", command=self.record_voice).pack()

    def upload_voice_note(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if file_path:
            self.process_audio(file_path)

    def record_voice(self):
        duration = 5  # seconds
        fs = 44100  # Sample rate
        messagebox.showinfo("Recording", "Recording will start now. Please speak for 5 seconds.")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        file_path = "recorded_voice.wav"
        sf.write(file_path, recording, fs)
        self.process_audio(file_path)

    def process_audio(self, file_path):
        gender = detect_gender(file_path)
        if gender != "female":
            messagebox.showerror("Error", "Please upload a female voice.")
            return

        features = load_and_preprocess_audio(file_path)
        if self.model:
            emotion_encoded = self.model.predict([features])
            emotion = self.le.inverse_transform(emotion_encoded)
            messagebox.showinfo("Emotion Detected", f"The detected emotion is: {emotion[0]}")
        else:
            messagebox.showerror("Error", "Model not trained yet.")

# Main execution
if __name__ == "__main__":
    # Load RAVDESS dataset
    dataset_path = "kaggle/input/ravdess-emotional-speech-audio/*/*.wav"
    audio_files = glob(dataset_path)

    # Extract labels from file names
    labels = [extract_emotion_label(file) for file in audio_files]

    # Filter out files with unknown emotions
    audio_files = [file for file, label in zip(audio_files, labels) if label != "unknown"]
    labels = [label for label in labels if label != "unknown"]

    # Train the model
    model, le = train_emotion_model(audio_files, labels)

    # Initialize GUI
    root = Tk()
    app = EmotionDetectionApp(root)
    app.model = model
    app.le = le
    root.mainloop()