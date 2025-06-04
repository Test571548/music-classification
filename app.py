import streamlit as st
import os
import requests
import librosa
import numpy as np
import pandas as pd
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

st.title("🎵 Artist Emotion Classifier via Deezer Audio Previews")

# List of artists
artists = [
    "Adele", "Ed Sheeran", "Taylor Swift", "Beyoncé", "Drake",
    "Ariana Grande", "Billie Eilish", "Shawn Mendes", "Dua Lipa", "The Weeknd"
]

selected_artists = st.multiselect("Select artists to process:", artists, default=artists[:1])

if st.button("Start Processing"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    all_data = []
    X, y = [], []

    for idx, artist in enumerate(selected_artists):
        try:
            status_text.text(f"Processing artist: {artist}")
            # Fetch previews
            preview_urls = []
            url = f"https://api.deezer.com/search/track"
            params = {"q": artist, "limit": 50}
            while url:
                response = requests.get(url, params=params).json()
                for track in response['data']:
                    if track['preview']:
                        preview_urls.append(track['preview'])
                url = response.get('next')
                params = None

            # Process top 10 previews
            for i, preview_url in enumerate(preview_urls[:1000]):
                response = requests.get(preview_url)
               
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tmp_file.write(response.content)
                    tmp_path = tmp_file.name

                try:
                    y_audio, sr = librosa.load(tmp_path, duration=3, offset=0.5)
                    y_audio = y_audio.astype(np.float32)
                    mfcc = np.mean(librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40).T, axis=0).astype(np.float32)
                    timbre = np.mean(mfcc) * 100

                    # Simple timbre-to-emotion mapping
                    if timbre < 60.81:
                        emotion = 'Reflective'
                    elif timbre < 249.23:
                        emotion = 'Calm'
                    elif timbre > 249.23:
                        emotion = 'Happy'
                    else:
                        emotion = 'Neutral'

                    all_data.append({
                        'Artist': artist,
                        'Audio Clip': f"{artist}_preview_{i+1}.mp3",
                        'Timbre Level': timbre,
                        'Emotion': emotion,
                       
                    })

                    X.append(mfcc)
                    y.append(emotion)

                finally:
                    os.remove(tmp_path)

            progress_bar.progress((idx + 1) / len(selected_artists))

        except Exception as e:
            st.error(f"Error processing {artist}: {e}")

    # Display results
    if all_data:
        df = pd.DataFrame(all_data)
        st.subheader("🎧 Audio Classification Results")
        st.dataframe(df)

        # Train KNN classifier
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
        # Convert to DataFrame and drop the 'support' column
        report_df = pd.DataFrame(report).transpose()
        if 'support' in report_df.columns:
            report_df = report_df.drop(columns=['support'])

        st.subheader("📊 Model Evaluation")
        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.write("**Classification Report:**")
        st.dataframe(pd.DataFrame(report_df).transpose())

    else:
        st.warning("No audio previews were processed.")
