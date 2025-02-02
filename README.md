# Sign Language to Speech and Text System

## Overview
This project is a Sign Language to Speech and Text recognition system that uses a webcam to capture hand gestures, classify them using a trained machine learning model, and convert the recognized gestures into text and speech. The system is built using OpenCV for image processing, MediaPipe for hand tracking, and a Random Forest classifier for gesture recognition. The backend is developed using Flask to facilitate interaction with a web-based interface.

## Features
- Real-time hand gesture detection and recognition
- Conversion of sign language to text
- Text-to-speech conversion using gTTS
- Web-based interface for user interaction
- Option to download recognized text as a file or as an audio output

## Technologies Used
- **Python**
- **OpenCV**
- **MediaPipe** 
- **Random Forest Classifier**
- **Flask** 
- **gTTS & pyttsx3** (for text-to-speech conversion)


## Running the System
1. **Start the Flask Application:**
   ```python
   python app.py
   ```
2. **Access the Web Interface:**
   Open a browser and go to `http://127.0.0.1:5000/`. (port number may differ)

3. **Using the System:**
   - Click "Start Camera" to begin capturing gestures.
   - Perform sign language gestures in front of the webcam.
   - The recognized gestures will be displayed as text.
   - Click "Download Text" to save the recognized text.
   - Click "Download Audio" to get the speech output.

## Generating Custom Dataset
To collect your own dataset, run:
```python
python collect_gestures.py
```
Follow the on-screen instructions to collect data for each class.

## Training a New Model
Modify `sign_model.py` to preprocess data and train a new Random Forest classifier. Save the trained model as `sign_model.p`.


## License
This project is open-source and free to use for educational purposes.

