# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/







import os
import json
import librosa
from torch.utils.data import Dataset
from moviepy.editor import VideoFileClip
from nltk.tokenize import word_tokenize


def extract_audio(video_file):
    video = VideoFileClip(video_file)
    audio = video.audio
    return audio


class VideoSummarizationDataset(Dataset):
    def __init__(self, video_dir, transcript_dir, summary_file, smrydata_file):
        self.video_dir = video_dir
        self.transcript_dir = transcript_dir
        self.summary_file = summary_file
        self.smrydata_file = smrydata_file
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_file = os.path.join(self.video_dir, item['videos'])
        transcript_file = os.path.join(self.transcript_dir, item['transcripts'])
        summary = item['summaries']
        audio = load_audio(video_file)
        transcript = load_transcript(transcript_file)
        return audio, transcript, summary

    def load_data(self):
        with open(self.smrydata_file, 'r') as f:
            data = json.load(f)
        return data


def load_audio(video_file):
    # Load audio from video file

    audio, _ = librosa.load(video_file, sr=16000)  # Set the desired sample rate
    mfcc = librosa.feature.mfcc(audio, sr=22050, n_mfcc=13)  # Extract 13 MFCC coefficients
    mfcc_features = np.mean(mfcc, axis=1)  # Take the mean along the time axis
    return mfcc_features


def load_transcript(transcript_file):
    # Load transcript from transcript file

    with open(transcript_file, 'r') as f:
        transcript = f.read()

    transcript = transcript.lower()  # Convert to lowercase
    transcript = word_tokenize(transcript)  # Tokenize the text
    processed_transcript = ' '.join(tokens)  # Join the tokens back into a string
    return processed_transcript


# Define the paths to the video directory, transcript directory, summary file, and smrydata file
video_dir = "videos"
transcript_dir = "transcripts"
summary_file = "summaries.json"
smrydata_file = "smrydata.json"

# Create an instance of the VideoSummarizationDataset
dataset = VideoSummarizationDataset(video_dir, transcript_dir, summary_file, smrydata_file)
for idx in range(len(dataset)):
    audio, transcript, summary = dataset[idx]