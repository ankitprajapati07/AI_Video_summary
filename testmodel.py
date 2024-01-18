import os
import librosa
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Define the path to the video file
video_file = "new_video.mp4"
transcript_file = "new_transcript.txt"  # Set to None if transcript is not available

# Load the BART tokenizer and model
custom_model = "fine-tuned_model"
tokenizer = BartTokenizer.from_pretrained(custom_model)
model = BartForConditionalGeneration.from_pretrained(custom_model)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the video audio
audio, _ = librosa.load(video_file, sr=16000)
audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)

# Load and tokenize the transcript (if available)
transcript_input_ids, transcript_attention_mask = None, None
if transcript_file is not None:
    transcript = open(transcript_file, 'r').read()
    encoding = tokenizer.encode_plus(
        transcript,
        add_special_tokens=True,
        max_length=512,
        padding='longest',
        truncation=True,
        return_tensors='pt'
    )
    transcript_input_ids = encoding['input_ids'].squeeze(0).to(device)
    transcript_attention_mask = encoding['attention_mask'].squeeze(0).to(device)

# Generate the summary
summary_ids = model.generate(
    audio=audio_tensor,
    input_ids=transcript_input_ids.unsqueeze(0) if transcript_input_ids is not None else None,
    attention_mask=transcript_attention_mask.unsqueeze(0) if transcript_attention_mask is not None else None,
    max_length=100,  # Adjust the maximum summary length as desired
    num_beams=4,  # Adjust the number of beams for beam search
    early_stopping=True
)

# Decode the summary
summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

# Print the generated summary
print("Generated Summary:")
print(summary)
