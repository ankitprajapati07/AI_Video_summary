import os
import json
import self as self
from moviepy.editor import VideoFileClip
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration


class CustomBartModel(BartForConditionalGeneration):
    def forward(
        self,
        audio=None,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        **kwargs
    ):
        return super().forward(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )


#  Load the BART tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')


class VideoSummarizationDataset(Dataset):
    def __init__(self, video_dir, transcript_dir, summary_file, mapping_file, summaries):
        self.video_dir = video_dir
        self.transcript_dir = transcript_dir
        self.summary_file = summary_file
        self.mapping_file = mapping_file
        self.summaries = summaries
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_file = os.path.join(self.video_dir, item['video_filename'])

        summary = self.summaries[idx]
        #print("summary string:", summary)

        audio = self.load_audio(video_file)

        transcript_input_ids = None
        transcript_attention_mask = None
        if 'transcript_filename' in item:
            transcript_file = os.path.join(self.transcript_dir, item['transcript_filename'])
            transcript_input_ids, transcript_attention_mask = self.load_transcript(transcript_file)


        return audio, transcript_input_ids, transcript_attention_mask, summary

    def load_data(self):
        with open(self.mapping_file, 'r') as f:
            data = json.load(f)
            # summaries = [list(item.values())[0] for item in data]
        return data

    def load_audio(self, video_file):
        # Load audio from video file and extract features (e.g., MFCC)
        clip = VideoFileClip(video_file)
        audio = clip.audio.to_soundarray()
        clip.close()
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # Convert numpy array to torch tensor
        return audio_tensor

    def load_transcript(self, transcript_file):
        # Load transcript from transcript file and tokenize using BART tokenizer
        transcript = open(transcript_file, 'r').read()
        encoding = tokenizer.encode_plus(
            transcript,
            add_special_tokens=False,
            max_length=512,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask


# Define the paths to the video directory, transcript directory, summary file, and mapping file
video_dir = "videos"
transcript_dir = "transcripts"
summary_file = "summaries.json"
mapping_file = "smrydata.json"

# Load the summaries from the summaries.json file
with open(summary_file, 'r') as f:
    summaries_data = json.load(f)
    summaries = [list(summary.values())[0] for summary in summaries_data]
    # print("summaries: ", summaries)

# Create an instance of the VideoSummarizationDataset
dataset = VideoSummarizationDataset(video_dir, transcript_dir, summary_file, mapping_file, summaries)
# print("dataset summmaries: ", dataset.summaries)
# print(type(summaries))
# Load the BART model
model = CustomBartModel.from_pretrained('facebook/bart-base')

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Fine-tuning hyperparameters
epochs = 10
batch_size = 1
learning_rate = 1e-4

# Define the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Create a data loader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the custom model directory
custom_model_dir = "CustomSummaryModel23"

# Fine-tuning loop
for epoch in range(epochs):
    total_loss = 0
    for audio, transcript_input_ids, transcript_attention_mask, summary in dataloader:
        audio = audio.to(device)
        if transcript_input_ids is not None:
            transcript_input_ids = transcript_input_ids.to(device)
        if transcript_attention_mask is not None:
            transcript_attention_mask = transcript_attention_mask.to(device)

        print("Summary111111:", summary)
        # Adjust the max_length parameter in the tokenizer's encode_plus method
        summary_input_ids = []
        for s in summary:
            encoding = tokenizer.encode_plus(
                s,
                add_special_tokens=False,
                max_length=1024,
                padding='longest',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)

            # Move the tensors to the device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            summary_input_ids.append((input_ids, attention_mask))

        decoder_input_ids = [input_ids.unsqueeze(0)[:, :-1] for input_ids, _ in summary_input_ids]
        print("shape of summary_input_ids", len(decoder_input_ids))
        print("Batch size of summary_input_ids:", len(summary_input_ids))

        # Convert the string elements of the tuple to integers
        # summary_input_ids = tokenizer.encode(summary, add_special_tokens=False)
        # print("summary_input_ids_tokenizer", summary_input_ids)

        # decoded_tokens = tokenizer.decode(summary_input_ids, skip_special_tokens=True)
        # print("Decoded summary_input_ids tokens:", decoded_tokens)
        # Convert the list to a tensor and move it to the device
        # summary_input_ids = torch.tensor(summary_input_ids).unsqueeze(0).to(device)

        print("Batch size of transcript_input_ids:", transcript_input_ids.size(0))
        # print("Batch size of summary_input_ids:", summary_input_ids.size(0))

        # Perform the slicing operation

        # decoder_input_ids = summary_input_ids[:, :-1]

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        decoder_input_ids = transcript_input_ids[:, :-1]
        print("shape of transcript_input_ids", decoder_input_ids)

        outputs = model(
            input_ids=transcript_input_ids,
            attention_mask=transcript_attention_mask,
            decoder_input_ids=decoder_input_ids,
            lm_labels=summary_input_ids[0][0].unsqueeze(0)[:, 1:].contiguous()
        )
        # print("Inputs",summary_input_ids[:, 1:].contiguous())
        loss = criterion(outputs.logits.view(-1, outputs.logits.shape[-1]), summary_input_ids[0][0].unsqueeze(0)[:, 1:].contiguous().view(-1))
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}: Average Loss = {average_loss:.4f}")

    # Save the trained model
    custom_model = "CustomSummaryModel23"
    os.makedirs(custom_model, exist_ok=True)
    model.save_pretrained(custom_model)
    model.tokenizer.save_pretrained(custom_model)

    # Save the training state
    training_state = {
      'epoch': epoch + 1,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': average_loss,
    }
    torch.save(training_state, os.path.join(custom_model_dir, 'training_state.pth'))