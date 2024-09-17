import os
from pydub import AudioSegment

# Define the path to the new dataset
new_root_dir = "/home/stemtec/Desktop/Wave-U-Net-Pytorch/data/new_archive"  # Path to your new dataset

# Function to trim or pad an audio segment to the target length
def adjust_audio_length(audio, target_length):
    current_length = len(audio)
    if current_length > target_length:
        # Trim the audio to the target length
        return audio[:target_length]
    elif current_length < target_length:
        # Pad the audio with silence to match the target length
        padding = AudioSegment.silent(duration=target_length - current_length)
        return audio + padding
    return audio

# Iterate through each split (train, test, valid)
for split in ["train", "test", "valid"]:
    split_path = os.path.join(new_root_dir, split)
    
    # Iterate through each track folder
    for track in os.listdir(split_path):
        track_path = os.path.join(split_path, track)
        
        if os.path.isdir(track_path):
            # Paths to all audio files in the track folder
            file_paths = {
                'mixture': os.path.join(track_path, "mixture.wav"),
                'vocals': os.path.join(track_path, "vocals.wav"),
                'drums': os.path.join(track_path, "drums.wav"),
                'bass': os.path.join(track_path, "bass.wav"),
                'other': os.path.join(track_path, "other.wav")
            }
            
            # Load all available audio files and find the maximum length
            audio_segments = {key: AudioSegment.from_wav(path) for key, path in file_paths.items() if os.path.exists(path)}
            lengths = [len(audio) for audio in audio_segments.values()]
            
            # Set the target length to the minimum or maximum length (choose one strategy)
            target_length = min(lengths)  # or use max(lengths) depending on your needs
            
            # Adjust all audio files to the target length
            for key, audio in audio_segments.items():
                adjusted_audio = adjust_audio_length(audio, target_length)
                adjusted_audio.export(file_paths[key], format="wav")
                print(f"Adjusted length of {file_paths[key]} to {target_length} ms.")

print("All audio files have been adjusted to the same length successfully.")

