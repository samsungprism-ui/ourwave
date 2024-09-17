import os
from pydub import AudioSegment
import random

# Define paths
original_root_dir = "/media/stemtec/Sid-HDD/archive"  # Original dataset path
new_root_dir = "/home/stemtec/Desktop/Wave-U-Net-Pytorch/data/new_archive"   # New dataset path where changes will be saved
wind_audios_dir = "/home/stemtec/Desktop/wind"  # Update this path to your wind audio folder
sample_rate = 44100  # Define a common sample rate for all audio files
wind_gain_reduction = -3  # Reduce wind intensity by 5 dB (adjust this value as needed)

# Function to resample audio to the target sample rate
def resample_audio(audio, target_sample_rate):
    return audio.set_frame_rate(target_sample_rate)

# Function to adjust the length of the audio to match the target length
def adjust_audio_length(audio, target_length):
    current_length = len(audio)
    if current_length < target_length:
        repeat_count = target_length // current_length + 1
        audio = audio * repeat_count
        audio = audio[:target_length]
    elif current_length > target_length:
        audio = audio[:target_length]
    return audio

# Function to load and convert wind audio to .wav if necessary
def load_wind_audio(file_path):
    # Convert to .wav if not already in .wav format
    if not file_path.endswith('.wav'):
        audio = AudioSegment.from_file(file_path)  # Automatically detects the format
        wav_path = file_path.rsplit('.', 1)[0] + ".wav"
        audio.export(wav_path, format="wav")  # Convert and save as .wav
        return AudioSegment.from_wav(wav_path)
    return AudioSegment.from_wav(file_path)

# Get all wind audio files, allowing for multiple audio formats
wind_audio_files = [os.path.join(wind_audios_dir, file) for file in os.listdir(wind_audios_dir) 
                    if file.endswith(('.wav', '.mp3', '.ogg', '.flac', '.m4a'))]  # Accept multiple formats

# Iterate through each split (train, test, valid)
for split in ["train", "test", "valid"]:
    split_path = os.path.join(original_root_dir, split)
    new_split_path = os.path.join(new_root_dir, split)
    os.makedirs(new_split_path, exist_ok=True)  # Create new split folder if it doesn't exist
    
    # Iterate through each track folder
    for track in os.listdir(split_path):
        track_path = os.path.join(split_path, track)
        new_track_path = os.path.join(new_split_path, track)
        
        if os.path.isdir(track_path):
            os.makedirs(new_track_path, exist_ok=True)  # Create new track folder
            
            # Define paths to the original audio files
            mixture_path = os.path.join(track_path, "mixture.wav")
            bass_path = os.path.join(track_path, "bass.wav")
            drums_path = os.path.join(track_path, "drums.wav")
            other_path = os.path.join(track_path, "other.wav")
            vocals_path = os.path.join(track_path, "vocals.wav")
            
            # Copy original audio files to the new directory
            for file_path in [mixture_path, bass_path, drums_path, other_path, vocals_path]:
                if os.path.exists(file_path):
                    file_name = os.path.basename(file_path)
                    new_file_path = os.path.join(new_track_path, file_name)
                    audio = AudioSegment.from_wav(file_path)
                    audio = resample_audio(audio, sample_rate)
                    audio.export(new_file_path, format="wav")
            
            # Check if mixture.wav exists and proceed with mixing
            if os.path.exists(mixture_path):
                # Load mixture audio and resample
                mixture = AudioSegment.from_wav(mixture_path)
                mixture = resample_audio(mixture, sample_rate)
                
                # Select a random wind audio from the list, convert to .wav if necessary, resample and adjust its length
                wind_path = random.choice(wind_audio_files)
                wind = load_wind_audio(wind_path)
                wind = resample_audio(wind, sample_rate)
                adjusted_wind = adjust_audio_length(wind, len(mixture))
                
                # Reduce the wind intensity by adjusting its gain
                adjusted_wind = adjusted_wind + wind_gain_reduction  # Reduce wind volume
                
                # Mix the adjusted wind with the mixture
                new_mixture = mixture.overlay(adjusted_wind)
                
                # Save the new mixed audio as 'new_mixture.wav'
                new_mixture_path = os.path.join(new_track_path, "new_mixture.wav")
                new_mixture.export(new_mixture_path, format="wav", parameters=["-ar", str(sample_rate)])
                
                # Save the adjusted wind audio as 'wind.wav'
                wind_output_path = os.path.join(new_track_path, "wind.wav")
                adjusted_wind.export(wind_output_path, format="wav", parameters=["-ar", str(sample_rate)])
                
                # Delete the old mixture file from the new directory
                old_mixture_path = os.path.join(new_track_path, "mixture.wav")
                if os.path.exists(old_mixture_path):
                    os.remove(old_mixture_path)
                    print(f"Deleted old mixture: {old_mixture_path}")
                
                print(f"Processed track: {track} with wind: {os.path.basename(wind_path)}")

print("All tracks processed and saved in the new directory structure successfully.")

