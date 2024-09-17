import os

# Define the path to the new dataset
new_root_dir = "/home/stemtec/Desktop/Wave-U-Net-Pytorch/data/new_archive"  # Path to your new dataset

# Iterate through each split (train, test, valid)
for split in ["train", "test", "valid"]:
    split_path = os.path.join(new_root_dir, split)
    
    # Iterate through each track folder
    for track in os.listdir(split_path):
        track_path = os.path.join(split_path, track)
        
        if os.path.isdir(track_path):
            # Define paths to the new and old mixture files
            new_mixture_path = os.path.join(track_path, "new_mixture.wav")
            mixture_path = os.path.join(track_path, "mixture.wav")
            
            # Check if new_mixture.wav exists and rename it to mixture.wav
            if os.path.exists(new_mixture_path):
                os.rename(new_mixture_path, mixture_path)
                print(f"Renamed {new_mixture_path} to {mixture_path}")

print("All new_mixture.wav files have been renamed to mixture.wav successfully.")

