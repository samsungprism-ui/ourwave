import os
import h5py
import numpy as np
from sortedcontainers import SortedList
from torch.utils.data import Dataset
from tqdm import tqdm
from data.utils import load

class SeparationDataset(Dataset):
    def __init__(self, dataset, partition, instruments, sr, channels, shapes, random_hops, hdf_dir, audio_transform=None, in_memory=False):
        '''
        Initializes a source separation dataset.
        '''
        super(SeparationDataset, self).__init__()

        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        self.hdf_dir = os.path.join(hdf_dir, partition + ".hdf5")

        self.random_hops = random_hops
        self.sr = sr
        self.channels = channels
        self.shapes = shapes
        self.audio_transform = audio_transform
        self.in_memory = in_memory
        self.instruments = instruments

        # Prepare HDF file
        if not os.path.exists(self.hdf_dir):
            # Create HDF file if it does not exist
            if not os.path.exists(hdf_dir):
                os.makedirs(hdf_dir)

            # Create HDF file
            with h5py.File(self.hdf_dir, "w") as f:
                f.attrs["sr"] = sr
                f.attrs["channels"] = channels
                f.attrs["instruments"] = instruments

                print("Adding audio files to dataset (preprocessing)...")
                for idx, example in enumerate(tqdm(dataset[partition])):
                    # Load mix
                    mix_audio, _ = load(example["mix"], sr=self.sr, mono=(self.channels == 1))

                    source_audios = []
                    # Get minimum length to ensure consistency
                    min_length = min([load(example[src], sr=self.sr, mono=(self.channels == 1))[0].shape[1] 
                                      for src in instruments if src in example])
                    min_length = min(min_length, mix_audio.shape[1])

                    # Trim or pad all sources and the mix to the minimum length
                    mix_audio = self._trim_or_pad(mix_audio, min_length)
                    for source in instruments:
                        if source in example:
                            source_audio, _ = load(example[source], sr=self.sr, mono=(self.channels == 1))
                            source_audio = self._trim_or_pad(source_audio, min_length)
                        else:
                            # If source audio is missing, create an empty array of the required length
                            source_audio = np.zeros((self.channels, min_length))
                        source_audios.append(source_audio)

                    # Concatenate source audios along the channel dimension
                    source_audios = np.concatenate(source_audios, axis=0)
                    assert source_audios.shape[1] == mix_audio.shape[1]

                    # Add to HDF5 file
                    grp = f.create_group(str(idx))
                    grp.create_dataset("inputs", shape=mix_audio.shape, dtype=mix_audio.dtype, data=mix_audio)
                    grp.create_dataset("targets", shape=source_audios.shape, dtype=source_audios.dtype, data=source_audios)
                    grp.attrs["length"] = mix_audio.shape[1]
                    grp.attrs["target_length"] = source_audios.shape[1]

        # Check compliance of the HDF file with input settings
        with h5py.File(self.hdf_dir, "r") as f:
            if f.attrs["sr"] != sr or \
                    f.attrs["channels"] != channels or \
                    list(f.attrs["instruments"]) != instruments:
                raise ValueError("Existing HDF file attributes do not match the expected settings.")

        # Set sampling positions
        with h5py.File(self.hdf_dir, "r") as f:
            lengths = [f[str(song_idx)].attrs["target_length"] for song_idx in range(len(f))]
            lengths = [(l // self.shapes["output_frames"]) + 1 for l in lengths]

        self.start_pos = SortedList(np.cumsum(lengths))
        self.length = self.start_pos[-1]

    def _trim_or_pad(self, audio, target_length):
        """Trim or pad an audio array to the target length."""
        current_length = audio.shape[1]
        if current_length > target_length:
            # Trim to the target length
            audio = audio[:, :target_length]
        elif current_length < target_length:
            # Pad to the target length
            pad_amount = target_length - current_length
            audio = np.pad(audio, ((0, 0), (0, pad_amount)), mode='constant')
        return audio

    def __getitem__(self, index):
        # Open HDF5 file
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_dir, 'r', driver=driver)

        # Determine which sample to read
        audio_idx = self.start_pos.bisect_right(index)
        if audio_idx > 0:
            index = index - self.start_pos[audio_idx - 1]

        # Get length of audio and target
        audio_length = self.hdf_dataset[str(audio_idx)].attrs["length"]
        target_length = self.hdf_dataset[str(audio_idx)].attrs["target_length"]

        # Random or sequential sampling of starting positions
        if self.random_hops:
            start_target_pos = np.random.randint(0, max(target_length - self.shapes["output_frames"] + 1, 1))
        else:
            start_target_pos = index * self.shapes["output_frames"]

        # Check front padding
        start_pos = start_target_pos - self.shapes["output_start_frame"]
        pad_front = max(0, -start_pos)
        start_pos = max(0, start_pos)

        # Check back padding
        end_pos = start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"]
        pad_back = max(0, end_pos - audio_length)
        end_pos = min(audio_length, end_pos)

        # Read inputs and targets
        audio = self.hdf_dataset[str(audio_idx)]["inputs"][:, start_pos:end_pos].astype(np.float32)
        targets = self.hdf_dataset[str(audio_idx)]["targets"][:, start_pos:end_pos].astype(np.float32)

        # Pad if necessary
        if pad_front > 0 or pad_back > 0:
            audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)
            targets = np.pad(targets, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        # Split targets into individual instruments
        targets = {inst: targets[idx * self.channels:(idx + 1) * self.channels] for idx, inst in enumerate(self.instruments)}

        # Apply optional audio transformations
        if hasattr(self, "audio_transform") and self.audio_transform is not None:
            audio, targets = self.audio_transform(audio, targets)

        return audio, targets

    def __len__(self):
        return self.length

