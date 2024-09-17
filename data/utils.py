import librosa
import numpy as np
import soundfile
import torch


def random_amplify(mix, targets, shapes, min, max):
    '''
    Data augmentation by randomly amplifying sources before adding them to form a new mixture
    :param mix: Original mixture
    :param targets: Source targets
    :param shapes: Shape dict from model
    :param min: Minimum possible amplification
    :param max: Maximum possible amplification
    :return: New data point as tuple (mix, targets)
    '''
    residual = mix  # start with original mix
    for key in targets.keys():
        if key != "mix":
            residual -= targets[key]  # subtract all instruments (output is zero if all instruments add to mix)
    mix = residual * np.random.uniform(min, max)  # also apply gain data augmentation to residual
    for key in targets.keys():
        if key != "mix":
            targets[key] = targets[key] * np.random.uniform(min, max)
            mix += targets[key]  # add instrument with gain data augmentation to mix
    mix = np.clip(mix, -1.0, 1.0)
    return crop_targets(mix, targets, shapes)


def crop_targets(mix, targets, shapes):
    '''
    Crops target audio to the output shape required by the model given in "shapes"
    :param mix: Mixture audio signal
    :param targets: Dictionary of target sources
    :param shapes: Dictionary defining shape properties for cropping
    :return: Cropped mixture and targets
    '''
    for key in targets.keys():
        if key != "mix":
            targets[key] = targets[key][:, shapes["output_start_frame"]:shapes["output_end_frame"]]
    return mix, targets


def load(path, sr=22050, mono=True, mode="numpy", offset=0.0, duration=None):
    '''
    Loads an audio file and optionally resamples it.
    :param path: Path to the audio file
    :param sr: Target sampling rate
    :param mono: Whether to convert the audio to mono
    :param mode: Output mode, either "numpy" or "pytorch" tensor
    :param offset: Start reading after this time (in seconds)
    :param duration: Only load up to this much audio (in seconds)
    :return: Loaded audio signal and sampling rate
    '''
    y, curr_sr = librosa.load(path, sr=sr, mono=mono, res_type='kaiser_fast', offset=offset, duration=duration)

    if len(y.shape) == 1:
        # Expand channel dimension if mono
        y = y[np.newaxis, :]

    if mode == "pytorch":
        y = torch.tensor(y)

    return y, curr_sr


def write_wav(path, audio, sr):
    '''
    Writes an audio file to disk
    :param path: Path to save the file
    :param audio: Audio data to write
    :param sr: Sampling rate
    '''
    soundfile.write(path, audio.T, sr, "PCM_16")


def resample(audio, orig_sr, new_sr, mode="numpy"):
    '''
    Resamples an audio signal to a new sampling rate
    :param audio: Input audio signal
    :param orig_sr: Original sampling rate
    :param new_sr: New sampling rate
    :param mode: Output mode, either "numpy" or "pytorch" tensor
    :return: Resampled audio signal
    '''
    if orig_sr == new_sr:
        return audio

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    out = librosa.resample(audio, orig_sr, new_sr, res_type='kaiser_fast')

    if mode == "pytorch":
        out = torch.tensor(out)
    return out

