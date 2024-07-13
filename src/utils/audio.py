import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn

def load_wave(wave_file_path:str, sample_rate:int, is_torch:bool=True, mono:bool=False):
    """
    Load a wave file from the given file path.

    Args:
        wave_file_path (str): The path to the wave file.
        sample_rate (int, optional): The desired sample rate of the loaded wave file. 
        is_torch (bool, optional): Whether to return the wave as a torch tensor. 
            If set to False, the wave will be converted to a numpy array. 
            Defaults to True.
        mono (bool, optional): Whether to convert the wave to mono. 
            If set to True, only the first channel will be returned. 
            Defaults to False.

    Returns:
        tuple: A tuple containing the loaded wave and the sample rate.
    """
    
    wave, sr = torchaudio.load(wave_file_path)
    if mono:
        wave = wave[0]
    if sample_rate > 0 and sample_rate != sr:
        wave = torchaudio.transforms.Resample(sr, sample_rate)(wave)
    else:
        sample_rate=sr
    if not is_torch:
        wave = wave.cpu().detach().numpy().copy()
    return wave, sample_rate

######################
# Mel Spec Transform #
######################

MAX_WAV_VALUE = 32768.0

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


mel_basis = {}
hann_windows = {}

def spectrogram_torch(y:torch.Tensor, n_fft, hop_size, win_size):
    """Convert waveform into Linear-frequency Linear-amplitude spectrogram.

    Args:
        y             :: (B, T) - Audio waveforms
        n_fft
        sampling_rate
        hop_size
        win_size
        center
    Returns:
        :: (B, Freq, Frame) - Linear-frequency Linear-amplitude spectrogram
    """
    # Validation
    if torch.min(y) < -1.07:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.07:
        print("max value is ", torch.max(y))
    if y.dim() == 3:
        # チャンネルを含んでいるwaveformは、チャンネル除去
        y = y.squeeze(1)
    # Window - Cache if needed
    global hann_windows
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_windows:
        hann_windows[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    # Padding
    y = torch.nn.functional.pad(
        y,
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )

    spec = torchaudio.functional.spectrogram(
        waveform=y,
        pad=0,
        window=hann_windows[wnsize_dtype_device],
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        power=2,
        normalized=False,
        center=False
    )
    
    # Linear-frequency Linear-amplitude spectrogram :: (B, Freq, Frame, RealComplex=2) -> (B, Freq, Frame)
    spec = torch.sqrt(spec + 1e-6)
    return spec

def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    # MelBasis - Cache if needed
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )

    # Mel-frequency Log-amplitude spectrogram :: (B, Freq=num_mels, Frame)
    melspec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    melspec = spectral_normalize_torch(melspec)
    return melspec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax
):
    """Convert waveform into Mel-frequency Log-amplitude spectrogram.

    Args:
        y       :: (B, T)           - Waveforms
    Returns:
        melspec :: (B, Freq, Frame) - Mel-frequency Log-amplitude spectrogram
    """
    # Linear-frequency Linear-amplitude spectrogram :: (B, T) -> (B, Freq, Frame)
    spec = spectrogram_torch(y, n_fft, hop_size, win_size)

    # Mel-frequency Log-amplitude spectrogram :: (B, Freq, Frame) -> (B, Freq=num_mels, Frame)
    melspec = spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)
    return melspec