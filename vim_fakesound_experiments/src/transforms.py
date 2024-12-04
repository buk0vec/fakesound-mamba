import torchaudio.transforms as T
import torchvision
import torch.nn.functional as F
import torch

def transform(audio, sample_rate, n_fft=2048, hop_length=160, n_mels=300, win_length=None):
    mel_spectrogram_32khz = T.MelSpectrogram(
            sample_rate=32000,
            n_fft=2048,
            hop_length=320,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=280,
            mel_scale="htk",
            f_min=0,
            f_max = 32000/2
        )
    train_mean = 2.5853811548103334
    train_std = 48.60284136954955
    normalize_t = torchvision.transforms.Normalize(mean=train_mean, std=train_std)
    s = mel_spectrogram_32khz(audio)
    # this truncates the last 10ms, but that's ok
    # eval is done in 1s and 20ms bins anyways, and a length of 1001 is cursed
    s = F.pad(s, (0, 1000 - s.shape[2]))
    s = normalize_t(s)
    return s

def augment_spec(spec):
    noise = torch.randn_like(spec) * 1e-4
    freq_masking = T.FrequencyMasking(freq_mask_param=140) # mask [0, 140) bins
    # mask up to 1 sec of audio. train set has 1-4 seconds of fake audio, so this shouldn't
    # mask all positive examples
    time_masking = T.TimeMasking(time_mask_param=1000) 
    spec = spec + noise # add noise
    spec = freq_masking(spec) # mask bins
    spec = time_masking(spec) # mask time
    return spec