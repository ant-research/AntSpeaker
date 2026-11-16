import os
import sys
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

import soundfile as sf

import antspeaker.models
from antspeaker.utils.registry import create_model
from antspeaker.models.backbones.mect import StreamingCache


def chunk_streaming(data, chunk_size, frame_length=25, frame_shift=10, sampling_rate=16000):
    chunk_size = int(chunk_size * sampling_rate)
    data_length = data.shape[-1]
    if data_length > chunk_size:
        # 【overlap carry】
        frame_length_samples = int(frame_length * sampling_rate / 1000)
        frame_shift_samples = int(frame_shift * sampling_rate / 1000)
        overlap = frame_length_samples - frame_shift_samples  # 240

        num_slices = int(data_length // chunk_size + 1)
        new_data = []
        for i in range(num_slices - 1):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            new_data.append(data[:, start:end + overlap])
    else:
        new_data = [data]

    return new_data


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model path '{model_path}' does not exist.")
    ckpt = torch.load(model_path, map_location="cpu")
    feat_config = ckpt['config']['feature']
    model = create_model(ckpt['config']['model'])
    model, _ = model.load_model(model_path)
    return model, feat_config


def load_audio(audio_path, sampling_rate):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio path '{audio_path}' does not exist.")
    waveform, sr = sf.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(waveform.T).unsqueeze(0) 
    if sr != sampling_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=sampling_rate)(waveform) # (1, T)
    return waveform, sr

def extract_fbank(waveform, feat_config):
    waveform = waveform * (1 << 15)
    fbank = kaldi.fbank(waveform,
        num_mel_bins=feat_config["num_mel_bins"],
        frame_length=feat_config["frame_length"],
        frame_shift=feat_config["frame_shift"],
        dither=feat_config["dither"],
        energy_floor=0.0,
        window_type="hamming",
        sample_frequency=feat_config['sampling_rate'])
    fbank = fbank.unsqueeze(0).transpose(-1, -2) # (1, D, T)
    return fbank
    

if __name__ == "__main__":
    ckpt_path = sys.argv[1]
    audio_path = sys.argv[2]
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    
    model, feat_config = load_model(ckpt_path)
    model = model.to(device)
    model.eval()
    
    waveform, _ = load_audio(
        audio_path, feat_config['sampling_rate'])

    waveform = waveform[:,:32000]
    
    chunk_streams = chunk_streaming(waveform, 0.1)

    backbone_cache = StreamingCache()
    pooling_cache = None
    for chunk in chunk_streams:
        fbank = extract_fbank(chunk, feat_config).to(device)
        embedding, backbone_cache, pooling_cache = model.forward_streaming_chunk(fbank, backbone_cache, pooling_cache)
        print(embedding.shape)
    print(embedding)
    