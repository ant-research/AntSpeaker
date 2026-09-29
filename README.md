# MECT: Mixture of Experts with CNN-Transformer Network for Speaker verification


<p align="center">
<a href="https://huggingface.co/AntResearch/AntSpeaker"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-AntSpeaker-blue.svg" alt="AntSpeaker on Hugging Face"></a>
<a href="https://arxiv.org/abs/2609.24061"><img src="https://img.shields.io/badge/arXiv-2609.24061-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv"></a>
<a href="https://github.com/ant-research/AntSpeaker"><img src="https://img.shields.io/badge/GitHub-AntSpeaker-181717.svg?logo=github&logoColor=white" alt="GitHub"></a>
<img src="https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-lightgrey" alt="License">
</p>

**Official repository for the paper: [MECT: Mixture of Experts with CNN-Transformer Network for Speaker verification](https://arxiv.org/abs/2609.24061)**



## 📖 Overview

**MECT** is a speaker verification model that integrates the Mixture-of-Experts (MoE) mechanism into a CNN-Transformer backbone with optimized block structure and stacking scheme. The model explores four MoE variants spanning utterance-level and frame-level granularity with dense and sparse routing strategies, proving that the MoE mechanism is effective over the baseline with only a small increase in parameters.

### ✨ Key Features

- **Mixture of Experts** — integrates the MoE mechanism into a CNN-Transformer backbone with optimized block structure and stacking scheme, exploring four MoE variants that span utterance-level and frame-level granularity with dense and sparse routing strategies
- **Lightweight & Scalable** — four model sizes ranging from 3.78M to 9.57M parameters
- **Strong Performance** — competitive EER on VoxCeleb1 benchmarks
- **Streaming support** — a streaming speaker verification model is also provided for real-time, low-latency scenarios such as live meetings, voice assistants, and on-device applications


## 📅 Roadmap

- **[Coming Soon]** Release streaming inference tutorial.
- **[2026.09.23]** Release streaming MECT-B2 checkpoint trained on VoxCeleb2.
- **[2026.09.21]** Paper published on arXiv.
- **[2026.09.15]** Release MECT-A1/A2/B1/B2 checkpoints trained on VoxCeleb2 and MECT-B2 checkpoint trained on VoxCeleb2 + VoxBlink2.
 

## 📦 Model Variants

This repository provides **4 model sizes** with **6 checkpoints** in total:

| Model | Parameters | Training Data | Checkpoint File |
|:-----:|:----------:|:-------------:|:---------------:|
| MECT-A1 | 3.78M | vc2 | [mect_a1_vc2.pt](https://huggingface.co/AntResearch/AntSpeaker/blob/main/mect_a1_vc2.pt) |
| MECT-A2 | 4.12M | vc2 | [mect_a2_vc2.pt](https://huggingface.co/AntResearch/AntSpeaker/blob/main/mect_a2_vc2.pt) |
| MECT-B1 | 8.26M | vc2 | [mect_b1_vc2.pt](https://huggingface.co/AntResearch/AntSpeaker/blob/main/mect_b1_vc2.pt) |
| MECT-B2 | 9.57M | vc2 | [mect_b2_vc2.pt](https://huggingface.co/AntResearch/AntSpeaker/blob/main/mect_b2_vc2.pt) |
| MECT-B2 | 9.57M | vc2+vb2 | [mect_b2_vb2.pt](https://huggingface.co/AntResearch/AntSpeaker/blob/main/mect_b2_vb2.pt) |
| MECT-B2-Causal | 9.57M | vc2 | [mect_b2_vc2_streaming.pt](https://huggingface.co/AntResearch/AntSpeaker/blob/main/mect_b2_vc2_streaming.pt) |

> **MECT-B2-Causal** is a streaming variant of MECT-B2 to support **streaming speaker verification**.


## 📊 Performance
### VoxCeleb1 Test Sets (EER % / minDCF)

| Model | Training Data | Calibration | Vox1-O | Vox1-E | Vox1-H |
|:-----:|:-------------:|:-------:|:------:|:------:|:------:|
| MECT-A1 | VoxCeleb2 | No | 0.44 / 0.038 | 0.60 / 0.060 | 1.05 / 0.108 |
| MECT-A2 | VoxCeleb2 | No | 0.37 / 0.032 | 0.54 / 0.056 | 0.97 / 0.096 |
| MECT-B1 | VoxCeleb2 | No | 0.37 / 0.029 | 0.50 / 0.051 | 0.94 / 0.092 |
| MECT-B2 | VoxCeleb2 | No | 0.27 / 0.024 | 0.46 / 0.048 | 0.85 / 0.082 |
| MECT-B2 | VoxCeleb2 + VoxBlink2 | No | 0.23 / 0.013 | 0.29 / 0.028 | 0.54 / 0.052 |
| MECT-B2 | VoxCeleb2 + VoxBlink2 | Yes | 0.22 / 0.012 | 0.28 / 0.026 | 0.52 / 0.048 |


## 🏗️ Architecture
MECT integrates CNN and Transformer components within a MoE framework. The framework supports four types of MoE configurations.

<div align="center">
  <img src="assets/block_architecture.png" alt="MECT Architecture" width="600">
  <p align="center"><b>Figure 1:</b> The architecture of MECT Block.</p>
</div>



## 🚀 Quick Start

### 1. Download Model Checkpoints

**From HuggingFace:**

```bash
# Clone the HuggingFace repo (requires Git LFS)
git lfs install
git clone https://huggingface.co/AntResearch/AntSpeaker

# Or download a single checkpoint
hf download AntResearch/AntSpeaker mect_b2_vc2.pt --local-dir ./
```

**From GitHub:**

Model checkpoints are available on [HuggingFace](https://huggingface.co/AntResearch/AntSpeaker). Source code is hosted on [GitHub](https://github.com/ant-research/AntSpeaker).

### 2. Installation

```bash
pip install torch torchaudio soundfile
```

### 3. Extract Speaker Embeddings

```python
import torch
import torchaudio
import soundfile as sf
import torchaudio.compliance.kaldi as kaldi
import antspeaker.models
from antspeaker.utils.registry import create_model

# 1. Load model
ckpt = torch.load("mect_b2_vc2.pt", map_location="cpu")
model = create_model(ckpt["config"]["model"])
model, _ = model.load_model("mect_b2_vc2.pt")
feat_config = ckpt["config"]["feature"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 2. Load audio (resample to model's sampling rate if needed)
waveform, sr = sf.read("example.wav", dtype="float32")
waveform = torch.from_numpy(waveform.T).unsqueeze(0) 
if sr != feat_config["sampling_rate"]:
    waveform = torchaudio.transforms.Resample(
        orig_freq=sr, new_freq=feat_config["sampling_rate"])(waveform)

# 3. Extract 80-dim Mel filterbank features
fbank = kaldi.fbank(
    waveform * (1 << 15),
    num_mel_bins=feat_config["num_mel_bins"],
    frame_length=feat_config["frame_length"],
    frame_shift=feat_config["frame_shift"],
    dither=feat_config["dither"],
    energy_floor=0.0,
    window_type="hamming",
    sample_frequency=feat_config["sampling_rate"],
)
fbank = fbank.unsqueeze(0).transpose(-1, -2)

# 4. Extract embedding
with torch.no_grad():
    embedding = model(fbank.to(device)).cpu()
```

### 4. Speaker Verification

```python
import torch.nn.functional as F

# Extract embeddings for two utterances
emb1 = model(fbank1.to(device)).cpu()
emb2 = model(fbank2.to(device)).cpu()

# Cosine similarity
cosine_sim = F.cosine_similarity(emb1, emb2)
print(f"Cosine similarity: {cosine_sim.item():.4f}")
```




## 📝 Citation
If you use MECT in your research, please cite:
```bibtex
@article{zheng2026mect,
    title={MECT: Mixture of Experts with CNN-Transformer Network for Speaker verification}, 
    author={Yu Zheng and Jinghan Peng and ChangHao Zhang and Jian Liu and Weiqiang Wang},
    journal={arXiv preprint arXiv:2609.24061},
    year={2026},
}
```

## 📜 License
CC-BY-NC-SA 4.0


## 📧 Contact
- **Yu Zheng**: [zy466683@antgroup.com](mailto:zy466683@antgroup.com)

