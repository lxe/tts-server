

import random
import torch
import torchaudio
import yaml
from collections import OrderedDict

import librosa
import numpy as np
import phonemizer
import nltk
from nltk.tokenize import word_tokenize

from cached_path import cached_path

import sys
import os

# Hack needed to make things imported inside styletts2 to work
current_dir = os.path.dirname(__file__)
styletts2_path = os.path.abspath(os.path.join(current_dir, 'styletts2'))
sys.path.append(styletts2_path)

from .styletts2.Modules.diffusion.sampler import (
    ADPM2Sampler,
    DiffusionSampler,
    KarrasSchedule,
)

from .styletts2.Utils.PLBERT.util import load_plbert

from .styletts2.models import build_model, load_ASR_models, load_F0_models
from .styletts2.text_utils import TextCleaner
from .styletts2.utils import recursive_munch

# Set device based on availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seeds()

if not nltk.find('tokenizers/punkt'):
    nltk.download('punkt')

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)

def preprocess_audio(wave, mean=-4, std=4):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def fix_path(path):
    return os.path.join(styletts2_path, path)

def load_model(
    config_path="hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/config.yml", 
    checkpoint_path="hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/epochs_2nd_00020.pth"
):
    config = yaml.safe_load(open(str(cached_path(config_path))))

    ASR_config = fix_path(config.get('ASR_config', False))
    ASR_path = fix_path(config.get('ASR_path', False))
    F0_path = fix_path(config.get('F0_path', False))
    BERT_path = fix_path(config.get('PLBERT_dir', False))
    
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    pitch_extractor = load_F0_models(F0_path)
    plbert = load_plbert(BERT_path)

    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    params_whole = torch.load(str(cached_path(checkpoint_path)), map_location='cpu')
    params = params_whole['net']

    for key in model:
        if key in params:
            state_dict = params[key]
            # Check and remove 'module.' prefix from the state dict keys
            if list(state_dict.keys())[0].startswith('module.'):
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
                    new_state_dict[name] = v
                state_dict = new_state_dict
            try:
                model[key].load_state_dict(state_dict)
            except RuntimeError as e:
                print(f"Error loading state dict for {key}: {e}")
                
    [model[key].eval() for key in model]
    [model[key].to(device) for key in model]
    
    return model_params, model

model_params, model = load_model()
text_cleaner = TextCleaner()
global_phonemizer = phonemizer.backend.EspeakBackend(
    language='en-us', 
    preserve_punctuation=True,  
    with_stress=True
)
schedule = KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0)
sampler = DiffusionSampler(
    model.diffusion.diffusion, 
    sampler=ADPM2Sampler(), 
    sigma_schedule=schedule,
    clamp=False
)

# Function for Length to Mask conversion
def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# Function to compute style from a given path
def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess_audio(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)

def LFinference(
    text, 
    s_prev, 
    ref_s, 
    alpha=0.3, 
    beta=0.7, 
    t=0.7, 
    diffusion_steps=5, 
    embedding_scale=1
):
    # Preprocess text
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = ' '.join(word_tokenize(ps[0])).replace('``', '"').replace("''", '"')
    
    # Prepare tokens
    tokens = torch.LongTensor([0] + text_cleaner(ps)).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        # Encode text
        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        # Predict style
        s_pred = sampler(
            noise=torch.randn((1, 256)).unsqueeze(1).to(device),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s,  # reference from the same speaker as the embedding
            num_steps=diffusion_steps
        ).squeeze(1)

        if s_prev is not None:
            # Convex combination of previous and current style
            s_pred = t * s_prev + (1 - t) * s_pred

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        s_pred = torch.cat([ref, s], dim=-1)

        # Predict duration
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        # Create alignment target
        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # Encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        # Predict F0 and N
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        # Decode
        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()[..., :-100], s_pred  # Fix weird pulse at the end later