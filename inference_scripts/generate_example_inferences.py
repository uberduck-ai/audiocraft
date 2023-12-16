import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio
import math
import os

# change save/load paths as necessary
prompt_path = "/home/ubuntu/datasets/metronome_pitches_bpms/metronome_a_flat_2_88.wav"
save_path = "/home/ubuntu/audiocraft_run_5bc3e300_2/"
save_name = prompt_path.split('/')[-1].split('.')[0]

if not os.path.exists(save_path):
    os.makedirs(save_path)

# change description as necessary
description = ["drums"]

history_prompt_wav, history_prompt_sr = torchaudio.load(prompt_path)

# truncate metronome sequence, optionally change if you're feeding in 
# an already short prompt
len_history_prompt_wav = history_prompt_wav.shape[-1]
history_prompt_wav = history_prompt_wav[:, :math.floor(len_history_prompt_wav / 3)]

### FINETUEND VS UNFINETUNED MODEL

# load UNFINETUNED model
# model = MusicGen.get_pretrained('/home/ubuntu/audiocraft_5bc3e300_2')
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=30)

converted_history_prompt_wav = convert_audio(history_prompt_wav, from_rate = history_prompt_sr, to_rate = model.sample_rate, to_channels=model.audio_channels)
audio_write(os.path.join(save_path, save_name), history_prompt_wav.cpu(), history_prompt_sr, strategy="loudness", loudness_compressor=True)

wav, tokens = model.generate_continuation(history_prompt_wav, history_prompt_sr, descriptions=description, return_tokens=True)

# wav = model.generate(description)
audio_write(os.path.join(save_path, f'continued_{save_name}_unfinetuned'), wav[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)






