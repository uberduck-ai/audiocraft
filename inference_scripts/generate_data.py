import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio
from stable_audio_tools.models import create_model_from_config
from collections import OrderedDict
import random
import AudioSegment

import ipdb
import os
import json
import glob
import math

'''
The purpose of this script is to take the input data, repeat it till it hits 60 seconds in length,
and then create random crops down to 30 seconds.
'''

# Change to point to the appropriate spots where the raw wav files live
raw_data_path  = "/home/ubuntu/datasets/drumloops/drumloops"
new_save_path = "/home/ubuntu/datasets/drumloops/extended_drumloops_60s"

if not os.path.exists(new_save_path):
    os.makedirs(new_save_path)

os.chdir(raw_data_path)

# loop through the raw data, and repeat it till it hits 60 seconds in length
for wav in glob.glob("*.wav"):

    # this is hardcoded to only pick out 20% of the input data since it was quite a lot
    # feel free to remove this if statement
    if random.random() < 0.2:
        print(wav)

        try:
            history_prompt_wav, history_prompt_sr = torchaudio.load(wav)
        except RuntimeError:
            print(f"failed to open {wav}, moving onto the next one")
            continue

        audio_length = math.floor(history_prompt_wav.shape[-1] / history_prompt_sr)
        if math.floor(audio_length) == 0:
            continue

        # make every clip approx 60s long
        total_gen_length = 60

        audio_repeat_factor = math.ceil(total_gen_length / audio_length)
        repeated_history_prompt = torch.cat([history_prompt_wav]*audio_repeat_factor, 1)
        audio_write(os.path.join(new_save_path, f'{wav[:-4]}_60s_extended'), repeated_history_prompt.cpu(), history_prompt_sr, strategy="loudness", loudness_compressor=True)


# move all the original and cropped clips to their respective directories
os.makedirs(os.path.join(new_save_path, "original"), exist_ok=True)
os.makedirs(os.path.join(new_save_path, "chunked"), exist_ok=True)


for filename in os.listdir(new_save_path):
    if filename.endswith(('.mp3', '.wav', '.flac')):

        curr_file_location = os.path.join(new_save_path, filename)
        new_file_location = os.path.join(new_save_path, "original", filename)
        # move original file out of the way
        os.rename(curr_file_location, new_file_location)
        audio = AudioSegment.from_file(new_file_location)

        # resample
        audio = audio.set_frame_rate(44100)

        # get three random 30 second chunks from each clip
        for clip in range(3):
            
            # make sure to not cut off the end, hence go past the halfway mark
            starting_point = np.random.randint(0, len(audio) - 35000)
            chunk = audio[starting_point:(starting_point + 30000)]

            chunked_file_location = os.path.join(new_save_path, "chunked", f"{filename[:-4]}_{clip}.wav")
            chunk.export(chunked_file_location, format="wav")
