import os
from tqdm import tqdm
import random
import librosa
import json
import random

os.makedirs("./egs/train", exist_ok=True)
os.makedirs("./egs/eval", exist_ok=True)

# dataset containing the audio data
dataset_path = "/home/ubuntu/datasets/metronome_pitches_bpms"
use_tqdm = True

train_len = 0 
eval_len = 0

with open("./egs/train/metronome_data.jsonl", "w") as train_file, open("./egs/eval/metronome_data.jsonl", "w") as eval_file:

    dset = os.listdir(dataset_path)
    random.shuffle(dset)

    for filename in tqdm(dset):

        y, sr = librosa.load(os.path.join(dataset_path, filename))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = round(tempo) # not usually accurate lol
        length = librosa.get_duration(y=y, sr=sr)

        entry = {
            "key": "",
            "artist": "",
            "sample_rate": 44100,
            "file_extension": "wav",
            "description": "drums",
            "keywords": "",
            "duration": length,
            "bpm": tempo,
            "genre": "",
            "title": "",
            "name": "",
            "instrument": "",
            "moods": [],
            "path": os.path.join(dataset_path, filename),
        }

        if random.random() < 0.85:
            train_len += 1
            train_file.write(json.dumps(entry) + '\n')
        else:
            eval_len += 1
            eval_file.write(json.dumps(entry) + '\n')

print(train_len)
print(eval_len)