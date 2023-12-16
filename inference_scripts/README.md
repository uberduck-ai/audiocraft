## Requirements

In addition to the requirements in the main readme, make sure you have the following other libaries installed:

- `librosa`
- `AudioSegment`
- `torchaudio`
- `dora`

## Data setup/inferencing

- To generate a dataset of 30-second clips, run `python generate_data.py`. You'll need to change the load and destination directories as necessary.
- To format the data in a manner that's ingestible by the model, run `python format_data.py`. Make sure the train/eval `.jsonl` files are saved respectively saved under `audiocraft/egs/train/data.jsonl` and `audiocraft/egs/eval/data.jsonl`. This is consistent with the naming conventions provided in the `audiocraft/config/dset/audio/train.yaml` file.
- To convert a a finetuned model checkpoint to one that's usable for inferencing,run `python convert_model_ckpt.py`. Change the model signature as ncessary, the trained model gets saved saved under `/tmp/audiocraft_ubuntu/`, which is created automatically.
- To inference an audio segment (e.g. continue it, conditioned based on text), run `python generate_example_inferences.py`. Change the `description` as you see fit, and the save/load locations.

## Training

Assuming all the train `.jsonl` and `.yaml` files are setup correctly, the following command should work for training:

```
dora -P audiocraft run solver=musicgen/musicgen_base_32khz model/lm/model_scale=small continue_from=//pretrained/facebook/musicgen-small conditioner=text2music dset=audio/train dataset.num_workers=2 dataset.valid.num_samples=1 dataset.batch_size=1 schedule.cosine.warmup=8 optim.optimizer=adamw optim.lr=1e-4 optim.epochs=20 optim.updates_per_epoch=1000 optim.adam.weight_decay=0.01 generate.lm.prompted_samples=False generate.lm.gen_gt_samples=True
```

Change `epochs` and `batch_size` as you see fit, on a `p3.2xlarge` I could not do more than a batch size of 1. Note that an `epoch` is *not* a run through the full dataset, rather, it's just the number times the model gets saved out, spaced out by the `updates_per_epoch`. E.g 20 epochs with `updates_per_epoch = 1000` means the model gets trained through `20,000` gradient descent steps, *not* twenty passes through the full training set. Further details available [here](https://github.com/facebookresearch/audiocraft/blob/main/docs/TRAINING.md), under `About Epochs`.

[This](https://colab.research.google.com/drive/13tbcC3A42KlaUZ21qvUXd25SFLu8WIvb?usp=sharing#scrollTo=uvtz2uKm59AR) training notebook is also very helpful as well.
