from audiocraft.utils import export
from audiocraft import train
import os
import ipdb

# obtained from training, need to manually set according to whatever the dora command sets as the LM signature
sig = "5bc3e300"
converted_ckpt_dir = f"./audiocraft_{sig}_2"

if not os.path.exists(f"./audiocraft_{sig}"):
    os.makedirs(f"./audiocraft_{sig}")

xp = train.main.get_xp_from_sig(sig)

export.export_lm(xp.folder / 'checkpoint.th', os.path.join(converted_ckpt_dir, "state_dict.bin"))
# You also need to bundle the EnCodec model you used !!
## Case 1) you trained your own
# xp_encodec = train.main.get_xp_from_sig('SIG_OF_ENCODEC')
# export.export_encodec(xp_encodec.folder / 'checkpoint.th', '/checkpoints/my_audio_lm/compression_state_dict.bin')
## Case 2) you used a pretrained model. Give the name you used without the //pretrained/ prefix.
## This will actually not dump the actual model, simply a pointer to the right model to download.
export.export_pretrained_compression_model('facebook/encodec_32khz', os.path.join(converted_ckpt_dir, "compression_state_dict.bin"))
