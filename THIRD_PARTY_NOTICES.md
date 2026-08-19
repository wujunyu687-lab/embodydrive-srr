# Third-Party Notices

## HorizonDrive and VideoX-Fun model code

The `videox_fun/` directory is derived from the HorizonDrive repository:

- Project: HorizonDrive
- Source: https://github.com/zcliangyue/HorizonDrive
- Upstream commit used by the local checkout: `8b13672`
- License: MIT, reproduced in `licenses/HorizonDrive-MIT.txt`

The local transformer includes robot-specific action and proprioception hooks
required by EmbodyDrive checkpoints. Those changes are visible in
`videox_fun/models/wan_transformer3d_unified_6v/model.py`.

## Wan2.1

`videox_fun/models/wan_vae.py` is derived from the Wan2.1 VAE implementation
and retains its upstream attribution header. No Wan model weights are included.

## DROID

The code reads DROID RLDS-style TFRecord shards. No DROID data is included.
Users are responsible for following the dataset's license and access terms.
