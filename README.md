# Uncertainty-aware-training

Use `train.py` to start training. 

The uncertainty loss is defined in `experiment_setup/UNetExperiment3D.py`. In `configs/config_unet3d.py`, set `loss_type = 0` for baseline loss and `loss_type = 1` to use uncertainty loss. Code is partially based on https://github.com/MIC-DKFZ/basic_unet_example.
