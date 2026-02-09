# Code for IROS 2025 Action Tokenizer paper on ManiSkill 



## Installation
Installation of ManiSkill is extremely simple, you only need to run a few pip installs and setup Vulkan for rendering.

```bash
# install the package
pip install --upgrade mani_skill
# install a version of torch that is compatible with your system
pip install torch
```

For more information as well as environment bug, refer to the [documentation](https://maniskill.readthedocs.io/en/latest/user_guide) for setup. For all experiments, remember to use RGBD expert demonstration.

Finally you also need to set up Vulkan with [instructions here](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan)

For more details about installation (e.g. from source, or doing troubleshooting) see [the documentation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html
)

## Collect data
Please refer to [official document](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html) on how to collect expert trajectory first.

## Train & Evaluation
```
cd examples/baselines/act
```
First replay the expert trajectory data
```
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state \
  --save-traj --num-procs 10 -b cpu
```

Then train and evaluate using:

```
python train_<action_tokenizer>.py --env-id PickCube-v1 \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pos.cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max_episode_steps 100 --total_iters 5000 --save_freq 5000 
```

where ```<action_tokenizer>``` is the name of the tokenizer you want to evaluate, for example ```bin, lfq_vae,...```.