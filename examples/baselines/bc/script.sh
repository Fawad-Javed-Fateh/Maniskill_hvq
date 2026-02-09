#!/bin/bash
python bc_transformer.py --env-id "PushCube-v1" \
  --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max-episode-steps 100 \
  --total-iters 30000


#!/bin/bash
python bc_transformer.py --env-id "PickCube-v1" \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max-episode-steps 100 \
  --total-iters 30000


#!/bin/bash
python bc_transformer.py --env-id "StackCube-v1" \
  --demo-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max-episode-steps 100 \
  --total-iters 30000