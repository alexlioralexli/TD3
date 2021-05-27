#!/bin/bash
#
#
#'dm.acrobot.swingup',
#'dm.cheetah.run',
#'dm.quadruped.run',
#'dm.quadruped.walk',
#'dm.finger.turn_hard',
#'dm.walker.run',


#'dm.humanoid.run',
#'dm.humanoid.stand',
#'dm.humanoid.walk',

#'dm.hopper.hop',
#'dm.fish.swim',
#'dm.swimmer.swimmer6',
#'dm.swimmer.swimmer15',



# no target network
#python plotting.py ../logs/1-24-2021-dm-control/dm.quadruped.walk-PytorchSAC-01-25-2021 -t "Quadruped Walk" -sp plots/quadruped_walk_no_target.pdf -y
#python plotting.py ../logs/1-24-2021-dm-control/dm.finger.turn_hard-PytorchSAC-01-25-2021  -t "Finger Turn (hard)" -sp plots/finger_turn_hard_no_target.pdf -y
#python plotting.py ../logs/1-24-2021-dm-control/dm.cheetah.run-PytorchSAC-01-25-2021 -t "Cheetah Run" -sp plots/cheetah_run_no_target.pdf -y
#python plotting.py ../logs/1-24-2021-dm-control/dm.humanoid.run-PytorchSAC-01-25-2021 -t "Humanoid Run" -sp plots/humanoid_run_no_target.pdf -y -l


# main figure
python plotting.py ../logs/1-25-2021-dm-control/dm.acrobot.swingup-PytorchSAC-01-26-2021 -t "Acrobot Swingup" -sp plots/acrobat_swingup.pdf -y
python plotting.py ../logs/1-25-2021-dm-control/dm.finger.turn_hard-PytorchSAC-01-26-2021 ../logs/2-4-dmcontrol/dm.finger.turn_hard-PytorchSAC-02-04-2021 -t "Finger Turn (hard)" -sp plots/finger_turn_hard.pdf -y
python plotting.py ../logs/1-25-2021-dm-control/dm.quadruped.run-PytorchSAC-01-27-2021 ../logs/2-4-dmcontrol/dm.quadruped.run-PytorchSAC-02-04-2021 -t "Quadruped Run" -sp plots/quadruped_run.pdf -y
python plotting.py ../logs/1-25-2021-dm-control/dm.quadruped.walk-PytorchSAC-01-27-2021 ../logs/2-4-dmcontrol/dm.humanoid.walk-PytorchSAC-02-04-2021 -t "Quadruped Walk" -sp plots/quadruped_walk.pdf -y
#python plotting.py ../logs/1-25-2021-dm-control/dm.cheetah.run-PytorchSAC-01-26-2021 ../logs/2-4-dmcontrol/dm.cheetah.run-PytorchSAC-02-04-2021 -t "Cheetah Run" -sp plots/cheetah_run.pdf -y
python plotting.py ../logs/2-03-dm-control/dm.hopper.hop-PytorchSAC-02-03-2021 ../logs/2-4-dmcontrol/dm.hopper.hop-PytorchSAC-02-04-2021 -t "Hopper Hop" -sp plots/hopper_hop.pdf -y
#python plotting.py ../logs/1-25-2021-dm-control/dm.walker.run-PytorchSAC-01-26-2021 ../logs/2-4-dmcontrol/dm.walker.run-PytorchSAC-02-04-2021 -t "Walker Run" -sp plots/walker_run.pdf -y
python plotting.py ../logs/1-25-2021-dm-control/dm.humanoid.run-PytorchSAC-01-26-2021 ../logs/2-4-dmcontrol/dm.humanoid.run-PytorchSAC-02-04-2021 -t "Humanoid Run" -sp plots/humanoid_run.pdf -y
python plotting.py ../logs/1-25-2021-dm-control/dm.humanoid.stand-PytorchSAC-01-26-2021 ../logs/2-4-dmcontrol/dm.humanoid.stand-PytorchSAC-02-04-2021 -t "Humanoid Stand" -sp plots/humanoid_stand.pdf -y
python plotting.py ../logs/1-25-2021-dm-control/dm.humanoid.walk-PytorchSAC-01-26-2021 ../logs/2-4-dmcontrol/dm.humanoid.walk-PytorchSAC-02-04-2021 -t "Humanoid Walk" -sp plots/humanoid_walk.pdf -y
python plotting.py ../logs/2-03-dm-control/dm.fish.swim-PytorchSAC-02-03-2021 ../logs/2-4-dmcontrol/dm.fish.swim-PytorchSAC-02-04-2021 -t "Fish Swim" -sp plots/fish_swim.pdf -y
python plotting.py ../logs/2-03-dm-control/dm.swimmer.swimmer6-PytorchSAC-02-03-2021 -t "Swimmer6" -sp plots/swimmer6.pdf -y
python plotting.py ../logs/2-03-dm-control/dm.swimmer.swimmer15-PytorchSAC-02-03-2021 ../logs/2-4-dmcontrol/dm.swimmer.swimmer15-PytorchSAC-02-04-2021 -t "Swimmer15" -sp plots/swimmer15.pdf -y -l


python plotting.py ../logs/logs/2-23-dm-control/dm.acrobot.swingup-PytorchSAC-02-24-2021 -t "Acrobot Swingup" -sp plots/acrobat_swingup_new.pdf -y


logs/2-23-dm-control

# main figure, updated runs
python plotting.py ../logs/1-25-2021-dm-control/dm.walker.run-PytorchSAC-01-26-2021 ../logs/2-4-dmcontrol/dm.walker.run-PytorchSAC-02-04-2021 ../logs/2-19-dm-control/dm.walker.run-PytorchSAC-02-20-2021 -t "Walker Run" -sp plots/walker_run.pdf -y

# ablations
python plot_ablations_temp.py ../logs/1-25-2021-dm-control/dm.finger.turn_hard-PytorchSAC-01-26-2021 ../logs/2-4-dmcontrol/dm.finger.turn_hard-PytorchSAC-02-04-2021 -t "Finger Turn (hard)" -sp plots/finger_ablation.pdf -y
python plot_ablations_temp.py ../logs/1-25-2021-dm-control/dm.walker.run-PytorchSAC-01-26-2021 ../logs/2-4-dmcontrol/dm.walker.run-PytorchSAC-02-04-2021 -t "Walker Run" -sp plots/walker_ablation.pdf -y -l
python plot_ablations_temp.py ../logs/1-25-2021-dm-control/dm.quadruped.run-PytorchSAC-01-27-2021 ../logs/2-4-dmcontrol/dm.quadruped.run-PytorchSAC-02-04-2021 -t "Quadruped Run" -sp plots/quadruped_ablation.pdf -y
python plot_ablations_temp.py ../logs/1-25-2021-dm-control/dm.humanoid.stand-PytorchSAC-01-26-2021 ../logs/2-4-dmcontrol/dm.humanoid.stand-PytorchSAC-02-04-2021 -t "Humanoid Stand" -sp plots/humanoid_ablation.pdf -y



# rebuttal
python plotting.py ../logs/3-11-dmcontrol/dm.swimmer.swimmer15-PytorchSAC-03-11-2021 ../logs/3-22-dmcontrol/dm.swimmer.swimmer15-PytorchSAC-03-22-2021 -t "Swimmer15" -sp plots/rebuttal/swimmer15.pdf -y -l
python plotting.py ../logs/2-13-dmcontrol/dm.cheetah.run-PytorchSAC-02-14-2021 ../logs/2-19-dm-control/dm.cheetah.run-PytorchSAC-02-20-2021 ../logs/2-23-dm-control/dm.cheetah.run-PytorchSAC-02-24-2021 ../logs/3-22-dmcontrol/dm.cheetah.run-PytorchSAC-03-22-2021 ../logs/3-24-dmcontrol/dm.cheetah.run-PytorchSAC-03-24-2021 ../logs/3-26-dmcontrol/dm.cheetah.run-PytorchSAC-03-26-2021 -t "Cheetah Run" -sp plots/rebuttal/cheetah_run.pdf -y
python plotting.py  ../logs/2-13-dmcontrol/dm.hopper.hop-PytorchSAC-02-14-2021 ../logs/3-22-dmcontrol/dm.hopper.hop-PytorchSAC-03-22-2021 ../logs/3-24-dmcontrol/dm.hopper.hop-PytorchSAC-03-24-2021 ../logs/3-25-dmcontrol/dm.hopper.hop-PytorchSAC-03-25-2021 ../logs/3-26-dmcontrol/dm.hopper.hop-PytorchSAC-03-26-2021  -t "Hopper Hop" -sp plots/rebuttal/hopper_hop.pdf -y
python plotting.py ../logs/3-11-dmcontrol/dm.quadruped.walk-PytorchSAC-03-11-2021 ../logs/3-22-dmcontrol/dm.quadruped.walk-PytorchSAC-03-22-2021 -t "Quadruped Walk" -sp plots/rebuttal/quadruped_walk.pdf -y
python plotting.py ../logs/3-22-dmcontrol/dm.humanoid.run-PytorchSAC-03-22-2021  -t "Humanoid Run (5M steps)" -sp plots/rebuttal/humanoid_run_rebuttal.pdf -y --xlimit 5000000

# more complicated rebuttal figures
# LFF, MLP, log-uniform. two will have long runs
python plotting.py ../logs/3-11-dmcontrol/dm.finger.turn_hard-PytorchSAC-03-11-2021 ../logs/3-22-dmcontrol/dm.finger.turn_hard-PytorchSAC-03-22-2021 ../logs/3-25-dmcontrol/dm.finger.turn_hard-PytorchSAC-03-25-2021 ../logs/3-26-dmcontrol/dm.finger.turn_hard-PytorchSAC-03-26-2021 -t "Finger Turn (hard)" -sp plots/rebuttal/finger_turn_hard_rebuttal.pdf -y
python plotting.py ../logs/2-19-dm-control/dm.walker.run-PytorchSAC-02-20-2021 ../logs/2-23-dm-control/dm.walker.run-PytorchSAC-02-24-2021 ../logs/3-22-dmcontrol/dm.walker.run-PytorchSAC-03-22-2021 ../logs/3-24-dmcontrol/dm.walker.run-PytorchSAC-03-24-2021 ../logs/3-25-dmcontrol/dm.walker.run-PytorchSAC-03-25-2021 ../logs/3-26-dmcontrol/dm.walker.run-PytorchSAC-03-26-2021 -t "Walker Run" -sp plots/rebuttal/walker_run_rebuttal.pdf -y
python plotting.py ../logs/2-19-dm-control/dm.quadruped.run-PytorchSAC-02-20-2021 ../logs/2-23-dm-control/dm.quadruped.run-PytorchSAC-02-24-2021 ../logs/3-22-dmcontrol/dm.quadruped.run-PytorchSAC-03-22-2021 ../logs/3-25-dmcontrol/dm.quadruped.run-PytorchSAC-03-25-2021 ../logs/3-26-dmcontrol/dm.quadruped.run-PytorchSAC-03-26-2021 -t "Quadruped Run (5M steps)" -sp plots/rebuttal/quadruped_run_rebuttal.pdf -y --xlimit 5000000 -l
python plotting.py ../logs/3-11-dmcontrol/dm.humanoid.stand-PytorchSAC-03-11-2021 ../logs/3-22-dmcontrol/dm.humanoid.stand-PytorchSAC-03-22-2021 ../logs/3-25-dmcontrol/dm.humanoid.stand-PytorchSAC-03-25-2021 ../logs/3-26-dmcontrol/dm.humanoid.stand-PytorchSAC-03-26-2021 -t "Humanoid Stand" -sp plots/rebuttal/humanoid_stand_rebuttal.pdf -y

# RAD figure
python plotting.py  ../logs/3-27-rad/dm.hopper.hop-SAC-03-27-2021  -t "Hopper Hop (pixels)" -sp plots/rebuttal/hopper_hop_pixels.pdf -y -l


# later RAD figures
python plotting.py  ../logs/3-26-rad/dm.cheetah.run-SAC-03-26-2021 ../logs/3-27-rad/dm.cheetah.run-SAC-03-27-2021 ../logs/4-2-rad-hc/dm.cheetah.run-SAC-04-02-2021 -t "Cheetah Run (pixels)" -sp plots/rebuttal/half_cheetah_pixels.pdf -y -l