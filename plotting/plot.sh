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
#python plotting.py ../logs/1-24-2021-dm-control/dm.finger.turn_hard-PytorchSAC-01-25-2021 -t "Finger Turn (hard)" -sp plots/finger_turn_hard_no_target.pdf -y
#python plotting.py ../logs/1-24-2021-dm-control/dm.cheetah.run-PytorchSAC-01-25-2021 -t "Cheetah Run" -sp plots/cheetah_run_no_target.pdf -y
#python plotting.py ../logs/1-24-2021-dm-control/dm.humanoid.run-PytorchSAC-01-25-2021 -t "Humanoid Run" -sp plots/humanoid_run_no_target.pdf -y -l


# main figure
python plotting.py ../logs/1-25-2021-dm-control/dm.acrobot.swingup-PytorchSAC-01-26-2021 -t "Acrobot Swingup" -sp plots/acrobat_swingup.pdf -y -l
python plotting.py ../logs/1-25-2021-dm-control/dm.finger.turn_hard-PytorchSAC-01-26-2021 -t "Finger Turn (hard)" -sp plots/finger_turn_hard.pdf -y
python plotting.py ../logs/1-25-2021-dm-control/dm.quadruped.run-PytorchSAC-01-27-2021 -t "Quadruped Run" -sp plots/quadruped_run.pdf -y
python plotting.py ../logs/1-25-2021-dm-control/dm.quadruped.walk-PytorchSAC-01-27-2021 -t "Quadruped Walk" -sp plots/quadruped_walk.pdf -y
python plotting.py ../logs/1-25-2021-dm-control/dm.cheetah.run-PytorchSAC-01-26-2021 -t "Cheetah Run" -sp plots/cheetah_run.pdf -y
python plotting.py ../logs/1-25-2021-dm-control/dm.walker.run-PytorchSAC-01-26-2021 -t "Walker Run" -sp plots/walker_run.pdf -y
python plotting.py ../logs/1-25-2021-dm-control/dm.humanoid.run-PytorchSAC-01-26-2021 -t "Humanoid Run" -sp plots/humanoid_run.pdf -y
python plotting.py ../logs/1-25-2021-dm-control/dm.humanoid.stand-PytorchSAC-01-26-2021 -t "Humanoid Stand" -sp plots/humanoid_stand.pdf -y
python plotting.py ../logs/1-25-2021-dm-control/dm.humanoid.walk-PytorchSAC-01-26-2021 -t "Humanoid Walk" -sp plots/humanoid_walk.pdf -y
python plotting.py ../logs/2-03-dm-control/dm.fish.swim-PytorchSAC-02-03-2021 -t "Fish Swim" -sp plots/fish_swim.pdf -y
python plotting.py ../logs/2-03-dm-control/dm.hopper.hop-PytorchSAC-02-03-2021 -t "Hopper Hop" -sp plots/hopper_hop.pdf -y
python plotting.py ../logs/2-03-dm-control/dm.swimmer.swimmer6-PytorchSAC-02-03-2021 -t "Swimmer6" -sp plots/swimmer6.pdf -y
python plotting.py ../logs/2-03-dm-control/dm.swimmer.swimmer15-PytorchSAC-02-03-2021 -t "Swimmer15" -sp plots/swimmer15.pdf -y