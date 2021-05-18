"""
Copy this file to config.py and modify as needed.
"""
import os
from os.path import join
import TD3

"""
`doodad.mount.MountLocal` by default ignores directories called "data"
If you're going to rename this directory and use EC2, then change
`doodad.mount.MountLocal.filter_dir`
"""
# The directory of the project, not source
# rlkit_project_dir = join(os.path.dirname(TD3.__file__), os.pardir)
rlkit_project_dir = os.path.dirname(TD3.__file__)
LOCAL_LOG_DIR = join(rlkit_project_dir, 'data')

"""
********************************************************************************
********************************************************************************
********************************************************************************

You probably don't need to set all of the configurations below this line,
unless you use AWS, GCP, Slurm, and/or Slurm on a remote server. I recommend
ignoring most of these things and only using them on an as-needed basis.

********************************************************************************
********************************************************************************
********************************************************************************
"""

"""
General doodad settings.
"""
CODE_DIRS_TO_MOUNT = [
    rlkit_project_dir,
    # '/home/user/python/module/one', Add more paths as needed
]

HOME = os.getenv('HOME') if os.getenv('HOME') is not None else os.getenv("USERPROFILE")

DIR_AND_MOUNT_POINT_MAPPINGS = [
    # dict(
    #     local_dir=join(HOME, 'mujoco_linux/'),
    #     mount_point='/root/.mujoco',
    # )
]
RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    join(rlkit_project_dir, 'scripts', 'run_experiment_from_doodad.py')
    # '/home/user/path/to/rlkit/scripts/run_experiment_from_doodad.py'
)
"""
AWS Settings
"""
# If not set, default will be chosen by doodad
# AWS_S3_PATH = 's3://bucket/directory  #todo: do this

# The docker image is looked up on dockerhub.com.
DOODAD_DOCKER_IMAGE = "'alexli1/td3:v3'" #"'jcoreyes/continualrl'"
INSTANCE_TYPE = 'c4.2xlarge'
SPOT_PRICE = 0.5 # 0.2 # 0.03

GPU_DOODAD_DOCKER_IMAGE = "'alexli1/td3:v3'" # "'jcoreyes/continualrl'"
GPU_INSTANCE_TYPE = 'g2.2xlarge'
GPU_SPOT_PRICE = 0.5

# You can use AMI images with the docker images already installed.
REGION_TO_GPU_AWS_IMAGE_ID = {
    # 'us-west-1': "ami-05e00979871a39188",
    # 'us-east-1': "ami-0b69ea66ff7391e80",
    'us-east-1': 'ami-084e787069ee27fb7',
    'us-west-2': "ami-076347b8649dddb00",
    'us-east-2': "ami-024def4b62933fe1c",
    # 'us-east-2': "ami-024f0e84c5a8282a8",
    'us-west-1': "ami-07ef6e8df35a09c16"
}

REGION_TO_GPU_AWS_AVAIL_ZONE = {
    'us-east-1': "us-east-1b",
    'us-west-2': "us-west-2d"
}

# This really shouldn't matter and in theory could be whatever
OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/doodad-output/'


"""
Slurm Settings
"""
SINGULARITY_IMAGE = '/home/PATH/TO/IMAGE.img'
# This assumes you saved mujoco to $HOME/.mujoco
SINGULARITY_PRE_CMDS = [
    'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin'
]
SLURM_CPU_CONFIG = dict(
    account_name='TODO',
    partition='savio',
    nodes=1,
    n_tasks=1,
    n_gpus=1,
)
SLURM_GPU_CONFIG = dict(
    account_name='TODO',
    partition='savio2_1080ti',
    nodes=1,
    n_tasks=1,
    n_gpus=1,
)


"""
Slurm Script Settings

These are basically the same settings as above, but for the remote machine
where you will be running the generated script.
"""
SSS_CODE_DIRS_TO_MOUNT = [
]
SSS_DIR_AND_MOUNT_POINT_MAPPINGS = [
    dict(
        local_dir='/global/home/users/USERNAME/.mujoco',
        mount_point='/root/.mujoco',
    ),
]
SSS_LOG_DIR = '/global/scratch/USERNAME/doodad-log'

SSS_IMAGE = '/global/scratch/USERNAME/TODO.img'
SSS_RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    '/global/home/users/USERNAME/path/to/rlkit/scripts'
    '/run_experiment_from_doodad.py'
)
SSS_PRE_CMDS = [
    'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/home/users/USERNAME'
    '/.mujoco/mjpro150/bin'
]

"""
GCP Settings
"""
GCP_IMAGE_NAME = 'TODO'
GCP_GPU_IMAGE_NAME = 'TODO'
GCP_BUCKET_NAME = 'TODO'

GCP_DEFAULT_KWARGS = dict(
    zone='us-west2-c',
    instance_type='n1-standard-4',
    image_project='TODO',
    terminate=True,
    preemptible=True,
    gpu_kwargs=dict(
        gpu_model='nvidia-tesla-p4',
        num_gpu=1,
    )
)

# try:
#     from rlkit.launchers.conf_private import *
# except ImportError:
#     print("No personal conf_private.py found.")
