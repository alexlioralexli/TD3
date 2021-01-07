import os
import json
import datetime


def create_env_folder(env_name, expID, algo):
    # might have diambiguation problems, but very unlikely to have match to the millisecond
    wrapper_folder = f'{env_name}-{algo}-{datetime.datetime.now().strftime("%m-%d-%Y")}'
    inner_folder = f'{env_name}-{algo}-exp{expID}-{datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S-%f")}'
    folder_path = os.path.join('logs', wrapper_folder, inner_folder)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_kwargs(kwargs_dict, dir):
    with open(os.path.join(dir, "variant.json"), "w") as f:
        json.dump(kwargs_dict, f, indent=2, sort_keys=True, cls=MyEncoder)
