# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
This is the main script to run on each MPI thread. It will spawn either a
Server or Worker object -- the former is responsible for orchestrating and
aggregating models, where as the latter processes clients' data to generate
a new model. The Server lives on the very first thread, whereas remaining
threads contain each a diferent Worker.
'''

import argparse
import os
import shutil
import yaml
from psutil import virtual_memory

import torch
from azureml.core import Run

from core import federated
from core.server import select_server
from core.client import Client
from core.globals import TRAINING_FRAMEWORK_TYPE, logging_level, define_file_type
from experiments import make_model
from utils import (
    make_optimizer,
    init_logging,
    print_rank,
    find_pretrained_model
)
from utils.dataloaders_utils import (
    make_train_dataloader,
    make_val_dataloader,
    make_test_dataloader,
)
from config_file_parser import (
    check_server_config,
    check_client_config
)

assert TRAINING_FRAMEWORK_TYPE == "mpi", "Unsupported platform {}".format(TRAINING_FRAMEWORK_TYPE)


def log_run_properties(config):
    """Log parameters on AzureML.
    
    Args:
        config (dict): config containing parameters to log.
    """

    properties = {}

    def lookup(key, cfg, default):
        """Look for key on dict"""
        keys = key.split(".")
        if len(keys) == 1:
            return cfg.get(key, default)
        if keys[0] in cfg:
            return lookup(".".join(keys[1:]), cfg[keys[0]], default)
        else:
            return default

    # Build properties dictionary
    mem = virtual_memory()
    properties["System memory (GB)"] = float(mem.total) / (1024**3)

    props = [
        ("server_config.num_clients_per_iteration", 0),
        ("server_config.max_iteration", 0),
        ("dp_config.eps", 0),
        ("dp_config.max_weight", 0),
        ("dp_config.min_weight", 0),
        ("server_config.optimizer_config.type", "sgd"),
        ("server_config.optimizer_config.lr", 1.0),
        ("server_config.optimizer_config.amsgrad", False),
        ("server_config.annealing_config.type", "step_lr"),
        ("server_config.annealing_config.step_interval", "epoch"),
        ("server_config.annealing_config.gamma", 1.0),
        ("server_config.annealing_config.step_size", 100),
    ]

    for (key, default) in props:
        properties[key] = lookup(key, config, default)

    # Log the properties dictionary into AzureML
    run = Run.get_context()
    for k in properties:
        run.log(k, properties[k])


def run_worker(model_path, config, task, data_path, local_rank):
    """Spawn worker object that lives throughout MPI thread.
    
    Args:
        model_path (str): path to the pretrained model.
        config (dict): dictionary containing parameters.
        task (str): what task to solve, must be a folder of :code:`experiments`.
        data_path (str): path to data.
        local_rank (int): the rank of the MPI thread.
    """
    model_config = config["model_config"]
    server_config = config["server_config"]
    define_file_type(data_path, config)

    # Get the rank on MPI
    rank = local_rank if local_rank > -1 else federated.rank()

    # Assign MPI thread to a specific GPU
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        torch.cuda.set_device(federated.local_rank() % n_gpus)
        print_rank(f"Assigning worker to GPU {federated.local_rank() % n_gpus}")

    # Make the Model to distribute to workers
    model = make_model(model_config)

    # Instantiate the Server object on the first thread
    if rank == 0:
        try:
            print_rank('Server data preparation')

            # pre-cache the training data and capture the number of clients for sampling
            training_filename = os.path.join(data_path, config["client_config"]["data_config"]["train"]["list_of_train_data"])
            config["server_config"]["data_config"]["num_clients"] = Client.get_num_users(training_filename)
            data_config = config['server_config']['data_config']

            # Make the Dataloaders
            if 'train' in data_config:
                server_train_dataloader = make_train_dataloader(data_config['train'], data_path, task=task, clientx=None)
            else:
                server_train_dataloader = None
            val_dataloader = make_val_dataloader(data_config["val"], data_path, task=task)
            test_dataloader = make_test_dataloader(data_config["test"], data_path, task=task)

            print_rank("Prepared the dataloaders")

            # Create the optimizer on the server
            optimizer = make_optimizer(server_config["optimizer_config"], model)

            # Load a model that's already trained
            best_trained_model = find_pretrained_model(model_path, model_config)
            if best_trained_model is not None:
                model_state_dict = torch.load(best_trained_model,
                    map_location=None if torch.cuda.is_available() else torch.device("cpu"))
                model.load_state_dict(model_state_dict)

            server_type = server_config["type"]
            server_setup = select_server(server_type, config)  # Return the server class
            server = server_setup(
                data_config["num_clients"],
                model,
                optimizer,
                None,
                data_path,
                model_path,
                server_train_dataloader,
                val_dataloader,
                test_dataloader,
                config,
                server_config
            )
            log_run_properties(config)

        except Exception as e:
            # Be sure the other workers are shut down.
            server.terminate_workers()
            raise e

        print_rank("Launching server")
        server.run()

    else:
        # Instantiate client-processing Worker on remaining threads
        print_rank("Worker on node {}: process started".format(rank))
        client_config = config["client_config"]
        worker = federated.Worker(
            model,
            data_path,
            do_profiling=client_config.get("do_profiling", False),
            clients_in_parallel=client_config.get("clients_in_parallel", None),
        )
        worker.run()


def _reconcile_args(args, config):
    '''Change parameters depending on command-line arguments'''

    if args.dp_config_grad_dir_eps:
        config["dp_config"]["grad_dir_eps"] = args.dp_config_grad_dir_eps
    if args.dp_config_grad_mag_eps:
        config["dp_config"]["grad_mag_eps"] = args.dp_config_grad_mag_eps
    if args.dp_config_weight_eps:
        config["dp_config"]["weight_eps"] = args.dp_config_weight_eps

    return config


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-config")
    parser.add_argument("-outputPath")
    parser.add_argument("-dataPath", default=None)
    parser.add_argument("-task", default=None, help="Define the task for the run")
    parser.add_argument("-num_skip_decoding", default=-1, type=int, help="Skip decoding in unsupervised learning mode")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dp_config_grad_dir_eps", default=None, type=float, help="DP direction epsilon")
    parser.add_argument("--dp_config_grad_mag_eps", default=None, type=float, help="DP magnitude epsilon")
    parser.add_argument("--dp_config_weight_eps", default=None, type=float, help="DP weight epsilon")

    args = parser.parse_args()
    data_path = args.dataPath
    task = args.task
    local_rank = args.local_rank

    # Create dictionaries w/ parameters
    default_data_conf = {
        "input_dim": 300,
        "batch_size": 40,
        "loader_type": "text",
        "prepend_datapath": False,
        "pin_memory": True,
        "num_frames": 0,
        "desired_max_samples": 300,
        "max_grad_norm": 5.0,        # max_grad_norm for gradient clipping
        "num_workers": 1,
        "max_batch_size": 0,         # maximum number of batch size; if 0, no limitation is applied
        "unsorted_batch": False      # do not sort when making batch; this is inefficient in terms of batch, but could be efficient in terms of accuracy
    }

    default_server_conf = {
        "val_freq": 1,
        "rec_freq": 8,
        "max_iteration": 100000000,
        "type": "optimization",
        "data_config": default_data_conf,
        "aggregate_median": None,
        "best_model_criterion": "loss",
        "fall_back_to_best_model": False,
        "num_clients_per_iteration": -1
    }

    default_client_conf = {
        "copying_train_jsonls": True,
        "type": "gradient_computation",
        "data_config": default_data_conf,
    }

   # The mount point can also be retrieved from input_datasets of the run context
    if data_path is None:
        data_path = Run.get_context().input_datasets["input"]
    print("The data can be found here: ", data_path)

    # Update the model path for the sake of AzureML
    id = Run.get_context().id
    experiment_name = "-".join(id.split("-")[-4:-2])
    experiment_root = os.path.join(args.outputPath, experiment_name)
    model_path = os.path.join(experiment_root, "models")
    log_path = os.path.join(experiment_root, "log")

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # Make a copy of the config file into the output folder, for future reference
    cfg_out = os.path.join(experiment_root, "FLUTE_config.yaml")
    if local_rank <= 0:
        shutil.copyfile(args.config, cfg_out)
    print("Copy created")

    # Initialize logging
    init_logging(log_path, loglevel=logging_level)

    with open(args.config) as f:
        config = yaml.safe_load(f)
        config = _reconcile_args(args, config)  # replace params. depending on CL args.

        assert "num_clients" not in config["server_config"]["data_config"], "Remove \"num_clients\" from server data_config since this is a reserved key"
        assert "num_clients" not in config["client_config"]["data_config"], "Remove \"num_clients\" from client data_config since this is a reserved key"

        # Make sure the pretrained model is found in the correct place
        if "pretrained_model_path" in config["model_config"]["model_type"]:
            config["model_config"]["model_type"]["pretrained_model_path"] = os.path.join(data_path, config["model_config"]["model_type"]["pretrained_model_path"])
        if "pretrained_model_path" in config["model_config"]:
            config["model_config"]["pretrained_model_path"] = os.path.join(data_path, config["model_config"]["pretrained_model_path"])

        config["data_path"] = data_path

        config = check_server_config(config, default_server_conf)
        config = check_client_config(config, default_client_conf)

        # Add task specification to client configuration
        config["client_config"]["task"] = task
        config["server_config"]["task"] = task

        # RL-related options
        if config["server_config"].get("wantRL", False):
            if config["server_config"]["RL"].get("RL_path_global", True):
                config["server_config"]["RL"]["RL_path"] = os.path.join(args.outputPath,
                                                                        config["server_config"]["RL"]["RL_path"])
            else:
                config["server_config"]["RL"]["RL_path"] = os.path.join(args.outputPath, experiment_name,
                                                                        config["server_config"]["RL"]["RL_path"])

        # Instantiate either Server or Worker on the thread
        run_worker(model_path, config, task, data_path, local_rank)