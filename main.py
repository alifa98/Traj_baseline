
import argparse
import os
import random
import numpy as np
import logging
import torch

from config.config import get_config
from data.utils import get_dataset
from models.deepmove_model import DeepMove
from utils.executor import TrajLocPredExecutor

logger = logging.getLogger(__name__)

general_arguments = {
    "gpu": {
        "type": "bool",
        "default": None,
        "help": "whether use gpu"
    },
    "gpu_id": {
        "type": "int",
        "default": None,
        "help": "the gpu id to use"
    },
    "train_rate": {
        "type": "float",
        "default": None,
        "help": "the train set rate"
    },
    "eval_rate": {
        "type": "float",
        "default": None,
        "help": "the validation set rate"
    },
    "batch_size": {
        "type": "int",
        "default": None,
        "help": "the batch size"
    },
    "learning_rate": {
        "type": "float",
        "default": None,
        "help": "learning rate"
    },
    "max_epoch": {
        "type": "int",
        "default": None,
        "help": "the maximum epoch"
    },
    "dataset_class": {
        "type": "str",
        "default": None,
        "help": "the dataset class name"
    },
    "executor": {
        "type": "str",
        "default": None,
        "help": "the executor class name"
    },
    "evaluator": {
        "type": "str",
        "default": None,
        "help": "the evaluator class name"
    },
}


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool value expected.')


def str2float(s):
    if isinstance(s, float):
        return s
    try:
        x = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError('float value expected.')
    return x


def add_general_args(parser):
    for arg in general_arguments:
        if general_arguments[arg]['type'] == 'int':
            parser.add_argument('--{}'.format(arg), type=int,
                                default=general_arguments[arg]['default'], help=general_arguments[arg]['help'])
        elif general_arguments[arg]['type'] == 'bool':
            parser.add_argument('--{}'.format(arg), type=str2bool,
                                default=general_arguments[arg]['default'], help=general_arguments[arg]['help'])
        elif general_arguments[arg]['type'] == 'float':
            parser.add_argument('--{}'.format(arg), type=str2float,
                                default=general_arguments[arg]['default'], help=general_arguments[arg]['help'])
        elif general_arguments[arg]['type'] == 'str':
            parser.add_argument('--{}'.format(arg), type=str,
                                default=general_arguments[arg]['default'], help=general_arguments[arg]['help'])
        elif general_arguments[arg]['type'] == 'list of int':
            parser.add_argument('--{}'.format(arg), nargs='+', type=int,
                                default=general_arguments[arg]['default'], help=general_arguments[arg]['help'])


def set_random_seed(seed):
    """
    重置随机数种子

    Args:
        seed(int): 种子数
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def run_model():
    config = get_config()
    config['task'] = "traj_loc_pred"
    config['model'] = "DeepMove"
    config['dataset'] = "foursquare_tky"
    config['saved_model'] = True
    config['train'] = True
    config['batch_size'] = 15

    # Experiment id
    config['exp_id'] = int(random.SystemRandom().random() * 100000)

    # logger
    logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(config['task']), str(config['model']), str(config['dataset']), str(config['exp_id'])))
    logger.info(config)

    # seed
    seed = config.get('seed', 0)
    set_random_seed(seed)

    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    model_save_path = './cache/{}/model_cache/{}_{}.m'.format(config['exp_id'],
                                                              str(config['model']), str(config['dataset']))
    model = DeepMove(config, data_feature)
    executor = TrajLocPredExecutor(config, model, data_feature)

    if config['train'] or not os.path.exists(model_save_path):
        executor.train(train_data, valid_data)
        if config['saved_model']:
            executor.save_model(model_save_path)
    else:
        executor.load_model(model_save_path)

    executor.evaluate(test_data)


if __name__ == '__main__':
    run_model()
