
import os
import random
import sys
import numpy as np
import logging
import torch

from config.config import get_config
from data.utils import get_dataset
from models.deepmove_model import DeepMove
from utils.executor import TrajLocPredExecutor


os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# logging config to console
root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

logger = logging.getLogger(__name__)

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
    config['batch_size'] = 50
    config['max_epoch'] = 1

    # # to evaluate a pre-trained model, set exp_id to the experiment id of the pre-trained model
    # config['exp_id'] = 90097
    # config['train'] = False 

    # new experiment id
    config['exp_id'] = config['exp_id'] if "exp_id" in config else int(random.SystemRandom().random() * 100000)

    # seed
    seed = config.get('seed', 0)
    set_random_seed(seed)

    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    model_save_path = './cache/{}/model_cache/{}_{}.m'.format(config['exp_id'],
                                                              str(config['model']), str(config['dataset']))
    
    if config['model'] == 'DeepMove':
        model = DeepMove(config, data_feature)
    else:
        raise ValueError("Invalid model name: {}".format(config['model']))
    
    executor = TrajLocPredExecutor(config, model, data_feature)

    logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(config['task']), str(config['model']), str(config['dataset']), str(config['exp_id'])))
    logger.info(config)

    if config['train'] or not os.path.exists(model_save_path):
        executor.train(train_data, valid_data)
        if config['saved_model']:
            executor.save_model(model_save_path)
    else:
        executor.load_model(model_save_path)

    executor.evaluate(test_data)


if __name__ == '__main__':
    run_model()
