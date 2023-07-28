from torch import device


def get_config():
    return {
        "task": "traj_loc_pred",
        "seed": 0,
        "train": True,
        "dataset_class": "TrajectoryDataset",
        "traj_encoder": "StandardTrajectoryEncoder",
        "executor": "TrajLocPredExecutor",
        "evaluator": "TrajLocPredEvaluator",
        "loc_emb_size": 200,
        "uid_emb_size": 40,
        "tim_emb_size": 20,
        "hidden_size": 128,
        "attn_type": "dot",
        "rnn_type": "GRU",
        "dropout_p": 0.5,
        "learning_rate": 0.001,
        "lr_step": 3,
        "max_epoch": 30,
        "optimizer": "adam",
        "window_size": 15,
        "min_session_len": 15,
        "max_session_len": 15, # used in cutting trajectories (including the target locations)
        "min_sessions": 1,
        "min_checkins": 10,
        "num_workers": 0,
        "cache_dataset": True,
        "train_rate": 0.7,
        "eval_rate": 0.1,
        "history_type": "splice",
        # "cut_method": "time_interval",
        "cut_method": "goto_else_block",
        "gpu": True,
        "gpu_id": 0,
        "L2": 1e-05,
        "lr_decay": 0.1,
        "clip": 5.0,
        "schedule_threshold": 0.001,
        "loss_func": "default",
        "load_best_epoch": True,
        "hyper_tune": False,
        "early_stop_lr": 9e-06,
        "debug": False,

        # evaluation config
        "metrics": [
            "Recall",
            "F1",
            "MRR",
            "MAP",
            "NDCG",
            # "ACC",
            # "BLEU"
        ],
        "evaluate_method": "popularity",
        "topk": [1, 3, 5],
        "predict_next_n": 5,
        "evaluate_steps_index": [4],

        # dataset config
        "geo": {
            "including_types": [
                "Point"
            ],
            "Point": {
                "venue_category_id": "enum",
                "venue_category_name": "enum"
            }
        },
        "usr": {
            "properties": {}
        },
        "dyna": {
            "including_types": [
                "trajectory"
            ],
            "trajectory": {
                "entity_id": "usr_id",
                "location": "geo_id"
            }
        },
        "distance_upper": 30.0,
        "device": device("cuda:0"),
        # "device": device("cpu"),
    }
