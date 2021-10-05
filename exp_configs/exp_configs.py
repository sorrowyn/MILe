from haven import haven_utils as hu

EXP_GROUPS = {

    
    "original_imagenet_real":  {
                # Hardware
                "ngpu": 1,
                "amp": 1,
                # model
                "pretrained": False,
                "ssl": "moco",
                "reweight": 'uniform',
                "binary_threshold": [0.25],
                "arch":"resnet50",
                "model":"supervised_real",
                "sigmoid_way": [True],
                "adjust_bias": [True],
                "rescale_sigmoid": [True],
                "label_smoothing": 0.0,
                "dropout_rate": [0.0],
                "teacher_max_pooling": "gap",
                "train_frac": [1.0],
                # dataset
                "dataset": "webvision",
                "num_classes": 1000,
                "relabel_root": None,
                # Optimization
                "batch_size": 196,
                "rescale_lr": True, # will rescale the lr wrt batch size
                "target_loss": "val_loss",
                "lr": [0.1],
                #"lr2": [0.0001],
                "decay":1e-4,
                "momentum": 0.9,
                "max_epoch": 110,
                "optimizer": ["sgd"],
                "scheduler": [{"name": "step",
                                "steps": [30, 60, 80]}]
    },

    
    "iterative_original_imagenet_real_k1k2":  {
                # Hardware
                "ngpu": 1,
                "amp": 1,
                # model
                "pretrained": False,
                "ssl": "moco",
                "use_gt_baseline": [False],
                "subset_path": [ "imagenet_subsets/10percent.txt"],
                "arch":["resnet50"],
                "model":"iterative_real_k1k2",
                "sigmoid_way": [True],
                "adjust_bias": [True],
                "rescale_sigmoid": [True],
                "label_smoothing": [0.0],
                "binary_threshold": [0.5],
                "warmup": [8],
                "iterative_groundtruth": [False],
                "iterative_copy_weights": [False],
                "jump": 2,
                "entropy_weight": [0.0],
                "temperature": [0.0],
                "sample_teacher":[False],
                "dropout_rate": [0.0],
                "k1": [2000],
                "k2": [2000],
                "teacher_max_pooling": "gap",
                "num_iters": 200000,
                "logging_iters": 100,
                "validation_iters": [6000],
                "evaluate_iter_batch": False,
                # dataset
                "dataset": ["real_imagenet_full"],
                "num_classes": [1000],
                "train_frac": [0.1],
                "relabel_root": None,
                # Optimization
                "batch_size": [164],
                "rescale_lr": True, # will rescale the lr wrt batch size
                "target_loss": "val_loss",
                "lr": [0.1],
                #"lr2": [0.00001],
                "decay":1e-4,
                "momentum": 0.9,
                "optimizer": ["sgd"]
    },
    }
EXP_GROUPS = {k:hu.cartesian_exp_group(v) for k,v in EXP_GROUPS.items()}