from . import supervised, supervised_relabel, supervised_real, iterative_real, iterative_real_k1k2, iterative_softmax
def get_model(exp_dict):
    # Exp dict is a dictionary with hyperparameters
    if exp_dict["model"] == 'supervised':
        return supervised.Model(exp_dict)
    elif exp_dict["model"] == 'supervised_relabel':
        return supervised_relabel.Model(exp_dict)
    elif exp_dict["model"] == 'supervised_real':
        return supervised_real.Model(exp_dict)
    elif exp_dict["model"] == 'iterative_real':
        return iterative_real.Model(exp_dict)
    elif exp_dict["model"] == 'iterative_real_k1k2':
        return iterative_real_k1k2.Model(exp_dict)
    elif exp_dict["model"] == 'iterative_softmax':
        return iterative_softmax.Model(exp_dict)
    else:
        raise FileNotFoundError(exp_dict["model"])