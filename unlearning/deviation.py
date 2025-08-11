import numpy as np
from model import set_model_parameters, get_model_parameters, train_one_epoch

def calibrate_ai(ai, Ek, data_loader_map, model_fn):
    ai_list = []
    for lj in Ek:
        model = model_fn()
        set_model_parameters(model, ai)
        train_one_epoch(model, data_loader_map[lj])
        ai_hat = get_model_parameters(model)
        norm = np.linalg.norm(ai_hat)
        ai_hat = ai_hat / norm if norm > 0 else ai_hat
        ai_proj = ai * ai_hat
        ai_list.append(ai_proj)
    return np.mean(ai_list, axis=0)
