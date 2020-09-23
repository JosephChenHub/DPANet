import time
import os
import torch
import logging as logger


def load_model(model, model_file, depth_input=False, is_restore=False):
    t_start = time.time()
    if isinstance(model_file, str):
        if not os.path.exists(model_file):
            logger.warning("Model file:%s does not exist!"%model_file)
            return
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    if depth_input:
        mean_w = state_dict['conv1.weight'].mean(dim=1, keepdim=True)
        state_dict['conv1.weight'] = mean_w

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        logger.warning('Model keys:{}, state dict: {}, missing key(s) in state_dict: {}'.format(len(own_keys), len(ckpt_keys),
                ', '.join('{}'.format(k) for k in missing_keys)
            ))

    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}'.format(
                    ', '.join('{}'.format(k) for k in unexpected_keys)
                ))

    del state_dict
    t_end = time.time()
    logger.info("Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(t_ioend - t_start, t_end - t_ioend))

    return model
