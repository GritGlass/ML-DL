from torch import optim


def adamw(train_params, cfg):
    optimizer = optim.AdamW(
        params=train_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
        eps=cfg.eps,
        amsgrad=cfg.amsgard,
    )
    return optimizer


def sgd(train_params, cfg):
    optimizer = optim.SGD(
        params=train_params,
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov,
    )
    return optimizer


def adam(train_params, cfg):
    optimizer = optim.Adam(
        params=train_params,
        lr=cfg.learning_rate,
        betas=cfg.betas,
        eps=cfg.eps,
        amsgrad=cfg.amsgard,
    )
    return optimizer
