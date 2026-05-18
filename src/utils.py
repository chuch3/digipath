import os

import torch


def load_checkpoint(
    model,
    optimizer=None,
    loss_history=None,
    acc_history=None,
    file_name=None,
):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if not os.path.isfile(file_name):
        print("=> No checkpoint found at '{}'".format(file_name))
    else:
        print("=> loading checkpoint '{}'".format(file_name))
        checkpoint = torch.load(file_name)
        start_epoch = checkpoint["epoch"]

        model.load_state_dict(checkpoint["state_dict"], strict=False)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if loss_history is not None:
            loss_history = checkpoint["loss_history"]

        if acc_history is not None:
            acc_history = checkpoint["acc_history"]

        print(
            "=> Loaded checkpoint '{}' (epoch {})".format(
                file_name, checkpoint["epoch"]
            )
        )

    return model, optimizer, start_epoch, loss_history, acc_history
