import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)

    model_S = model.Model(args, checkpoint)

    args.model = 'lrdn'
    model_R = model.Model(args, checkpoint)

    args.model = 'lrdn'
    model_I = model.Model(args, checkpoint)

    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model_S, model_I, model_R, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

