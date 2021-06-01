import torch

import utility
import data
import model
import loss
from option import args
from trainer_test import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)

    args.model = args.s_model
    model_S = model.Model(args, checkpoint)
    model_S.model.load_state_dict(torch.load(args.s_path))


    args.model = args.r_model
    model_R = model.Model(args, checkpoint)
    model_R.model.load_state_dict(torch.load(args.r_path))

    args.n_colors = 3
    args.model = args.i_model
    model_I = model.Model(args, checkpoint)
    model_I.model.load_state_dict(torch.load(args.i_path))

    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model_S, model_I, model_R, loss, checkpoint)
    while not t.terminate():
        t.train()

        import datetime
        starttime = datetime.datetime.now()

        t.test()
        endtime = datetime.datetime.now()
        print (endtime - starttime).seconds


    checkpoint.done()

