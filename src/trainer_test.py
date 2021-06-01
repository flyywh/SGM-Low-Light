import os
import math
from decimal import Decimal

import utility
import IPython
import torch.nn.functional as F

import torch
from torch.autograd import Variable
from tqdm import tqdm
import pytorch_ssim
import numpy as np

def loss_package(hr_x1, hr_x2, hr_x3, hr_x4, epoch, ep):

    v_vec = torch.Tensor([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]])
    h_vec = torch.Tensor([[ 1, 1,  1],
                          [ 0, 0,  0],
                          [-1,-1, -1]])

    v_vec  = v_vec.view((1,1,3,3)).cuda()
    h_vec  = h_vec.view((1,1,3,3)).cuda()

    hr_x1_G_v  = F.conv2d(hr_x1, v_vec)
    hr_x1_G_h  = F.conv2d(hr_x1, h_vec)

    hr_x2_G_v  = F.conv2d(hr_x2, v_vec)
    hr_x2_G_h  = F.conv2d(hr_x2, h_vec)

    hr_x3_G_v  = F.conv2d(hr_x3, v_vec)
    hr_x3_G_h  = F.conv2d(hr_x3, h_vec)

    hr_x4_G_v  = F.conv2d(hr_x4, v_vec)
    hr_x4_G_h  = F.conv2d(hr_x4, h_vec)
    
    ep1 = ep[0]
    ep2 = ep[1]
    ep3 = ep[2]
    ep4 = ep[3]

    hr_x1_G_v_weight = 1
    hr_x1_G_h_weight = 1

    hr_x2_G_v_weight = 1/(torch.abs(hr_x1_G_v)+ep1)
    hr_x2_G_h_weight = 1/(torch.abs(hr_x1_G_h)+ep1)

    hr_x2_G_v_weight = hr_x2_G_v_weight.detach()
    hr_x2_G_h_weight = hr_x2_G_h_weight.detach()

    hr_x3_G_v_weight = 1/(torch.abs(hr_x2_G_v)+ep2)
    hr_x3_G_h_weight = 1/(torch.abs(hr_x2_G_h)+ep2)

    hr_x3_G_v_weight = hr_x3_G_v_weight.detach()
    hr_x3_G_h_weight = hr_x3_G_h_weight.detach()

    hr_x4_G_v_weight = 1/(torch.abs(hr_x3_G_v)+ep3)
    hr_x4_G_h_weight = 1/(torch.abs(hr_x3_G_h)+ep3)

    hr_x4_G_v_weight = hr_x4_G_v_weight.detach()
    hr_x4_G_h_weight = hr_x4_G_h_weight.detach()

    ep = 0.001

    x1_sparse_loss = (1-0.9**(epoch-1))*torch.sum(torch.abs(hr_x1_G_v)+ep+torch.abs(hr_x1_G_h))
    x2_sparse_loss = (1-0.9**(epoch-1))*torch.sum(torch.abs(hr_x2_G_v*hr_x2_G_v_weight+ep)+torch.abs(hr_x2_G_h*hr_x2_G_h_weight+ep))
    x3_sparse_loss = (1-0.9**(epoch-1))*torch.sum(torch.abs(hr_x3_G_v*hr_x3_G_v_weight+ep)+torch.abs(hr_x3_G_h*hr_x3_G_h_weight+ep))
    x4_sparse_loss = (1-0.9**(epoch-1))*torch.sum(torch.abs(hr_x4_G_v*hr_x4_G_v_weight+ep)+torch.abs(hr_x4_G_h*hr_x4_G_h_weight+ep))

    return x1_sparse_loss + x2_sparse_loss + x3_sparse_loss + x4_sparse_loss


class Trainer():
    def __init__(self, args, loader, my_model, I_model, R_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.model = my_model
        self.model_I = I_model
        self.model_R = R_model

        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.I_optimizer = utility.make_optimizer(args, self.model_I)
        self.R_optimizer = utility.make_optimizer(args, self.model_R)

        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.I_scheduler = utility.make_scheduler(args, self.I_optimizer)
        self.R_scheduler = utility.make_scheduler(args, self.R_optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.I_scheduler.step()
        self.R_scheduler.step()

        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()

        self.model.train()
        self.model_I.train()
        self.model_R.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        all_sparse_loss = 0
        all_rect_loss = 0

        criterion_ssim = pytorch_ssim.SSIM(window_size = 11)

        for batch, (lr, hr,  _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            self.I_optimizer.zero_grad()
            self.R_optimizer.zero_grad()

            lr = lr/255.0
            hr = hr/255.0


            hr_max = torch.max(hr, dim=1)[0].unsqueeze(1)
            lr_max = torch.max(lr, dim=1)[0].unsqueeze(1)

            bs = hr_max.shape

            hr_max = hr_max.detach()
            lr_max = lr_max.detach()

            hr_x1, hr_x2, hr_x3, hr_x4 = self.model(hr_max.detach(), idx_scale)
            lr_x1, lr_x2, lr_x3, lr_x4 = self.model(lr_max.detach(), idx_scale)

            hr_l = hr_x4
            lr_l = lr_x4

            lr_l[lr_l<0.01] = 0.01
            lr_l[lr_l>1] = 1
            hr_l[hr_l<0.01] = 0.01
            hr_l[hr_l>1] = 1

            hr_r = hr.clone()
            lr_r = lr.clone()

            lr_r[:,0:1,:,:] = torch.div(lr[:,0:1,:,:], lr_l.detach())
            lr_r[:,1:2,:,:] = torch.div(lr[:,1:2,:,:], lr_l.detach())
            lr_r[:,2:3,:,:] = torch.div(lr[:,2:3,:,:], lr_l.detach())

            hr_r[:,0:1,:,:] = torch.div(hr[:,0:1,:,:], hr_l.detach())
            hr_r[:,1:2,:,:] = torch.div(hr[:,1:2,:,:], hr_l.detach())
            hr_r[:,2:3,:,:] = torch.div(hr[:,2:3,:,:], hr_l.detach())

            hr_l_pre= self.model_I(lr_l, idx_scale)+lr_l
            hr_r_pre = self.model_R(lr, idx_scale)+lr_r
            hr_r_pre[hr_r_pre<0] = 0

            loss_I = -criterion_ssim(hr_l_pre, hr_l.detach())

            bg_x = hr[:, :, :,:-1] - hr[:, :, :, 1:]
            bg_y = hr[:, :, :-1,:] - hr[:, :, 1:, :]
            hr_pred = hr_l_pre*lr_r
                    
            gradient_loss = (hr_pred[:,:,:,:-1] - hr_pred[:,:,:,1:] - (bg_x)).norm() +(hr_pred[:,:,:-1,:] - hr_pred[:,:,1:,:] - (bg_y)).norm()
            content_loss = -criterion_ssim(hr_pred, hr)

            loss_R = content_loss + 0.005*gradient_loss

            ep_list_lr = [0.01, 0.03, 0.05, 0.05, 0.05, 0.05, 0.05]
            ep_list_hr = [0.01, 0.05, 0.10, 0.15, 0.15, 0.15, 0.15]

            sparse_loss_lr = loss_package(lr_x1, lr_x2, lr_x3, lr_x4, epoch, ep_list_lr)
            sparse_loss_hr = loss_package(hr_x1, hr_x2, hr_x3, hr_x4, epoch, ep_list_hr)

            rect_loss = self.loss(hr_x1, hr_max)+self.loss(hr_x2, hr_max)+self.loss(hr_x3, hr_max)+self.loss(hr_x4, hr_max) + self.loss(lr_x1, lr_max)+self.loss(lr_x2, lr_max)+self.loss(lr_x3, lr_max)+self.loss(lr_x4, lr_max)
            map_loss = loss_I + loss_R

            sparse_coeff = 0.0001
            loss = rect_loss + sparse_coeff*(sparse_loss_lr + sparse_loss_hr) + map_loss*0.1

            all_sparse_loss += sparse_coeff*(sparse_loss_lr + sparse_loss_hr)
            all_rect_loss += rect_loss

            loss.backward()

            if loss.item() < self.args.skip_threshold * self.error_last:
                if epoch<100:
                    self.optimizer.step()

                self.I_optimizer.step()
                self.R_optimizer.step()

            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{}\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                 
                    loss_R.data[0],
                    loss_I.data[0],
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        print(all_sparse_loss.item())
        print(all_rect_loss.item())

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()
        self.model_R.eval()
        self.model_I.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)

                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    lr = lr/255.0
                    hr = hr/255.0

                    lr = (lr+0.05)/1.05
                    hr = (hr+0.05)/1.05

                    hr_max = torch.max(hr, dim=1)[0].unsqueeze(1)
                    lr_max = torch.max(lr, dim=1)[0].unsqueeze(1)
                    hr_x1, hr_x2, hr_x3, hr_x4 = self.model(hr_max, idx_scale)
                    lr_x1, lr_x2, lr_x3, lr_x4 = self.model(lr_max, idx_scale)

                    lr_l = lr_x4
                    hr_l = hr_x4

                    lr_l[lr_l<0.05/1.05] = 0.05/1.05
                    lr_l[lr_l>1] = 1
                    hr_l[hr_l<0.05/1.05] = 0.05/1.05
                    hr_l[hr_l>1] = 1

                    lr_l = torch.cat((lr_l, lr_l, lr_l), 1)
                    hr_l = torch.cat((hr_l, hr_l, hr_l), 1)

                    lr_r = lr.clone()
                    hr_r = hr.clone()

                    lr_r = lr/lr_l.detach()
                    hr_r = hr/hr_l.detach()

                    hr_l_pre = self.model_I(lr, idx_scale)

                    hr_l_pre[hr_l_pre<0.05/1.05] = 0.05/1.05
                    hr_l_pre[hr_l_pre>1] = 1

                    hr_r_pre = self.model_R(lr_r, idx_scale)
                    hr_r_pre = hr_r_pre

                    hr_pre = torch.mul(hr_l_pre, hr_r_pre)

                    lr = lr*1.05-0.05
                    hr = hr*1.05-0.05
                    hr_pre = hr_pre*1.05-0.05

                    lr_max = torch.cat((lr_max, lr_max, lr_max),1)
                    hr_max = torch.cat((hr_max, hr_max, hr_max),1)

                    lr_x1 = torch.cat((lr_x1, lr_x1, lr_x1),1)
                    lr_x2 = torch.cat((lr_x2, lr_x2, lr_x2),1)
                    lr_x3 = torch.cat((lr_x3, lr_x3, lr_x3),1)
                    lr_x4 = torch.cat((lr_x4, lr_x4, lr_x4),1)

                    hr_x1 = torch.cat((hr_x1, hr_x1, hr_x1),1)
                    hr_x2 = torch.cat((hr_x2, hr_x2, hr_x2),1)
                    hr_x3 = torch.cat((hr_x3, hr_x3, hr_x3),1)
                    hr_x4 = torch.cat((hr_x4, hr_x4, hr_x4),1)                    
                    
                    hr_x1 = utility.quantize(hr_x1*255, self.args.rgb_range)
                    hr_x2 = utility.quantize(hr_x2*255, self.args.rgb_range)
                    hr_x3 = utility.quantize(hr_x3*255, self.args.rgb_range)
                    hr_x4 = utility.quantize(hr_x4*255, self.args.rgb_range)

                    hr_max = utility.quantize(hr_max*255, self.args.rgb_range)

                    hr = utility.quantize(hr*255, self.args.rgb_range)

                    lr_x1 = utility.quantize(lr_x1*255, self.args.rgb_range)
                    lr_x2 = utility.quantize(lr_x2*255, self.args.rgb_range)
                    lr_x3 = utility.quantize(lr_x3*255, self.args.rgb_range)
                    lr_x4 = utility.quantize(lr_x4*255, self.args.rgb_range)

                    lr_max = utility.quantize(lr_max*255, self.args.rgb_range)
                    lr = utility.quantize(lr*255, self.args.rgb_range)

                    lr_l = utility.quantize(lr_l*255, self.args.rgb_range)
                    lr_r = utility.quantize(lr_r*255, self.args.rgb_range)
                    hr_l = utility.quantize(hr_l*255, self.args.rgb_range)
                    hr_r = utility.quantize(hr_r*255, self.args.rgb_range)
                    hr_l_pre = utility.quantize(hr_l_pre*255, self.args.rgb_range)
                    hr_r_pre = utility.quantize(hr_r_pre*255, self.args.rgb_range)
                    hr_pre = utility.quantize(hr_pre*255, self.args.rgb_range)

                    save_list = [hr, hr_max, hr_x1, hr_x2, hr_x3, hr_x4, lr, lr_max, lr_x1, lr_x2, lr_x3, lr_x4, lr_r, hr_l_pre, hr_r_pre, hr_pre, hr_l, hr_r, lr_l]

                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            hr_pre, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale, epoch)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)

                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

