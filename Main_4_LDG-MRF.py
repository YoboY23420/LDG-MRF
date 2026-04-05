# -*- coding: utf-8 -*-
import os, losses, utils, time, torch, dataloader
from datetime import datetime
import torch.nn as nn
from torch import optim
from argparse import ArgumentParser
from model import Model

def init_weights(m):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.xavier_uniform_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias, 0)

def main(device, dataset, channel_num, cluster_num):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    if dataset == 'OASIS':
        train_dir = '/Medical_Image_Registration/3D_brain_MRI/affine_img/'
        valid_dir = '/Medical_Image_Registration/3D_brain_MRI/affine_seg/'
        img_size = (160, 192, 224)
        max_epoch = 1
        loader_train = dataloader.torch_Dataloader_OASIS(train_dir, valid_dir, 'train', batch_size=1, inshape=img_size)
        loader_validation = dataloader.torch_Dataloader_OASIS(train_dir, valid_dir, 'test', batch_size=1, inshape=img_size)
        save_exp = '/saved_models/experiments_OASIS'
    elif dataset == 'LPBA40':
        train_dir = '/Medical_Image_Registration/3D_LPBA40/affine_img/'
        valid_dir = '/Medical_Image_Registration/3D_LPBA40/affine_seg/'
        img_size = (160, 192, 224)
        max_epoch = 30
        loader_train = dataloader.torch_Dataloader_LPBA40(train_dir, valid_dir, 'train', batch_size=1, inshape=img_size)
        loader_validation = dataloader.torch_Dataloader_LPBA40(train_dir, valid_dir, 'test', batch_size=1, inshape=img_size)
        save_exp = '/saved_models/experiments_LPBA40'
    elif dataset == 'IXI':
        train_dir = '/Medical_Image_Registration/3D_IXI_dataset/IXI_data/Train/'
        valid_dir = '/Medical_Image_Registration/3D_IXI_dataset/IXI_data/Test/'
        img_size = (160, 192, 224)
        max_epoch = 1
        loader_train = dataloader.torch_Dataloader_IXI(train_dir, valid_dir, 'train', batch_size=1, inshape=img_size)
        loader_validation = dataloader.torch_Dataloader_IXI(train_dir, valid_dir, 'test', batch_size=1, inshape=img_size)
        save_exp = '/saved_models/experiments_IXI'
    elif dataset == 'Mind_101':
        data_dir = '/Medical_Image_Registration/Mindboggle-101/data_used/'
        img_size = (160, 192, 160)
        max_epoch = 30
        loader_train = dataloader.torch_Dataloader_Mind101(data_dir, 'train', batch_size=1)
        loader_validation = dataloader.torch_Dataloader_Mind101(data_dir, 'test', batch_size=1)
        save_exp = '/saved_models/experiments_Mind101'
    elif dataset == 'Lung_CT':
        train_dir = '/Medical_Image_Registration/Lung_CT_dataset/affine_img_ordered/'
        valid_dir = '/Medical_Image_Registration/Lung_CT_dataset/affine_seg_ordered/'
        img_size = (160, 192, 224)
        max_epoch = 30
        loader_train = dataloader.torch_Dataloader_LungCT(train_dir, valid_dir, 'train', batch_size=1)
        loader_validation = dataloader.torch_Dataloader_LungCT(train_dir, valid_dir, 'test', batch_size=1)
        save_exp = '/saved_models/experiments_LungCT'
    else:
        return

    timestamp = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
    if os.path.exists(save_exp):
        os.makedirs(os.path.join(save_exp, timestamp))
    else:
        os.makedirs(os.path.join(save_exp))
        os.makedirs(os.path.join(save_exp, timestamp))

    model = Model(channel_num, cluster_num)
    # model.apply(init_weights)
    model.cuda()
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    loss_weights = [1.0, 1.0] # first term for similarity, second term for regularization
    lr = 0.0001
    epoch_start = 0

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    criterion = losses.NCC(win=9).loss
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    for epoch in range(epoch_start, max_epoch):
        step_t = 0
        for pair_t, mi_t, fi_t, ml_t, fl_t in loader_train:
            step_t += 1
            model.train()
            mi_t = mi_t.unsqueeze(0).cuda()
            fi_t = fi_t.unsqueeze(0).cuda()
            output = model(mi_t, fi_t)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], fi_t) * loss_weights[n]
                loss_vals.append(curr_loss.item())
                loss += curr_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('\rEpoch {}<-->Image {} to {}<-->Iter {}/{}<-->TotalLoss {:.4f} = SimLoss {:.4f} + {:.4f} SmoLoss'.format(epoch, pair_t[0], pair_t[1], step_t, len(loader_train), loss.item(), loss_vals[0], loss_vals[1]))

            if dataset == 'OASIS':
                total_iter = 64770
                testing_iter = 15000
            elif dataset == 'LPBA40':
                total_iter = 756
                testing_iter = 757
            elif dataset == 'IXI':
                total_iter = 162006
                testing_iter = 20000
            elif dataset == 'Mind_101':
                total_iter = 1722
                testing_iter = 1723
            elif dataset == 'Lung_CT':
                total_iter = 1560
                testing_iter = 1561
            else:
                return
            if step_t == total_iter or step_t % testing_iter == 0:
                eval_dsc = utils.AverageMeter()
                eval_njd = utils.AverageMeter()
                eval_hd95 = utils.AverageMeter()
                eval_assd = utils.AverageMeter()
                eval_time = utils.AverageMeter()
                with torch.no_grad():
                    step_v = 0
                    for pair_v, mi_v, fi_v, ml_v, fl_v in loader_validation:
                        step_v += 1
                        model.eval()
                        mi_v = mi_v.unsqueeze(0).cuda()
                        fi_v = fi_v.unsqueeze(0).cuda()
                        ml_v = ml_v.unsqueeze(0).cuda()
                        fl_v = fl_v.unsqueeze(0).cuda()
                        # grid_img = mk_grid_img(8, 1, img_size)
                        time_start = time.time()
                        output = model(mi_v, fi_v)
                        time_end = time.time()
                        warped_ml_v = reg_model([ml_v, output[1]])
                        warped_ml_v = warped_ml_v.detach().cpu().numpy()[0, 0, ...]
                        fl_v = fl_v.detach().cpu().numpy()[0, 0, ...]
                        dsc = utils.dice_val_ROI(warped_ml_v, fl_v, dataset=dataset)
                        eval_dsc.update(dsc.item(), n=1)
                        njd = utils.jacobian_determinant_vxm(output[1].detach().cpu().numpy().squeeze())
                        eval_njd.update(njd.item(), n=1)
                        hd95 = utils.hd95_val_ROI(warped_ml_v, fl_v, dataset=dataset)
                        eval_hd95.update(hd95.item(), n=1)
                        assd = utils.assd_val_ROI(warped_ml_v, fl_v, dataset=dataset)
                        eval_assd.update(assd.item(), n=1)
                        time_usage = time_end - time_start
                        eval_time.update(time_usage, n=1)
                        print(
                            '\rEpoch {}<-->Image {} to {}<-->Iter {}/{}<-->DSC {:.4f}<-->NJD {:.4f}<-->HD95 {:.4f}<-->ASSD {:.4f}<-->Time {:.4f}'.format(
                                epoch, pair_v[0], pair_v[1], step_v, len(loader_validation), dsc, njd, hd95, assd,
                                time_usage
                            ), end='')
                torch.save(model.state_dict(), '{}/{}_Dice{:.4f}_NJD{:.4f}_HD95{:.4f}_ASSD{:.4f}_Time{:.4f}.pth'.format(os.path.join(save_exp, timestamp), step_t, eval_dsc.avg, eval_njd.avg, eval_hd95.avg, eval_assd.avg, eval_time.avg))
    print('All Finished!')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='OASIS')
    parser.add_argument('--channel_num', type=int, default=1)
    parser.add_argument('--cluster_num', type=int, default=1)
    args = parser.parse_args()
    main(**vars(args))
