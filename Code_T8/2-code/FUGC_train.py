import argparse
from asyncore import write
from decimal import ConversionSyntax
import logging
from multiprocessing import reduction
import os
import random
import shutil
import sys
import time
import pdb
import cv2
import matplotlib.pyplot as plt
import imageio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, ThreeStreamBatchSampler)
from networks.net_factory import net, net_factory
import losses, ramps, feature_memory, contrastive_losses, val_2d


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ISBI/train', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='ours', help='experiment_name')
parser.add_argument('--model', type=str, default='resnet101', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=8000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=18000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[336, 544, 3], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=3, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8, help='labeled_batch_size per gpu')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int,  default=6, help='multinum of random masks')
#resnet
# parser.add_argument('--replace_stride_with_dilation', default=[False,False,True])
# parser.add_argument('--dilations', default=[6, 12, 18])
# parser.add_argument('--early_stop', type=int, default=20000)

args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=args.num_classes)

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

def get_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert(labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        class_list.append(largestCC * i)
    largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(largestCC).cuda()

def get_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 3):
            temp_seg = segmentation[i] #== c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)          
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = class_list[0] + class_list[1]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()
    

def get_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    avg_prob = _.mean()
    if nms == 1:
        probs = get_2DLargestCC(probs)
    return avg_prob, probs

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5* args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)

def update_model_ema_dynamic(
        model,
        ema_model,
        confidence: float,
        base_alpha: float = 0.99,
        max_alpha: float = 0.9999,
        min_alpha: float = 0.8
):
    model_state = model.state_dict()
    ema_state = ema_model.state_dict()
    new_state = {}

    dynamic_alpha = base_alpha * (1.0 - confidence)
    dynamic_alpha = max(min_alpha, min(max_alpha, dynamic_alpha))

    for key in model_state:
        if key not in ema_state:
            new_state[key] = model_state[key].clone()
            continue

        new_state[key] = dynamic_alpha * ema_state[key] + (1 - dynamic_alpha) * model_state[key]

    ema_model.load_state_dict(new_state, strict=False)

def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()


def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x*2/(3*shrink_param)), int(img_y*2/(3*shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s*x_split, (x_s+1)*x_split-patch_x)
            h = np.random.randint(y_s*y_split, (y_s+1)*y_split-patch_y)
            mask[w:w+patch_x, h:h+patch_y] = 0
            loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y *4/9)
    h = np.random.randint(0, img_y-patch_y)
    mask[h:h+patch_y, :] = 0
    loss_mask[:, h:h+patch_y, :] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False, hd_weight=0.05, iter_num = 0):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce

    return loss_dice, loss_ce

def pre_train(args, snapshot_path, fold_num):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     
    # model = DeepLabV3Plus(args)
    # model.to(torch.device("cuda"))
    model = net(in_chns=3, class_num=num_classes)


    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]), fold_num=fold_num)
    db_val = BaseDataSets(base_dir=args.root_path, split="val", fold_num=fold_num)
    total_slices = len(db_train)
    labeled_slice = 40
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=1, pin_memory=True, worker_init_fn=random.seed(args.seed))

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=3e-5, amsgrad=False)

    # writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    # best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            # print(iter_num)
            volume_batch, label_batch = sampled_batch['image_weak'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            # img_mask, loss_mask = generate_fmix_mask(img_a)
            img_mask, loss_mask = generate_mask(img_a)
            # gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            #-- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl = model(net_input)
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            # writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            # writer.add_scalar('info/mix_ce', loss_ce, iter_num)
            # writer.add_scalar('info/total_loss', loss, iter_num)

            # logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
                
            # if iter_num % 20 == 0:
            #     image = net_input[1, 0:1, :, :]
            #     writer.add_image('pre_train/Mixed_Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
            #     writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
            #     labs = gt_mixl[1, ...].unsqueeze(0) * 50
            #     writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                # for class_i in range(num_classes-1):
                #     writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                #     writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                hd95 = np.mean(metric_list, axis=0)[1]
                # writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'fold_{}_iter_{}_dice_{}.pth'.format(fold_num, iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'fold_{}_{}_best_model.pth'.format(fold_num, args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info(
                    'fold: %d   iteration %d : mean_dice : %f mean_hd95 : %f' % (fold_num, iter_num, performance, hd95))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    # writer.close()

def self_train(args ,pre_snapshot_path, snapshot_path, fold_num):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'fold_{}_{}_best_model.pth'.format(fold_num, args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)

    # model = DeepLabV3Plus(args)
    # model.to(torch.device("cuda"))
    # ema_model = DeepLabV3Plus(args)
    # ema_model.to(torch.device("cuda"))
    # for param in ema_model.parameters():
    #     param.detach_()
    model = net(in_chns=3, class_num=num_classes)
    ema_model = net(in_chns=3, class_num=num_classes, ema=True)


    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]), fold_num=fold_num)
    db_val = BaseDataSets(base_dir=args.root_path, split="val", fold_num=fold_num)
    total_slices = len(db_train)
    labeled_slice = 40
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=1, pin_memory=True, worker_init_fn=random.seed(args.seed))

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=3e-5, amsgrad=False)
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    ema_model.train()

    # ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    # best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_weak_batch, label_batch, volume_strong_batch = sampled_batch['image_weak'], sampled_batch['label'], sampled_batch['image_strong']
            volume_weak_batch, label_batch, volume_strong_batch = volume_weak_batch.cuda(), label_batch.cuda(), volume_strong_batch.cuda()

            limg = volume_weak_batch[:args.labeled_bs]
            uimg_weak = volume_weak_batch[args.labeled_bs:]
            uimg_strong = volume_strong_batch[args.labeled_bs:]
            lab = label_batch[:args.labeled_bs]
            with torch.no_grad():
                pre_weak = ema_model(uimg_weak)
                avg_prob, plab_weak = get_masks(pre_weak, nms=1)
                avg_prob = avg_prob.cpu().detach().numpy()
                img_mask, loss_mask = generate_mask(uimg_weak)
                # img_mask, loss_mask = generate_fmix_mask(img_a)


            net_input = uimg_strong * img_mask + limg * (1 - img_mask)
            out_mix = model(net_input)
            pre_strong = model(uimg_strong)
            mix_dice, mix_ce = mix_loss(out_mix, plab_weak, lab, loss_mask, u_weight=args.u_weight, unlab=True, iter_num= iter_num)
            consistency_loss = (dice_loss(pre_strong, plab_weak.unsqueeze(1)) + F.mse_loss(pre_strong, pre_weak))/2
            loss_ce = mix_dice
            loss_dice = mix_ce

            loss = loss_dice + loss_ce + consistency_loss * 0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            # update_model_ema(model, ema_model, 0.99)
            update_model_ema_dynamic(model, ema_model, avg_prob)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                # for class_i in range(num_classes-1):
                #     writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                #     writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                hd95 = np.mean(metric_list, axis=0)[1]
                # writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'fold_{}_iter_{}_dice_{}.pth'.format(fold_num, iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'fold_{}_{}_best_model.pth'.format(fold_num, args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info(
                    'fold: %d   iteration %d : mean_dice : %f mean_hd95 : %f' % (fold_num, iter_num, performance, hd95))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    pre_snapshot_path = "./model/ours/{}_labeled/pre_train".format(args.exp)
    self_snapshot_path = "./model/ours/{}_labeled/self_train".format(args.exp)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)


    # Pre_train
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    for fold_num in range(1, 6):
        logging.info("=========================== {} fold pre-train start======================".format(fold_num))
        logging.info(str(args))
        pre_train(args, pre_snapshot_path, fold_num)

    #Self_train
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    for fold_num in range(1, 6):
        logging.info("=========================== {} fold self-train start======================" .format(fold_num))
        logging.info(str(args))
        self_train(args, pre_snapshot_path, self_snapshot_path, fold_num)

    


