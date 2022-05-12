# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import warnings
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter

import _init_paths
import models

from config import cfg
from config import update_config
from core.loss import MultiLossFactory
from core.trainer import do_train
from dataset import make_dataloader
from fp16_utils.fp16util import network_to_half
from fp16_utils.fp16_optimizer import FP16_Optimizer
from utils.utils import create_logger
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general，指定yaml文件的路径
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='../experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml',
                        type=str)
    # 暂时没有具体实现
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu', help='gpu id for multiprocessing training', type=str)
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')

    args = parser.parse_args()

    return args


def main():
    # 对输入参数进行解析
    args = parse_args()

    # 根据输入参数对默认的cfg配置进行更新
    update_config(cfg, args)

    cfg.defrost()
    cfg.RANK = args.rank
    cfg.freeze()

    # 创建logger，用于记录训练过程的打印信息
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # 判断输入参数中是否指定了GPU id
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or cfg.MULTIPROCESSING_DISTRIBUTED

    ngpus_per_node = torch.cuda.device_count()

    # 判断是否采用分布式训练
    if cfg.MULTIPROCESSING_DISTRIBUTED:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # 多采用分布式训练，会根据gpu数量启用相应个数的main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, final_output_dir, tb_log_dir))
    else:
        # 没有采用分布式训练，则只需调用 main_worker 函数
        main_worker(','.join([str(i) for i in cfg.GPUS]), ngpus_per_node, args, final_output_dir, tb_log_dir)


def main_worker(gpu, ngpus_per_node, args, final_output_dir, tb_log_dir):
    '''第一部分 cudnn的设置，以及cfg的更新'''
    # cudnn related setting
    # 使用GPU的一些相关设置
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # fp16 模式需要启用 cudnn 后端
    if cfg.FP16.ENABLED:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    # 警告：如果不使用 --fp16，static_loss_scale 将被忽略
    if cfg.FP16.STATIC_LOSS_SCALE != 1.0:
        if not cfg.FP16.ENABLED:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if cfg.MULTIPROCESSING_DISTRIBUTED:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            # 对于多进程分布式训练，rank需要是所有进程中的全局rank
            args.rank = args.rank * ngpus_per_node + gpu
        print('Init process group: dist_url: {}, world_size: {}, rank: {}'.
              format(args.dist_url, args.world_size, args.rank))
        dist.init_process_group(backend=cfg.DIST_BACKEND, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)

    # 根据输入参数对默认的cfg配置进行更新
    update_config(cfg, args)

    # setup logger
    logger, _ = setup_logger(final_output_dir, args.rank, 'train')

    '''第二部分 开始创建model文件'''
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)
    # torch.save(model.state_dict(), 'model.pth')
    # print(model)

    '''第三部分 设置日志，model的属性，以及多卡，单卡或cpu的选择。'''
    # copy model file
    if not cfg.MULTIPROCESSING_DISTRIBUTED or (
            cfg.MULTIPROCESSING_DISTRIBUTED and args.rank % ngpus_per_node == 0):
        this_dir = os.path.dirname(__file__)
        shutil.copy2(os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'), final_output_dir)

    # 日志
    writer_dict = {'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0, 'valid_global_steps': 0,}

    if not cfg.MULTIPROCESSING_DISTRIBUTED or (
            cfg.MULTIPROCESSING_DISTRIBUTED
            and args.rank % ngpus_per_node == 0):
        dump_input = torch.rand((1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE))
        # 输出日志
        writer_dict['writer'].add_graph(model, (dump_input, ))
        # logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.MODEL.SYNC_BN and not args.distributed:
        print('Warning: Sync BatchNorm is only supported in distributed training.')

    if args.distributed:
        if cfg.MODEL.SYNC_BN:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            # 如果没有设置 device_ids，DistributedDataParallel 将划分并分配 batch_size 给所有可用的 GPU
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        # 单GPU工作

        # torch.cuda.set_device(args.gpu)
        # model = model.cuda(args.gpu)
        torch.cuda.set_device(0)
        model = model.cuda(0)
    else:
        # 让模型支持CPU训练
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    '''第四部分 定义损失函数（标准）'''
    # 调用了core/loss.py的MultiLossFactory()函数,里面定义了热图损失函数和分组损失函数
    loss_factory = MultiLossFactory(cfg).cuda()

    # Data loading code
    '''第五部分 生成训练数据，创建训练数据的迭代器'''
    train_loader = make_dataloader(cfg, is_train=True, distributed=args.distributed)
    logger.info(train_loader.dataset)
    '''第六部分 实例化优化器'''
    best_perf = -1          # 最优得分初始化
    best_model = False      # 是否最佳模型的初始化
    last_epoch = -1         # 当前最新的epoch初始化
    # 定义优化器
    optimizer = get_optimizer(cfg, model)

    '''第七部分 训练前的设置部分'''
    if cfg.FP16.ENABLED:
        optimizer = FP16_Optimizer(optimizer,
            static_loss_scale=cfg.FP16.STATIC_LOSS_SCALE,
            dynamic_loss_scale=cfg.FP16.DYNAMIC_LOSS_SCALE)

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')    # 加载中间节点的模型权重信息（接续训练）
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)    # 载入节点权重信息
        begin_epoch = checkpoint['epoch']         # 开始的批次数
        best_perf = checkpoint['perf']                # 最优的结果？
        last_epoch = checkpoint['epoch']
        # 加载模型权重  state_dict是一个Python字典，将每一层映射成它的参数张量。
        #  注意只有带有可学习参数的层（卷积层、全连接层等），
        # 以及注册的缓存（batchnorm的运行平均值）在state_dict 中才有记录
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])    # 加载权重
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    if cfg.FP16.ENABLED:
        # pytorch动态调整学习率
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer.optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch)
    else:
        # pytorch动态调整学习率
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch)
    '''第八部分 训练部分
         Epoch：一个Epoch就是将所有训练样本训练一次的过程。
         Batch（批 / 一批样本）：将整个训练样本分成若干个Batch。
         Batch_Size（批大小）： 每批样本的大小。
         Iteration（一次迭代）：训练一个Batch就是一次Iteration。
    '''
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        # train one epoch
        '''
         do_train()函数内实现了输入数据，返回检测后的结果，并进行损失函数计算，更新参数。
         do_train()函数后就是更新最优模型，若该epoch是最优模型，则会保存此epoch的节点和模型参数'''
        do_train(cfg, model, train_loader, loss_factory, optimizer, epoch,
                 final_output_dir, tb_log_dir, writer_dict, fp16=cfg.FP16.ENABLED)

        # In PyTorch 1.1.0 and later, you should call `lr_scheduler.step()` after `optimizer.step()`.
        # 是用来更新优化器的学习率的，一般是按照epoch为单位进行更换，
        # 即多少个epoch后更换一次学习率，因而scheduler.step()放在epoch这个大循环下。
        lr_scheduler.step()

        perf_indicator = epoch
        # 更新最优模型
        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        if not cfg.MULTIPROCESSING_DISTRIBUTED or (
                cfg.MULTIPROCESSING_DISTRIBUTED
                and args.rank == 0):
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            # checkpoint检查点：不仅保存模型的参数，优化器参数，还有loss，epoch等（相当于一个保存模型的文件夹）
            # 在保存用于推理或者继续训练的常规检查点的时候，除了模型的state_dict之外，还必须保存其他参数。
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),}, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir, 'final_state{}.pth.tar'.format(gpu))

    logger.info('saving final model state to {}'.format(final_model_state_file))
    # 保存整个模型
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
