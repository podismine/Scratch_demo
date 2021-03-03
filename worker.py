import argparse
import os
import random
import shutil
import time
import warnings
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast
import torch
import torch as t
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from utils.visualize import Visualizer
from dataset.dataset import MultiBranch_Data
from apex import amp
from apex.parallel import DistributedDataParallel
import warnings 
from warmup_scheduler import GradualWarmupScheduler
#yours
from models.sfcn import SFCN
from dataset.brain_age_dataset import CombinedData

vis = Visualizer(args.env_name)

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            #data,label,sex, y1, bc1, y2, bc2,y3,bc3
            self.next_input, self.next_target, self.next_sex, self.next_y, self.next_bc = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_sex = None
            self.next_y = None
            self.next_bc = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_sex = self.next_sex.cuda(non_blocking=True)
            self.next_y = self.next_y.cuda(non_blocking=True)
            self.next_bc = self.next_bc.cuda(non_blocking=True)

            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_target = self.next_target.float()
            self.next_sex = self.next_sex.float()
            self.next_y = self.next_y.float()
            self.next_bc = self.next_bc.float()
            #self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        sex = self.next_sex
        y = self.next_y
        bc = self.next_bc
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if sex is not None:
            sex.record_stream(torch.cuda.current_stream())
        if y is not None:
            y.record_stream(torch.cuda.current_stream())
        if bc is not None:
            bc.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target, sex, y, bc


def run_main(args):
    
    args.nprocs = torch.cuda.device_count()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args.local_rank, args.nprocs, args)

def loss_func(x, y):
        return torch.nn.L1Loss().cuda()(x, y) +  torch.nn.L1Loss().cuda()(x, y)

def main_worker(local_rank, nprocs, args):
    best_mae = 99.0

    dist.init_process_group(backend='nccl')

    model = SFCN()

    torch.cuda.set_device(local_rank)
    model.cuda()
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / nprocs)

    # define loss function (criterion) and optimizer
    criterion = loss_func

    #optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = t.optim.Adam(model.parameters(),lr = args.lr,weight_decay = args.weight_decay)
    model, optimizer = amp.initialize(model, optimizer)

    model = DistributedDataParallel(model)
    
    cudnn.benchmark = True

    # Data loading code
    train_data = CombinedData('',train = True)
    val_data = CombinedData('',train = False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

    train_loader = DataLoader(train_data,args.batch_size,
                        shuffle=False,num_workers=4,pin_memory = True, sampler = train_sampler)
    val_loader = DataLoader(val_data,args.batch_size,
                        shuffle=False,num_workers=4,pin_memory = True)



    if args.evaluate:
        validate(val_loader, model, criterion, local_rank, args)
        return


    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, local_rank, args)

        # evaluate on validation set
            
        mae= validate(val_loader, model, criterion, local_rank, args)

        # remember best acc@1 and save checkpoint
        is_best = mae < best_mae
        best_mae = min(mae, best_mae)


        if not os.path.exists("checkpoints/%s" % args.env_name):
            os.makedirs("checkpoints/%s" % args.env_name)

        if is_best:
            if local_rank == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'best_mae': best_mae,
                        'amp': amp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, True ,'./checkpoints/%s/%s_epoch_%s_%s' % (args.env_name, args.env_name, epoch, best_mae))



def train(train_loader, model, criterion, optimizer, epoch, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    loss_mae = AverageMeter('mae', ':6.2f')
    loss_kl = AverageMeter('kl', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, loss_mae, loss_kl],
                             prefix="Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()
    prefetcher = data_prefetcher(train_loader)
    images, target, sex, y, bc= prefetcher.next()
    i = 0
    optimizer.zero_grad()
    optimizer.step()
    while images is not None:
        # measure data loading time
        data_time.update(time.time() - end)
        
        # compute output
        with autocast():
            out = model(images)
            prob = torch.exp(out)
            pred = torch.sum(prob * bc, dim = 1)
            
            kl_loss = dpl.my_KLDivLoss(out, y) 
            mae = torch.nn.L1Loss()(pred, target) 


        loss_all = kl_loss + mae # criterion(x, y)
        torch.distributed.barrier() 

        reduced_loss = reduce_mean(loss_all, args.nprocs)
        reduced_mae = reduce_mean(mae, args.nprocs)
        reduced_kl = reduce_mean(loss, args.nprocs)

        losses.update(reduced_loss.item(), images.size(0))
        loss_mae.update(reduced_mae.item(), images.size(0))
        loss_kl.update(reduced_kl.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        with amp.scale_loss(loss_all, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        i += 1

        images, target, sex, y, bc = prefetcher.next()
    vis.plot('train_mae_loss', float(loss_mae.avg))
    vis.plot('train_kl_loss', float(loss_kl.avg))
    vis.plot('train_loss', float(losses.avg))

def validate(val_loader, model, criterion, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    loss_mae = AverageMeter('mae', ':6.2f')
    loss_kl = AverageMeter('kl', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, loss_mae, loss_kl], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        prefetcher = data_prefetcher(val_loader)
        images, target, sex, y, bc = prefetcher.next()
        i = 0
        while images is not None:

            # compute output
            with autocast():
                out = model(images)
                prob = torch.exp(ou1)
                pred = torch.sum(prob * bc, dim = 1)

                loss = dpl.my_KLDivLoss(out, y)
                mae = torch.nn.L1Loss()(pred, target) 


            torch.distributed.barrier()
            loss_all = loss #+ mae

            reduced_loss = reduce_mean(loss_all, args.nprocs)
            reduced_mae = reduce_mean(mae, args.nprocs)
            reduced_kl = reduce_mean(loss, args.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            loss_mae.update(reduced_mae.item(), images.size(0))
            loss_kl.update(reduced_kl.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

            i += 1

            images, target, sex, y, bc = prefetcher.next()

        # TODO: this should also be done with the ProgressMeter
        print(' * MAE@ {loss_mae.avg:.3f} KL@ {loss_kl.avg:.3f}'.format(loss_mae=loss_mae, loss_kl=loss_kl))
        vis.plot('val_mae_loss', float(loss_mae.avg))
        vis.plot('val_kl_loss', float(loss_kl.avg))
        vis.plot('val_loss', float(losses.avg))
    return loss_mae.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
    


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_warmup = [i / 500. * args.lr for i in range(1, 510)]
    #lr = args.lr * (0.1**(epoch // 30))
    if epoch < 500:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_warmup[epoch]
    else:
        for param_group in optimizer.param_groups:
                param_group['lr'] = min(1e-6, (args.epochs - epoch) / args.epochs * args.lr)