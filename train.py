import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
from tensorboardX import SummaryWriter

from model import PTModel
from loss import ssim
from data import getTrainingTestingData, loss_scale
from utils import AverageMeter, DepthNorm, colorize
import cv2
import numpy as np

eval_losses = AverageMeter()

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    args = parser.parse_args()

    # Create model
    model = PTModel().cuda()
    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam( model.parameters(), args.lr )
    batch_size = args.bs
    prefix = 'densenet_' + str(batch_size)

    # Load data
    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)

    test_loader = iter(test_loader)
    sample_batched = next(test_loader)
    # Logging
    writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)

    # Loss
    l1_criterion = nn.L1Loss()
    tf = open('eval/train_loss_avg.txt','w')
    ef = open('eval/eval_loss_avg.txt','w')
    # Start training...
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)

        # Switch to train mode
        model.train()

        end = time.time()

        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            # # Normalize depth
            # depth_n = DepthNorm( depth )

            # print("-----------------")
            # print("image shape: ",depth.shape)
            # print("depth shape: ",depth.shape)
            # print("depth_n shape: ",depth_n.shape)
            # print("depth",depth)
            # print("depth_n",depth_n)
            # print("__________________")

            # Predict%
            output = model(image)
            # print("output.shape:",output.shape)
            # print("depth.shape:",depth.shape)
            # exit()
            # Compute the loss
            # l_depth = l1_criterion(output, depth_n)
            # l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)
            l_depth = l1_criterion(output, depth)
            l_ssim = torch.clamp((1 - ssim(output, depth, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)

            loss = (1.0 * l_ssim) + (0.1 * l_depth)
            # print("loss: ",loss)
            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))
        
            # Log progress
            niter = epoch*N+i
            if i % 5 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                'ETA {eta}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'
                .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

                # Log to tensorboard
                writer.add_scalar('Train/Loss', losses.val, niter)

            # if i % 300 == 0:
            if i % 20 == 0:
                LogProgress(model, writer, sample_batched, niter)
                # ImageLogProgress(model, writer, test_loader, niter)
            # if i % 1200 == 0:    
        # Record epoch's intermediate results
        LogProgress(model, writer, sample_batched, niter)
        writer.add_scalar('Train/Loss.avg', losses.avg, epoch)
        writer.add_scalar('Test/Loss.avg', eval_losses.avg, epoch)
        print(losses.avg,file=tf, flush=True)
        print(eval_losses.avg,file=ef, flush=True)
        print("Train Loss: ",losses.avg)
        print("Eval  Loss: ",eval_losses.avg)
        ImageLogProgress(model, writer, sample_batched, epoch)
        sample_batched = next(test_loader)
        torch.save(model,"eval/"+str(epoch)+"model.h5")

def LogProgress(model, writer, test_loader, epoch):
    global eval_losses
    l1_criterion = nn.L1Loss()
    # sequential = test_loader
    # sample_batched = next(sequential)
    image = torch.autograd.Variable(test_loader['image'].cuda())
    depth = torch.autograd.Variable(test_loader['depth'].cuda(non_blocking=True))
    depth_n = DepthNorm( depth )
    if epoch == 0: writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
    if epoch == 0: writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth_n.data, nrow=6, normalize=False)), epoch)
    output = model(image)
    output_n = DepthNorm( output )
    l_depth = l1_criterion(output, depth)
    l_ssim = torch.clamp((1 - ssim(output, depth, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)
    loss = (1.0 * l_ssim) + (0.1 * l_depth)
    eval_losses.update(loss.data.item(), image.size(0))
    writer.add_scalar('Test/Loss', loss, epoch)
    writer.add_image('Train.3.Ours', colorize(vutils.make_grid(output_n.data, nrow=6, normalize=False)), epoch)
    writer.add_image('Train.3.Diff', colorize(vutils.make_grid(torch.abs(output_n-depth_n).data, nrow=6, normalize=False)), epoch)
    # model.eval()
    del image
    del depth
    del output

def ImageLogProgress(model, writer, test_loader, epoch):
    model.eval()
    # sequential = test_loader
    # sample_batched = next(sequential)
    image = torch.autograd.Variable(test_loader['image'].cuda())
    depth = torch.autograd.Variable(test_loader['depth'].cuda(non_blocking=True))
    output = model(image)
    depth_scale = 0.0002500000118743628 * loss_scale
    output_n = output.detach().cpu().numpy()[0][0] * depth_scale
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(output_n, alpha=255/output_n.max()), cv2.COLORMAP_JET)
    depth_n = depth.detach().cpu().numpy()[0][0] * depth_scale
    depth_true_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_n, alpha=255/depth_n.max()), cv2.COLORMAP_JET)
    diff = depth_n - output_n
    diff_colormap = cv2.applyColorMap(cv2.convertScaleAbs(diff, alpha=255/diff.max()), cv2.COLORMAP_JET)
    cv2.imwrite("eval/"+str(epoch)+'.png',depth_colormap)
    cv2.imwrite("eval/"+str(epoch)+'_diff.png',diff_colormap)
    cv2.imwrite("eval/"+str(epoch)+"depth_truth.png",depth_true_colormap)
    del image
    del depth
    del output_n

if __name__ == '__main__':
    main()
