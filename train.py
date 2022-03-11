import argparse
import collections
import torch
import data_loader.data_loaders as module_data
# import data_loader.numpy_dataset as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.network as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from math import ceil


torch.backends.cudnn.benchmark = True

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances

    tr_dataset = config.init_obj('tr_dataset', module_data, sample_rate=config['sample_rate'], T=config['num_samples'])
    val_dataset = config.init_obj('val_dataset', module_data, sample_rate=config['sample_rate'], T=config['num_samples'])
    tr_data_loader = config.init_obj('data_loader', module_data, tr_dataset)
    val_data_loader = config.init_obj('data_loader', module_data, val_dataset)

    logger.info('Finish initializing datasets')
    #
    device, device_ids = prepare_device(config['n_gpu'])

    logger.info('Finish preparing gpu')

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch, num_samples=config['num_samples'])

    # prepare for (multi-device) GPU training
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    logger.info(model)

    # get function handles of loss and metrics
    criterion = config.init_obj('loss', module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    n_encode_channels = config['arch']['args']['n_encode_channels']
    k1 = config['lr_scheduler']['k1']
    num_warmups = config['lr_scheduler']['num_warmups']
    k2 = config['lr_scheduler']['k2']
    lr_end_of_warmup = k1 * (n_encode_channels ** -0.5) * (num_warmups**-1.5) * num_warmups
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params, lr=lr_end_of_warmup)
    len_epoch = len(tr_data_loader)
    # after num_warmups, the lr is set back to init_lr
    lambda_warmup = lambda step: step/num_warmups
    warmup_scheduler =torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_warmup)
    lambda_epoch = lambda epoch: k2/lr_end_of_warmup * (0.98 ** ((epoch+ceil(num_warmups/len_epoch))//2))
    epoch_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=tr_data_loader,
                      valid_data_loader=val_data_loader,
                      num_warmups = num_warmups,
                      warmup_scheduler=warmup_scheduler,
                      epoch_scheduler=epoch_scheduler
                      )

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Speech denoising diffusion model')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
