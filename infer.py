import argparse
import torch
import torchaudio
from tqdm import tqdm
import data_loader.data_loaders as module_data
# import data_loader.numpy_dataset as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch

import model.diffusion as module_diffusion
import model.network as module_network
from evaluate_results import evaluate

from parse_config import ConfigParser

torch.backends.cudnn.benchmark = True


def main(config):
    logger = config.get_logger('infer')

    # setup data_loader instances

    sample_rate = config['sample_rate']

    infer_dataset = config.init_obj('infer_dataset', module_data, sample_rate=sample_rate, T=config['num_samples'] )
    infer_data_loader = config.init_obj('infer_data_loader', module_data, infer_dataset)

    logger.info('Finish initializing datasets')

    # build model architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = config.init_obj('diffusion', module_diffusion, device=device)
    network = config.init_obj('network', module_network, num_samples=config['num_samples'])
    model = config.init_obj('arch', module_arch, diffusion, network)
    # prepare model for testing
    model = model.to(device)
    model.eval()
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)



    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])

    total_loss = 0.0

    sample_path = config.save_dir/'samples'
    sample_path.mkdir(parents=True, exist_ok=True)

    target_path = sample_path/'target'
    output_path = sample_path/'output'
    condition_path = sample_path/'condition'
    target_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    condition_path.mkdir(parents=True, exist_ok=True)

    n_samples = len(infer_data_loader)
    with torch.no_grad():
        for i, (target, condition, index) in tqdm(enumerate(infer_data_loader), desc='infer process', total=n_samples):
            target, condition = target.to(device), condition.to(device)

            # infer from conditional input only

            output = model.infer(condition)

            # save samples, or do something with output here

            real_batch_size = target.shape[0]
            batch_index_temp = []
            previous_index = -1
            for b in range(real_batch_size):
                ind = index[b]
                if ind == previous_index:
                    batch_index_temp.append(b)
                    continue
                else:
                    if previous_index > -1:
                        name = infer_dataset.getName(previous_index)
                        # stack back to full audio
                        torchaudio.save(output_path/f'{name}.wav',
                                        output[batch_index_temp, :, :].view(1, -1).cpu(), sample_rate)
                        torchaudio.save(target_path/f'{name}.wav',
                                        target[batch_index_temp, :, :].view(1, -1).cpu(), sample_rate)
                        torchaudio.save(condition_path/f'{name}.wav',
                                        condition[batch_index_temp, :, :].view(1, -1).cpu(), sample_rate)

                    batch_index_temp = [b]
                    previous_index = ind


            # computing loss, metrics on test set

            loss = loss_fn(output, target)
            total_loss += loss.item()

    log = {'loss': total_loss / n_samples}
    logger.info(log)

    # evaluate results
    metrics = {'pesq_wb', 'sisnr', 'stoi'}
    datatype = config['infer_dataset']['args']['datatype']
    evaluate(sample_path, datatype, sample_rate, metrics, logger)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
