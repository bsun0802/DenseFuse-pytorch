import time

from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader

from args import Args
from dataset import *
from model import *
from utils import *

from pytorch_msssim import MS_SSIM


def main():
    torch.cuda.manual_seed_all(Args.seed)

    train_transform = transforms.Compose([
        transforms.Resize(Args.resized),
        transforms.Grayscale(Args.num_channel),
        transforms.ToTensor()
    ])
    coco_train = COCO(Args.train_path, transform=train_transform)
    trainloader = DataLoader(coco_train, batch_size=Args.batch_size, shuffle=True,
                             num_workers=min(4, Args.batch_size), pin_memory=True)
    loaders = {'train': trainloader}

    model = DenseFuse(num_channel=Args.num_channel)
    model.to(Args.device)

    optimizer = optim.Adam(model.parameters(), lr=Args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     patience=1, factor=Args.lr_decay_factor)

    ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=1)
    criterions = {'ms_ssim': ms_ssim}

    ckpt_name = time.ctime().replace(' ', '-').replace(':', '-')
    ckptPath = Args.ckptPath.joinpath(ckpt_name)
    train(loaders, model, criterions, optimizer, Args.num_epochs, ckptPath,
          scheduler=scheduler)


def train(loaders, model, criterions, optimizer, num_epochs, ckptPath, **kwargs):
    print('{} -- Start Training, Params: (LR, Weight Decay) = {}'.format(
        time.ctime(),
        str([(group['lr'], group['weight_decay']) for group in optimizer.param_groups])))

    scheduler = kwargs.get('scheduler', None)

    model.train()
    best_loss = float('inf')
    history = {'mse': [], 'ssim': [], 'total': []}

    for epoch in range(num_epochs):
        logger = Logger(len(history))
        num_batches = len(loaders['train'])
        for i, batch in enumerate(loaders['train']):
            optimizer.zero_grad()

            batch = batch.to(Args.device)
            enc = model.encoder(batch)
            dec = model.decoder(enc)

            mse_loss = F.mse_loss(dec, batch)
            ssim_loss = 1 - criterions['ms_ssim'](dec, batch)
            loss = mse_loss + Args.w_ssim * ssim_loss

            logger.update([l.item() for l in (mse_loss, ssim_loss, loss)])

            loss.backward()
            optimizer.step()

            if (i + 1) % logger.interval == 0:  # log loss history
                history['mse'].append(mse_loss.item())
                history['ssim'].append(ssim_loss.item())
                history['total'].append(loss.item())
                print(logger.TRAIN_INFO.format(
                    logger.now(), epoch + 1, num_epochs, i, num_batches,
                    ', '.join(['{:.3f}'.format(l * 1e3) for l in logger.val]), '1000x',
                    ', '.join(['{:.3f}'.format(l * 1e3) for l in logger.avg]), '1000x'))

            if (i + 1) % (200 * logger.interval) == 0:  # save checkpoint and loss
                is_best = loss.item() < best_loss
                best_loss = min(loss.item(), best_loss)
                save_ckpt({
                    'epoch': epoch,
                    'iter': i,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, ckptPath, is_best, epoch, i)

                if is_best:
                    loss_fp = ckptPath.joinpath('best_model_loss_hist.npz')
                    np.savez_compressed(str(loss_fp), mse=np.array(history['mse']),
                                        ssim=np.array(history['ssim']), total=np.array(history['total']))

            gstep = epoch * num_batches + (i + 1)
            if scheduler and gstep % 5000 == 0:
                scheduler.step(loss.item())


if __name__ == '__main__':
    main()
