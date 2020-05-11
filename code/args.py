from pathlib import Path
import pkg_resources
import sys
from subprocess import check_call


class Args:
    train_path = '../data/COCO/images/train2014'  # 82000+ images
    resized = (256, 256)
    mode = 'L'
    num_channel = 1 if mode == 'L' else 3

    num_epochs = 2
    batch_size = 4
    lr = 1e-4
    lr_decay_step = 5000  # reduce lr if plateau per 20000 images
    lr_decay_factor = 0.97

    w_ssim = 1.  # multiplier for MS_SSIM loss

    device = 1
    seed = 42

    ckptPath = Path('../train-jobs/ckpt')
    ckptPath.mkdir(parents=True, exist_ok=True)

    _installed = {pkg.key for pkg in pkg_resources.working_set}
    if 'pytorch-msssim' not in _installed:
        _python = sys.executable
        check_call(['echo', '[INFO] Install pytorch-msssim'])
        _pipinstall = ['sudo', _python, '-m', 'pip', 'install', 'pytorch-msssim']
        check_call(_pipinstall)
