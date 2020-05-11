from pathlib import Path


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

    w_ssim = 2  # multiplier for MS_SSIM loss

    device = 0
    seed = 42

    path_to_ckpt = Path('../train-jobs/ckpt')
    path_to_ckpt.mkdir(parents=True, exist_ok=True)
