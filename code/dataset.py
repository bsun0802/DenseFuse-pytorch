from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from itertools import chain


class COCO(Dataset):
    def __init__(self, path, sfxes=('.jpg', '.jpeg', '.png'), transform=None):
        self.path = Path(path)
        self.images = list(chain.from_iterable(self.path.rglob('*' + sfx) for sfx in sfxes))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img
