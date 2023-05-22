import os

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import numpy as np


class PlanktonSegmentationDataset(Dataset):
    def __init__(self, data_dir, annotation_file='_annotations.coco.json', transform=None):
        super(PlanktonSegmentationDataset, self).__init__()
        self.data_dir = data_dir
        self.annotations = COCO(os.path.join(data_dir, annotation_file))
        self.image_ids = self.annotations.getImgIds()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_info = self.annotations.loadImgs(self.image_ids[item])[0]
        image_path = os.path.join(self.data_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        annotation_ids = self.annotations.getAnnIds(imgIds=self.image_ids[item])
        annotations = self.annotations.loadAnns(annotation_ids)
        mask = Image.new('L', image.size)
        for ann in annotations:
            seg_mask = self.annotations.annToMask(ann)
            seg_mask = Image.fromarray(seg_mask.astype('uint8') * 255)
            mask.paste(seg_mask, (0, 0), mask=seg_mask)

        image = np.array(image, dtype=np.float32) / 255
        mask = np.array(mask, dtype=np.float32) / 255

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        return image, mask