import numpy as np
import os
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from mypath import Path
from dataloaders import custom_transforms as tr
from PIL import Image

class FashionDataset(Dataset):

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('fashion_clothes'),
                 mode='train'
                 ):

        self.args = args
        self._base_dir = base_dir
        self._mode = mode
        self._image_dir = os.path.join(self._base_dir, mode,'images')
        self._cat_dir = os.path.join(self._base_dir, mode, 'labels')
        self.nclass = 7

        print("dataset image path {}".format(self._image_dir))
        print("dataset labels path {}".format(self._cat_dir))


    def __len__(self):
        return 300

    def __getitem__(self, idx):

        if type(idx) != int:
            idx = int(idx.item())

        idx = idx + 1
        if (self._mode == 'test'):
            idx+= 300

        labelImage = '_clothes.png'

        originalImagePath = os.path.join(self._image_dir, str(idx).zfill(4) + '.jpg')
        labelImagePath = os.path.join(self._cat_dir, str(idx).zfill(4) + labelImage)


        _img = Image.open(originalImagePath).convert('RGB')

        label_img = self.ground_truth(labelImagePath)

        _target = Image.fromarray(label_img, mode='P')

        sample = {'image': _img, 'label': _target}

        if self._mode == "train":
            return self.transform_tr(sample)
        elif self._mode == 'test':
            return self.transform_val(sample)


    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    @staticmethod
    def convert2lable(p):
        """
        :param: image path
        :return: pixel class
        """
        background = (0, 0, 0)  # 0
        skin = (128, 0, 0)  # 1
        hair = (0, 128, 0)
        tshirt = (128, 128, 0)
        shoes = (0, 0, 128)
        pants = (128, 0, 128)
        dress = (0, 128, 128)

        if p == background:
            return 0
        elif p == skin:
            return 1
        elif p == hair:
            return 2
        elif p == tshirt:
            return 3
        elif p == shoes:
            return 4
        elif p == pants:
            return 5
        elif p == dress:
            return 6


    def ground_truth(self, img_file):
        """
        :param img_file: image path
        :return: label each piexl
        """
        # read the file
        img = io.imread(img_file)

        if (img.shape[-1] == 4):
            img = img[:, :, :-1]

        background = np.array([0, 0, 0])  # 0
        skin = np.array([128, 0, 0])  # 1
        hair = np.array([0, 128, 0])
        tshirt = np.array([128, 128, 0])
        shoes = np.array([0, 0, 128])
        pants = np.array([128, 0, 128])
        dress = np.array([0, 128, 128])

        label_seg = np.zeros((img.shape[:2]), dtype=np.uint8)
        label_seg[(img == background).all(axis=2)] = np.array([0])
        label_seg[(img == skin).all(axis=2)] = np.array([1])
        label_seg[(img == hair).all(axis=2)] = np.array([2])
        label_seg[(img == tshirt).all(axis=2)] = np.array([3])
        label_seg[(img == shoes).all(axis=2)] = np.array([4])
        label_seg[(img == pants).all(axis=2)] = np.array([5])
        label_seg[(img == dress).all(axis=2)] = np.array([6])

        return label_seg.reshape(600, 400)


if __name__ == "__main__":
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    dataset = FashionDataset(args, Path.db_root_dir('fashion_clothes'), 'train')

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()

            tmp = np.array(gt[jj]).astype(np.uint8)

            segmap = decode_segmap(tmp, dataset='fashion_clothes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)
        break

    plt.show(block=True)