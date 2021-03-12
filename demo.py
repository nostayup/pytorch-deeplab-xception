import os
import cv2
import time
import argparse
import numpy as np
from PIL import Image
from modeling.deeplab import *
from eval_metrics import *
from dataloaders import custom_transforms as tr
from torchvision import transforms


class Tester(object):
    def __init__(self, args):
        # define network
        self.args = args
        print('--- model initing and loading weights ---')
        self.model = DeepLab(
                num_classes=21,
            backbone=self.args.backbone,
            output_stride=self.args.out_stride,
            sync_bn=self.args.sync_bn,
            freeze_bn=self.args.freeze_bn
        )

        # load weights
        self.model.load_state_dict(torch.load(self.args.weights)['state_dict'])

        # using cuda
        if self.args.cuda:
            self.model.cuda(1)
        self.model.train(False)
        self.model.eval()
        print('--- model initing down ------------------')

    def transform_ts(self, sample):
        composed_transforms = transforms.Compose([
            # tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def infer(self, sample):
        image, label = sample['image'], sample['label']
        image = image.unsqueeze(0)
        if self.args.cuda:
            image = image.cuda(1)
        with torch.no_grad():
            pred = self.model(image)
        
        pred = pred.data.cpu().numpy()  #output.shape = 1,21,w,h
        pred = np.argmax(pred, axis=1)  #output.shape = 1,w,h
        print(pred.shape)
        pred = np.where(pred == 1, 255, 0)
        pred = pred.transpose((1,2,0))
        return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Testing")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--weights', type=str, default='weights/deeplab-resnet.pth', help='weights path')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--sync-bn', type=bool, default=True,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                    testing (default: auto)')
    parser.add_argument('--no-cuda', action='store_true', default=
    False, help='disables CUDA training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    test = Tester(args)

    image_dir = 'data/JPEGImages'
    result_dir = 'data/results_test'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    img_list = os.listdir(image_dir)
    for i, item in enumerate(img_list):
        image = Image.open(os.path.join(image_dir, item)).convert('RGB')
        label = Image.open(os.path.join(image_dir, item))
        sample = {'image': image, 'label': label}
        sample = test.transform_ts(sample)
        start_time = time.time()
        pred = test.infer(sample)
        end_time = time.time()
        print('[%d/%d]Infering %s Using Time: %.4f' % (i+1, len(img_list), item, (end_time - start_time)))
        cv2.imwrite(os.path.join(result_dir, item), pred)

        #控制测试图片数量
        if i > 10:
            break

    gt_dir = 'data/GTs'
    result_list = os.listdir(result_dir)
    sum_miou = 0
    sum_acc = 0
    for i, item in enumerate(result_list):
        imgPredict = cv2.imread(os.path.join(result_dir, item))
        imgLabel = cv2.imread(os.path.join(gt_dir, item))
        imgPredict = cv2.resize(imgPredict, (1920, 1080))
        imgLabel = cv2.resize(imgLabel, (1920, 1080))
        imgLabel = np.where(imgLabel > 0, 1, 0)
        imgPredict = np.where(imgPredict >= 255, 1, 0)

        metric = SegmentationMetric(2)
        metric.addBatch(imgPredict, imgLabel)
        acc = metric.pixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        sum_miou += mIoU
        sum_acc += acc
        print('[%d/%d]%s : mIoU %.4f | acc %.4f' % ((i+1), len(result_list), item, mIoU, acc))
    print('Average mIoU: %.4f' % (sum_miou / len(result_list)))
    print('Average acc : %.4f' % (sum_acc / len(result_list)))
