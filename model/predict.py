import torch
import argparse
import os
from dataloaders.utils import decode_seg_map_sequence
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from modeling.deeplab import DeepLab

def transform_val(sample):
    composed_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    return composed_transforms(sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Segmentation")

    parser.add_argument('--input', '-i', metavar='input_path',
                        help='Input image ', required=True)

    parser.add_argument('--output', '-o', metavar='output_path',
                        help='Output image', required=True)

    args = parser.parse_args()

    dataset = "fashion_clothes"
    path = "./bestmodels/deep_clothes/checkpoint.pth.tar"
    nclass = 7

    #Initialize the DeeplabV3+ model
    model = DeepLab(num_classes=nclass, output_stride=8)

    #run model on CPU
    model.cpu()
    torch.set_num_threads(8)

    #error checking
    if not os.path.isfile(path):
        raise RuntimeError("no model found at'{}'".format(path))

    if not os.path.isfile(args.input):
        raise RuntimeError("no image found at'{}'".format(input))

    if os.path.exists(args.output):
        raise RuntimeError("Existed file or dir found at'{}'".format(args.output))

    #load model
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    #load image
    img = Image.open(args.input).convert('RGB').resize((400, 600), Image.BILINEAR)
    img = transform_val(img)

    #notify all your layers that you are in eval mode, that way, 
    #batchnorm or dropout layers will work in eval mode instead of training mode.
    model.eval()

    print("Start Processing")
    with torch.no_grad():
        img = torch.unsqueeze(img, 0)
        output = model(img)
        prediction = decode_seg_map_sequence(torch.max(output[:], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset)

        prediction = prediction.squeeze(0)
        save_image(prediction, args.output, normalize=False)
        print("Finished Processing")
        print("Output Saved")


