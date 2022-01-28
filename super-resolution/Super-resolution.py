import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio as im
from PIL import Image
from scipy import ndimage
from torchvision import transforms
import torchvision
from torchvision.utils import save_image
import skimage
from skimage import metrics
import matplotlib.pyplot as plt

impath = 'path_to/butterfly_GT.bmp'
modelpath = 'path_to/model.pth'

"""Part 1: Defining the functions used in pre-processing the image. Bicubic interpolation is used to to create a low resolution image for use by SRCNN later on."""

def imread(path, is_grayscale):

    """ This reads the image from a given path. Will convert to grayscale (as_gray) if desired and image is read by YCbCr format as the paper."""

    # print both the size and shape of the original (groundtruth) image.
    print("\nOriginal image size", im.imread(path).size)
    print("Original image shape:", im.imread(path).shape)

    # convert to grayscale if desired.
    if is_grayscale == True:

        return im.imread(path, as_gray=True, pilmode='YCbCr').astype(np.float32)
    else:
        return im.imread(path, pilmode='YCbCr').astype(np.float32)


def imcrop(image, scale):

    """Need to ensure image size/shape is divisible by scale factor before down and up-scaling"""
    
    # np.mod return the remainder
    if len(image.shape) == scale:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image

def preprocess(path, scale, is_grayscale):
    """
    Preprocess single image file
      (1) Read original image as YCbCr format
      (2) Normalize
      (3) Apply image file with bicubic interpolation
    Args:
      path: file path of desired file
      input_: image applied bicubic interpolation (low-resolution)
      label_: image with original resolution (high-resolution)
    """
    image = imread(path, is_grayscale)
    label_ = imcrop(image, scale)

    # Must be normalised. This converts pixel brightness from 0-255 to 0-1.
    label_ = label_ / 255.

    # this scales down image by 'scale' and then scales up the image by 'scale'. Uses bicubic interpolation.
    input_ = ndimage.interpolation.zoom(label_, (1. / scale), prefilter=False)
    input_ = ndimage.interpolation.zoom(input_, (scale / 1.), prefilter=False)
    
    return input_, label_



"""Part 2: Define the model weights and biases."""

# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
# padding is needed to maintain image dimensions. padding = (kernel-1)/2

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, padding=4, padding_mode='replicate')
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0, padding_mode='replicate')
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=2, padding_mode='replicate')

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        return out


# Load the pre-trained model file. Need to include path

model = SRCNN()
model.load_state_dict(torch.load(modelpath))
model.eval()


"""Part 3: Read the test image. Scale is set to 3 and convert to grayscale set to true"""

LR_image, HR_image = preprocess(impath,3,1)

#print('LR shape:', LR_image.shape)
#print('HR shape:', HR_image.shape)

# transform the input to 4-D tensor
input_ = np.expand_dims(np.expand_dims(LR_image, axis=0), axis=0)
input_ = torch.from_numpy(input_)


"""Part 4: Run the model and get the SR image. Save and show all images"""

with torch.no_grad():
    output_ = model(input_)

# check input and output images have same dimensions
print('----------------------------------------------')
print('input_ shape:', input_.shape)
print('output_ shape:', output_.shape)



# select first image from 4D tensor
SR_image=output_[0]

# save images to current folder. SuperRes is SRCNN image, HiRes is groundtruth image and LowRes is bicubic interpolation image.
save_image(SR_image, './SuperRes.png')
save_image(torch.from_numpy(HR_image), './HiRes.png')
save_image(torch.from_numpy(LR_image), './LowRes.png')

# convert from torch tensor to nump array.
SR_image=SR_image[0]
SR_image=SR_image.numpy()

# show images from array. Need to 'un-normalise' pixel brightness.
Image.fromarray(HR_image*255.).show()
Image.fromarray(LR_image*255.).show()
Image.fromarray(SR_image*255.).show()


"""Part 5: comparing image quality using PSNR"""

psnr=skimage.metrics.peak_signal_noise_ratio(HR_image, LR_image)
psnr1=skimage.metrics.peak_signal_noise_ratio(HR_image, SR_image)

print('----------------------------------------------')
print('Ground truth vs bicubic interpolation PSNR:', psnr,'dB')
print('Ground truth vs SRCNN PSNR:', psnr1,'dB')
