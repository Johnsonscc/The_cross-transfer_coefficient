import numpy as np
import imageio.v2 as iio
from skimage.io import imsave
from skimage.color import rgb2gray

def load_image(path,grayscale=True):
    image = iio.imread(path)
    if grayscale and len(image.shape)>2:#将彩色通道图像转化为灰度图
        image = rgb2gray(image)
    return image

def binarize_image(image):
    threshold = 0.5 * np.max(image)
    return (image>threshold).astype(np.uint8)#threshold为设定阈值，对图像进行二值化

def save_image(image,path):
    if image.dtype != np.uint8:
        image_normalized = (255*((image - image.min())/(image.max()-image.min())).astype(np.uint8))#归一化处理
        imsave(path, image_normalized)
    else:
        imsave(path,image)