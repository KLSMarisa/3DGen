from skimage.metrics import structural_similarity as ssim
import numpy as np
def compare_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def compare_ssim(img1, img2, multichannel=True):
    return ssim(img1, img2, multichannel=multichannel,channel_axis=-1)
