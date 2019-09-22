import os
import sys
import cv2
import numpy as np
import torch

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
    from utils import util
except ImportError:
    pass


def generate_mod_LR_bic():
    # set parameters
    up_scale = 4
    mod_scale = 4
    # set data dir
    sourcedir = '/mnt/yjchai/SR_data/Flickr2K/Flickr2K_HR' #'/mnt/yjchai/SR_data/DIV2K_test_HR' #'/mnt/yjchai/SR_data/Flickr2K/Flickr2K_HR'
    savedir = '/mnt/yjchai/SR_data/Flickr2K_train' #'/mnt/yjchai/SR_data/DIV2K_test' #'/mnt/yjchai/SR_data/Flickr2K_train'

    # load PCA matrix of enough kernel
    print('load PCA matrix')
    pca_matrix = torch.load('media/sdc/yjchai/IKC/codes/pca_matrix.pth', map_location=lambda storage, loc: storage)
    print('PCA matrix shape: {}'.format(pca_matrix.shape))

    saveHRpath = os.path.join(savedir, 'HR', 'x' + str(mod_scale))
    saveLRpath = os.path.join(savedir, 'LR', 'x' + str(up_scale))
    saveBicpath = os.path.join(savedir, 'Bic', 'x' + str(up_scale))

    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if not os.path.isdir(os.path.join(savedir, 'HR')):
        os.mkdir(os.path.join(savedir, 'HR'))
    if not os.path.isdir(os.path.join(savedir, 'LR')):
        os.mkdir(os.path.join(savedir, 'LR'))
    if not os.path.isdir(os.path.join(savedir, 'Bic')):
        os.mkdir(os.path.join(savedir, 'Bic'))

    if not os.path.isdir(saveHRpath):
        os.mkdir(saveHRpath)
    else:
        print('It will cover ' + str(saveHRpath))

    if not os.path.isdir(saveLRpath):
        os.mkdir(saveLRpath)
    else:
        print('It will cover ' + str(saveLRpath))

    if not os.path.isdir(saveBicpath):
        os.mkdir(saveBicpath)
    else:
        print('It will cover ' + str(saveBicpath))

    kernel_list = []

    filepaths = sorted([f for f in os.listdir(sourcedir) if f.endswith('.png')])
    print(filepaths)
    num_files = len(filepaths)

    # prepare data with augementation
    for i in range(num_files):
        filename = filepaths[i]
        print('No.{} -- Processing {}'.format(i, filename))
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename))
        # gaussian random kernel
        if up_scale == 2:
            prepro = util.SRMDPreprocessing(up_scale, pca_matrix, para_input=15, kernel=21, noise=False, cuda=True,
                                        sig_min=0.2, sig_max=2.0, rate_iso=1.0, scaling=3,
                                        rate_cln=0.2, noise_high=0.0)
        elif up_scale == 3:
            prepro = util.SRMDPreprocessing(up_scale, pca_matrix, para_input=15, kernel=21, noise=False, cuda=True,
                                        sig_min=0.2, sig_max=3.0, rate_iso=1.0, scaling=3,
                                        rate_cln=0.2, noise_high=0.0)
        elif up_scale == 4:
            prepro = util.SRMDPreprocessing(up_scale, pca_matrix, para_input=15, kernel=21, noise=False, cuda=True,
                                        sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3,
                                        rate_cln=0.2, noise_high=0.0)

        width = int(np.floor(image.shape[1] / mod_scale))
        height = int(np.floor(image.shape[0] / mod_scale))
        # modcrop
        if len(image.shape) == 3:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width, :]
        else:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width]
        image_HR_blurred = cv2.filter2D(image_HR, -1, blur_ker)
        # LR
        image_LR = imresize_np(image_HR_blurred, 1 / up_scale, True)
        # bic
        image_Bic = imresize_np(image_LR, up_scale, True)

        cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)
        cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)
        cv2.imwrite(os.path.join(saveBicpath, filename), image_Bic)
        kernel_list.append(blur_ker)


    print("Down smaple Done: X"+str(up_scale))

if __name__ == "__main__":
    generate_mod_LR_bic()
