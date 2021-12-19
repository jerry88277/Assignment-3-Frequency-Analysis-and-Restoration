# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 18:05:55 2021

@author: Jerry
"""

import os
import cv2
# import math
import BM3D
import array
import struct
import numpy as np
import matplotlib.pyplot as plt


from tqdm import tqdm

# In[]

def read_image(path):  #path = os.path.join('data/', images[0])
    file_extention = path.split('.')[-1]
    
    if file_extention == 'bmp':
        f = open(path, 'rb')
        bmp_header_b = f.read(0x36) # read top 54 byte of bmp file 
        bmp_header_s = struct.unpack('<2sI2H4I2H6I', bmp_header_b) # parse data with struct.unpack()
        pixel_index = bmp_header_s[4] - 54
        bmp_rgb_data_b = f.read()[pixel_index:] # read pixels of bmp file 
        list_b = array.array('B', bmp_rgb_data_b).tolist()
        rgb_data_3d_list  = np.reshape(list_b, (bmp_header_s[6], bmp_header_s[7], bmp_header_s[8])).tolist() # reshape pixel with height, width, RGB channel of image
        
        image = []
        for row in range(len(rgb_data_3d_list)):
            image.insert(0, rgb_data_3d_list[row])
        
        
    elif file_extention == 'raw':
        f = open(path, 'rb').read()
        image = reshape_byte(f) # reshape byte
        
    image = np.array(image) # store into np.array
    
    if len(image.shape) != 3:
        image = np.reshape(image, (image.shape[0], image.shape[1]))
    
    return image

def reshape_byte(byte, size=[512,512]): 
    new_img = []
    
    for row in range(size[0]):
        new_img_row = []
        
        for col in range(size[1]):
            new_img_row.append(byte[row * size[1] + col])
            
        new_img.append(new_img_row)
    
    return new_img

# In[] Setting

path = 'data/'

# In[] Task1 Discrete Fourier Transform

def dft_matrix(input_img):
    rows = input_img.shape[0]
    cols = input_img.shape[1]
    t = np.zeros((rows,cols), complex)
    output_img = np.zeros((rows, cols), complex)
    m = np.arange(rows)
    n = np.arange(cols)
    x = m.reshape((rows, 1))
    y = n.reshape((cols, 1))
    for row in tqdm(range(0, rows)):
        M1 = 1j * np.sin(-2*np.pi*y*n/cols) + np.cos(-2*np.pi*y*n/cols)
        t[row] = np.dot(M1, input_img[row])
    for col in tqdm(range(0, cols)):
        M2 = 1j * np.sin(-2*np.pi*x*m/cols) + np.cos(-2*np.pi*x*m/cols)
        output_img[:, col] = np.dot(M2, t[:, col])
    return output_img

def shift_dft(dft): # dft = out_dftma
    h, w = dft.shape
    half_h = h // 2
    
    zone_1 = dft[0:half_h, 0:half_h]
    zone_2 = dft[half_h:, 0:half_h]
    zone_3 = dft[half_h:, half_h:]
    zone_4 = dft[0:half_h, half_h:]
    
    top_zone = np.hstack((zone_3, zone_2))
    bottom_zone = np.hstack((zone_4, zone_1))
    
    shift = np.vstack((top_zone, bottom_zone))
    
    return shift


task1_imagelist = ['baboon', 'F16', 'lena', 'Noisy']

save_path = 'output/task1'
if not os.path.exists(save_path):
    os.makedirs(save_path)

## save original images
for i_image_name in task1_imagelist:
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    image = read_image(i_image_path)
    
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(f'original_{i_image_name}.raw')
    plt.savefig(os.path.join(save_path, f'{i_image_name}.png'))
    plt.close()

## save DFT images
for i_image_name in tqdm(task1_imagelist):
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    image = read_image(i_image_path)
    
    DFT_image = dft_matrix(image)
    out_dftma = np.log(np.abs(DFT_image))
    
    plt.imshow(out_dftma, cmap='gray')
    plt.axis('off')
    plt.title(f'DFT_{i_image_name}.raw')
    plt.savefig(os.path.join(save_path, f'DFT_{i_image_name}.png'))
    plt.close()
    
    
    out_dftma_shift = shift_dft(out_dftma)
    
    plt.imshow(out_dftma_shift, cmap='gray')
    plt.axis('off')
    plt.title(f'Shift_DFT_{i_image_name}.raw')
    plt.savefig(os.path.join(save_path, f'Shift_DFT_{i_image_name}.png'))
    plt.close()
    
    hist, _ = np.histogram(image, bins=256, range=(0,255))
    plt.bar(np.arange(256), hist)
    plt.xticks([])
    plt.text(x = np.min(hist), y = -np.max(hist)*0.05, s = 'lowest frequency', wrap = False)
    print(np.min(hist))
    plt.title(f'Histrogram of {i_image_name}.raw')
    plt.savefig(os.path.join(save_path, f'Histrogram of DFT_{i_image_name}.png'))
    plt.close()
    
    


# In[] Low-pass filter

def add_noisy(image, noise_typ):
    if noise_typ == 'gauss&uniform':
        h, w = image.shape
        mean = 5
        var = 0.1
        sigma = var**0.5
        prob = 0.2
        levels = int((prob * 255) // 2)
        unform = np.random.randint(-levels, levels, (h, w))
        gauss = np.random.normal(mean, sigma, (h, w))
        gauss = gauss.reshape(h, w)
        noisy = image + gauss + unform
        return noisy
    
    elif noise_typ == 's&p':
        row, col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1
          
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    

def idft_matrix(dft_image):
    rows = dft_image.shape[0]
    cols = dft_image.shape[1]
    t = np.zeros((rows,cols), complex)
    inverse_img = np.zeros((rows, cols), complex)
    m = np.arange(rows)
    n = np.arange(cols)
    x = m.reshape((rows, 1))
    y = n.reshape((cols, 1))
    for row in tqdm(range(0, rows)):
        M1 = 1j * np.sin(2*np.pi*y*n/cols) + np.cos(2*np.pi*y*n/cols)
        t[row] = np.dot(M1, dft_image[row])
    for col in tqdm(range(0, cols)):
        M2 = 1j * np.sin(2*np.pi*x*m/cols) + np.cos(2*np.pi*x*m/cols)
        inverse_img[:, col] = np.dot(M2, t[:, col])
    return inverse_img

def ishift_dft(dft): # dft = out_dftma
    h, w = dft.shape
    half_h = h // 2
    
    zone_1 = dft[0:half_h, 0:half_h]
    zone_2 = dft[half_h:, 0:half_h]
    zone_3 = dft[half_h:, half_h:]
    zone_4 = dft[0:half_h, half_h:]
    
    top_zone = np.hstack((zone_3, zone_2))
    bottom_zone = np.hstack((zone_4, zone_1))
    
    shift = np.vstack((top_zone, bottom_zone))
    
    return shift

def check_limit(image):
    
    image[image > 255] = 255
    image[image < 0] = 0
    
    return image

def frequency_filter(image, D0, N=2, type='lp', filter='butterworth'):
    '''
    频域滤波器
    Args:
        img: 灰度图片
        D0: 截止频率
        N: butterworth的阶数(默认使用二阶)
        type: lp-低通 hp-高通
        filter:butterworth、ideal、Gaussian即巴特沃斯、理想、高斯滤波器
    Returns:
        imgback：滤波后的图像
    '''
    # 离散傅里叶变换
    dft = dft_matrix(image)
    # 中心化
    dtf_shift = shift_dft(dft)

    rows, cols = image.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 计算频谱中心
    mask = np.zeros((rows, cols))  # 生成rows行cols列的二维矩阵

    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) # 计算D(u,v)
            if (filter.lower() == 'butterworth'):  # 巴特沃斯滤波器
                if (type == 'lp'):
                    mask[i, j] = 1 / (1 + (D / D0) ** (2 * N))
                elif (type == 'hp'):
                    mask[i, j] = 1 / (1 + (D0 / D) ** (2 * N))
                else:
                    assert ('type error')
            elif (filter.lower() == 'ideal'):  # 理想滤波器
                if (type == 'lp'):
                    if (D <= D0):
                        mask[i, j] = 1
                elif (type == 'hp'):
                    if (D > D0):
                        mask[i, j] = 1
                else:
                    assert ('type error')
            elif (filter.lower() == 'gaussian'):  # 高斯滤波器
                if (type == 'lp'):
                    mask[i, j] = np.exp(-(D * D) / (2 * D0 * D0))
                elif (type == 'hp'):
                    mask[i, j] = (1 - np.exp(-(D * D) / (2 * D0 * D0)))
                else:
                    assert ('type error')

    fshift = dtf_shift * mask

    f_ishift = ishift_dft(fshift)
    img_back = idft_matrix(f_ishift)
    # img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])  # 计算像素梯度的绝对值
    img_back = np.abs(img_back)
    # img_back = (img_back - np.amin(img_back)) / (np.amax(img_back) - np.amin(img_back))
    return img_back

task2_imagelist = ['baboon', 'F16', 'lena']

save_path = 'output/task2'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
for i_image_name in tqdm(task2_imagelist):
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    image = read_image(i_image_path)
    
    noisy_image = add_noisy(image, 'gauss&uniform')
    noisy_image = check_limit(noisy_image)
    
    plt.imshow(noisy_image, cmap='gray')
    plt.axis('off')
    plt.title(f'Noisy_{i_image_name}.raw')
    plt.savefig(os.path.join(save_path, f'Noisy_{i_image_name}.png'))
    plt.close()
    
    D0 = 100
    denoisy_image_1 = frequency_filter(noisy_image, D0, type='lp', filter='ideal')
    plt.imshow(denoisy_image_1, cmap='gray')
    plt.axis('off')
    plt.title(f'ILPF(D = {D0})_{i_image_name}.raw')
    plt.savefig(os.path.join(save_path, f'ILPF(D = {D0})_{i_image_name}.png'))
    plt.close()
    
    
    D0 = 100
    denoisy_image_2 = frequency_filter(noisy_image, D0, type='lp', filter='gaussian')
    plt.imshow(denoisy_image_2, cmap='gray')
    plt.axis('off')
    plt.title(f'GLPF(D = {D0})_{i_image_name}.raw')
    plt.savefig(os.path.join(save_path, f'GLPF(D = {D0})_{i_image_name}.png'))
    plt.close()
    


# In[] High-pass filter

def laplacian_filter(image, kernel_size, stride = 1):
    # Formatted this way for readability
    if kernel_size == 3:
        laplacian_kernel = np.array((
                           	[0, 1, 0],
                           	[1, -4, 1],
                           	[0, 1, 0]), dtype="int")
        
    elif kernel_size == 5:
        laplacian_kernel = np.array((
                            [0, -1, -1, -1, 0],
                            [-1, -1, -1, -1, -1],
                            [-1, -1, 21, -1, -1],
                            [-1, -1, -1, -1, -1],
                            [0, -1, -1, -1, 0]), dtype="int")
        
    elif kernel_size == 7:
        laplacian_kernel = np.array((
                            [0, 0, -1, -1, -1, 0, 0],
                            [0, -1, -3, -3, -3, -1, 0],
                            [-1, -3, 0, 7, 0, -3, -1],
                            [-1, -3, 7, 24, 7, -3, -1],
                            [-1, -3, 0, 7, 0, -3, -1],
                            [0, -1, -3, -3, -3, -1, 0],
                            [0, 0, -1, -1, -1, 0, 0],), dtype="int")
    
    height, width = image.shape
    padding_size = np.floor((height * (stride - 1) - stride + kernel_size) / 2).astype(int)
    
    temp_image = np.zeros([height + 2 * padding_size, height + 2 * padding_size])
    temp_image[padding_size : -padding_size, padding_size : -padding_size] = image
    
    new_image = np.zeros([height, height])
    
    for row in range(height):
        for col in range(height):
            new_image[row, col] = np.sum(temp_image[row : row + kernel_size, col : col + kernel_size] * laplacian_kernel)
    
    return new_image

task3_imagelist = ['baboon', 'F16', 'lena']

save_path = 'output/task3'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
for i_image_name in tqdm(task3_imagelist):
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    image = read_image(i_image_path)

    kernel_size = 7
    image_LF = laplacian_filter(image, kernel_size = kernel_size)
    plt.imshow(image_LF, cmap='gray')
    plt.axis('off')
    plt.title(f'{i_image_name}_kernel size = {kernel_size}')
    plt.savefig(os.path.join(save_path, f'{i_image_name}_LF_{kernel_size}.png'))
    plt.close()

    IHPF_image = frequency_filter(image, 5, type='hp', filter='ideal')
    plt.imshow(IHPF_image, cmap='gray')
    plt.axis('off')
    plt.title(f'IHPF(D = {D0})_{i_image_name}.raw')
    plt.savefig(os.path.join(save_path, f'IHPF(D = {D0})_{i_image_name}.png'))
    plt.close()
    
    BHPF_image = frequency_filter(image, 5, type='hp', filter='butterworth')
    plt.imshow(BHPF_image, cmap='gray')
    plt.axis('off')
    plt.title(f'BHPF(D = {D0})_{i_image_name}.raw')
    plt.savefig(os.path.join(save_path, f'BHPF(D = {D0})_{i_image_name}.png'))
    plt.close()


# In[] Image denoising

def generate_noisy(image, noise_typ):
    if noise_typ == 'gauss&uniform':
        h, w = image.shape
        mean = 5
        var = 0.1
        sigma = var**0.5
        prob = 0.2
        levels = int((prob * 255) // 2)
        unform = np.random.randint(-levels, levels, (h, w))
        gauss = np.random.normal(mean, sigma, (h, w))
        gauss = gauss.reshape(h, w)
        noisy = gauss + unform
        return noisy
    
    elif noise_typ == 's&p':
        row, col = image.shape
        s_vs_p = 0.8
        amount = 0.1
        out = np.zeros(image.shape)
        # out = np.copy(image)
        
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[tuple(coords)] = 1
          
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[tuple(coords)] = 0
        
        plt.imshow(out, cmap = 'gray')
        out_dft = dft_matrix(out)
        out_dft_shift = shift_dft(out_dft)
        plt.imshow(np.log(np.abs(out_dft_shift)), cmap = 'gray')
        
        return out

def create_Gaussian_kernel(kernel_size = 7, sigma = 1):
    
    half_length = kernel_size // 2
    x, y = np.mgrid[-half_length : half_length + 1, -half_length : half_length + 1]
    gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp((-(x**2+y**2)) / (2 * sigma)) * 1500
    gaussian_kernel = gaussian_kernel.astype(np.uint8)
    
    return gaussian_kernel


def inverse_filter(image, kernel): 
    k_h, k_w = kernel.shape
    kernel_padded = np.zeros_like(image)
    kernel_padded[0:k_h, 0:k_w] = kernel
    kernel_padded = kernel_padded / np.sum(kernel_padded)
    
    image_dft_shift = dft_matrix(image)
    
    kernel_dft_shift = dft_matrix(kernel_padded)
    
    restoration_image_dft_shift = image_dft_shift / kernel_dft_shift
    restoration_image = np.abs(idft_matrix(restoration_image_dft_shift))
    
    return restoration_image

def wiener_filter(image, kernel, eps, K = 0.01):
    k_h, k_w = kernel.shape
    kernel_padded = np.zeros_like(image)
    kernel_padded[0:k_h, 0:k_w] = kernel
    kernel_padded = kernel_padded / np.sum(kernel_padded)
    
    image_dft_shift = dft_matrix(image)
    
    kernel_dft_shift = dft_matrix(kernel_padded) + eps
    kernel_dft_shift_1 = np.conj(kernel_dft_shift) / (np.abs(kernel_dft_shift)**2 + K)
    
    restoration_image_dft_shift = image_dft_shift / kernel_dft_shift_1
    restoration_image = np.abs(idft_matrix(restoration_image_dft_shift))
    
    return restoration_image

def padding_constant(image, pad_size, constant_value=0):
    """
    Padding with constant value.
    Parameters
    ----------
    image: NDArray
        Image to padding. Only support 2D(gray) or 3D(color)
    pad_size: tuple
        Padding size for height and width axis respectively
    Returns
    -------
    ret: NDArray
        Image after padding
    """
    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    h, w = pad_size
    ret = np.zeros((shape[0]+2*h, shape[1]+2*w, shape[2]))
    ret[h:-h, w:-w, :] = image

    ret[:h, :, :] = constant_value
    ret[-h:, :, :] = constant_value
    ret[:, :w, :] = constant_value
    ret[:, -w:, :] = constant_value
    return ret if is_3D else np.squeeze(ret, axis=2)


def padding_reflect(image, pad_size):
    """
    Padding with reflection to image by boarder
    Parameters
    ----------
    image: NDArray
        Image to padding. Only support 2D(gray) or 3D(color)
    pad_size: tuple
        Padding size for height adn width axis respectively
    Returns
    -------
    ret: NDArray
        Image after padding
    """
    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    h, w = pad_size
    ret = np.zeros((shape[0]+2*h, shape[1]+2*w, shape[2]))

    for i in range(shape[0]+2*h):
        for j in range(shape[1]+2*w):
            if i < h:
                if j < w:
                    ret[i, j, :] = image[h-1-i, w-1-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[h-1-i, j-w, :]
                else:
                    ret[i, j, :] = image[h-1-i, w+2*shape[1]-1-j, :]
            elif h <= i <= h + shape[0] - 1:
                if j < w:
                    ret[i, j, :] = image[i-h, w-1-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[i-h, j-w, :]
                else:
                    ret[i, j, :] = image[i-h, w+2*shape[1]-1-j, :]
            else:
                if j < w:
                    ret[i, j, :] = image[h+2*shape[0]-1-i, w-1-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[h+2*shape[0]-1-i, j-w, :]
                else:
                    ret[i, j, :] = image[h+2*shape[0]-1-i, w+2*shape[1]-1-j, :]

    return ret if is_3D else np.squeeze(ret, axis=2)


def padding_reflect_101(image, pad_size):
    """
    Padding with reflection to image by boarder
    Parameters
    ----------
    image: NDArray
        Image to padding. Only support 2D(gray) or 3D(color)
    pad_size: tuple
        Padding size for height adn width axis respectively
    Returns
    -------
    ret: NDArray
        Image after padding
    """
    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    h, w = pad_size
    ret = np.zeros((shape[0]+2*h, shape[1]+2*w, shape[2]))

    for i in range(shape[0]+2*h):
        for j in range(shape[1]+2*w):
            if i < h:
                if j < w:
                    ret[i, j, :] = image[h-i, w-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[h-i, j-w, :]
                else:
                    ret[i, j, :] = image[h-i, w+2*shape[1]-2-j, :]
            elif h <= i <= h + shape[0] - 1:
                if j < w:
                    ret[i, j, :] = image[i-h, w-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[i-h, j-w, :]
                else:
                    ret[i, j, :] = image[i-h, w+2*shape[1]-2-j, :]
            else:
                if j < w:
                    ret[i, j, :] = image[h+2*shape[0]-2-i, w-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[h+2*shape[0]-2-i, j-w, :]
                else:
                    ret[i, j, :] = image[h+2*shape[0]-2-i, w+2*shape[1]-2-j, :]

    return ret if is_3D else np.squeeze(ret, axis=2)


def padding_edge(image, pad_size):
    """
    Padding with edge
    Parameters
    ----------
    image: NDArray
        Image to padding. Only support 2D(gray) or 3D(color)
    pad_size: tuple
        Padding size for height adn width axis respectively
    Returns
    -------
    ret: NDArray
        Image after padding
    """
    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    h, w = pad_size
    ret = np.zeros((shape[0]+2*h, shape[1]+2*w, shape[2]))

    for i in range(shape[0]+2*h):
        for j in range(shape[1]+2*w):
            if i < h:
                if j < w:
                    ret[i, j, :] = image[0, 0, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[0, j-w, :]
                else:
                    ret[i, j, :] = image[0, shape[1]-1, :]
            elif h <= i <= h + shape[0] - 1:
                if j < w:
                    ret[i, j, :] = image[i-h, 0, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[i-h, j-w, :]
                else:
                    ret[i, j, :] = image[i-h, shape[1]-1, :]
            else:
                if j < w:
                    ret[i, j, :] = image[shape[0]-1, 0, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[shape[0]-1, j-w, :]
                else:
                    ret[i, j, :] = image[shape[0]-1, shape[1]-1, :]

    return ret if is_3D else np.squeeze(ret, axis=2)

def to_32F(image):
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(np.float32(image), 0, 1)

def box_filter(I, r, normalize=True, border_type='reflect_101'):
    """
    Parameters
    ----------
    I: NDArray
        Input should be 3D with format of HWC
    r: int
        radius of filter. kernel size = 2 * r + 1
    normalize: bool
        Whether to normalize
    border_type: str
        Border type for padding, includes:
        edge        :   aaaaaa|abcdefg|gggggg
        zero        :   000000|abcdefg|000000
        reflect     :   fedcba|abcdefg|gfedcb
        reflect_101 :   gfedcb|abcdefg|fedcba
    Returns
    -------
    ret: NDArray
        Output has same shape with input
    """
    I = I.astype(np.float32)
    shape = I.shape
    assert len(shape) in [2, 3], \
        "I should be NDArray of 2D or 3D, not %dD" % len(shape)
    is_3D = True

    if len(shape) == 2:
        I = np.expand_dims(I, axis=2)
        shape = I.shape
        is_3D = False

    (rows, cols, channels) = shape

    tmp = np.zeros(shape=(rows, cols+2*r, channels), dtype=np.float32)
    ret = np.zeros(shape=shape, dtype=np.float32)

    # padding
    if border_type == 'reflect_101':
        I = padding_reflect_101(I, pad_size=(r, r))
    elif border_type == 'reflect':
        I = padding_reflect(I, pad_size=(r, r))
    elif border_type == 'edge':
        I = padding_edge(I, pad_size=(r, r))
    elif border_type == 'zero':
        I = padding_constant(I, pad_size=(r, r), constant_value=0)
    else:
        raise NotImplementedError

    I_cum = np.cumsum(I, axis=0) # (rows+2r, cols+2r)
    tmp[0, :, :] = I_cum[2*r, :, :]
    tmp[1:rows, :, :] = I_cum[2*r+1:2*r+rows, :, :] - I_cum[0:rows-1, :, :]

    I_cum = np.cumsum(tmp, axis=1)
    ret[:, 0, :] = I_cum[:, 2*r, :]
    ret[:, 1:cols, :] = I_cum[:, 2*r+1:2*r+cols, :] - I_cum[:, 0:cols-1, :]
    if normalize:
        ret /= float((2*r+1) ** 2)

    return ret if is_3D else np.squeeze(ret, axis=2)


def guided_filter(I, p, radius, eps):
    """
    Parameters
    ----------
    p: NDArray
        Filtering input of 2D
    Returns
    -------
    q: NDArray
        Filtering output of 2D
    """
    # step 1
    I = to_32F(I)
    meanI  = box_filter(I, r = radius)
    meanp  = box_filter(p, r = radius)
    corrI  = box_filter(I * I, r = radius)
    corrIp = box_filter(I * p, r = radius)
    # step 2
    varI   = corrI - meanI * meanI
    covIp  = corrIp - meanI * meanp
    # step 3
    a      = covIp / (varI + eps)
    b      = meanp - a * meanI
    # step 4
    meana  = box_filter(I=a, r = radius)
    meanb  = box_filter(I=b, r = radius)
    # step 5
    q = meana * I + meanb

    return q



task4_imagelist = ['Noisy']

save_path = 'output/task4'
if not os.path.exists(save_path):
    os.makedirs(save_path)

gaussian_kernel = create_Gaussian_kernel(kernel_size = 7, sigma = 1)


# radius_list = [2, 8, 32]
radius_list = [32]
# eps_list = [0.01, 0.16, 0.36]
eps_list = [0.36]


for i_image_name in tqdm(task4_imagelist):
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    image = read_image(i_image_path)
    
    # noisy = generate_noisy(image, 's&p')
    
    # inverse_image = inverse_filter(image, noisy)
    inverse_image = inverse_filter(image, gaussian_kernel)
    
    plt.imshow(inverse_image, cmap='gray')
    plt.axis('off')
    plt.title(f'IF_{i_image_name}.raw')
    plt.savefig(os.path.join(save_path, f'IF_{i_image_name}.png'))
    plt.close()

    wiener_image = wiener_filter(image, gaussian_kernel, eps = 1e-3)
    
    plt.imshow(wiener_image, cmap='gray')
    plt.axis('off')
    plt.title(f'WF_{i_image_name}.raw')
    plt.savefig(os.path.join(save_path, f'WF_{i_image_name}.png'))
    plt.close()
    
    Basic_img = BM3D.BM3D_1st_step(image)
    Final_img = BM3D.BM3D_2nd_step(Basic_img, image)
    
    plt.imshow(Final_img, cmap='gray')
    plt.axis('off')
    plt.title(f'BM3D_{i_image_name}.raw')
    plt.savefig(os.path.join(save_path, f'BM3D_{i_image_name}.png'))
    plt.close()
    
    # cv2.imwrite(os.path.join(save_path, f'BM3D_{i_image_name}.png'), Final_img)
    
    # maxx = np.max(gaussian_kernel)
    # out = guidedfilter(gaussian_kernel/maxx, image, 150, 0.04)
    
    for i_radius in radius_list:
        for i_eps in eps_list:
            out = guided_filter(image, image, i_radius, i_eps)
            plt.imshow(Final_img, cmap='gray')
            plt.axis('off')
            plt.title(f'GF_{i_image_name}_R{i_radius}_E{i_eps}.raw')
            plt.savefig(os.path.join(save_path, f'GF_{i_image_name}_R{i_radius}_E{i_eps}.png'))
            plt.close()
            
            # cv2.imwrite(os.path.join(save_path, f'GF_{i_image_name}_R{i_radius}_E{i_eps}.png'), out)
    
    

# In[] DCT as the image restoration domain
def image_dct(image):
    m, n = image.shape
    C_temp = np.zeros(image.shape)
    dct = np.zeros(image.shape)
    
    image_f32 = image.astype('float')
    
    N = n
    C_temp[0, :] = 1 * np.sqrt(1/N)
     
    for i in range(1, m):
         for j in range(n):
              C_temp[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N)
    ) * np.sqrt(2 / N )
     
    dct = np.dot(C_temp , image_f32)
    dct = np.dot(dct, np.transpose(C_temp))
     
    dct1= np.log(abs(dct))  #进行log处理
    
    # D0 = 5.5
    # crow, ccol = int(m / 2), int(n / 2)
    # mask = np.zeros((m, n))  # 生成rows行cols列的二维矩阵

    # for i in range(m):
    #     for j in range(n):
    #         D = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) # 计算D(u,v)
    #         mask[i, j] = np.exp(-(D * D) / (2 * D0 * D0))
            
    #         # if (D <= D0):
    #             # mask[i, j] = 1

    # dct = dct * mask
    
    img_recor = np.dot(np.transpose(C_temp), dct)
    dct_image = np.dot(img_recor, C_temp)
    
    return dct_image, dct1

def block_image_dct(image_f32):
    height, width = image_f32.shape[:2]
    block_y = height // 8
    block_x = width // 8
    height_ = block_y * 8
    width_ = block_x * 8
    img_f32_cut = image_f32[:height_, :width_]
    img_dct = np.zeros((height_, width_), dtype = np.float32)
    new_img = img_dct.copy()
    for h in range(block_y):
        for w in range(block_x):
            # # 对图像块进行dct变换
            # img_block = img_f32_cut[8 * h: 8 * (h+1), 8 * w : 8 * (w+1)]
            # img_dct[8*h : 8 * (h+1), 8 * w : 8 * (w+1)] = cv2.dct(img_block)

            # # 进行 idct 反变换
            # dct_block = img_dct[8 * h : 8 * (h+1), 8 * w : 8 * (w+1)]
            # img_block = cv2.idct(dct_block)
            # new_img[8 * h : 8 * (h+1), 8 * w : 8 * (w+1)] = img_block
            
            img_block = img_f32_cut[8 * h: 8 * (h+1), 8 * w : 8 * (w+1)]
            new_img[8 * h : 8 * (h+1), 8 * w : 8 * (w+1)], img_dct[8*h : 8 * (h+1), 8 * w : 8 * (w+1)] = image_dct(img_block)
            
    img_dct_log2 = np.log(abs(img_dct))
    return img_dct_log2, new_img


task5_imagelist = ['Noisy']

save_path = 'output/task5'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i_image_name in tqdm(task5_imagelist):
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    image = read_image(i_image_path)
    
    img_dct_log, new_img = block_image_dct(image.astype(np.float))
    
    plt.imshow(img_dct_log, cmap='gray')
    plt.axis('off')
    plt.title(f'DCT_{i_image_name}.raw')
    plt.savefig(os.path.join(save_path, f'DCT_{i_image_name}.png'))
    plt.close()
    
    plt.imshow(new_img, cmap='gray')
    plt.axis('off')
    # plt.title(f'DCT_{i_image_name}.raw')
    plt.savefig(os.path.join(save_path, f'new_{i_image_name}.png'))
    plt.close()



# In[] other test

test_image = cv2.imread(r'D:\NCKU\Course\Digital Image Processing And Computer vision\HW3\image-restoration-master\kernels\5.bmp')

plt.imshow(test_image)



