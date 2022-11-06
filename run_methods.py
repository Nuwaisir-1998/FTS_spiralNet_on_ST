# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import numpy as np
from imageio import imread
from imageio import imwrite
import os
import random
from PIL import Image as im


# %%
def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()


# %%
def make_row_col_power_of_2(x):
    n = x.shape[0]
    m = x.shape[1]
    n_new = next_power_of_2(n)
    m_new = next_power_of_2(m)
    extra_n = n_new - n
    extra_m = m_new - m
    x_new = np.pad(x, ((extra_n//2, extra_n-extra_n//2), (extra_m//2, extra_m-extra_m//2), (0, 0)), 'constant')
    return x_new


# %%
def normalize_and_scale_and_reshape(x):
    x = make_row_col_power_of_2(x)
    mins = x.min(axis=2)
    new_matrix = x - mins[:, :, np.newaxis]
    sums = new_matrix.sum(axis=2)
    sums += 1
    new_matrix = x / sums[:, :, np.newaxis]
    return np.round(new_matrix * 255).astype('uint8')


# %%
# v = np.array([
#     [[1,2,3], [3,4,1]],
#     [[1,5,3], [4,3,1]]
# ])
# # normalize_and_scale(v)
# # np.argwhere(v == 2)
# a = np.array([[1, 2, 3], [1, 4, 5]])
# v = make_row_col_power_of_2(v)
# v.shape
# # a.shape


# %%
def make_dlpfc_dataset_images():
    samples = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']
    for sample in samples:
        pixel_npy = np.load(f'/content/drive/MyDrive/Nuwaisir/Thesis_updated/ScribbleSeg_updated/Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/Human_DLPFC/{sample}/Npys/mapped_15.npy')
        pixel_npy_pc3 = pixel_npy[:, :, :3]
        pixel_npy_pc3 = normalize_and_scale_and_reshape(pixel_npy_pc3)
        data = im.fromarray(pixel_npy_pc3)
        data.save(f'./datasets/Human_DLPFC_3pc_images/{sample}.png')


# %%
# make_dlpfc_dataset_images()


# %%
def make_dlpfc_dataset():
    samples = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']
    for sample in samples:
        pixel_npy = np.load(f'/content/drive/MyDrive/Nuwaisir/Thesis_updated/ScribbleSeg_updated/Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/Human_DLPFC/{sample}/Npys/mapped_15.npy')
        pixel_npy_pc3 = pixel_npy[:, :, :3]
        np.save(f'./datasets/Human_DLPFC/{sample}.npy', pixel_npy_pc3)


# %%
# make_dlpfc_dataset()


# %%
def make_train_list(from_folder, train_list_folder_path, top):
    with open(f'{train_list_folder_path}/train.flist', 'a') as the_file:
        for file_name in os.listdir(from_folder)[:top]:
            abs_path = os.path.abspath(f'{from_folder}/{file_name}')
            the_file.write(f'{abs_path}\n')
        


# %%
def make_mask_list(from_folder, mask_list_folder_path):
    with open(f'{mask_list_folder_path}/mask.flist', 'a') as the_file:
        for file_name in os.listdir(from_folder):
            abs_path = os.path.abspath(f'{from_folder}/{file_name}')
            the_file.write(f'{abs_path}\n')


# %%
# make_train_list('./datasets/Human_DLPFC_3pc_images', './flists/dlpfc', top=10)


# %%
# make_mask_list('./masks/dlpfc', './flists/dlpfc')


# %%
def gen_mask(src_folder_path, file_name, slice_len, dest_folder_path):
    data = imread(f'{src_folder_path}/{file_name}')
    n = data.shape[0]
    m = data.shape[1]
    pos_r = random.randint(0, n - slice_len)
    pos_c = random.randint(0, m - slice_len)
    mask = np.zeros((n, m), dtype='uint8')
    mask[pos_r: pos_r + slice_len, pos_c: pos_c + slice_len] = 255
    file_name_splitted = file_name.split('.')
    new_file_name = file_name_splitted[0] + '_mask.' + file_name_splitted[1]
    imwrite(f'{dest_folder_path}/{new_file_name}', mask)
    # print(data.shape)


# %%
# gen_mask('./datasets/Human_DLPFC_3pc_images', '151507.png', 64, './masks/dlpfc')
# gen_mask('./datasets/Human_DLPFC_3pc_images', '151670.png', 64, './masks/dlpfc')
# gen_mask('./datasets/Human_DLPFC_3pc_images', '151673.png', 64, './masks/dlpfc')
# gen_mask('./datasets/Human_DLPFC_3pc_images', '151674.png', 64, './masks/dlpfc')


# %%

# m = np.load('/content/drive/MyDrive/Nuwaisir/FTS/spiralnet/checkpoints/SliceGAN/dlpfc_test/result_slice/metrics.npz')
# # for i in m:
# #     print(i)
# print(m['names'])

