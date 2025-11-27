from PIL import Image
import numpy as np


def load_image(path):

    img = Image.open(path).convert('L')

    # Convert image to numpy array
    img_array = np.array(img, dtype=np.float32)

    return img_array



def divide_into_blocks(image, block_h, block_w):
    H, W = image.shape

    pad_h = (block_h - (H % block_h)) % block_h
    pad_w = (block_w - (W % block_w)) % block_w

    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    new_H, new_W = padded_image.shape
    blocks = []
    for i in range(0, new_H, block_h):
        for j in range(0, new_W, block_w):
            block = padded_image[i:i + block_h, j:j + block_w]
            blocks.append(block.flatten())

    return np.array(blocks), new_H, new_W




def lbg_algorithm(blocks, num_codevectors, epsilon=1e-5):

    codebook = [np.mean(blocks, axis=0)]

    while len(codebook) < num_codevectors:

        new_codebook = []
        for vec in codebook:
            new_codebook.append(vec * (1 + epsilon))
            new_codebook.append(vec * (1 - epsilon))
        codebook = new_codebook

        prev_distortion = float('inf')
        while True:

            distances = np.linalg.norm(blocks[:, None] - codebook, axis=2)
            indices = np.argmin(distances, axis=1)


            new_codebook = []
            distortion = 0
            for i in range(len(codebook)):
                assigned_blocks = blocks[indices == i]
                if len(assigned_blocks) > 0:
                    new_vec = np.mean(assigned_blocks, axis=0)
                    new_codebook.append(new_vec)
                    distortion += np.sum((assigned_blocks - new_vec) ** 2)
                else:
                    new_codebook.append(codebook[i])
            codebook = new_codebook
            distortion /= blocks.shape[0]


            if abs(prev_distortion - distortion) / distortion < epsilon:
                break
            prev_distortion = distortion

    return np.array(codebook)


def compress_image(blocks, codebook):
    distances = np.linalg.norm(blocks[:, None] - codebook, axis=2)
    indices = np.argmin(distances, axis=1)
    return indices


def save_codebook(codebook, filename="codebook.txt"):
    np.savetxt(filename, codebook, fmt='%.4f')


def save_compressed(indices, filename="compressed.txt"):
    np.savetxt(filename, indices, fmt='%d')

