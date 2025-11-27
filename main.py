import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox


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
    return np.array(blocks), new_H, new_W, padded_image

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

def decompress_image(indices, codebook, block_h, block_w, H, W):
    blocks = np.array([codebook[i] for i in indices])
    reconstructed = np.zeros((H, W), dtype=np.float32)
    idx = 0
    for i in range(0, H, block_h):
        for j in range(0, W, block_w):
            block = blocks[idx].reshape((block_h, block_w))
            reconstructed[i:i+block_h, j:j+block_w] = block
            idx += 1
    return reconstructed

def save_codebook(codebook, padded_H, padded_W, filename="codebook.txt"):
    with open(filename, "w") as f:
        f.write(f"{int(padded_H)} {int(padded_W)}\n")
        np.savetxt(f, codebook, fmt='%.4f')

def load_codebook(filename="codebook.txt"):
    with open(filename, "r") as f:
        first_line = f.readline()
        padded_H, padded_W = map(int, first_line.split())
        codebook = np.loadtxt(f, dtype=np.float32)
    return codebook, padded_H, padded_W

def save_compressed(indices, filename="compressed.txt"):
    np.savetxt(filename, indices, fmt='%d')

def load_compressed(filename="compressed.txt"):
    return np.loadtxt(filename, dtype=int)

def compression_ratio(original_image, indices, codebook):
    original_size = original_image.size
    codebook_size = codebook.size
    compressed_size = indices.size + codebook_size
    return original_size / compressed_size

def mean_squared_error(img1, img2):
    return np.mean((img1 - img2) ** 2)


class VQ_GUI:
    def __init__(self, master):
        self.master = master
        master.title("Vector Quantization Compression")
        master.geometry("1400x700")
        master.resizable(False, False)

        self.img_array = None

        self.control_frame = tk.Frame(master)
        self.control_frame.pack(side=tk.LEFT, padx=20, pady=20, anchor='n')

        self.load_button = tk.Button(self.control_frame, text="Load Image", width=22, command=self.load_image)
        self.load_button.pack(pady=5)

        tk.Label(self.control_frame, text="Block Height:").pack(pady=(10,0))
        self.block_h_entry = tk.Entry(self.control_frame, width=10)
        self.block_h_entry.insert(0, "8")
        self.block_h_entry.pack()

        tk.Label(self.control_frame, text="Block Width:").pack(pady=(10,0))
        self.block_w_entry = tk.Entry(self.control_frame, width=10)
        self.block_w_entry.insert(0, "8")
        self.block_w_entry.pack()

        tk.Label(self.control_frame, text="Number of Codevectors:").pack(pady=(10,0))
        self.num_vectors_entry = tk.Entry(self.control_frame, width=10)
        self.num_vectors_entry.insert(0, "256")
        self.num_vectors_entry.pack()

        self.compress_button = tk.Button(self.control_frame, text="Compress & Reconstruct", width=28, command=self.compress_and_reconstruct)
        self.compress_button.pack(pady=15)

        self.info_label = tk.Label(self.control_frame, text="", justify=tk.LEFT)
        self.info_label.pack(pady=10)

        self.image_frame = tk.Frame(master)
        self.image_frame.pack(side=tk.RIGHT, padx=20, pady=20)

        self.left_image_frame = tk.Frame(self.image_frame)
        self.left_image_frame.pack(side=tk.LEFT, padx=10)

        self.original_label = tk.Label(self.left_image_frame, text="Original Image")
        self.original_label.pack()
        self.original_canvas = tk.Label(self.left_image_frame, bd=2, relief="sunken")
        self.original_canvas.pack(pady=5)

        self.right_image_frame = tk.Frame(self.image_frame)
        self.right_image_frame.pack(side=tk.LEFT, padx=10)

        self.reconstructed_label = tk.Label(self.right_image_frame, text="Reconstructed Image")
        self.reconstructed_label.pack()
        self.reconstructed_canvas = tk.Label(self.right_image_frame, bd=2, relief="sunken")
        self.reconstructed_canvas.pack(pady=5)


    def display_image(self, array, canvas, max_size=600):

        img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        w, h = img.size
        scale = min(max_size / w, max_size / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        tk_img = ImageTk.PhotoImage(img)
        canvas.imgtk = tk_img
        canvas.configure(image=tk_img)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select Image")
        if not file_path:
            return
        img = Image.open(file_path).convert("L")
        self.img_array = np.array(img, dtype=np.float32)

        self.display_image(self.img_array, self.original_canvas)
        self.info_label.config(text="Image loaded successfully.")

    def compress_and_reconstruct(self):

        block_h = int(self.block_h_entry.get())
        block_w = int(self.block_w_entry.get())
        num_codevectors = int(self.num_vectors_entry.get())

        if block_h <= 0 or block_w <= 0 or num_codevectors <= 0:
            messagebox.showerror("Error", "Block sizes and number of codevectors must be positive.")
            return

        blocks, padded_H, padded_W, padded_image = divide_into_blocks(self.img_array, block_h, block_w)


        codebook = lbg_algorithm(blocks, num_codevectors)


        indices = compress_image(blocks, codebook)


        save_codebook(codebook, padded_H, padded_W)
        save_compressed(indices)


        reconstructed = decompress_image(indices, codebook, block_h, block_w, padded_H, padded_W)


        H_orig, W_orig = self.img_array.shape
        reconstructed_cropped = reconstructed[:H_orig, :W_orig]


        self.display_image(self.img_array, self.original_canvas)
        self.display_image(reconstructed_cropped, self.reconstructed_canvas)


        Image.fromarray(np.clip(reconstructed_cropped, 0, 255).astype(np.uint8)).save("reconstructed.png")


        ratio = compression_ratio(self.img_array, indices, codebook)
        mse = mean_squared_error(self.img_array, reconstructed_cropped)
        self.info_label.config(text=f"Compression Ratio: {ratio:.2f}\nMSE: {mse:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    gui = VQ_GUI(root)
    root.mainloop()
