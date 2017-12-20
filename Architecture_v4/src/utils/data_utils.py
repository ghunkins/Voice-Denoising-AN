import numpy as np
import h5py
import os
import librosa
from scipy.io.wavfile import write

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pylab as plt
from matplotlib.colors import Normalize

norm = Normalize(vmin=0.0,vmax=7.7, clip=True)
norm2 = Normalize(vmin=-6.0, vmax=1.0, clip=True)
# constants
WINDOW_LENGTH = 0.032
HOP_SIZE = 0.016
nperseg = int(WINDOW_LENGTH / (1. / 16000.))
noverlap = int(HOP_SIZE / (1. / 16000.))


def normalization(X):
    """Normalization for Facades."""
    return X / 127.5 - 1

def inverse_normalization(X):
    """Inverse Normalization for Facades."""
    return norm.inverse((X + 1.) / 2.)

def normalization_audio(X):
    """Normalization for audio STFT."""
    X[X == 0] = np.finfo(dtype='float32').tiny
    X = np.log10(X)
    return (norm2(X) * 2.) - 1. 

def inverse_normalization_audio(X):
    """Inverse normalization for audio STFT."""
    X = ((X * 3.5) - 2.5)
    return np.power(10.0, X)


def get_nb_patch(img_dim, patch_size, image_data_format):

    assert image_data_format in ["channels_first", "channels_last"], "Bad image_data_format"

    if image_data_format == "channels_first":
        assert img_dim[1] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[2] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[1] // patch_size[0]) * (img_dim[2] // patch_size[1])
        img_dim_disc = (img_dim[0], patch_size[0], patch_size[1])

    elif image_data_format == "channels_last":
        assert img_dim[0] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[1] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[0] // patch_size[0]) * (img_dim[1] // patch_size[1])
        img_dim_disc = (patch_size[0], patch_size[1], img_dim[-1])

    return nb_patch, img_dim_disc


def extract_patches(X, image_data_format, patch_size):

    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        X = X.transpose(0,2,3,1)

    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] // patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])

    if image_data_format == "channels_first":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0,3,1,2)

    return list_X


def load_data(dset, image_data_format):

    with h5py.File("../../data/processed/%s_data.h5" % dset, "r") as hf:

        X_full_train = hf["train_data_full"][:].astype(np.float16)
        #X_full_train = normalization(X_full_train)

        X_sketch_train = hf["train_data_sketch"][:].astype(np.float16)
        #X_sketch_train = normalization(X_sketch_train)

        if image_data_format == "channels_last":
            X_full_train = X_full_train.transpose(0, 2, 3, 1)
            X_sketch_train = X_sketch_train.transpose(0, 2, 3, 1)

        X_full_val = hf["val_data_full"][:].astype(np.float16)
        #X_full_val = normalization(X_full_val)

        X_sketch_val = hf["val_data_sketch"][:].astype(np.float16)
        #X_sketch_val = normalization(X_sketch_val)

        if image_data_format == "channels_last":
            X_full_val = X_full_val.transpose(0, 2, 3, 1)
            X_sketch_val = X_sketch_val.transpose(0, 2, 3, 1)

        return X_full_train, X_sketch_train, X_full_val, X_sketch_val

def load_data_audio(dset, image_data_format):
    with h5py.File("../../../../%s_data.h5" % dset, "r") as hf:

        X_clean_train = hf["clean_train"][:].astype(np.float16)
        X_clean_train = normalization_audio(X_clean_train)

        X_noisy_train = hf["mag_train"][:].astype(np.float16)
        X_noisy_train = normalization_audio(X_noisy_train)

        X_phase_train = hf["phase_train"][:].astype(np.float16)

        X_str_train = hf["train_str"][:]

        if image_data_format == "channels_last":
            X_clean_train = X_clean_train.transpose(0, 2, 3, 1)
            X_noisy_train = X_noisy_train.transpose(0, 2, 3, 1)
            X_phase_train = X_phase_train.transpose(0, 2, 3, 1)

        X_clean_val = hf["clean_val"][:].astype(np.float16)
        X_clean_val = normalization_audio(X_clean_val)

        X_noisy_val = hf["mag_val"][:].astype(np.float16)
        X_noisy_val = normalization_audio(X_noisy_val)

        X_phase_val= hf["phase_val"][:].astype(np.float16)

        X_str_val = hf["val_str"][:]

        if image_data_format == "channels_last":
            X_clean_val = X_clean_val.transpose(0, 2, 3, 1)
            X_noisy_val = X_noisy_val.transpose(0, 2, 3, 1)
            X_phase_val = X_phase_val.transpose(0, 2, 3, 1)

        return X_clean_train, X_noisy_train, X_phase_train, X_str_train, \
               X_clean_val, X_noisy_val, X_phase_val, X_str_val


def gen_batch(X1, X2, X3, X4, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False)
        yield X1[idx], X2[idx], X3[idx], X4[idx]


def get_disc_batch(X_full_batch, X_sketch_batch, generator_model, batch_counter, patch_size,
                   image_data_format, label_smoothing=False, label_flipping=0):

    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        X_disc = generator_model.predict(X_sketch_batch)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        X_disc = X_full_batch
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form X_disc
    X_disc = extract_patches(X_disc, image_data_format, patch_size)

    return X_disc, y_disc


def plot_generated_batch(X_full, X_sketch, X_phase, X_str, generator_model, batch_size, image_data_format, suffix, dirs):

    # Generate images
    X_gen = generator_model.predict(X_sketch)

    # Inverse Normalization
    X_sketch = inverse_normalization_audio(X_sketch)
    X_full = inverse_normalization_audio(X_full)
    X_gen = inverse_normalization_audio(X_gen)

    # Directory to save
    dir_to_save = dirs + suffix
    if not os.path.isdir(dir_to_save):
        os.makedirs(dir_to_save)

    for i in range(X_gen.shape[0]):
        # example folder
        dir_i = dir_to_save + '/' + str(i+1)
        if not os.path.isdir(dir_i):
            os.makedirs(dir_i)
        # seperate relevant parts
        gen = X_gen[i, :, :, 0]
        noisy = X_sketch[i, :, :, 0]
        clean = X_full[i, :, :, 0]
        phase = X_phase[i, :, :, 0]
        # re-compute the complex
        c_gen = gen * np.exp(phase * 1j)
        c_noisy = noisy * np.exp(phase * 1j)
        c_clean = clean * np.exp(phase * 1j)
        # transform from (256, 256) --> (257, 256)
        c_gen = np.append(np.zeros((1, 256)), c_gen, axis=0)
        c_noisy = np.append(np.zeros((1, 256)), c_noisy, axis=0)
        c_clean = np.append(np.zeros((1, 256)), c_clean, axis=0)
        # open files, compute ISTFT, and write WAV
        f_gen = open(dir_i + '/{}_{}_gen{}.wav'.format(X_str[i], suffix, str(i)), 'w')
        f_noisy = open(dir_i + '/{}_{}_noisy{}.wav'.format(X_str[i], suffix, str(i)), 'w')
        f_clean = open(dir_i + '/{}_{}_clean{}.wav'.format(X_str[i], suffix, str(i)), 'w')
        y_gen = librosa.istft(c_gen, hop_length=noverlap, win_length=nperseg, window="hamming")
        y_noisy = librosa.istft(c_noisy, hop_length=noverlap, win_length=nperseg, window="hamming")
        y_clean = librosa.istft(c_clean, hop_length=noverlap, win_length=nperseg, window="hamming")
        write(f_gen, 16000, y_gen)
        write(f_noisy, 16000, y_noisy)
        write(f_clean, 16000, y_clean)
        # save figures in log format
        plt.pcolormesh(np.log10(gen), cmap="gnuplot2")
        plt.colorbar()
        plt.savefig(dir_i + '/{}_gen{}.png'.format(suffix, str(i)))
        plt.clf()
        plt.pcolormesh(np.log10(noisy), cmap="gnuplot2")
        plt.colorbar()
        plt.savefig(dir_i + '/{}_noisy{}.png'.format(suffix, str(i)))
        plt.clf()
        plt.pcolormesh(np.log10(clean), cmap="gnuplot2")
        plt.colorbar()
        plt.savefig(dir_i + '/{}_clean{}.png'.format(suffix, str(i)))
        plt.clf()

    return
