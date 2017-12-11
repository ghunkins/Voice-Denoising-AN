from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import h5py
import os
import re
import librosa
from time import gmtime, strftime
from scipy.io.wavfile import read, write

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pylab as plt
from matplotlib.colors import LogNorm, Normalize

norm = Normalize(vmin=0.0,vmax=7.7, clip=True)
# constants
WINDOW_LENGTH = 0.032
HOP_SIZE = 0.016
nperseg = int(WINDOW_LENGTH / (1. / 16000.))
noverlap = int(HOP_SIZE / (1. / 16000.))


def normalization(X):

    return X / 127.5 - 1


def inverse_normalization(X):

    return norm.inverse((X + 1.) / 2.)

def normalization_audio(X):
    #X = X.clip(min=0)
    return (norm(X) * 2.) - 1.

def inverse_normalization_audio(X):
    
    return ((X + 1.) * 7.7)


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

        if image_data_format == "channels_last":
            print "CHANNELS LAST"
            X_clean_train = X_clean_train.transpose(0, 2, 3, 1)
            X_noisy_train = X_noisy_train.transpose(0, 2, 3, 1)

        X_clean_val = hf["clean_val"][:].astype(np.float16)
        X_clean_val = normalization_audio(X_clean_val)

        X_noisy_val = hf["mag_val"][:].astype(np.float16)
        X_noisy_val = normalization_audio(X_noisy_val)

        X_phase_val= hf["phase_val"][:].astype(np.float16)

        if image_data_format == "channels_last":
            X_clean_val = X_clean_val.transpose(0, 2, 3, 1)
            X_noisy_val = X_noisy_val.transpose(0, 2, 3, 1)

        return X_clean_train, X_noisy_train, X_phase_train, X_clean_val, X_noisy_val, X_phase_val


def load_test_audio(size=10000, train_pct=0.8):
    ROOT = '/scratch/ghunkins/Combined/'
    # compile regexes
    mag_r = re.compile('\d+_mag*')
    phase_r = re.compile('\d+_phase_*')
    clean_r = re.compile('\d+_clean_*')
    # get full list of files
    full_dir = os.listdir(ROOT)
    # filter
    mag_dir = filter(mag_r.match, full_dir)
    phase_dir = filter(phase_r.match, full_dir)
    clean_dir = filter(clean_r.match, full_dir)
    # sort
    mag_dir.sort()
    phase_dir.sort()
    clean_dir.sort()
    # get indices
    train_indices = (0, int(size * train_pct))
    test_indices = (train_indices, train_indices + int(size * (1-train_pct)))

    #X_full_train = np.array


def gen_batch(X1, X2, X3, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False)
        yield X1[idx], X2[idx], X3[idx]


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


def plot_generated_batch(X_full, X_sketch, X_phase, generator_model, batch_size, image_data_format, suffix):

    # Generate images
    X_gen = generator_model.predict(X_sketch)

    X_sketch = inverse_normalization_audio(X_sketch)
    X_full = inverse_normalization_audio(X_full)
    X_gen = inverse_normalization_audio(X_gen)

    dir_to_save = "../../figures/" + suffix + "_" + strftime("%Y-%m-%d %H:%M:%S", gmtime())
    os.makedirs(dir_to_save)
    #np.save(dir_to_save + "/{}_noisy.npy".format(suffix), X_sketch)
    #np.save(dir_to_save + "/{}_gen.npy".format(suffix), X_gen)
    #np.save(dir_to_save + "/{}_clean.npy".format(suffix), X_full)

    for i in range(X_gen.shape[0]):
        # seperate relevant parts
        gen = X_gen[i, :, :, 0]
        noisy = X_sketch[i, :, :, 0]
        clean = X_full[i, :, :, 0]
        phase = X_phase[i, :, :, 0]
        # save wav
        c_gen = gen * np.exp(phase * 1j)
        c_noisy = noisy * np.exp(phase * 1j)
        c_clean = clean * np.exp(phase * 1j)
        c_gen = np.append(np.zeros((1, 256)), c_gen, axis=0)
        c_noisy = np.append(np.zeros((1, 256)), c_noisy, axis=0)
        c_clean = np.append(np.zeros((1, 256)), c_clean, axis=0)
        print c_gen.shape, c_noisy.shape, c_clean.shape

        f_gen = open(dir_to_save + '/{}_gen{}.wav'.format(suffix, str(i)), 'w')
        f_noisy = open(dir_to_save + '/{}_noisy{}.wav'.format(suffix, str(i)), 'w')
        f_clean = open(dir_to_save + '/{}_clean{}.wav'.format(suffix, str(i)), 'w')
        y_gen = librosa.istft(c_gen, hop_length=noverlap, win_length=nperseg, window="hamming")
        y_noisy = librosa.istft(c_noisy, hop_length=noverlap, win_length=nperseg, window="hamming")
        y_clean = librosa.istft(c_clean, hop_length=noverlap, win_length=nperseg, window="hamming")
        write(f_gen, 16000, y_gen)
        write(f_noisy, 16000, y_noisy)
        write(f_clean, 16000, y_clean)
        # save figures
        plt.pcolormesh(gen, cmap="gnuplot2")
        plt.colorbar()
        plt.savefig(dir_to_save + '/{}_gen{}.png'.format(suffix, str(i)))
        plt.clf()
        plt.pcolormesh(noisy, cmap="gnuplot2")
        plt.colorbar()
        plt.savefig(dir_to_save + '/{}_noisy{}.png'.format(suffix, str(i)))
        plt.clf()
        plt.pcolormesh(clean, cmap="gnuplot2")
        plt.colorbar()
        plt.savefig(dir_to_save + '/{}_clean{}.png'.format(suffix, str(i)))
        plt.clf()

    return

    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    np.save("../../figures/current_batch_{}_{}.npy".format('noisy', time), X_sketch)
    np.save("../../figures/current_batch_{}_{}.npy".format('gen', time), X_gen)
    np.save("../../figures/current_batch_{}_{}.npy".format('clean', time), X_full)

    for i in range(X_gen.shape[0]):
        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    #np.save("../../figures/current_batch_%s.png" % suffix)

    Xs = X_sketch[:8]
    Xg = X_gen[:8]
    Xr = X_full[:8]

    np.save("../../figures/current_batch_{}_{}.npy".format('noisy', suffix), X_sketch)
    np.save("../../figures/current_batch_{}_{}.npy".format('gen', suffix), X_gen)
    np.save("../../figures/current_batch_{}_{}.npy".format('clean', suffix), X_full)

    if image_data_format == "channels_last":
        X = np.concatenate((Xs, Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] // 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=0)

    if image_data_format == "channels_first":
        X = np.concatenate((Xs, Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] // 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=1)
        Xr = Xr.transpose(1,2,0)

    if Xr.shape[-1] == 1:
        #plt.imshow(Xr[:, :, 0], cmap="gray")
        plt.pcolormesh(Xr[:, :, 0], cmap="gnuplot2")
    else:
        print "NOT IN PCOLORMESH"
        plt.imshow(Xr)
    plt.axis("off")
    plt.savefig("../../figures/current_batch_%s.png" % suffix)
    plt.clf()
    plt.close()
