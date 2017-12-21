import sys
sys.path.append("../utils")
import data_utils
import os
from keras.models import load_model

os.environ["KERAS_BACKEND"] = "tensorflow"
image_data_format = "channels_last"
batch_size = 4
save_dir = '/scratch/ghunkins/DCGAN_RESULTS/'
dset = 'audio_10000_new'

generator_model = load_model('/scratch/ghunkins/DCGAN_RESULTS/DCGAN_RUN_2017-12-21_03:59:11/GENERATOR.h5')


X_full_train, X_sketch_train, X_phase_train, X_str_train, X_full_val, X_sketch_val, X_phase_val, X_str_val = data_utils.load_data_audio(dset, image_data_format)


data_utils.plot_generated_batch(X_full_val, X_sketch_val, X_phase_val, X_str_val, generator_model,
                                    batch_size, image_data_format, "Full_Validation", save_dir)