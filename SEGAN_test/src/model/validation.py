import sys
sys.path.append("../utils")
import data_utils
import os
from keras.models import load_model

os.environ["KERAS_BACKEND"] = "tensorflow"
image_data_format = "channels_last"
batch_size = 4
save_dir = '/scratch/ghunkins/SEGAN_RESULTS/'
dset = 'SEGAN'

generator_model = load_model('/scratch/ghunkins/DCGAN_RESULTS/DCGAN_RUN_2017-12-21_03:59:11/GENERATOR.h5')


X_clean_mag_train, X_noisy_mag_train, X_clean_phase_train,\
X_noisy_phase_train, X_str_train, X_clean_mag_test,\
X_noisy_mag_test, X_clean_phase_test, X_noisy_phase_test, X_str_test = data_utils.load_data_audio(dset, image_data_format)


data_utils.plot_generated_batch(X_clean_mag_test, X_noisy_mag_test, X_clean_phase_test,
							    X_noisy_phase_test, X_str_test, generator_model,
                                batch_size, image_data_format, "SEGAN_Validation", save_dir)