# Source-Denoising-Pix2Pix-cGAN

### Basic Information
---
**Author:** Gregory Hunkins

**Organization:** University of Rochester

**License:** MIT

**Abstract:** An Conditional Generative Adverserial Network (cGAN) was adapted for the task of source de-noising of noise voice auditory images. The base architecture is adapted from [Pix2Pix](https://phillipi.github.io/pix2pix/). The cGAN inputs fixed-length short-time Fourier Transform (STFT) magnitude features of the noisy speech and returns a denoised image that can be converted to the time-domain via IFFT. The dataset is created by randomly combined TIMIT speaker samples and non-stationary noise. For the non-stationary noise, Duan et al.'s dataset is used: http://www2.ece.rochester.edu/~zduan/data/noise/.

### Running The Code
---

Reference: https://cs.rochester.edu/~cxu22/t/577F17/bluehive_tutorial.html

For the most recent architecture, navigate into the `Architecture_v4` folder. Submit job.sh to train the architecture and save results.

```bash
sbatch src/model/job.sh
```
### Subjective Evaluation
---

The full validation set can be downloaded in two different ways: via validation example or seperated into dB noise classes. The first link contains all validation examples with both the appropriate WAV and PNG files. The second link only contains the relevant WAV files. Both are ~2GB.

[Full Validation (WAV and PNG)](https://drive.google.com/file/d/1Q9RVQda8osre-gyCvmQ4d5q-Xjludj7G/view?usp=sharing) and [dB Separated Validation Set (WAV)](https://drive.google.com/file/d/1xJhkaXLT0_APNMek7Nr_MZlK3KCKjs81/view?usp=sharing)


### Data
---
The data is available in HDF5 format. 

[NoisyAudImg10K](NoisyAudImg10K)