This folder contains the details of developing Generative adversarial network to restorate fluorescence lifetime images with few photons.
# FLIM Images Restoration using GAN CNN
## 1. Simulated dataset generation
Two datasets with different photon numbers (peak of the decay curve) will be generated using the same a1, a2, t1, t2

### 1.1 Ideal FLIM Images (photon number at the peak position > 80)
- Fitting Peak 5 ~ 25
- Normalized percentage a1 = 0.65 ~ 0.85 Real a1 = 3 ~ 20
- Normalized percentage a2 = 0.15 ~ 0.35 Real a2 = 1 ~ 8.75
- t1 = 0.15 ~ 0.45
- t2 = 1.5 ~ 3.5

### 1.2 FLIM with fewer photons and low SNR
- Fitting Peak 1 ~ 5
- Normalized percentage a1 = 0.65 ~ 0.85 Real a1 = 0.65 ~ 4.25
- Normalized percentage a2 = 0.15 ~ 0.35 Real a2 = 0.15 ~ 1.75
- t1 = 0.15 ~ 0.45
- t2 = 1.5 ~ 3.5

### 1.3 Notes
- Time: 0 – 12.5ns with 256 time frames
- Use the same a1, a2 (percentage), and t1, t2 values to generate TPSF images.
- The real a1 and a2 of the ideal FLIM are five times the real a1 and a2 of the FLIM with low SNR
- The TPSF images were generated by the equation of α_1 e^(-t/τ_1 )+ α_2 e^(-t/τ_2 ) , and then convoluted with IRF, and added the Poisson noises.
- 3000 of The MINIST dataset (28 x 28) was applied as the original mask.

### 1.4 Results
- Representative simulated decay curve. The photon number of the ideal decay = 2532. The photon number of the few-photon decay = 460

![image](https://github.com/walshlab/FLIM-fit/assets/49083235/0f1c62d4-237e-4f0f-9708-d8248aa30d27)
- FLIM fitting result after deconvolution

![image](https://github.com/walshlab/FLIM-fit/assets/49083235/9b2707ee-ca88-4dc9-8628-5a03ee4a14b3)

Conclusion: Images with fewer photon does not give a good fitting result compared to the images with more photon, and it is necessary to develop a CNN model to restorat the images with fewer photons.

## 2. GAN CNN Development

A [generative adversarial network](https://github.com/walshlab/FLIM-fit/blob/main/GAN%20Model%20Development/FLIM_GAN_Demo5.ipynb) was developed to restorate the images. 
This idea comes from the [paper](https://doi.org/10.1038/s42003-021-02938-w).
### 2.1 Image Preprocessing
The size of MNIST image is 28 x 28, and we will pad the images to be 32 x 32 to fit our FLIM dataset (256 x 256)
### 2.2 CNN architecture illustration
The GAN model was developed based on the [turtoial](https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/).
We applied the classical Pix2Pix GAN to achieve this translation with adjustment to 3D imgeas.

The architecture is comprised of two models: the discriminator and the generator.

The discriminator is a deep convolutional neural network that performs image classification. Specifically, conditional-image classification. It takes both the source image (low photon) and the target image (high photon) as input and predicts the likelihood of whether target image is real or a fake translation of the source image. The discriminator design is based on the effective receptive field of the model, which defines the relationship between one output of the model to the number of pixels in the input image. 

The generator is an encoder-decoder model using a U-Net architecture. The model takes a source image (low photon) and generates a target image (high photon). It does this by first downsampling or encoding the input image down to a bottleneck layer, then upsampling or decoding the bottleneck representation to the size of the output image. 
The encoder and decoder of the generator are comprised of standardized blocks of convolutional, batch normalization, dropout, and activation layers.

![image](https://github.com/walshlab/FLIM-fit/assets/49083235/ac1e2834-befe-4328-8af7-87e1327326d3)

### 2.2 GAN Training
Training involves a fixed number of training iterations. There are 3000 TPSF images in the training dataset. One epoch is one iteration through this number of examples, with a batch size of 10 means 300 training steps. The model will run for 50 epochs, or a total of 15000 training steps.

Each training step involves first selecting a batch of real examples, then using the generator to generate a batch of matching fake samples using the real source images. The discriminator is then updated with the batch of real images and then fake images.

Next, the generator model is updated providing the real source images as input and providing class labels of 1 (real) and the real target images as the expected outputs of the model required for calculating loss. The generator has two loss scores as well as the weighted sum score returned from the call to train_on_batch(). We are only interested in the weighted sum score (the first value returned) as it is used to update the model weights.

Finally, the loss for each update is reported to the console each training iteration.

### 2.3 GAN Evaluation
The trained GAN was saved, and the model was tested with other new 100 TPSF images to visualize its performance.
