# Using a transformer encoder to identify speakers

## Task
The task at hand is to identify speakers in the voice recordings from the [VoxCeleb2 Dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) using the transformer encoder.

## Dataset
- Label: Each voice recording is assigned a label corresponding to one of the 600 speakers (classes).
- Training data: 90% of the 56,666 voice recordings are used for training and have assigned labels.
- Validating data: 10% of the 56,666 voice recordings are used for validation and have assigned labels.
<!--- Testing data: There are 4,000 voice recordings without labels for testing.-->

## Voice prepocessing
Each voice recording is represented as a mel-spectrogram with 40 dimensions. Because the length of each voice recording varies, a fixed segment of 128x15ms is randomly extracted from each recording to serve as input data.

## Model architecture

## Results
The accuracy rate reaches **53.94%** when implementing only one layer of the transformer encoder.  
![image](https://github.com/Wen-ChuangChou/Speaker-identification/blob/main/doc/fig/acc_1_transform_layer.png?raw=true)

When implementing two layers of the transformer encoder, the accuracy rate increases to **66.49%**.  
![image](https://github.com/Wen-ChuangChou/Speaker-identification/blob/main/doc/fig/acc_2_transform_layers.png?raw=true)

By implementing two layers of the transformer encoder and increasing the number of expected features in the input (d_model) from 80 to 256, the accuracy rate further improves to **72.95%**.  
![image](https://github.com/Wen-ChuangChou/Speaker-identification/blob/main/doc/fig/acc_2_transform_layers_increase_d_model.png?raw=true)

Maintaining the above hyperparameters but increasing the number of heads from 4 to 64 results in an accuracy rate of **77.24%**.  
![image](https://github.com/Wen-ChuangChou/Speaker-identification/blob/main/doc/fig/acc_2_transform_layers_increase_d_model_increase_head.png?raw=true)

It is important to note that further increasing the values of the above hyperparameters does not significantly improve the performance of the model.

## Requirements
PyTorch 1.8.1  
tqdm 4.65.0

## Acknowledgment
I an grateful to the Center for Information Services and High Performance Computing [Zentrum für Informationsdienste und Hochleistungsrechnen (ZIH)] at TU Dresden for providing its facilities for high throughput calculations. Part of code snippets are retrieved from the assignment of machine learning course lectured by Prof. Hung-yi Lee at National Taiwan University.

## Reference
1. A. Vaswani et al., “Attention Is All You Need,” arXiv:1706.03762, Jun 2017. https://arxiv.org/abs/1706.03762
2. A. Gulati et al., “Conformer: Convolution-augmented Transformer for Speech Recognition,” arXiv:2005.08100 , May 2020, https://arxiv.org/abs/2005.08100

‌
