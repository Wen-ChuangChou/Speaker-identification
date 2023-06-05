# Speaker Identification

## Task
The task at hand is to identify speakers in the voice recordings from the [VoxCeleb2 Dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) using the transformer encoder.
Identifying speakers of the voice from  by the encoder of the transformer.

## Dataset
- Label: Each voice recording is assigned a label corresponding to one of the 600 speakers (classes).
- Training data: There are 56,666 voice recordings with assigned labels for training.
- Testing data: There are 4,000 voice recordings without labels for testing.

## Voice prepocessing
Each voice recording is represented as a mel-spectrogram with 40 dimensions. Because the length of each voice recording varies, a fixed segment of 128x15ms is randomly extracted from each recording to serve as input data.

## Model architecture

## Results
The accurancy rate reaches 53.94% when one transformer encoder layer is implemented.
![image](https://github.com/Wen-ChuangChou/Speaker-identification/blob/main/doc/fig/acc_1_transform_layer.png?raw=true)

The accurancy rate reaches 66.49% when two transformer encoder layers are implemented.
![image](https://github.com/Wen-ChuangChou/Speaker-identification/blob/main/doc/fig/acc_2_transform_layers.png?raw=true)


## Requirements
PyTorch 1.8.1  
tqdm 4.65.0

## Acknowledgment
I an grateful to the Center for Information Services and High Performance Computing [Zentrum für Informationsdienste und Hochleistungsrechnen (ZIH)] at TU Dresden for providing its facilities for high throughput calculations. Part of code snippets are retrieved from the assignment of machine learning course lectured by Prof. Hung-yi Lee at National Taiwan University.

## Reference
1. A. Vaswani et al., “Attention Is All You Need,” arXiv:1706.03762, Jun 2017. https://arxiv.org/abs/1706.03762
2. A. Gulati et al., “Conformer: Convolution-augmented Transformer for Speech Recognition,” arXiv:2005.08100 , May 2020, https://arxiv.org/abs/2005.08100

‌
