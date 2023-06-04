# Speaker-identification

## Task
Identifying speakers of the voice from [VoxCeleb2 Dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) by the encoder of the transformer.

## Dataset
- Label: Each voice is labelled by one of 600 speakers (class).
- Training data: 56666 voice with labels are training data.
- Testing data: 4000 voice without labels are testing data.

## Voice prepocessing
Each voice is represented in mel-spectrogram with 40 dimensions. Because the length of each voice is different, fixed segment (128x15ms) is randomly extracted in each voice as input data.

## Model architecture

## Results
The best results of accurancy rate so far reaches 53.94%

![image](https://github.com/Wen-ChuangChou/Speaker-identification/blob/main/results.png)

## Acknowledgment
I an grateful to the Center for Information Services and High Performance Computing [Zentrum für Informationsdienste und Hochleistungsrechnen (ZIH)] at TU Dresden for providing its facilities for high throughput calculations. Part of code snippets are retrieved from the assignment of machine learning course lectured by Prof. Hung-yi Lee at National Taiwan University.

## Reference
1. A. Vaswani et al., “Attention Is All You Need,” arXiv:1706.03762, Jun 2017. https://arxiv.org/abs/1706.03762
2. A. Gulati et al., “Conformer: Convolution-augmented Transformer for Speech Recognition,” arXiv:2005.08100 , May 2020, https://arxiv.org/abs/2005.08100

‌
