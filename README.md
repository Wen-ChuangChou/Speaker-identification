# Speaker-identification

## Task
Identifying speakers of the voice from [VoxCeleb2 Dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) by the encoder of the transformer.

## Dataset
- Label: 600 speakers are used in this project, and each voice is labelled by one speaker (class).
- Training data: 56666 voice with labels are training data.
- Testing data: 4000 voice without labels are testing data.

## Voice prepocessing
Each voice is represented in mel-spectrogram with 40 dimensions. Because the length of each voice is different, fixed segment (128x15ms) is randomly extracted in each voice as input data.

## Model architecture

## Results

## 
