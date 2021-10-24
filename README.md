# Problem-Based-AI

## Problem 1 
  [Robot Game](https://github.com/sydneymagee/Problem-Based-AI/tree/master/Problem1)  
  * Simple bot written in Python to play RobotGame
  * https://web.archive.org/web/20160311213016/https://robotgame.net/rules
## Problem 2
  [Terrain Model](https://github.com/sydneymagee/Problem-Based-AI/tree/master/Problem2)
  * Data provided by United States Geographical Services.
  * Data description: topographic map files. Cropped and downsampled images.
  * Deep Learning model for elevation data compression.
  * Trained the model on a single cropped or downsampled image that takes a coordinate and produces an elevation value.
  * Used the minimum description length (MDL) principle to compare the compressed tile to the original tile.
  * Altered hyperparameters to find the lowest mean squared error and greatest improvement.
  * [Terrain Model](https://github.com/sydneymagee/Problem-Based-AI/blob/master/Problem2/Terrain/model.py)
## Problem 3
  [Spotify Loudness Model](https://github.com/sydneymagee/Problem-Based-AI/tree/master/Problem3)
  * Data collected from the Echo Nest Audio Analysis tool provided by Spotify.
  * Data description: 129-vector containing the average absolute frequency spectrum for a segment.
  * Deep Learning Model for average loudness.
  * Altered hyperparameters to find the lowest mean squared error and greatest improvement.
  * [Loudness Model](https://github.com/sydneymagee/Problem-Based-AI/blob/master/Problem3/model0.py)
## Problem 4
  [Bee Pollination Behavior Model](https://github.com/sydneymagee/Problem-Based-AI/tree/master/Problem4)
  * Data description: Squash images with or without bee interaction. 1000 images for training and 400 images for validation.
  * Convolutional Neural Network that classifies squash images according to whether or not a bee is actively interacting with the flower.
  * Altered hyperparameters to find the lowest validation loss, high sensitivity and high positive predictive value.
  * Utilized preprocessing techniques to reduce overfitting.
  * Utilized data augmentation to allow the model to learn from more data.
  * Architecture inspired from LeNet-5 and AlexNet.
  * [Pollination Model](https://github.com/sydneymagee/Problem-Based-AI/blob/master/Problem4/squash1.py)
## Problem 5
  [Natural Language Processing Model](https://github.com/sydneymagee/Problem-Based-AI/tree/master/Problem5)
  * Data derived from Roatan Marine Park. Dr. Sarah Beth Hopton and R. Mitchell Parry classified the data into training and validation datasets.
  * Data description: Roatan Marine Park social media posts that are categorized into three sections:'information', 'community', and 'action'.
  * Convolutional Neural Network for text classification.
  * Utilized a parameter grid to increase testing ability and efficiency.
  * Altered hyperparameters to find the lowest validation loss and high validation accuracy.
  * [Text Model](https://github.com/sydneymagee/Problem-Based-AI/blob/master/Problem5/roatan_sjm.py)
