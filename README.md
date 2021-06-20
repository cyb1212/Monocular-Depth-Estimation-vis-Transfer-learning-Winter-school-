# Monocular-Depth-Estimation-vis-Transfer-learning-Winter-school-
This project is a learning material for 2021 Winter school on SLAM in deformable environments.
The cited reference is: Paper: Alhashim I, Wonka P. High quality monocular depth estimation via transfer learning[J]. arXiv preprint arXiv:1812.11941, 2018. https://arxiv.org/abs/1812.11941

The network parts of this code has been delated as a potential homework for the winter school. The pre-operations, including training data and data loading, and the later-operations with  have been provided. 
The readers can complete the following steps:

## adding the network part based on the following structure:

Encoder and decorder: the input RGB image is encoded into a feature vector using the DenseNet-169 network [1] pretrained on ImageNet [2].

Network Architecture\
![图片](https://user-images.githubusercontent.com/32351126/122663989-db2df100-d1e1-11eb-8f7c-d87d70a35352.png)

DenseNet-169 network\
![图片](https://user-images.githubusercontent.com/32351126/122664008-f7ca2900-d1e1-11eb-981c-ebd7b372c711.png)

Decoder sub-block\
![图片](https://user-images.githubusercontent.com/32351126/122664144-dd447f80-d1e2-11eb-8752-1076bfc41338.png)

## improving the loss function

The provided code is only based on the point-wise L1 loss defined on the depth values:\
![图片](https://user-images.githubusercontent.com/32351126/122664217-665bb680-d1e3-11eb-89bc-6a14d588d7a4.png)

The readers are encourage to test the other loss in the cited reference including the differences in image gradient and structural similarity (SSIM). Some other loss functions are also encouraged.\
![图片](https://user-images.githubusercontent.com/32351126/122664294-f1d54780-d1e3-11eb-9687-6afba80ffacb.png)
\
![图片](https://user-images.githubusercontent.com/32351126/122664245-a6229e00-d1e3-11eb-9474-60c18fb2ff60.png) 
\
![图片](https://user-images.githubusercontent.com/32351126/122664240-9dca6300-d1e3-11eb-9f92-60864177b6a1.png)

## Adding data augmentation

The data augmentation is not provided in this code. The readers can test some classical augmentation approach for the image dataset, including: Flip, Rotation, Scale, Crop, Translation, Gussian Noise, and Salt-and-pepper Noise, to fully use the offered dataset.\
![图片](https://user-images.githubusercontent.com/32351126/122664412-8fc91200-d1e4-11eb-8f94-3b0010de9adb.png)
![图片](https://user-images.githubusercontent.com/32351126/122664400-7c1dab80-d1e4-11eb-8b6c-f3fc2e7c5079.png)
![图片](https://user-images.githubusercontent.com/32351126/122664499-2eee0980-d1e5-11eb-8f7e-97114d50b158.png)

## Design new encoder and decorder network (optional)

A new network with this encoder and decorder structure is also encouraged. The readers can desin their own network to reach a better performance.


## Reference
[1] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger. Densely connected convolutional networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), pages 2261–2269, 2017. 2, 3, 5, 11
[2] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. 2009 IEEE Conference on Computer Vision and Pattern Recognition,
pages 248–255, 2009. 3, 5
