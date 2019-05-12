# Rigid Batch Norm

VERYDEEPCONVOLUTIONALNETWORKSFORLARGE-SCALEIMAGERECOGNITION # https://arxiv.org/pdf/1409.1556.pdf
Batch Normalization: Accelerating Deep Network Training byReducing Internal Covariate Shift # https://arxiv.org/pdf/1502.03167.pdf


### Data
1. Mnist
2. Cifar10
3. Cifar100
4. Fashion mnist


### MODEL
1. VGG16
2. conv 5 * 5 * 32 -> max pool 2 -> conv 5 * 5 * 64 -> max pool 2 -> fc: 1024 -> fc: case


### Criteria
1. log loss
2. acc (= top 1 erorr)


### FIXED
1. 50 epoch
2. relu
3. sgd
4. xavier
5. bn momentum 0.99


### Comparison
1. Normalization: None, BN, RBN
2. lr: 1e-2, 5e-3, 1e-3
3. bound: 5, 7, 10
4. reg cf: 0.5, 1, 1.5, 2
5. batch size: 50, 100, 200
