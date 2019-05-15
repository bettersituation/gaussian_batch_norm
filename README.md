# Rigid Batch Norm


### Data
1. Mnist
2. Cifar10
3. Cifar100
4. Fashion mnist


### MODEL
1. simple
2. small resnet


### Criteria
1. log loss
2. acc (= top 1 erorr)


### FIXED
1. relu
2. sgd
3. xavier


### IDEA
1. There is a possibility that there exist outliers, too big or small value. <br>
If it is, batch mean is corrupted. So I think it is need to varianc