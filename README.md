# Semi-Supervised Learning for image classification on CIFAR-10

The goal of this project was to study a simple but efficient model for Semi-Supervised Learning : [FixMatch](https://arxiv.org/pdf/2001.07685.pdf). <br>
This [very good blogpost by Amit Chaudhary](https://amitness.com/2020/03/fixmatch-semi-supervised/) explains the main points of the approach, which allows to reach 95% top-1 accuracy for classification by using only 250 labelled images (25 randomly selected images per class) out of the 60 000 of the dataset.
