# Semi-Supervised Learning for image classification on CIFAR-10

The goal of this project was to study a simple but effective model for Semi-Supervised Learning : **FixMatch. <br>
This [very good blogpost by Amit Chaudhary](https://amitness.com/2020/03/fixmatch-semi-supervised/) explains the main points of the approach, which allows to reach **95% top-1 accuracy** for classification by **using only 250 labelled images** (25 randomly selected images per class) out of the 60 000 of the CIFAR-10 dataset.

Of particular interest to me was understanding what made this implementation so effective. Indeed, FixMatch relies on a Wide ResNet, which is not a particulary exotic model, and two key, but classical, principles of Semi-Supervised Learning :
* **Pseudo-labelling** : assignation of "pseudo-label" to unlabelled data after prediction by the model, if the classifier is sufficiently confident about the class to which they belong.
* **Consistency regularization** : the perturbation of an image must not modify its label : a severely perturbed image must therefore be classified in the same way as a weakly perturbed image.

I reimplemented a FixMatch version specific to CIFAR-10 with 250 labelled images by using [kekmodel's Github](https://github.com/kekmodel/FixMatch-pytorch) as inspiration. This allowed me to gain a better understanding of the functioning of this algorithm and improve my software engineering skills
* My reimplementation is available in the [SSL_CIFAR](./SSL_CIFAR.py) python file
* The results of my analysis is available in the [Report](./Report.pdf) PDF file
* The Wide ResNet model used is available in the [wideresnet](./wideresnet.py) Python file

## References
1. Kihyuk Sohn et al. *FixMatch : Simplifying Semi-Supervised Learning with Consistency and Confidence*, 2020, eprint : [arXiv:2001.07685](https://arxiv.org/pdf/2001.07685.pdf)
2. Ilya Loshchilov et Frank Hutter. *SGDR : Stochastic Gradient Descent with Warm Restarts*, 2016, eprint : [arXiv:1608.03983](https://arxiv.org/pdf/1608.03983.pdf)
3. Sergey Zagoruyko et Nikos Komodakis. *Wide Residual Networks*, 2016, eprint : [arXiv:1605.07146](https://arxiv.org/pdf/1605.07146.pdf)
4. Twan van Laarhoven. *L2 Regularization versus Batch and Weight Normalization*, 2017, eprint : [arXiv: 1706.05350](https://arxiv.org/pdf/1706.05350.pdf)
5. Nicholas Carlini, Úlfar Erlingsson et Nicolas Papernot. *Distribution Density, Tails, and Outliers in Machine Learning : Metrics and Applications*, 2019, eprint : [arXiv:1910.13427](https://arxiv.org/pdf/1910.13427.pdf)
