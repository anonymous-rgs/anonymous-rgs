# Randomized Gradient Adversarial Training

This repository provides an implementation of Randomized Gradient Step adversarial attack and defense method on CIFAR10 dataset with ResNet model. Implementation is based on [MadryLab CIFAR10's challenge](https://github.com/MadryLab/cifar10_challenge). `rgs_attack.py` file provides the Random Gradient Step attack. We have modified `config.json` to take an additional parameter `attack_type` which is either **PGD** or **RGS** and used this parameter in `train.py` to initialize adversarial attack from `pgd_attack.py` or `rgs_attack.py` accordingly.
