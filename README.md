# PV021 Neural Networks — Fashion-MNIST Classifier

This repository contains my project for the **PV021 Neural Networks** course
at **FI MUNI**. The goal is to implement a feed-forward neural network from
scratch in **C++** — no high-level machine learning, matrix, or autodiff
libraries — and train it with backpropagation to classify images from the
Fashion-MNIST dataset.

Trained on Fashion-MNIST: 28x28 grayscale images of clothing items,
60,000 training examples and 10,000 test examples, split across 10 classes.

**Target:** at least 88% accuracy on the test set, with the whole pipeline
(loading data, training, evaluation, exporting results) finishing within
10 minutes.

## Project structure

```
.
├── data/               # Fashion-MNIST CSV files (train/test vectors and labels)
├── src/                # Neural network source code
├── evaluator/          # Script to check predictions against ground truth
├── run.sh              # Compiles and runs the project end-to-end
└── example_test_predictions.csv
```

## How to compile and run

Everything is driven by a single script:

```bash
./run.sh
```
