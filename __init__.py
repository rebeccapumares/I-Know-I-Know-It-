"""TOTNet: a feedforward neural network and backpropagation engine written
from scratch in NumPy, applied to tip-of-the-tongue (TOT) state prediction."""
from .data import (CLASS_NAMES, Preprocessor, encode_labels, load_csv, save_csv,
                   simulate_tot_dataset, split_by_participant)
from .gradcheck import gradient_check
from .layers import Dense, Dropout, LeakyReLU, ReLU, Sigmoid, Tanh
from .losses import BinaryCrossEntropy, SoftmaxCrossEntropy, softmax
from .metrics import accuracy, confusion_matrix, per_class_report, roc_auc
from .network import FeedforwardNet, build_mlp
from .optim import SGD, Adam

__version__ = "0.1.0"
