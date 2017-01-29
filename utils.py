import os
import sys
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
import msgpack
import time
from sklearn.learning_curve import learning_curve
from colorama import Fore, Back, Style, init as colorama_init

reload(sys)
sys.setdefaultencoding('utf-8')

colorama_init()

code_folder = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(code_folder, '../data/')
chunks_folder = data_folder + 'chunks/'
sents_folder = data_folder + 'sents/'
clean_folder = data_folder + 'clean/'
trained_folder = data_folder + 'trained/'
result_folder = data_folder + 'result/'
raw_data_folder = os.path.join(code_folder, '../raw src data/')
subsets_folder = os.path.join(code_folder, '../raw subsets/')

log_root_folder = ''


class AllSentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            log('reading file ' + fname)
            with open(os.path.join(self.dirname, fname), 'rb') as f:
                sentences = msgpack.unpack(f)
                for sentence in sentences:
                    yield sentence


def get_all_sentences():
    return AllSentences(os.path.join(code_folder, '../raw src data/toks'))


def log(message, color=None):
    txt = "at " + time.strftime('%H:%M:%S') + " " + str(message)
    if not color:
        # %Y-%m-%d
        print txt
    else:
        print color + txt + Style.RESET_ALL

    with open(log_root_folder + 'text.log', 'a') as myfile:
        myfile.write(txt + '\n')


def get_not_none(items):
    return [item for item in items if item is not None]


def create_log_folder(script_file):
    global log_root_folder

    # create log directory
    time_str = time.strftime('%Y_%m_%d--%H_%M_%S')
    log_root_folder = './log/' + time_str + '/'
    os.makedirs(os.path.dirname(log_root_folder))

    # copy current file to log folder in order to see what code was running
    copyfile(script_file, log_root_folder + os.path.basename(script_file))

    return log_root_folder


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes = np.linspace(.1, 1.0, 5)):

    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='f1')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
