import numpy as np
from utils.fileUtil import FileUtil
from matplotlib import pyplot as plt

FILE_PATH = "./Data/history.txt"

def show_figures(train_metric,
                valid_metric,
                title,
                y_label, 
                x_label,
                legend,
                show_figure = True,
                save_file_path = None):
    plt.clf()
    plt.plot(train_metric)
    plt.plot(valid_metric)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(legend, loc='upper left')

    if(save_file_path is not None):
        plt.savefig(save_file_path)

    if(show_figure):
        plt.show()

if __name__ == '__main__':

    file_helper = FileUtil()
    metrics = file_helper.load_file(FILE_PATH)

    
    train_accuracy_metric = []
    valid_accuracy_metric = []
    train_loss_metric = []
    valid_loss_metric = []
    train_time_metric = []
    valid_time_metric = []

    for fold in metrics:
        fold_train_accuracy_metric = metrics[fold]['train_acc']
        fold_valid_accuracy_metric = metrics[fold]['valid_acc']

        fold_train_loss_metric = metrics[fold]['train_loss']
        fold_valid_loss_metric = metrics[fold]['valid_loss']

        fold_train_time_metric = metrics[fold]['train_time']
        fold_valid_time_metric = metrics[fold]['valid_time']


        train_accuracy_metric.append(fold_train_accuracy_metric)
        valid_accuracy_metric.append(fold_valid_accuracy_metric)
        train_loss_metric.append(fold_train_loss_metric)
        valid_loss_metric.append(fold_valid_loss_metric)
        train_time_metric.append(fold_train_time_metric)
        valid_time_metric.append(fold_valid_time_metric)


    avg_train_accuracy_metric = np.sum(train_accuracy_metric, axis=0) / len(train_accuracy_metric)
    avg_valid_accuracy_metric = np.sum(valid_accuracy_metric, axis=0) / len(valid_accuracy_metric)
    avg_train_loss_metric = np.sum(train_loss_metric, axis=0) / len(train_loss_metric)
    avg_valid_loss_metric = np.sum(valid_loss_metric, axis=0) / len(valid_loss_metric)
    avg_train_time_metric = np.sum(train_time_metric, axis=0) / len(train_time_metric)
    avg_valid_time_metric = np.sum(valid_time_metric, axis=0) / len(valid_time_metric)


    print(avg_train_accuracy_metric)
    print(avg_valid_accuracy_metric)
    print(avg_train_loss_metric)
    print(avg_valid_loss_metric)
    print(avg_train_time_metric)
    print(avg_valid_time_metric)


    show_figures(
        train_metric=avg_train_accuracy_metric,
        valid_metric=avg_valid_accuracy_metric,
        title='model accuracy',
        y_label='accuracy',
        x_label='epoch',
        legend=['train', 'val'],
        save_file_path="./Data/accuracy.eps",
        show_figure=False
    )

    show_figures(
        train_metric=avg_train_loss_metric,
        valid_metric=avg_valid_loss_metric,
        title='model loss',
        y_label='loss',
        x_label='epoch',
        legend=['train', 'val'],
        save_file_path="./Data/loss.eps",
        show_figure=False
    )

    show_figures(
        train_metric=avg_train_time_metric,
        valid_metric=avg_valid_time_metric,
        title='training time',
        y_label='time',
        x_label='epoch',
        legend=['train', 'val'],
        save_file_path="./Data/time.eps",
        show_figure=True
    )






