import numpy as np
from functions.utilities.file_util import FileUtil
from matplotlib import pyplot as plt
from re import search

FILE_PATH = "./data/filter_history_hyper_parameter_configuration_22.json"

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
    file = file_helper.load_file(FILE_PATH)

    
    avg_fold_train_accuracy_metrics = []
    avg_valid_accuracy_metrics = []
    avg_train_loss_metrics = []
    avg_valid_loss_metrics = []
    avg_train_time_metrics = []
    avg_valid_time_metrics = []

    for attribute in file:
        if not search("fold", attribute):
            continue
        fold_train_accuracy_metrics = file[attribute]['train_accs']
        fold_valid_accuracy_metrics = file[attribute]['valid_accs']

        fold_train_loss_metrics = file[attribute]['train_losses']
        fold_valid_loss_metrics = file[attribute]['valid_losses']

        fold_train_time_metrics = file[attribute]['train_times']
        fold_valid_time_metrics = file[attribute]['valid_times']

        avg_fold_train_accuracy_metric = np.sum(fold_train_accuracy_metrics, axis=0) / len(fold_train_accuracy_metrics)
        avg_fold_valid_accuracy_metrics = np.sum(fold_valid_accuracy_metrics, axis=0) / len(fold_valid_accuracy_metrics)
        avg_fold_train_loss_metrics = np.sum(fold_train_loss_metrics, axis=0) / len(fold_train_loss_metrics)
        avg_fold_valid_loss_metrics = np.sum(fold_valid_loss_metrics, axis=0) / len(fold_valid_loss_metrics)
        avg_fold_train_time_metrics = np.sum(fold_train_time_metrics, axis=0) / len(fold_train_time_metrics)
        avg_fold_valid_time_metrics = np.sum(fold_valid_time_metrics, axis=0) / len(fold_valid_time_metrics)

        avg_fold_train_accuracy_metrics.append(avg_fold_train_accuracy_metric)
        avg_valid_accuracy_metrics.append(avg_fold_valid_accuracy_metrics)
        avg_train_loss_metrics.append(avg_fold_train_loss_metrics)
        avg_valid_loss_metrics.append(avg_fold_valid_loss_metrics)
        avg_train_time_metrics.append(avg_fold_train_time_metrics)
        avg_valid_time_metrics.append(avg_fold_valid_time_metrics)


    avg_across_folds_train_accuracy_metric = np.sum(avg_fold_train_accuracy_metrics, axis=0) / len(avg_fold_train_accuracy_metrics)
    avg_across_folds_valid_accuracy_metric = np.sum(avg_valid_accuracy_metrics, axis=0) / len(avg_valid_accuracy_metrics)
    avg_across_folds_train_loss_metric = np.sum(avg_train_loss_metrics, axis=0) / len(avg_train_loss_metrics)
    avg_across_folds_valid_loss_metric = np.sum(avg_valid_loss_metrics, axis=0) / len(avg_valid_loss_metrics)
    avg_across_folds_train_time_metric = np.sum(avg_train_time_metrics, axis=0) / len(avg_train_time_metrics)
    avg_across_folds_valid_time_metric = np.sum(avg_valid_time_metrics, axis=0) / len(avg_valid_time_metrics)


    print(avg_across_folds_train_accuracy_metric)
    print(avg_across_folds_valid_accuracy_metric)
    print(avg_across_folds_train_loss_metric)
    print(avg_across_folds_valid_loss_metric)
    print(avg_across_folds_train_time_metric)
    print(avg_across_folds_valid_time_metric)


    show_figures(
        train_metric=avg_across_folds_train_accuracy_metric,
        valid_metric=avg_across_folds_valid_accuracy_metric,
        title='model accuracy',
        y_label='accuracy',
        x_label='epoch',
        legend=['train', 'val'],
        save_file_path="./data/accuracy.eps",
        show_figure=False
    )

    show_figures(
        train_metric=avg_across_folds_train_loss_metric,
        valid_metric=avg_across_folds_valid_loss_metric,
        title='model loss',
        y_label='loss',
        x_label='epoch',
        legend=['train', 'val'],
        save_file_path="./data/loss.eps",
        show_figure=False
    )

    show_figures(
        train_metric=avg_across_folds_train_time_metric,
        valid_metric=avg_across_folds_valid_time_metric,
        title='training time',
        y_label='time',
        x_label='epoch',
        legend=['train', 'val'],
        save_file_path="./data/time.eps",
        show_figure=True
    )






