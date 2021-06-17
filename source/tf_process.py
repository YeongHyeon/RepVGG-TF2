import os
import scipy.ndimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import source.utils as utils

def perform_from_confmat(confusion_matrix, num_class, verbose=False):

    dict_perform = {'accuracy':0, 'precision':0, 'recall':0, 'f1score':0}

    for idx_c in range(num_class):
        precision = np.nan_to_num(confusion_matrix[idx_c, idx_c] / np.sum(confusion_matrix[:, idx_c]))
        recall = np.nan_to_num(confusion_matrix[idx_c, idx_c] / np.sum(confusion_matrix[idx_c, :]))
        f1socre = np.nan_to_num(2 * (precision * recall / (precision + recall)))

        dict_perform['accuracy'] += confusion_matrix[idx_c, idx_c]
        dict_perform['precision'] += precision
        dict_perform['recall'] += recall
        dict_perform['f1score'] += f1socre

        if(verbose):
            print("Class-%d | Precision: %.5f, Recall: %.5f, F1-Score: %.5f" \
                %(idx_c, precision, recall, f1socre))

    for key in list(dict_perform.keys()):
        if('accuracy' == key): dict_perform[key] = dict_perform[key] / np.sum(confusion_matrix)
        else: dict_perform[key] = dict_perform[key] / num_class
        print("%s: %.5f" %(key.upper(), dict_perform[key]))

    return dict_perform

def training(agent, dataset, batch_size, epochs):

    print("\n** Training of the CNN to %d epoch | Batch size: %d" %(epochs, batch_size))
    iteration = 0

    for epoch in range(epochs):

        list_loss = []
        while(True):
            minibatch = dataset.next_batch(batch_size=batch_size, tt=0)
            if(len(minibatch['x'].shape) == 1): break
            step_dict = agent.step(minibatch=minibatch, iteration=iteration, training=True)
            iteration += 1
            list_loss.append(step_dict['losses']['entropy'])
            if(minibatch['t']): break

        print("Epoch [%d / %d] | Loss: %f" %(epoch, epochs, np.average(list_loss)))
        agent.save_params(model='model_0_finepocch')

def test(agent, dataset, batch_size):

    savedir = 'results_te'
    utils.make_dir(path=savedir, refresh=True)

    list_model = utils.sorted_list(os.path.join('Checkpoint', 'model*'))
    for idx_model, path_model in enumerate(list_model):
        list_model[idx_model] = path_model.split('/')[-1]

    for idx_model, path_model in enumerate(list_model):

        print("\n** Test with %s" %(path_model))
        agent.load_params(model=path_model)
        utils.make_dir(path=os.path.join(savedir, path_model), refresh=False)

        confusion_matrix = np.zeros((dataset.num_class, dataset.num_class), np.int32)
        while(True):
            minibatch = dataset.next_batch(batch_size=batch_size, tt=1)
            if(len(minibatch['x'].shape) == 1): break
            step_dict = agent.step(minibatch=minibatch, training=False)

            for idx_y, _ in enumerate(minibatch['y']):
                y_true = np.argmax(minibatch['y'][idx_y])
                y_pred = np.argmax(step_dict['y_hat'][idx_y])
                confusion_matrix[y_true, y_pred] += 1

            if(minibatch['t']): break

        dict_perform = perform_from_confmat(confusion_matrix=confusion_matrix, num_class=dataset.num_class, verbose=True)
        np.save(os.path.join(savedir, path_model, 'conf_mat.npy'), confusion_matrix)
