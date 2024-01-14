import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import copy


def train_model(model, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs, save_path, training_log):
    since = time.time()
    # training_log = {'epoch': [], 'training_loss': [], 'val_loss': []}
    best_loss = float("inf")

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(100, num_epochs):
        training_log['epoch'].append(epoch)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_loss_b = 0.0
            running_loss_i = 0.0

            # Iterate over data.
            for inputs, binarys, instances in dataloaders[phase]:
                inputs = inputs.type(torch.FloatTensor).to(device)
                binarys = binarys.type(torch.LongTensor).to(device)
                instances = instances.type(torch.FloatTensor).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, instances)
                    loss = outputs['loss']

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_loss_b += outputs['seg_loss'].item() * inputs.size(0)
                running_loss_i += (outputs['var_loss'].item() + outputs['dist_loss'].item()) * inputs.size(0)

            if phase == 'train':
                if scheduler != None:
                    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            binary_loss = running_loss_b / dataset_sizes[phase]
            instance_loss = running_loss_i / dataset_sizes[phase]
            print(
                '{} Total Loss: {:.4f} Binary Loss: {:.4f} Instance Loss: {:.4f}'.format(phase, epoch_loss, binary_loss,
                                                                                         instance_loss))

            # deep copy the model
            if phase == 'train':
                training_log['training_loss'].append(epoch_loss)
            if phase == 'val':
                training_log['val_loss'].append(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 5 == 0:
            model_save_filename = os.path.join(save_path, 'epoch_{:d}_model.pth'.format(epoch + 1))
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'log': training_log},
                       model_save_filename)
            print("model is saved: {}".format(model_save_filename))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val_loss: {:4f}'.format(best_loss))
    training_log['training_loss'] = np.array(training_log['training_loss'])
    training_log['val_loss'] = np.array(training_log['val_loss'])

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_log


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
