import torch
from torch import nn

#from tqdm.notebook import tqdm

def model_train(model, optimizer, dataloader, device):

    model.train() #model to train mode

    criterion = nn.BCELoss() #binary cross-entropy

    #tot_itr = len(dataloader.dataset.data)//dataloader.batch_size #total train iterations

    #pbar = tqdm(total = tot_itr, ncols=700) #progress bar

    beta = 0.98 #beta of running average, don't change

    avg_loss = 0. #average loss

    all_predictions = []

    for itr_idx, (tensors, labels, misc_data, variant_meta) in enumerate(dataloader):

        #if itr_idx==10:
        #    break
        tensors = tensors.to(device)
        labels = labels.to(device)

        misc_data = misc_data.to(device)

        outputs = model((tensors, misc_data))

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #exponential moving evaraging of loss
        avg_loss = beta * avg_loss + (1-beta)*loss.item()
        smoothed_loss = avg_loss / (1 - beta**(itr_idx+1))

        outputs = outputs.cpu().tolist()
        labels = labels.cpu().tolist()

        current_predictions = list(zip(outputs, labels, variant_meta)) #(tensors_dataset_idx, prediction, true_label)
        all_predictions.extend(current_predictions)

        #pbar.update(1)
        #pbar.set_description(f"Running loss:{smoothed_loss:.4}")

    return smoothed_loss, all_predictions #return average loss and predictions

def model_eval(model, optimizer, dataloader, device, inference_mode=False):

    model.eval() #model to evaluation mode

    criterion = nn.BCELoss() #binary cross-entropy

    #tot_itr = len(dataloader.dataset.data)//dataloader.batch_size #total evaluation iterations

    #pbar = tqdm(total = tot_itr, ncols=700) #progress bar

    all_loss = 0. #all losses, for simple averaging

    all_predictions = []

    with torch.no_grad():

        for itr_idx, (tensors, labels, misc_data, variant_meta) in enumerate(dataloader):

            #if itr_idx==10:
            #    break
            tensors = tensors.to(device)

            misc_data = misc_data.to(device)

            outputs = model((tensors, misc_data))

            if not labels.isnan().sum():

                #in inference mode, all labels are None

                labels = labels.to(device)

                loss = criterion(outputs, labels)

                all_loss += loss.item()

                labels = labels.cpu().tolist()

            outputs = outputs.cpu().tolist()

            current_predictions = list(zip(outputs, labels, variant_meta)) #(tensors_dataset_idx, prediction, true_label)
            all_predictions.extend(current_predictions)

            #pbar.update(1)
            #pbar.set_description(f"Running loss:{all_loss/(itr_idx+1):.4}")

    return all_loss/(itr_idx+1), all_predictions #return average loss and predictions
