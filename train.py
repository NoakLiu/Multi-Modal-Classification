# train.py
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from torch.autograd import Variable

def train_iter(log_interval, model, optimizer, loss_func, img, text_inputs, target):
    optimizer.zero_grad()
    output_before_sigmoid, output = model(img, text_inputs)
    loss = loss_func(output, target.float()) if loss_func.__class__.__name__ == 'BCELoss' else loss_func(output_before_sigmoid, target.float())
    loss.backward()
    optimizer.step()
    return loss

def validation(model, device, val_loader, loss_func, threshold):
    model.eval()
    val_loss = 0
    correct = 0
    all_preds = None
    all_targets = None
    with torch.no_grad():
        for sample in val_loader:
            img = sample['image'].to(device)
            text_inputs = sample['text'].to(device)
            target = sample['target'].to(device)
            output_before_sigmoid, output = model(img, text_inputs)
            pred = np.where(output.cpu().data.numpy() >= threshold, 1, 0)
            if all_preds is None:
                all_preds = pred
                all_targets = target.cpu().data.numpy()
            else:
                all_preds = np.concatenate((all_preds, pred), axis=0)
                all_targets = np.concatenate((all_targets, target.cpu().data.numpy()), axis=0)
            val_loss += target.size(0) * loss_func(output, target.float()).item()
            correct += accuracy_score(target.cpu().data.numpy(), pred) * target.size(0)
    f1score = f1_score(all_targets, all_preds, average='samples')
    val_loss /= len(val_loader.dataset)
    print(classification_report(all_targets, all_preds))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n, F1 score: {:.4f}'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset),
        f1score))
    return f1score

def train_epoch(log_interval, model, train_loader, optimizer, epoch, loss_func):
    start_time = time.time()
    model.train()
    train_loss_batch = 0
    for batch_idx, sample in enumerate(train_loader):
        img = sample['image']
        text_inputs = sample['text']
        target = sample['target']
        loss = train_iter(log_interval, model, optimizer, loss_func, img, text_inputs, target)
        train_loss_batch += loss.item()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Time: {: 2f}'.format(
                epoch, batch_idx * len(target), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                time.time() - start_time))
    return train_loss_batch/len(train_loader)

def test_pred(model, testloader, threshold):
    model.eval()
    img_paths = []
    pred_labels = []
    with torch.no_grad():
        for batch, sample in enumerate(testloader):
            img = sample['image']
            text_inputs = sample['text']
            path = sample['image_path']
            img_paths.extend(path)
            output_before_sigmoid, output = model(img, text_inputs)
            pred = np.where(output.cpu().data.numpy() >= threshold, 1, 0)
            for line in pred:
                label_list = [str(x+1) for x in range(len(line)) if line[x] == 1]
                labels = ' '.join(label_list)
                pred_labels.append(labels)
    submission_df = pd.DataFrame({'ImageID': img_paths, 'Labels': pred_labels})
    return submission_df
