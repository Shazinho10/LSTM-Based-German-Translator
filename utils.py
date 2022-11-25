from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def Valid_Accuracy(model, loss_fn, valid_iterator):
  losses, y_true, y_pred = [], [], []
  for data in valid_iterator:
    src = data.src
    src = src.to(device)

    trg = data.trg
    trg = trg.to(device)

    preds = model(src, trg)
    preds = preds.reshape(-1, preds.shape[2])
    trg = trg.view(-1)

    loss = loss_fn(preds, trg)
    losses.append(loss.item())

    y_true.append(trg)
    y_pred.append(preds.argmax(1)) #appending the best guessed values out of all of the possibilities

  y_true = torch.cat(y_true)
  y_pred = torch.cat(y_pred)


  losses = torch.tensor(losses)
  accuracy = accuracy_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

  print(f'Valid Loss: {losses.mean()}, Valid Accuracy: {accuracy}')

def Train_model(model, loss_fn, epochs, optimizer, train_iterator, valid_iterator):
 
  for epoch in range(epochs):
    losses, y_true, y_pred = [], [], []
    for data in tqdm(train_iterator):
      src = data.src
      src = src.to(device)

      trg = data.trg
      trg =trg.to(device)

      preds = model(src, trg)
      preds = preds.reshape(-1, preds.shape[2])  #all of the possible classes for each token shape --> (no. of tokens, classes)
      trg = trg.view(-1)

      loss = loss_fn(preds, trg)
      losses.append(loss.item())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      y_true.append(trg)
      y_pred.append(preds.argmax(1)) #appending the best guessed values out of all of the possibilities

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)


    losses = torch.tensor(losses)
    accuracy = accuracy_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    
    print(f'Train Loss: {losses.mean()}, Train Accuracy: {accuracy}')
    Valid_Accuracy(model, loss_fn, valid_iterator)

