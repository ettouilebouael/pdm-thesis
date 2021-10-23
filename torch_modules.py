from torch.utils.data import Dataset, DataLoader
import torch
from livelossplot import PlotLosses
import numpy as np


class TurboFanDataset(Dataset):
  
  
  def __init__(self, X, y):
    self.X, self.y = X, y
    
  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, ix):
    return torch.from_numpy(self.X[ix,:,:]).float() , torch.from_numpy(np.array(self.y[ix])).float()

  
def fit(encoder, rul_predictor, dataloader, epochs, learning_rate, data_num):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    liveloss = PlotLosses()
    encoder.to(device)
    rul_predictor.to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(rul_predictor.parameters()), lr=learning_rate,weight_decay=0)
    criterion = torch.nn.MSELoss()
    best_loss = 30
    for epoch in range(1, epochs+1):
        encoder.train()
        rul_predictor.train()
        train_loss = []
        eval_loss = []
        logs = {}
        for batch in dataloader['train']:
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            h_hat = encoder(X)
            y_hat = rul_predictor(h_hat)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        encoder.eval()
        rul_predictor.eval()
        with torch.no_grad():
          for batch in dataloader['test']:
            X, y = batch
            X, y = X.to(device), y.to(device)
            h_hat = encoder(X)
            y_hat = rul_predictor(h_hat)
            loss = criterion(y_hat, y)
            eval_loss.append(loss.item())

            rmse_train = np.sqrt(np.mean(train_loss))
            rmse_val = np.sqrt(np.mean(eval_loss))

          logs["loss"] = rmse_train
          logs["val_loss"] = rmse_val

          liveloss.update(logs)
          liveloss.send()

          if epoch > 25 :
            if best_loss > rmse_val:
              best_loss = rmse_val
              torch.save(encoder.state_dict(), f"models/fd00{data_num}/encoder_fd00{data_num}_rmse_{rmse_val}")
              torch.save(rul_predictor.state_dict(), f"models/fd00{data_num}/rul_predictor_fd00{data_num}_rmse_{rmse_val}")
              print(f'best model saved with rmse = {rmse_val} at epoch {epoch}')

