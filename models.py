import torch
import torch.nn.functional as F

class Encoder(torch.nn.Module):


  def __init__(self, input_size, n_hidden_lstm, n_lstm_layers, dropout):
    super().__init__()
    self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=n_hidden_lstm, num_layers = n_lstm_layers, batch_first = True)
    self.n_hidden_lstm = n_hidden_lstm

  def forward(self, x):
    output, (h, c) = self.lstm(x)
    x = h[-1].reshape(-1,self.n_hidden_lstm)
    x = F.relu(x)
    return x

class RULPredictor(torch.nn.Module):


  def __init__(self, n_hidden_lstm,dropout):
    super().__init__()
    self.n_hidden_lstm = n_hidden_lstm
    self.fc1 = torch.nn.Linear(n_hidden_lstm, 64)
    self.fc2 = torch.nn.Linear(64, 32)
    self.fc3 = torch.nn.Linear(32, 16)
    self.output = torch.nn.Linear(16, 1)
    self.dropout = torch.nn.Dropout(p=dropout)

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x - self.dropout(x)
    x = self.fc2(x)
    x = F.relu(x)
    x - self.dropout(x)
    x = self.fc3(x)
    x = F.relu(x)
    x - self.dropout(x)
    x = self.output(x)
    return x
  
class Discriminator(torch.nn.Module):


  def __init__(self, n_hidden_lstm,dropout):
    super().__init__()
    self.n_hidden_lstm = n_hidden_lstm
    self.fc1 = torch.nn.Linear(n_hidden_lstm, 64)
    self.fc2 = torch.nn.Linear(64, 32)
    self.fc3 = torch.nn.Linear(32, 16)
    self.output = torch.nn.Linear(16, 1)
    self.dropout = torch.nn.Dropout(p=dropout)

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x - self.dropout(x)
    x = self.fc2(x)
    x = F.relu(x)
    x - self.dropout(x)
    x = self.fc3(x)
    x = F.relu(x)
    x - self.dropout(x)
    x = self.output(x)
    F.sigmoid(x)
    return x
