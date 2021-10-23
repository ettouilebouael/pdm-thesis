from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np

def load_data(data_num, online = False):
  sensor_col_names = ['s_%i' % i for i in range(1,22)]
  col_names = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2','setting_3'] + sensor_col_names
  
  train_path = "CMAPSSData/train_FD00{}.txt".format(str(data_num))
  test_path = "CMAPSSData/test_FD00{}.txt".format(str(data_num))
  RUL_path = "CMAPSSData/RUL_FD00{}.txt".format(str(data_num))

  if online == True:
    test  = pd.read_csv(test_path , sep = "\s+", header=None, names = col_names, index_col=False).dropna(axis = 1)
    RUL_test = pd.read_csv(RUL_path, sep = "\s+", header= None, names = ["true_rul"], index_col= False).reset_index().rename(columns = {"index" :"unit_number"})
    RUL_test.unit_number = RUL_test.unit_number + 1 
    return test , RUL_test
  else:
    train  = pd.read_csv(train_path , sep = " ", header=None, names = col_names, index_col=False).dropna(axis = 1)
    return train

def compute_rul(df):
    a = df[['unit_number' , 'time_in_cycles']].groupby('unit_number').max()
    a = a.rename({"time_in_cycles" : "max_cyles"}, axis = 1)
    b = df.join(a, on = 'unit_number', how='outer', lsuffix=" ")
    df['rul'] = b.max_cyles - b.time_in_cycles
    df['rul'] = df['rul'].clip(upper = 125)
    return df

def gen_sequence(df, unit, seq_len, features):
  X =  df[df.unit_number == unit][features].values
  num_elements = X.shape[0]
  for start, stop in zip(range(0, num_elements-seq_len + 1), range(seq_len, num_elements +1)):
       yield X[start:stop]

def gen_labels(df, seq_len):
    y = df['rul'].values
    return y[seq_len - 1:]

def get_lstm_data(df, features, seq_len):
  X = [[seq.tolist() for seq in gen_sequence(df,unit ,seq_len,features)] for unit in df.unit_number.unique() if len(df[df.unit_number == unit]) > seq_len]
  y = [gen_labels(df[df.unit_number == unit], seq_len).tolist() for unit in df.unit_number.unique() if len(df[df.unit_number == unit]) > seq_len]

  return np.concatenate(X).astype(float), np.concatenate(y).astype(float)

def condition_scaler(df):
  scaler = RobustScaler()
  sensor_col_names = ['s_%i' % i for i in range(1,22)]

  df['setting_1'] = df['setting_1'].round(1)
  df['setting_2'] = df['setting_2'].round(1)
  df['setting_3'] = df['setting_3'].round(1)
  
  df['op_cond'] = df['setting_1'].astype(str) + '_' + \
                    df['setting_2'].astype(str) + '_' + \
                      df['setting_3'].astype(str)

  for condition in df['op_cond'].unique():
      scaler.fit(df.loc[df['op_cond']==condition, sensor_col_names])
      df.loc[df['op_cond']==condition, sensor_col_names] = scaler.transform(df.loc[df['op_cond']==condition, sensor_col_names])
    
  return df

def denoise(df, features):
  df[features] = df.groupby('unit_number')[features].apply(lambda x: x.ewm(30).mean()).values
  def create_mask(data, samples):
    result = np.ones_like(data)
    result[0:samples] = 0
    return result
    
  mask = df.groupby('unit_number')['unit_number'].transform(create_mask, samples=0).astype(bool)
  df = df[mask]
    
  return df

def train_preprocessing(data_num,features, seq_len):
  train = load_data(data_num)
  X = compute_rul(train)
  X = denoise(X,features)
  X = condition_scaler(X)
  X_train, y_train = get_lstm_data(X, features, seq_len)
  return X_train, y_train.reshape(-1,1)

def get_online_test_unit(df, features, seq_len):
  X = []
  for i in features:
    online_test_unit = []
    x = []
    for unit in df.unit_number.unique():
      signal = df[i][df.unit_number == unit].values.tolist()
      if len(signal) >= seq_len :
        x.append(signal[-seq_len:])
        online_test_unit.append(unit)
    X.append(x)
  return np.array(X).transpose(1,2,0), list(set(online_test_unit))

def online_preprocessing(data_num, features, seq_len):
  online_df , rul = load_data(data_num=data_num, online = True)
  X = denoise(online_df,features)
  X_online_scaled = condition_scaler(online_df)
  X_online, online_test_unit = get_online_test_unit(X_online_scaled, features, seq_len)
  y_online = rul.true_rul[rul.unit_number.isin(online_test_unit)].clip(upper = 125).values
  return X_online , y_online.reshape(-1,1)

def get_online_indexes(df,seq_len):
  ids = []
  for unit in df.unit_number.unique():
    if len(df[df.unit_number == unit]) > seq_len :
      indexes = df[df.unit_number == unit].index.values
      indexes = indexes[seq_len - 1:]
      ids.extend(indexes)
  return ids
 
def annotate_online_data(X,y):
  rul = []
  for unit in X.unit_number.unique():
    last_rul = y[y.unit_number == unit].true_rul.values[0]
    first_rul = last_rul + len(X[X.unit_number == unit]) + 1
    unit_rul = np.arange(first_rul, last_rul + 1, -1).tolist()
    rul.extend(unit_rul)
  return np.array(rul).clip(1,125)

def get_online_unit(df, seq_len):
  units = []
  for unit in df.unit_number.unique():
    if len(df[df.unit_number == unit]) > seq_len:
      units.extend(unit)
  return units

def online_pipeline(data_num,features, seq_len):
  online_df , y = load_data(data_num,online=True)
  online_df["rul"] = annotate_online_data(online_df , y)
  online_df = denoise(online_df,features)
  online_df = condition_scaler(online_df)
  ids_in = get_online_indexes(online_df, seq_len)
  ids_all = online_df.index.values
  online_df_pred = online_df[np.isin(ids_all,ids_in)]
  X , y = get_lstm_data(online_df, features, seq_len)
  return X, y.reshape(-1,1), online_df_pred 
