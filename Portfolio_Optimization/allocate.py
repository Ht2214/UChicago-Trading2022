import numpy as np
import pandas as pd
import scipy

##bug, weights dont add to 1

#########################################################################
## Change this code to take in all asset price data and predictions    ##
## for one day and allocate your portfolio accordingly.                ##
#########################################################################

# empty data frames
asset_prices_df = pd.DataFrame(columns=list(range(9)))
asset_pred_1_df = pd.DataFrame(columns=list(range(9)))
asset_pred_2_df = pd.DataFrame(columns=list(range(9)))
asset_pred_3_df = pd.DataFrame(columns=list(range(9)))

fin_weights_list = [np.repeat(1 / 9, 9)]

#shares outstanding
#just change path
shares = pd.read_csv("/Shares Outstanding.csv")
shares_arr = shares.iloc[[0],1:].to_numpy()

#tau - tuning constant
tau = 0.05

#risk aversion parameter 2.15 to 2.65, just to simplify
rap = 2.4

#P - picking matrix
P = np.array(
  [
      [1, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 1],
  ]
)

#previous sharpe
p_s = 0

#to think, update each time considering previous weight values?

def allocate_portfolio(asset_prices, asset_price_predictions_1, 
                       asset_price_predictions_2,
                       asset_price_predictions_3):
  
    #update dataframes
    asset_prices_df.loc[len(asset_prices_df.index)] = asset_prices
    asset_pred_1_df.loc[len(asset_pred_1_df.index)] = asset_price_predictions_1
    asset_pred_2_df.loc[len(asset_pred_2_df.index)] = asset_price_predictions_2
    asset_pred_3_df.loc[len(asset_pred_3_df.index)] = asset_price_predictions_3


    #check that there are at least 2 days worth of data
    if (len(asset_prices_df.index) < 3):
      n_assets = len(asset_prices)
      return np.repeat(1 / n_assets, n_assets)
    
    #create return series - output DataFrame
    #to consider for improvement: using log instead of raw return
    ret_real = convert_returns(asset_prices_df)
    ret_df1 = convert_returns(asset_pred_1_df)
    ret_df2 = convert_returns(asset_pred_2_df)
    ret_df3 = convert_returns(asset_pred_3_df)

    #var-covar matrices - output NumPy array
    covar_real = np.cov(ret_real.to_numpy().T)
    covar_1 = np.cov(ret_df1.to_numpy().T)
    covar_2 = np.cov(ret_df2.to_numpy().T)
    covar_3 = np.cov(ret_df3.to_numpy().T)

    #final day idx
    idx = len(asset_prices_df.index) - 1

    #final day's price
    fin_price_real = one_day_price(asset_prices_df, idx)
    fin_price_1 = one_day_price(asset_pred_1_df, idx)
    fin_price_2 = one_day_price(asset_pred_2_df, idx)
    fin_price_3 = one_day_price(asset_pred_3_df, idx)

    #implied returns for curr day
    imp_real = implied_market_ret(fin_price_real, covar_real, rap, shares_arr)
    imp_1 = implied_market_ret(fin_price_1, covar_1, rap, shares_arr)
    imp_2 = implied_market_ret(fin_price_2, covar_2, rap, shares_arr)
    imp_3 = implied_market_ret(fin_price_3, covar_3, rap, shares_arr)

    #confidence matrix
    conf_real = conf_mat(tau, P, covar_real)
    conf_1 = conf_mat(tau, P, covar_1)
    conf_2 = conf_mat(tau, P, covar_2)
    conf_3 = conf_mat(tau, P, covar_3)

    #view for curr day
    view_1 = one_day_view(ret_df1, len(ret_real.index) - 1)
    view_2 = one_day_view(ret_df2, len(ret_real.index) - 1)
    view_3 = one_day_view(ret_df3, len(ret_real.index) - 1)

    #latest real price
    rp = one_day_price(asset_prices_df, idx)

    #expected returns
    er1 = exp_ret(tau, covar_1, P, conf_1, shares_arr, view_1, rp)
    er2 = exp_ret(tau, covar_2, P, conf_2, shares_arr, view_2, rp)
    er3 = exp_ret(tau, covar_3, P, conf_3, shares_arr, view_3, rp)

    #individual weights
    fw1 = normalize(fin_weights(tau, covar_1, er1))
    fw2 = normalize(fin_weights(tau, covar_2, er2))
    fw3 = normalize(fin_weights(tau, covar_3, er3))

    #average weights
    fin_res = (fw1 + fw2 + fw3)
    fin_res = fin_res.T[0]
    #print(fin_res)
    #fin_res = normalize(fin_res)
    if len(fin_weights_list) > 9:
      fin_weights_list.clear()
      fin_weights_list.append(np.repeat(1 / 9, 9))

    start = fin_weights_list[0]
    for f in range(1, len(fin_weights_list)):
      start = start + fin_weights_list[f]

    last_ret = start * one_day_price(ret_real, idx - 1)
    curr_ret = fin_res * one_day_price(ret_real, idx - 1)
    sharpe1 = np.mean(last_ret) / np.std(last_ret)
    sharpe2 = np.mean(curr_ret) / np.std(curr_ret)

    if sharpe2 > sharpe1:
      fin_weights_list.append(fin_res)
      # start = fin_weights_list[0]
      # for f in range(1, len(fin_weights_list)):
      #   start = start + fin_weights_list[f]
    else:
      fin_res = start

    #print(max(sharpe1, sharpe2))
    return fin_res

def convert_returns(data):
  return_series = []
  for i in range(1, data.shape[0]):
    row_i = one_day_price(data, i)
    row_j = one_day_price(data, i - 1)
    return_series.append(np.log(row_i[0] / row_j[0]))
  return pd.DataFrame(return_series)

#calculate market implied returns
def implied_market_ret(day_data, var_covar, delta, shares_arr):

  #calculate market-cap weight ?? for each day

  #for day one (index: 1)
  #one_day = df1.iloc[[1],[1,2,3,4,5,6,7,8,9]].to_numpy()
  mcap_w = day_data * shares_arr
  mcap_w = (mcap_w / np.sum(mcap_w)).T

  # final step
  implied_ret = np.matmul((delta * var_covar), mcap_w) 
  return implied_ret

def one_day_price(data, day):
  #day - index start with 1
  return data.iloc[[day]].to_numpy()

def one_day_view(data, day):
  #data should be return series
  #day - index start with 0
  return data.iloc[[day]].to_numpy().reshape(-1, 1)

def conf_mat(tau, pick, covar):
  #confidence matrix
  #tau * (picking matrix * covariance matrix * picking transposed)
  return tau * np.matmul(np.matmul(pick, covar), pick.T)

def exp_ret(tau, covar, pick, conf, shares_arr, view, real_price):
  #one day view (latest day?)
  #real_price (corresponding day)
  part1 = get_inverse(tau * covar)
  inv_conf = get_inverse(conf)
  part1 += np.matmul(np.matmul(pick.T, inv_conf), pick) 
  part1 = get_inverse(part1)

  part2 = get_inverse(tau * covar)
  imp = implied_market_ret(real_price, covar, tau, shares_arr)
  part2 = np.matmul(part2, imp)
  part2 += np.matmul(np.matmul(pick.T, inv_conf), view)
  er = np.matmul(part1, part2)
  return er

def fin_weights(tau, covar, er):
  #er - expected return
  w1 = get_inverse(tau * covar)
  w1 = np.matmul(w1, er) * 2
  return w1

def normalize(arr):
  l1 = np.linalg.norm(arr)
  arr = arr / l1
  return arr

def get_inverse(arr):
  try:
    arr = np.linalg.inv(arr)
  except:
    arr = np.linalg.lstsq(arr, P, rcond=None)[0]
  return arr