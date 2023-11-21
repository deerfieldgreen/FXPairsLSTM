## Version
# Forex LSTM V 1.0

##-##
IS_LIVE = False

TO_SAVE_DATA = False
# if IS_LIVE:
#     TO_SAVE_DATA = False
# else:
#     TO_SAVE_DATA = True
##-##


from AlgorithmImports import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import pickle
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

from config import (
    general_setting,
    consolidator_settings,
    indicator_settings,
    signal_settings,
    model_settings,
)

from data_classes import (
    SymbolData,
    MarketHours,
)

from signal_classes import (
    FxLstmSignal,
)

from model_functions import (
    get_threshold,
    set_seed,
)

from model_classes_both import get_torch_rnn_dataloaders as get_torch_rnn_dataloaders_both
from model_classes_both import get_rnn_model as get_rnn_model_both
from model_classes_both import get_predictions as get_predictions_both

from model_classes_hybrid import get_torch_rnn_dataloaders as get_torch_rnn_dataloaders_hybrid
from model_classes_hybrid import get_rnn_model as get_rnn_model_hybrid
from model_classes_hybrid import get_predictions as get_predictions_hybrid
from model_classes_hybrid import get_regression_pred_decision, get_prediction_hybrid_regression
from sklearn.metrics import mean_squared_error


signal_mapping = {
    "FxLstm_Both_EURUSD": FxLstmSignal,
    "FxLstm_Both_EURUSD_Trail": FxLstmSignal,
    "FxLstm_Both_EURUSD_F": FxLstmSignal,

    "FxLstm_Hybrid_EURUSD": FxLstmSignal,
    "FxLstm_Hybrid_EURUSD_Trail": FxLstmSignal, 

    "FxLstm_Both_USDJPY": FxLstmSignal,
    "FxLstm_Both_USDJPY_Trail": FxLstmSignal, 

    "FxLstm_Hybrid_GBPUSD": FxLstmSignal,
    "FxLstm_Hybrid_GBPUSD_Trail": FxLstmSignal,

    "FxLstm_Both_AUDUSD": FxLstmSignal,
    "FxLstm_Both_AUDUSD_Trail": FxLstmSignal,
    
    "FxLstm_Hybrid_AUDUSD": FxLstmSignal,
    "FxLstm_Hybrid_AUDUSD_Trail": FxLstmSignal,
}


from QuantConnect.DataSource import *





## Performance Tracking

## Futures (Experiment)
# Baseline:                             0.546
# *+ FxLstm_Both_AUDUSD:                0.644
# ++ FxLstm_Hybrid_AUDUSD:              0.781
# ++ FxLstm_Hybrid_AUDUSD_Trail:        0.813
# ++ Both (Scaling 0.5)                 0.832
# ++ Both (Scaling 0.6)                 0.948
# ++ Both (Scaling 0.7)                 1.061
# ++ Both (Scaling 0.8)                 1.066
# ++ Both (Scaling 0.9)                 1.16
# ++ Both (Scaling 1.0)                 1.086


## Spot
# 2018: 1.085
# 2019: 1.072
# 2021: 1.365
# 2022: 1.684

## Spot (Exc USDJPY)
# 2018: 1.006
# 2019: 0.906
# 2021: 1.019
# 2022: 1.405
## Futures
# 2018: 0.215
# 2019: 0.301
# 2021: 0.757
# 2022: 1.248


# Original: 0.597
# D1: 0.669
# D30: 0.534
# No shift: 0.56

# 2018: 0.984
# 2019: 0.915
# 2021: 0.951

# 2018: 0.984
# 2019: 0.915
# 2021: 0.951
# 2022: 1.024


## IB
# 2022: 0.951

# FxLstm_Hybrid_GBPUSD_Trail: 1.054
# FxLstm_Hybrid_EURUSD: 0.618
# FxLstm_Both_EURUSD_Trail: 0.412
# FxLstm_Both_USDJPY_Trail: 0.26
# FxLstm_Both_EURUSD: 0.126

# FxLstm_Both_AUDUSD: -0.003
# FxLstm_Both_USDJPY: -0.071
# FxLstm_Hybrid_AUDUSD: -0.24

# FxLstm_Hybrid_AUDUSD_Trail: -0.4


# FxLstm_Hybrid_GBPUSD_Trail: 1.054
# FxLstm_Hybrid_EURUSD: 1.198
# FxLstm_Both_EURUSD_Trail: 1.242
# FxLstm_Both_USDJPY_Trail: 1.444

## FxLstm_Both_EURUSD: 1.09
# FxLstm_Both_AUDUSD: 1.18, 1.555
# FxLstm_Both_USDJPY: 1.598

## FxLstm_Hybrid_AUDUSD: 1.46, 1.491
## FxLstm_Hybrid_AUDUSD_Trail: 1.547



## Original Spot OANDA
# 2018: 1.085
# 2019: 1.072
# 2021: 1.365
# 2022: 1.684

## New Spot IB (Excldue Both) **
# 2018: 0.999
# 2019: 1.033
# 2021: 1.291
# 2022: 1.278

## New Spot IB (Excldue Single)
# 2019: 1.017
# 2021: 1.398
# 2022: 1.332




##-##

## Future
# 2018: 0.303
# 2022: 1.017

## Spot
# 2018: 0.972
# 2022: 1.245


## Large Risk Pct, Combined
# 2018: 1.088, 27.366%, 19.700%
# 2022: 1.654, 47.471%, 13.900%

## Small Risk Pct, Combined
# 2018: 0.987, 21.964%, 18.400%
# 2022: 1.653, 41.143%, 11.100%

## ?? Risk Pct, Combined
# 2018: 
# 2019: 1.025
# 2021: 
# 2022: 

##-##





class FxLstmAlgo(QCAlgorithm):

    def Initialize(self):

        self.SetTimeZone(TimeZones.Johannesburg)

        if TO_SAVE_DATA:
            self.SetStartDate(2013, 1, 1)       ## SR: 0.574, 0.811, 0.714            self.SetStartDate(2013, 1, 1)       ## SR: 0.574, 0.811, 0.714

            # self.SetEndDate(2017, 12, 31)
            # self.SetEndDate(2018, 12, 31)
            # self.SetEndDate(2020, 12, 31)
            # self.SetEndDate(2021, 12, 31)
        else:
            # self.SetStartDate(2018, 1, 1)         ## SR: 0.799, 1.094, 1.16, 1.054, 1.078, 1.097, 1.087, 0.847
            # self.SetStartDate(2019, 1, 1)         ## SR: 0.573, 0.908, 0.986, 0.997, 1.028, 1.076
            # self.SetStartDate(2021, 1, 1)         ## SR: 0.489, 1.108, 1.256, 1.333, 1.333, 1.374
            self.SetStartDate(2022, 1, 1)         ## SR: 0.783, 1.49, 1.729, 1.674, 1.674, 1.696, 1.708


        self.SetCash(50000)

        if TO_SAVE_DATA:
            self.SetWarmUp(int(12 * 20 * 24 * 60), Resolution.Minute)
        else:
            self.SetWarmUp(int(6 * 20 * 24 * 60), Resolution.Minute)

        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

        self.to_plot = False
        self.general_setting = general_setting
        self.consolidator_settings = consolidator_settings
        self.indicator_settings = indicator_settings
        self.signal_settings = signal_settings
        self.model_settings = model_settings
        self.ref_ticker = "EURUSD"
        self.ref_symbol = None

        self.model_name = general_setting["model_name"]
        self.month_start_date = None
        self.IS_LIVE = IS_LIVE
        self.TO_SAVE_DATA = TO_SAVE_DATA
        self.signals = self.general_setting["signals"]
        self.lstm_tickers = self.general_setting["lstm_tickers"] 

        self.prediction_dict = {}
        for _signal in self.signals:
            self.prediction_dict[_signal] = {}
            for ticker in self.lstm_tickers:
                if ticker in self.signal_settings[_signal]['valid_tickers']:
                    self.prediction_dict[_signal][ticker] = 1
                    
        # Data Initialization
        self.Data = {}
        self.Signal = {}
        self.Counter = {}
        self.SymbolMarketHours = {}
        self.symbol_ticker_map = {}     
        self.ticker_symbol_map = {}
        self.output_data_dict = {}
        self.Futures = {}
        self.FuturesTracker = {}   
        self.FuturesSymbol = {}   
        self.FuturesRefSymbol = {}  

        for ticker in self.general_setting["tickers"]:

            if general_setting["tickers"][ticker]["type"] == "equity":
                symbol = self.AddEquity(
                    ticker,
                    Resolution.Minute,
                    dataNormalizationMode=DataNormalizationMode.Raw,
                ).Symbol
            elif general_setting["tickers"][ticker]["type"] == "forex":
                symbol = self.AddForex(
                    ticker,
                    Resolution.Minute,
                    Market.Oanda,
                ).Symbol
            elif general_setting["tickers"][ticker]["type"] == "cfd":
                symbol = self.AddCfd(
                    ticker,
                    Resolution.Minute,
                    Market.Oanda,
                ).Symbol
            elif general_setting["tickers"][ticker]["type"] == "forex_futures":
                if ticker == "EURUSD_F":
                    self.Futures[ticker] = self.AddFuture(
                        Futures.Currencies.MicroEUR,
                        Resolution.Minute,
                        dataNormalizationMode=DataNormalizationMode.BackwardsRatio,
                        dataMappingMode=DataMappingMode.LastTradingDay,
                        extendedMarketHours=True,
                    )
                if ticker == "GBPUSD_F":
                    self.Futures[ticker] = self.AddFuture(
                        Futures.Currencies.MicroGBP,
                        Resolution.Minute,
                        dataNormalizationMode=DataNormalizationMode.BackwardsRatio,
                        dataMappingMode=DataMappingMode.LastTradingDay,
                        extendedMarketHours=True,
                    )
                if ticker == "AUDUSD_F":
                    self.Futures[ticker] = self.AddFuture(
                        Futures.Currencies.MicroAUD,
                        Resolution.Minute,
                        dataNormalizationMode=DataNormalizationMode.BackwardsRatio,
                        dataMappingMode=DataMappingMode.LastTradingDay,
                        extendedMarketHours=True,
                    )

                symbol = self.Futures[ticker].Symbol
                self.Futures[ticker].SetFilter(0, 182)
                self.FuturesSymbol[ticker] = symbol
                self.FuturesRefSymbol[ticker] = None
                self.FuturesTracker[ticker] = None

            if ticker == self.ref_ticker:
                self.ref_symbol = symbol

            self.Data[symbol] = SymbolData(
                self,
                symbol,
                ticker,
                general_setting,
                consolidator_settings,
                indicator_settings,
            )

            self.Counter[symbol] = {}
            self.Counter[symbol]["counter"] = 0
            self.Counter[symbol]["last_order_counter"] = 0
            self.SymbolMarketHours[symbol] = MarketHours(self, symbol)
            self.symbol_ticker_map[symbol] = ticker
            self.ticker_symbol_map[ticker] = symbol

            self.Signal[symbol] = {}
            for _signal in self.signals:
                self.Signal[symbol][_signal] = signal_mapping[_signal](
                    self, symbol, ticker, self.general_setting, self.signal_settings[_signal]
                )


        # Model Initialization
        self.Models = {}
        self.Scalers = {}
        self.ModelParams = {}   
        for lstm_ticker in self.lstm_tickers:
            self.Models[lstm_ticker] = {}
            self.Scalers[lstm_ticker] = {}
            self.ModelParams[lstm_ticker] = {}
            for model_type in self.general_setting["model_types"]:
                self.Models[lstm_ticker][model_type] = {}
                self.Scalers[lstm_ticker][model_type] = {}
                self.ModelParams[lstm_ticker][model_type] = {}

                if model_type == 'both':
                    self.Models[lstm_ticker][model_type]['both'] = None
                    self.Scalers[lstm_ticker][model_type]['both'] = None 

                if model_type == 'hybrid':
                    self.Models[lstm_ticker][model_type]['fundamental'] = None
                    self.Models[lstm_ticker][model_type]['technical'] = None
                    self.Scalers[lstm_ticker][model_type]['fundamental'] = None
                    self.Scalers[lstm_ticker][model_type]['technical'] = None


        # Model Data Initialization
        self.data_list_tickers = {}
        self.has_initialized_model_data = {}
        if self.TO_SAVE_DATA:
            self.ModelData = {} 
        else:
            self.ModelData = pickle.loads(bytes(self.ObjectStore.ReadBytes(f"MODEL_DATA_{self.model_name}")))

        for lstm_ticker in self.lstm_tickers:
            self.data_list_tickers[lstm_ticker] = []
            self.has_initialized_model_data[lstm_ticker] = False
            if lstm_ticker not in self.ModelData:
                self.ModelData[lstm_ticker] = pd.DataFrame()

        self.Schedule.On(
            self.DateRules.EveryDay(self.ref_symbol),
            self.TimeRules.AfterMarketOpen(self.ref_symbol),
            self.Start_Of_Day,
        )

        self.Schedule.On(
            self.DateRules.EveryDay(self.ref_symbol),
            self.TimeRules.At(max(self.general_setting['FxLstm_prediction_hour'].values()), 5, 0),
            self.Prepare_Model_Data,
        )

        self.Schedule.On(
            self.DateRules.MonthStart(self.ref_symbol),
            self.TimeRules.AfterMarketOpen(self.ref_symbol, 1),
            self.Get_Month_Start_Date,
        )

        self.Schedule.On(
            self.DateRules.EveryDay(self.ref_symbol),
            self.TimeRules.AfterMarketOpen(self.ref_symbol, 2),
            self.Train_Model_Both,
        )

        self.Schedule.On(
            self.DateRules.EveryDay(self.ref_symbol),
            self.TimeRules.AfterMarketOpen(self.ref_symbol, 3),
            self.Train_Model_Hybrid,
        )

        self.external_data = {}
        for _dn in self.general_setting["external_data"]:
            self.external_data[_dn] = {}
            self.external_data[_dn]['time'] = None
            self.external_data[_dn]['value'] = None
            source = self.general_setting["external_data"][_dn]['source']

            if source == 'gsheet':
                self.Log(f"{str(self.Time)}: {_dn}: Loading Initial GSheet Data")
                link = self.general_setting["external_data"][_dn]['link']
                col_date = self.general_setting["external_data"][_dn]['col_date']
                col_val = self.general_setting["external_data"][_dn]['col_val']
                to_run = True
                while to_run:
                    try:
                        data = self.Download(link)
                        rows = []
                        for row in data.split('\n'):
                            rows.append(row.replace('\r', '').lower().split(','))
                        data_df = pd.DataFrame(np.array(rows[1:]), columns=rows[0])
                        data_df[col_date] = data_df[col_date].apply(lambda s: datetime.strptime(s, '%Y-%m-%d'))
                        data_df[col_val] = data_df[col_val].astype(float)
                        self.external_data[_dn]['data'] = data_df.copy()
                        to_run = False
                    except:
                        pass

                self.Log(f"{str(self.Time)}: {_dn}: Initial GSheet Data Loaded")

            if source == 'NasdaqDataLink':
                ref = self.general_setting["external_data"][_dn]['ref']
                self.external_data[_dn]['symbol'] = self.AddData(NasdaqDataLink, ref, Resolution.Daily).Symbol

            if source == 'equity':
                ticker = self.general_setting["external_data"][_dn]['ticker']
                self.external_data[_dn]['symbol'] = self.AddEquity(ticker, Resolution.Daily).Symbol

            if source == 'USTreasuryYieldCurveRate':
                ref = self.general_setting["external_data"][_dn]['ref']
                self.external_data[_dn]['symbol'] = self.AddData(USTreasuryYieldCurveRate, ref).Symbol


    def Start_Of_Day(self):

        if self.IS_LIVE and (not self.IsWarmingUp):
            for _dn in self.general_setting["external_data"]:
                source = self.general_setting["external_data"][_dn]['source']
                if source == 'gsheet':
                    self.Log(f"{str(self.Time)}: {_dn}: Loading GSheet Data")

                    link = self.general_setting["external_data"][_dn]['link']
                    col_date = self.general_setting["external_data"][_dn]['col_date']
                    col_val = self.general_setting["external_data"][_dn]['col_val']

                    to_run = True
                    while to_run:
                        try:
                            data = self.Download(link)
                            rows = []
                            for row in data.split('\n'):
                                rows.append(row.replace('\r', '').lower().split(','))
                            data_df = pd.DataFrame(np.array(rows[1:]), columns=rows[0])
                            data_df[col_date] = data_df[col_date].apply(lambda s: datetime.strptime(s, '%Y-%m-%d'))
                            data_df[col_val] = data_df[col_val].astype(float)
                            self.external_data[_dn]['data'] = data_df.copy()
                            to_run = False
                        except:
                            pass

                self.Log(f"{str(self.Time)}: {_dn}: GSheet Data Loaded")

        for _dn in self.general_setting["external_data"]:
            source = self.general_setting["external_data"][_dn]['source']
            if source == 'gsheet':
                col_date = self.general_setting["external_data"][_dn]['col_date']
                col_val = self.general_setting["external_data"][_dn]['col_val']
                lag_days = self.general_setting["external_data"][_dn]['lag_days']

                data = self.external_data[_dn]['data'][self.external_data[_dn]['data'][col_date] < (self.Time - timedelta(days=lag_days))]
                if len(data) > 0:
                    self.external_data[_dn]['time'] = data[col_date].values[-1]
                    self.external_data[_dn]['value'] = data[col_val].values[-1]

            if source == 'USTreasuryYieldCurveRate':
                col_date = self.general_setting["external_data"][_dn]['col_date']
                col_val = self.general_setting["external_data"][_dn]['col_val']
                symbol = self.external_data[_dn]['symbol']
                history = self.History(USTreasuryYieldCurveRate, symbol, 1, Resolution.Daily)
                history = history.reset_index()
                if len(history) > 0:
                    if col_val in history.columns:
                        self.external_data[_dn]['time'] = pd.to_datetime(history[col_date], utc=True).iloc[0].replace(tzinfo=None)
                        self.external_data[_dn]['value'] = history[col_val].values[0]


    def Prepare_Model_Data(self):
        # self.Log(f"{str(self.Time)}: Preparing Model Data")
        col_price = self.model_settings['col_price']
        col_price_cur = self.model_settings['col_price_cur']
        cols_data = self.model_settings['cols_data']
        col_fundamental = self.model_settings['col_fundamental']
        col_technical = self.model_settings['col_technical']
        start_year = self.model_settings['start_year']
        scaled_tickers = self.model_settings['scaled_tickers']
        prediction_lookforward_days = self.model_settings['prediction_lookforward_days']
        col_target_gains = f"gains_N{self.model_settings['prediction_lookforward_days']}D"
        inflation_map_dict = self.model_settings['inflation_map_dict']

        to_save_data = False
        for lstm_ticker in self.lstm_tickers:
            data_df = self.ModelData[lstm_ticker].copy()

            has_new_data = False
            if len(self.data_list_tickers[lstm_ticker]) > 0:
                has_new_data = True

                data_df_new = pd.DataFrame(self.data_list_tickers[lstm_ticker]).copy()

                if lstm_ticker in scaled_tickers:
                    data_df_new[col_price] = data_df_new[col_price] / 100
                    data_df_new[col_price_cur] = data_df_new[col_price_cur] / 100

                data_df_new = data_df_new[cols_data]

                data_df_new['year'] = data_df_new['datetime'].dt.year
                data_df_new['hour'] = data_df_new['datetime'].dt.hour
                data_df_new['month'] = data_df_new['datetime'].dt.month
                data_df_new['year_month'] = data_df_new['year'].astype(str) + "-" + data_df_new['month'].astype(str).apply(lambda s: s.zfill(2))

                data_df = pd.concat([data_df, data_df_new])

            if len(data_df) > 0:
                if (not self.has_initialized_model_data[lstm_ticker]) or has_new_data:
                    self.has_initialized_model_data[lstm_ticker] = True

                    data_df.reset_index(drop=True, inplace=True)
                    if self.TO_SAVE_DATA:
                        data_df.drop_duplicates('datetime', keep='last', inplace=True)
                    else:
                        data_df.drop_duplicates('datetime', keep='first', inplace=True)

                    data_df.reset_index(drop=True, inplace=True)
                    data_df.sort_values('datetime', ascending=True, inplace=True)
                    data_df.reset_index(drop=True, inplace=True)

                    for col in col_fundamental + col_technical:
                        data_df[col] = data_df[col].fillna(method='ffill')

                    data_df = data_df[data_df['year'] >= start_year]
                    data_df = data_df[data_df['hour'] == self.general_setting['FxLstm_prediction_hour'][lstm_ticker]]
                    data_df.reset_index(drop=True, inplace=True)

                    for col in inflation_map_dict:
                        col_cpi = inflation_map_dict[col]

                        ## FRED CPI value is contribution to inflation. To test using it directly without differencing, as well as  d1, d30 difference on BOTH
                        # data_df[f"{col}_d1"] = (data_df[col_cpi] - data_df[col_cpi].shift(1)) / data_df[col_cpi].shift(1)
                        # data_df[f"{col}_d30"] = (data_df[col_cpi] - data_df[col_cpi].shift(30)) / data_df[col_cpi].shift(30)
                        data_df[f"{col}_d1"] = data_df[col_cpi] - data_df[col_cpi].shift(1)
                        data_df[f"{col}_d30"] = data_df[col_cpi] - data_df[col_cpi].shift(1)  

                    data_df[col_target_gains] = data_df[col_price].shift(-prediction_lookforward_days) - data_df[col_price]

                    self.ModelData[lstm_ticker] = data_df.copy()
                    self.data_list_tickers[lstm_ticker] = []

                    to_save_data = True

        if to_save_data and self.TO_SAVE_DATA:
            self.ObjectStore.SaveBytes(f"MODEL_DATA_{self.model_name}", pickle.dumps(self.ModelData))
            self.Log(f"{str(self.Time)}: Model Data Saved At: MODEL_DATA_{self.model_name}")

        # self.Log(f"{str(self.Time)}: Model Data Prepared")


    def Get_Month_Start_Date(self):
        self.month_start_date = self.Time


    def Train_Model_Both(self):
        model_type = "both"
        # self.Log(f"{str(self.Time)}: {model_type}: Training Model")
        model_setting = self.model_settings[f"model_settings_{model_type}"]

        col_date = self.model_settings['col_date']
        col_price = self.model_settings['col_price']
        col_price_cur = self.model_settings['col_price_cur']
        col_target = self.model_settings['col_target']
        prediction_lookforward_days = self.model_settings['prediction_lookforward_days']
        col_target_gains = f"gains_N{self.model_settings['prediction_lookforward_days']}D"

        use_gru_model = model_setting['use_gru_model']
        use_dual_lstm = model_setting['use_dual_lstm']
        epochs = model_setting['epochs']
        hidden_size = model_setting['hidden_size']
        window_size = model_setting['window_size']
        thres_multiplier = model_setting['thres_multiplier']
        use_early_stop = model_setting['use_early_stop']
        learning_rate = model_setting['learning_rate']
        batch_size = model_setting['batch_size']
        use_weighted_sampler = model_setting['use_weighted_sampler']
        volatility_type = model_setting['volatility_type']
        valid_lookback_months = model_setting['valid_lookback_months']
        train_lookback_months = model_setting['train_lookback_months']
        inflation_map_dict = self.model_settings['inflation_map_dict']

        for lstm_ticker in self.lstm_tickers:

            if self.month_start_date is None:
                continue
            else:
                month_start_year = self.month_start_date.year
                month_start_month = self.month_start_date.month
                month_start_day = self.month_start_date.day

            model_train_day = month_start_day + self.general_setting["lstm_model_training_displace_days"][lstm_ticker]
            to_train = (self.Time.year == month_start_year) and (self.Time.month == month_start_month) and (self.Time.day == model_train_day)
            if not to_train:
                continue

            data_df = self.ModelData[lstm_ticker].copy()
            if len(data_df) == 0:
                continue

            col_feature_both = model_setting["col_feature_dict"][lstm_ticker]
            year_month_list = sorted(list(set(data_df['year_month'])))
            year_month_vec = np.array(year_month_list)
            year_vec = np.array(sorted(list(set(data_df['year']))))

            test_year_month = f"{self.Time.year}-{str(self.Time.month).zfill(2)}"
            valid_year_month_list = list(year_month_vec[year_month_vec < test_year_month][-valid_lookback_months:])
            if len(valid_year_month_list) < valid_lookback_months:
                continue

            if np.sum(year_month_vec < min(valid_year_month_list)) == 0:
                continue

            train_year_month_list = list(year_month_vec[year_month_vec < min(valid_year_month_list)][-train_lookback_months:])
            if len(train_year_month_list) < train_lookback_months:
                continue

            data_df_temp = data_df.copy()

            if volatility_type == 'thres_v1':
                col_target_gains_thres = 0.00200
                data_df_temp[col_target] = 1
                data_df_temp.loc[data_df_temp[col_target_gains] < -col_target_gains_thres, col_target] = 0
                data_df_temp.loc[data_df_temp[col_target_gains] > col_target_gains_thres, col_target] = 2

            if volatility_type == 'thres_v2':
                col_target_gains_thres = 0.00235
                data_df_temp[col_target] = 1
                data_df_temp.loc[data_df_temp[col_target_gains] < -col_target_gains_thres, col_target] = 0
                data_df_temp.loc[data_df_temp[col_target_gains] > col_target_gains_thres, col_target] = 2

            if volatility_type == 'thres_auto_v1':
                thres_df = data_df_temp[data_df_temp['year_month'].isin(train_year_month_list)].copy()
                thres_df.reset_index(drop=True, inplace=True)
                col_target_gains_thres = get_threshold(thres_df[col_price]) * thres_multiplier
                data_df_temp[col_target] = 1
                data_df_temp.loc[data_df_temp[col_target_gains] < -col_target_gains_thres, col_target] = 0
                data_df_temp.loc[data_df_temp[col_target_gains] > col_target_gains_thres, col_target] = 2

            if volatility_type == 'thres_auto_v2':
                thres_df = data_df_temp[data_df_temp['year_month'].isin(train_year_month_list+valid_year_month_list)].copy()
                thres_df.reset_index(drop=True, inplace=True)
                col_target_gains_thres = get_threshold(thres_df[col_price]) * thres_multiplier
                data_df_temp[col_target] = 1
                data_df_temp.loc[data_df_temp[col_target_gains] < -col_target_gains_thres, col_target] = 0
                data_df_temp.loc[data_df_temp[col_target_gains] > col_target_gains_thres, col_target] = 2

            self.ModelParams[lstm_ticker][model_type]['col_target_gains_thres'] = col_target_gains_thres

            data_df_temp = data_df_temp.dropna()
            data_df_temp.reset_index(drop=True, inplace=True)

            train_df = data_df_temp[data_df_temp['year_month'].isin(train_year_month_list)].copy()
            valid_df = data_df_temp[data_df_temp['year_month'].isin(valid_year_month_list)].copy()

            valid_df_windowed = pd.concat([train_df,valid_df]).copy()
            valid_df_windowed = valid_df_windowed.tail(len(valid_df) + window_size-1)

            set_seed(100)
            (train_loader, val_loader, 
            _, scaler, weighted_sampler, class_weights) = get_torch_rnn_dataloaders_both(
                [col_price] + col_feature_both, col_target, train_df, valid_df_windowed, None, window_size, batch_size,
                use_weighted_sampler=use_weighted_sampler,
                has_test_data=False,
            )
            self.Scalers[lstm_ticker][model_type]['both'] = None
            self.Scalers[lstm_ticker][model_type]['both'] = scaler
            self.Models[lstm_ticker][model_type]['both'] = None
            self.Models[lstm_ticker][model_type]['both'] = get_rnn_model_both(
                [col_price] + col_feature_both, train_loader, val_loader, 
                epochs, batch_size, learning_rate, window_size, hidden_size, device, 
                use_early_stop=use_early_stop, use_weighted_sampler=use_weighted_sampler, class_weights=class_weights,
                use_dual_lstm=use_dual_lstm, use_gru_model=use_gru_model,
            )
            self.Log(f"{str(self.Time)}: {model_type}: {lstm_ticker}: Model Trained")


    def Train_Model_Hybrid(self):
        model_type = "hybrid"
        # self.Log(f"{str(self.Time)}: {model_type}: Training Model")
        model_setting = self.model_settings[f"model_settings_{model_type}"]

        col_date = self.model_settings['col_date']
        col_price = self.model_settings['col_price']
        col_price_cur = self.model_settings['col_price_cur']
        col_target = self.model_settings['col_target']
        prediction_lookforward_days = self.model_settings['prediction_lookforward_days']
        col_target_gains = f"gains_N{self.model_settings['prediction_lookforward_days']}D"

        use_gru_model = model_setting['use_gru_model']
        use_dual_lstm = model_setting['use_dual_lstm']
        epochs = model_setting['epochs']
        hidden_size = model_setting['hidden_size']
        window_size = model_setting['window_size']
        thres_multiplier = model_setting['thres_multiplier']
        learning_rate = model_setting['learning_rate']
        batch_size = model_setting['batch_size']
        volatility_type = model_setting['volatility_type']
        valid_lookback_months = model_setting['valid_lookback_months']
        train_lookback_months = model_setting['train_lookback_months']
        inflation_map_dict = self.model_settings['inflation_map_dict']

        for lstm_ticker in self.lstm_tickers:

            if self.month_start_date is None:
                continue
            else:
                month_start_year = self.month_start_date.year
                month_start_month = self.month_start_date.month
                month_start_day = self.month_start_date.day

            model_train_day = month_start_day + self.general_setting["lstm_model_training_displace_days"][lstm_ticker]
            to_train = (self.Time.year == month_start_year) and (self.Time.month == month_start_month) and (self.Time.day == model_train_day)
            if not to_train:
                continue

            data_df = self.ModelData[lstm_ticker]
            if len(data_df) == 0:
                continue

            col_feature_fundamental = model_setting["col_feature_fundamental_dict"][lstm_ticker]
            col_feature_technical = model_setting["col_feature_technical_dict"][lstm_ticker]

            year_month_list = sorted(list(set(data_df['year_month'])))
            year_month_vec = np.array(year_month_list)
            year_vec = np.array(sorted(list(set(data_df['year']))))

            test_year_month = f"{self.Time.year}-{str(self.Time.month).zfill(2)}"
            valid_year_month_list = list(year_month_vec[year_month_vec < test_year_month][-valid_lookback_months:])
            if len(valid_year_month_list) < valid_lookback_months:
                continue

            if np.sum(year_month_vec < min(valid_year_month_list)) == 0:
                continue

            train_year_month_list = list(year_month_vec[year_month_vec < min(valid_year_month_list)][-train_lookback_months:])
            if len(train_year_month_list) < train_lookback_months:
                continue

            data_df_temp = data_df.copy()

            if volatility_type == 'thres_v1':
                col_target_gains_thres = 0.00200

            if volatility_type == 'thres_v2':
                col_target_gains_thres = 0.00235

            if volatility_type == 'thres_auto_v1':
                thres_df = data_df_temp[data_df_temp['year_month'].isin(train_year_month_list)].copy()
                thres_df.reset_index(drop=True, inplace=True)
                col_target_gains_thres = get_threshold(thres_df[col_price]) * thres_multiplier

            if volatility_type == 'thres_auto_v2':
                thres_df = data_df_temp[data_df_temp['year_month'].isin(train_year_month_list+valid_year_month_list)].copy()
                thres_df.reset_index(drop=True, inplace=True)
                col_target_gains_thres = get_threshold(thres_df[col_price]) * thres_multiplier

            self.ModelParams[lstm_ticker][model_type]['col_target_gains_thres'] = col_target_gains_thres

            data_df_temp[col_target] = data_df_temp[col_price].shift(-prediction_lookforward_days)
            data_df_temp = data_df_temp.dropna()
            data_df_temp.reset_index(drop=True, inplace=True)

            train_df = data_df_temp[data_df_temp['year_month'].isin(train_year_month_list)].copy()
            valid_df = data_df_temp[data_df_temp['year_month'].isin(valid_year_month_list)].copy()

            set_seed(100)
            (train_loader, val_loader, _, scaler) = get_torch_rnn_dataloaders_hybrid(
                [col_price] + col_feature_fundamental, col_target, train_df, valid_df, None, window_size, batch_size,
                has_test_data=False,
            )

            self.Scalers[lstm_ticker][model_type]['fundamental'] = None
            self.Scalers[lstm_ticker][model_type]['fundamental'] = scaler

            self.Models[lstm_ticker][model_type]['fundamental'] = None
            self.Models[lstm_ticker][model_type]['fundamental'] = get_rnn_model_hybrid(
                [col_price] + col_feature_fundamental, train_loader, val_loader, 
                epochs, learning_rate, hidden_size, device, 
                use_dual_lstm=use_dual_lstm, use_gru_model=use_gru_model,
            )

            y_pred_val = get_predictions_hybrid(
                val_loader, 
                self.Models[lstm_ticker][model_type]['fundamental'], 
                self.Scalers[lstm_ticker][model_type]['fundamental'], 
                [col_price] + col_feature_fundamental, 
                device,
            )
            valid_df['pred_price_fundamental'] = y_pred_val
            valid_df['pred_fundamental'] = (valid_df['pred_price_fundamental'] - valid_df[col_price]).apply(get_regression_pred_decision, col_target_gains_thres=col_target_gains_thres)

            set_seed(100)
            (train_loader, val_loader, _, scaler) = get_torch_rnn_dataloaders_hybrid(
                [col_price] + col_feature_technical, col_target, train_df, valid_df, None, window_size, batch_size,
                has_test_data=False,
            )

            self.Scalers[lstm_ticker][model_type]['technical'] = scaler

            self.Models[lstm_ticker][model_type]['technical'] = get_rnn_model_hybrid(
                [col_price] + col_feature_technical, train_loader, val_loader, 
                epochs, learning_rate, hidden_size, device, 
                use_dual_lstm=use_dual_lstm, use_gru_model=use_gru_model,
            )

            y_pred_val = get_predictions_hybrid(
                val_loader, 
                self.Models[lstm_ticker][model_type]['technical'], 
                self.Scalers[lstm_ticker][model_type]['technical'], 
                [col_price] + col_feature_technical, 
                device,
            )
            valid_df['pred_price_technical'] = y_pred_val
            valid_df['pred_technical'] = (valid_df['pred_price_technical'] - valid_df[col_price]).apply(get_regression_pred_decision, col_target_gains_thres=col_target_gains_thres)

            fundamental_mse = mean_squared_error(valid_df['pred_price_fundamental'], valid_df[col_target])
            technical_mse = mean_squared_error(valid_df['pred_price_technical'], valid_df[col_target])

            self.ModelParams[lstm_ticker][model_type]['fundamental_mse'] = fundamental_mse
            self.ModelParams[lstm_ticker][model_type]['technical_mse'] = technical_mse
            self.Log(f"{str(self.Time)}: {model_type}: {lstm_ticker}: Model Trained")


    def OnData(self, data):

        # Prepare external Data
        for _dn in self.general_setting["external_data"]:
            source = self.general_setting["external_data"][_dn]['source']

            if source == 'NasdaqDataLink':
                symbol = self.external_data[_dn]['symbol']
                if data.ContainsKey(symbol):
                    self.external_data[_dn]['value'] = data[symbol].Value

            if source == 'equity':
                symbol = self.external_data[_dn]['symbol']
                if data.ContainsKey(symbol):
                    if data[symbol] is not None:
                        self.external_data[_dn]['time'] = data[symbol].Time
                        self.external_data[_dn]['value'] = data[symbol].Price

        # Roll over futures contract
        for ticker in self.lstm_tickers:
            if ticker in self.FuturesTracker:
                if self.FuturesTracker[ticker] is not None: 
                    if self.Portfolio[self.FuturesTracker[ticker]].Quantity == 0:
                        self.FuturesTracker[ticker] = None

                futures_chain = data.FuturesChains.get(self.FuturesSymbol[ticker])
                if futures_chain:
                    # Select the contract with the greatest open interest
                    ref_futures_contract = sorted(futures_chain, key=lambda contract: contract.OpenInterest, reverse=True)[0]
                    self.FuturesRefSymbol[ticker] = ref_futures_contract.Symbol

                if data.SymbolChangedEvents.ContainsKey(self.FuturesSymbol[ticker]):
                    futures_changed_event = data.SymbolChangedEvents[self.FuturesSymbol[ticker]]
                    old_futures_symbol = futures_changed_event.OldSymbol
                    new_futures_symbol = futures_changed_event.NewSymbol
                    rollover_futures_symbol = None
                    if self.FuturesRefSymbol[ticker] != old_futures_symbol:
                        rollover_futures_symbol = self.FuturesRefSymbol[ticker]
                    else:
                        rollover_futures_symbol = new_futures_symbol

                    if self.FuturesTracker[ticker] is not None: 
                        if old_futures_symbol == self.FuturesTracker[ticker]: 

                            rollover_quantity = self.Portfolio[old_futures_symbol].Quantity
                            if rollover_quantity != 0:
                                self.MarketOrder(old_futures_symbol, -rollover_quantity)
                                self.MarketOrder(rollover_futures_symbol, rollover_quantity)
            
                            self.FuturesTracker[ticker] = rollover_futures_symbol   


        FxLstm_SymbolQuantity = {}
        for symbol, symbolData in self.Data.items():
            if not (
                data.ContainsKey(symbol)
                and data[symbol] is not None
                and symbolData.IsReady
            ):
                continue

            ticker = self.symbol_ticker_map[symbol]

            is_valid_time = self.Time.minute == 0
            is_valid_time = is_valid_time and (self.Time.hour in [self.general_setting['FxLstm_prediction_hour'][ticker]])

            if is_valid_time:
                if ticker in self.lstm_tickers:

                    data_dict = {}
                    data_dict["datetime"] = self.Time
                    data_dict["price"] = np.round(data[symbol].Price, 6)

                    # Daily Data
                    _consolidator = symbolData.consolidators["D1"]
                    data_dict["close_D1"] = _consolidator.close[0]

                    # External Data
                    for _dn in self.general_setting["external_data"]:  
                        data_dict[_dn] = self.external_data[_dn]['value']

                    # Technical Features
                    for _tf in self.general_setting["features"]:
                        _consolidator = symbolData.consolidators[_tf]
                        for _in in self.general_setting["features"][_tf]:
                            _indicator = _consolidator.indicators[_in]

                            if _in in self.general_setting["features_val_map"]:
                                for _v in self.general_setting["features_val_map"][
                                    _in
                                ]:
                                    data_dict[f"{_tf}-{_in}-{_v}"] = np.round(
                                        _indicator[_v][0], 5
                                    )

                    if self.TO_SAVE_DATA:
                        if not self.IsWarmingUp:    
                            self.data_list_tickers[ticker] += [data_dict]
                    else:
                        self.data_list_tickers[ticker] += [data_dict]

                    col_price = self.model_settings['col_price']
                    col_price_cur = self.model_settings['col_price_cur']
                    cols_data = self.model_settings['cols_data']
                    col_fundamental = self.model_settings['col_fundamental']
                    col_technical = self.model_settings['col_technical']
                    start_year = self.model_settings['start_year']
                    col_target = self.model_settings['col_target']
                    scaled_tickers = self.model_settings['scaled_tickers']
                    inflation_map_dict = self.model_settings['inflation_map_dict']
                    max_window_size = self.model_settings['max_window_size']

                    test_df = pd.DataFrame()
                    if len(self.data_list_tickers[ticker]) > 0:
                        data_df_new = pd.DataFrame(self.data_list_tickers[ticker]).copy()
                        if ticker in scaled_tickers:
                            data_df_new[col_price] = data_df_new[col_price] / 100
                            data_df_new[col_price_cur] = data_df_new[col_price_cur] / 100

                        data_df_new = data_df_new[cols_data]

                        data_df_new['year'] = data_df_new['datetime'].dt.year
                        data_df_new['hour'] = data_df_new['datetime'].dt.hour
                        data_df_new['month'] = data_df_new['datetime'].dt.month
                        data_df_new['year_month'] = data_df_new['year'].astype(str) + "-" + data_df_new['month'].astype(str).apply(lambda s: s.zfill(2))

                        data_df = self.ModelData[ticker].copy()
                        if len(data_df) > 0:
                            idx = data_df['datetime'] < self.Time
                            data_df = data_df[idx]
                            data_df.reset_index(drop=True, inplace=True)

                        data_df = pd.concat([data_df, data_df_new])
                        data_df.reset_index(drop=True, inplace=True)

                        data_df.drop_duplicates('datetime', keep='last', inplace=True)
                        data_df.reset_index(drop=True, inplace=True)

                        data_df.sort_values('datetime', ascending=True, inplace=True)
                        data_df.reset_index(drop=True, inplace=True)

                        for col in col_fundamental + col_technical:
                            data_df[col] = data_df[col].fillna(method='ffill')

                        data_df = data_df[data_df['year'] >= start_year]
                        data_df = data_df[data_df['hour'] == self.general_setting['FxLstm_prediction_hour'][ticker]]
                        data_df.reset_index(drop=True, inplace=True)

                        for col in inflation_map_dict:
                            col_cpi = inflation_map_dict[col]

                            ## FRED CPI value is contribution to inflation. To test using it directly without differencing, as well as  d1, d30 difference on BOTH
                            # data_df[f"{col}_d1"] = (data_df[col_cpi] - data_df[col_cpi].shift(1)) / data_df[col_cpi].shift(1)
                            # data_df[f"{col}_d30"] = (data_df[col_cpi] - data_df[col_cpi].shift(30)) / data_df[col_cpi].shift(30)
                            data_df[f"{col}_d1"] = data_df[col_cpi] - data_df[col_cpi].shift(1)
                            data_df[f"{col}_d30"] = data_df[col_cpi] - data_df[col_cpi].shift(1)

                        test_df = data_df.tail(max_window_size).copy()
                        test_df.reset_index(drop=True, inplace=True)


                    for model_type in self.general_setting["model_types"]:
                        if len(test_df) == 0:
                            continue

                        if model_type == 'both':
                            if self.Models[ticker][model_type]['both'] is None:
                                continue

                        if model_type == 'hybrid':
                            if self.Models[ticker][model_type]['fundamental'] is None:
                                continue
                            if self.Models[ticker][model_type]['technical'] is None:
                                continue

                        model_setting = self.model_settings[f"model_settings_{model_type}"]

                        test_df_windowed = test_df.tail(model_setting['window_size']).copy()
                        test_df_windowed.reset_index(drop=True, inplace=True)

                        if len(test_df_windowed) != model_setting['window_size']:
                            continue

                        if model_type == 'both':
                            col_feature_both = model_setting["col_feature_dict"][ticker]
                            test_df_windowed[col_target] = 1

                            (_, _, test_loader, _, _, _) = get_torch_rnn_dataloaders_both(
                                [col_price] + col_feature_both, col_target, None, None, test_df_windowed.copy(), 
                                model_setting['window_size'], model_setting['batch_size'],
                                use_weighted_sampler=False,
                                has_test_data=True,
                                is_training=False,
                                scaler=self.Scalers[ticker][model_type]['both'],
                            )

                            (y_pred_list, y_score_list) = get_predictions_both(test_loader, self.Models[ticker][model_type]['both'], device)
                            y_pred = y_pred_list[-1]

                        if model_type == 'hybrid':
                            col_feature_fundamental = model_setting["col_feature_fundamental_dict"][ticker]
                            col_feature_technical = model_setting["col_feature_technical_dict"][ticker]
                            test_df_windowed[col_target] = 1
                            ref_price = test_df_windowed[col_price].values[-1]
                            col_target_gains_thres = self.ModelParams[ticker][model_type]['col_target_gains_thres']
                            fundamental_mse = self.ModelParams[ticker][model_type]['fundamental_mse']
                            technical_mse = self.ModelParams[ticker][model_type]['technical_mse']

                            (_, _, test_loader, _) = get_torch_rnn_dataloaders_hybrid(
                                [col_price] + col_feature_fundamental, col_target, None, None, test_df_windowed.copy(), 
                                model_setting['window_size'], model_setting['batch_size'],
                                has_test_data=True,
                                is_training=False,
                                scaler=self.Scalers[ticker][model_type]['fundamental'],
                            )

                            y_pred_val = get_predictions_hybrid(
                                test_loader, 
                                self.Models[ticker][model_type]['fundamental'], 
                                self.Scalers[ticker][model_type]['fundamental'],
                                [col_price] + col_feature_fundamental,
                                device,
                            )
                            pred_price_fundamental = y_pred_val[-1]
                            pred_fundamental = get_regression_pred_decision(pred_price_fundamental - ref_price, col_target_gains_thres)

                            (_, _, test_loader, _) = get_torch_rnn_dataloaders_hybrid(
                                [col_price] + col_feature_technical, col_target, None, None, test_df_windowed.copy(), 
                                model_setting['window_size'], model_setting['batch_size'],
                                has_test_data=True,
                                is_training=False,
                                scaler=self.Scalers[ticker][model_type]['technical'],
                            )

                            y_pred_val = get_predictions_hybrid(
                                test_loader, 
                                self.Models[ticker][model_type]['technical'], 
                                self.Scalers[ticker][model_type]['technical'],
                                [col_price] + col_feature_technical,
                                device,
                            )
                            pred_price_technical = y_pred_val[-1]
                            pred_technical = get_regression_pred_decision(pred_price_technical - ref_price, col_target_gains_thres)
                            y_pred = get_prediction_hybrid_regression(pred_fundamental, pred_technical, fundamental_mse, technical_mse)


                        for _signal in self.signals:
                            if ticker in self.signal_settings[_signal]['valid_tickers']:
                                pred_type = self.signal_settings[_signal]['pred_type']
                                lstm_ticker = self.signal_settings[_signal]['lstm_ticker']   

                                if (pred_type == model_type) and (lstm_ticker == ticker):
                                    self.prediction_dict[_signal][ticker] = y_pred
                                    self.Signal[symbol][_signal].update_prediction_direction(self.prediction_dict[_signal][ticker])


            symbolQuantity = 0
            for _signal in self.signals:
                if ticker in self.signal_settings[_signal]['valid_tickers']:

                    to_exit = self.Signal[symbol][_signal].check_exit(symbolData, data[symbol].Price, data[symbol].Time)
                    if to_exit:
                        self.Signal[symbol][_signal].update_exit()

                    has_enter = self.Signal[symbol][_signal].enter(symbolData, data[symbol].Price, data[symbol].Time)


                    if general_setting["tickers"][ticker]["type"] == "forex":
                        fx_spot_scaling_factor = self.general_setting["fx_spot_scaling_factor"][ticker]
                        quantity = self.Signal[symbol][_signal].quantity * self.Signal[symbol][_signal].allocation_multiplier * fx_spot_scaling_factor
                    else:
                        quantity = self.Signal[symbol][_signal].quantity * self.Signal[symbol][_signal].allocation_multiplier
           
                    quantity = int(np.ceil(quantity))
                    symbolQuantity += quantity

            FxLstm_SymbolQuantity[symbol] = symbolQuantity


        ## Aggregate symbol quantities across strategies
        for symbol, symbolData in self.Data.items():
            if not (
                data.ContainsKey(symbol)
                and data[symbol] is not None
                and symbolData.IsReady
            ):
                continue

            ticker = self.symbol_ticker_map[symbol]
            self.Counter[symbol]["counter"] += 1

            symbolQuantity = 0
            if symbol in FxLstm_SymbolQuantity:
                symbolQuantity += FxLstm_SymbolQuantity[symbol]

            symbolQuantityFutures = 0
            if general_setting["tickers"][ticker]["type"] == "forex_futures":
                if symbolQuantity != 0:
                    fx_future_units = self.general_setting["fx_future_units"][self.symbol_ticker_map[symbol]]
                    fx_future_scaling_factor = self.general_setting["fx_future_scaling_factor"][self.symbol_ticker_map[symbol]]
                    symbolQuantityFutures = int(np.sign(symbolQuantity) * np.sign(fx_future_units) * np.ceil(abs(symbolQuantity) * fx_future_scaling_factor / abs(fx_future_units)))

            if not self.IsWarmingUp:
                # In case orders takes longer than 1 bar to be filled. Only send market orders every 3 minutes
                if (self.Counter[symbol]["counter"] - self.Counter[symbol]["last_order_counter"]) >= self.general_setting["order_counter_diff"]:

                    if general_setting["tickers"][ticker]["type"] == "forex_futures":
                        if ticker in self.lstm_tickers:
                            if self.FuturesTracker[ticker] is None:
                                if (symbolQuantityFutures != 0) and (self.FuturesRefSymbol[ticker] is not None):
                                    self.FuturesTracker[ticker] = self.FuturesRefSymbol[ticker]
                                    self.MarketOrder(self.FuturesTracker[ticker], symbolQuantityFutures)
                                    self.Counter[symbol]["last_order_counter"] = self.Counter[symbol]["counter"]
                            else:
                                if (symbolQuantityFutures - self.Portfolio[self.FuturesTracker[ticker]].Quantity) != 0:
                                    self.MarketOrder(self.FuturesTracker[ticker], symbolQuantityFutures - self.Portfolio[self.FuturesTracker[ticker]].Quantity)
                                    self.Counter[symbol]["last_order_counter"] = self.Counter[symbol]["counter"]

                    elif general_setting["tickers"][ticker]["type"] == "forex":
                        quantityDiff = symbolQuantity - self.Portfolio[symbol].Quantity
                        quantityDiffAdjusted = quantityDiff
                        fx_spot_min_order_units = self.general_setting["fx_spot_min_order_units"][self.symbol_ticker_map[symbol]]

                        if (quantityDiff != 0) and (fx_spot_min_order_units is not None) and (abs(quantityDiff) < fx_spot_min_order_units):
                            quantityDiffAdjusted = int(np.sign(quantityDiff) * np.round(abs(quantityDiff) / fx_spot_min_order_units)) * fx_spot_min_order_units

                        if quantityDiffAdjusted != 0:
                            self.MarketOrder(symbol, quantityDiffAdjusted)
                            self.Counter[symbol]["last_order_counter"] = self.Counter[symbol]["counter"]

                    else:
                        quantityDiff = symbolQuantity - self.Portfolio[symbol].Quantity
                        if quantityDiff != 0:
                            self.MarketOrder(symbol, quantityDiff)
                            self.Counter[symbol]["last_order_counter"] = self.Counter[symbol]["counter"]



