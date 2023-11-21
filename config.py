#region imports
from AlgorithmImports import *
#endregion


## General Settings
general_setting = {
    "tickers": {
        "EURUSD": {"type": "forex"},
        "EURUSD_F": {"type": "forex_futures"},
        "GBPUSD": {"type": "forex"},
        "AUDUSD": {"type": "forex"},
        "USDJPY": {"type": "forex"},
    },
    
    "model_name": "ForexLSTM_V2_03",
    "consolidator_timeframes": ["D1", "W1"],
    "order_counter_diff": 3,
    "model_types": ["both","hybrid"],

    "lstm_tickers": ['EURUSD','EURUSD_F','GBPUSD','AUDUSD','USDJPY'],

    "lstm_model_training_displace_days": {
        'EURUSD': 0,
        'EURUSD_F': 0,
        'AUDUSD': 0,
        'USDJPY': 1,
        'GBPUSD': 1,
    },

    "fx_future_units": {
        'EURUSD_F': 12500,
        'GBPUSD_F': 6250,
        'AUDUSD_F': 10000,
        'USDJPY_F': -62500,
    },

    "fx_future_scaling_factor": {
        'EURUSD_F': 0.9,
        'GBPUSD_F': 0.9,
        'AUDUSD_F': 0.9,
        'USDJPY_F': 0.9,
    },

    "fx_spot_scaling_factor": {
        'EURUSD': 2,
        'GBPUSD': 2,
        'AUDUSD': 2,
        'USDJPY': 2,
    },

    "fx_spot_min_order_units": {
        'EURUSD': 20000,
        'GBPUSD': 20000,
        'AUDUSD': 25000,
        'USDJPY': 25000,
    },

    "signals": [
        "FxLstm_Both_EURUSD_Trail",
        "FxLstm_Both_USDJPY_Trail",
        "FxLstm_Hybrid_GBPUSD_Trail",
        "FxLstm_Both_AUDUSD",
        "FxLstm_Both_EURUSD_F",
    ], 

    "FxLstm_prediction_hour": {
        "EURUSD": 1,
        "EURUSD_F": 2,
        "GBPUSD": 1,
        "AUDUSD": 1,
        "USDJPY": 1, 
    },

    "external_data": {
        # SP500
        'spy': {
            'source': 'equity',
            'ticker': 'SPY',
        },

        # Global X DAX Germany ETF
        'dax': {
            'source': 'equity',
            'ticker': 'DAX',
        },

        # US Treasury
        'us_treasury': {
            'source': 'USTreasuryYieldCurveRate',
            'ref': 'USTYCR',
            'col_date': 'time',
            'col_val': 'onemonth',
        },

        # # Consumer Price Index for Inflation Rate
        # # https://data.nasdaq.com/data/RATEINF-inflation-rates
        # 'cpi_usa': {
        #     'source': 'NasdaqDataLink',
        #     'ref': "RATEINF/CPI_USA",
        # },
        # 'cpi_eur': {
        #     'source': 'NasdaqDataLink',
        #     'ref': "RATEINF/CPI_EUR",
        # },
        # 'cpi_deu': {
        #     'source': 'NasdaqDataLink',
        #     'ref': "RATEINF/CPI_DEU",
        # },
        # 'cpi_gbr': {
        #     'source': 'NasdaqDataLink',
        #     'ref': "RATEINF/CPI_GBR",
        # },
        # 'cpi_chf': {
        #     'source': 'NasdaqDataLink',
        #     'ref': "RATEINF/CPI_CHE",
        # },
        # 'cpi_jpn': {
        #     'source': 'NasdaqDataLink',
        #     'ref': "RATEINF/CPI_JPN",
        # },
        # 'cpi_can': {
        #     'source': 'NasdaqDataLink',
        #     'ref': "RATEINF/CPI_CAN",
        # },
        # 'cpi_aus': {
        #     'source': 'NasdaqDataLink',
        #     'ref': "RATEINF/CPI_AUS",
        # },

        # Consumer Price Index: All Items: Total for United States
        # https://fred.stlouisfed.org/series/USACPALTT01CTGYM
        'cpi_usa': {
            'source': 'gsheet',
            'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vRvNxE4jbVrfSsTTEgiDNEDexnYZb2masiq6TRUcOkhPZkGmv_HkqIoxjPWhvo9EhNWlL-J_KBLPHol/pub?gid=146996374&single=true&output=csv",
            'col_date': 'date',
            'col_val': 'USACPALTT01CTGYM'.lower(),
            'lag_days': 10,
        },

        # Consumer Price Index: All Items: Total: Total for the Euro Area (19 Countries)
        # https://fred.stlouisfed.org/series/EA19CPALTT01GYM
        'cpi_eur': {
            'source': 'gsheet',
            'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vRvf0cVUhP7zHjYYdC-G-xclugSxFvlwYD-pwwDfVjAZBS1QmXp-8ZHu76mapn2fVaz1C-qQBavO3C7/pub?gid=949965906&single=true&output=csv",
            'col_date': 'date',
            'col_val': 'EA19CPALTT01GYM'.lower(),
            'lag_days': 10,
        },

        # Consumer Price Index: All Items: Total for Germany
        # https://fred.stlouisfed.org/series/DEUCPALTT01CTGYM
        'cpi_deu': {
            'source': 'gsheet',
            'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vR7kCXUDxqhaNOuaVBqJRf8aC7-Jn10gYshNih733SJfOgZ_nFKatRMoAK9S7Ba5aSNx6NzTVZdrf-T/pub?gid=1418680730&single=true&output=csv",
            'col_date': 'date',
            'col_val': 'DEUCPALTT01CTGYM'.lower(),
            'lag_days': 10,
        },

        # Consumer Price Index: All Items: Total for United Kingdom
        # https://fred.stlouisfed.org/series/GBRCPALTT01CTGYM
        'cpi_gbr': {
            'source': 'gsheet',
            'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vRI9OhFmNZp-bOj7cEqWuVYLZVuoDUUOnDNXE89FoX8T7_kq8zQ_cCgObYY6SjkeOgoGxgcEibCCnbk/pub?gid=1762567982&single=true&output=csv",
            'col_date': 'date',
            'col_val': 'GBRCPALTT01CTGYM'.lower(),
            'lag_days': 10,
        },

        # Consumer Price Index: All Items: Total for Switzerland 
        # https://fred.stlouisfed.org/series/CHECPALTT01CTGYM
        'cpi_chf': {
            'source': 'gsheet',
            'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vR-Y9ggMABaJNckdttoV1rC4RgVbAq4feCR85uNAbJub6aa5iLEUSSjm2c-jgTeT8JoZpOSF8_TYtST/pub?gid=134775832&single=true&output=csv",
            'col_date': 'date',
            'col_val': 'CHECPALTT01CTGYM'.lower(),
            'lag_days': 10,
        },

        # Consumer Price Index: All Items: Total for Japan 
        # https://fred.stlouisfed.org/series/CPALTT01JPM659N
        'cpi_jpn': {
            'source': 'gsheet',
            'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vSPuyftpjMlII8T-4LkXeuX6qc5LNWuG8tlWfQ-Q9qmK7SvHlIrgyp0hwKQcCuJwo69KmC4QoeU_LMO/pub?gid=683346418&single=true&output=csv",
            'col_date': 'date',
            'col_val': 'CPALTT01JPM659N'.lower(),
            'lag_days': 10,
        },

        # # Consumer Price Index: All Items: Total for Canada 
        # # https://fred.stlouisfed.org/series/CANCPALTT01CTGYM
        # 'cpi_can': {
        #     'source': 'gsheet',
        #     'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vQLYucCiesuHZDBvnDX1cQqZ3LXlSXa8kOBe21RHC6pAtRiyhSMphTB7MwmNtdlb8hJr8te380HgNfR/pub?gid=1792688787&single=true&output=csv",
        #     'col_date': 'date',
        #     'col_val': 'CANCPALTT01CTGYM'.lower(),
        #     'lag_days': 10,
        # },

        # Consumer Price Index: All Items: Total for Australia 
        # https://fred.stlouisfed.org/series/CPALTT01AUQ659N
        'cpi_aus': {
            'source': 'gsheet',
            'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vT7AXyI8YZzfoUW-DackiJ13BfIldHZLgtJ1ixqScNT8qHhF8dmFdk73yJeD-gPmcplpIlIZzp7wBXl/pub?gid=88209986&single=true&output=csv",
            'col_date': 'date',
            'col_val': 'CPALTT01AUQ659N'.lower(),
            'lag_days': 10,
        },

        # Federal Funds Effective Rate (DFF)
        # https://fred.stlouisfed.org/series/DFF
        'dff': {
            'source': 'gsheet',
            'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vT5lyey5dhfrZifoZFuDwlQDOz6oILyUyAHTLVe2eqiLv9jWkNeIFITIeKqwBOtS8oEUOoZ2zXX1De7/pub?gid=1400614786&single=true&output=csv",
            'col_date': 'date',
            'col_val': 'dff',
            'lag_days': 1,
        },

        # Interest Rates: Long-Term Government Bond Yields: 10-Year: Main (Including Benchmark) for the Euro Area (19 Countries)
        # https://fred.stlouisfed.org/series/IRLTLT01EZM156N
        'rate_eur_lt_gov': {
            'source': 'gsheet',
            'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vSl_hxRnfcXnFly0Gh1vyZYNTRW6VTv-FQDlXuNUR1090RIst2a01nyhGl7tPR4VIcrgFfGBc3OSD72/pub?gid=1026565438&single=true&output=csv",
            'col_date': 'date',
            'col_val': 'IRLTLT01EZM156N'.lower(),
            'lag_days': 10,
        },

        # Interest Rates: 3-Month or 90-Day Rates and Yields: Interbank Rates: Total for the Euro Area (19 Countries)
        # https://fred.stlouisfed.org/series/IR3TIB01EZM156N
        'rate_eur_3m_bank': {
            'source': 'gsheet',
            'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vSkVfnj8N9AIsVF5PJN0JzU9ahw71nK_sTwY2qLKtNNxs1JI0STexUPEW15dY9bDUN8Fwql7_WUiKhK/pub?gid=2059310805&single=true&output=csv",
            'col_date': 'date',
            'col_val': 'IR3TIB01EZM156N'.lower(),
            'lag_days': 10,
        },

        # Interest Rates: Long-Term Government Bond Yields: 10-Year: Main (Including Benchmark) for Germany
        # https://fred.stlouisfed.org/series/IRLTLT01DEM156N
        'rate_deu_lt_gov': {
            'source': 'gsheet',
            'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vToUOn242L-w9ZWUXz_fU59aUc6oN5tDJEG8fu207zO7jMyfy5y7VesxH0mzEKaqwuU7WGOq7_xxDSu/pub?gid=2099864712&single=true&output=csv",
            'col_date': 'date',
            'col_val': 'IRLTLT01DEM156N'.lower(),
            'lag_days': 10,
        },

        # Interest Rates: 3-Month or 90-Day Rates and Yields: Interbank Rates: Total for Germany
        # https://fred.stlouisfed.org/series/IR3TIB01DEM156N
        'rate_deu_3m_bank': {
            'source': 'gsheet',
            'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vTswIuhg3-tLwgP6RWSSPRyyLDpvHNqdlSgSNk91_SkUjKAD9_lyvhI84MAHRHzYdrIho1Narccx_w1/pub?gid=1568788544&single=true&output=csv",
            'col_date': 'date',
            'col_val': 'IR3TIB01DEM156N'.lower(),
            'lag_days': 10,
        },

        # Interest Rates: 3-Month or 90-Day Rates and Yields: Interbank Rates: Total for Switzerland
        # https://fred.stlouisfed.org/series/IR3TIB01CHM156N
        'rate_chf_3m_bank': {
            'source': 'gsheet',
            'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vRRVvpohXIZOGQ4HpAjTTMeZ6cTat0wZ1gOxpUR_5E3pDuDCHDppiRnV9GQNK33jWJ3pYxAjvOvmerO/pub?gid=1734297228&single=true&output=csv",
            'col_date': 'date',
            'col_val': 'IR3TIB01CHM156N'.lower(),
            'lag_days': 10,
        },

        #  3-Month or 90-day Rates and Yields: Interbank Rates for Japan
        # https://fred.stlouisfed.org/series/IR3TIB01JPM156N
        'rate_jpn_3m_bank': {
            'source': 'gsheet',
            'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vTjcVbe63Ea3BoVxTpTBNcaEICdI11DhVmZ6Qxb-_GcuP8VbemKreHWNEu5id0ZviHPk7PAtLHqdBGr/pub?gid=1682849610&single=true&output=csv",
            'col_date': 'date',
            'col_val': 'IR3TIB01JPM156N'.lower(),
            'lag_days': 10,
        },

        # Interest Rates: 3-Month or 90-Day Rates and Yields: Interbank Rates: Total for Australia
        # https://fred.stlouisfed.org/series/IR3TIB01AUM156N
        'rate_aus_3m_bank': {
            'source': 'gsheet',
            'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vS73ca8pMDndu3lH5SjmrIJS-HwWfDdqS2mh1YQkQwhGj3UtIauP12xjhmLusXag9ibZJE3YZsWLERT/pub?gid=1639615970&single=true&output=csv",
            'col_date': 'date',
            'col_val': 'IR3TIB01AUM156N'.lower(),
            'lag_days': 10,
        },

        # Interest Rates: 3-Month or 90-Day Rates and Yields: Interbank Rates: Total for Canada 
        # https://fred.stlouisfed.org/series/IR3TIB01CAM156N
        'rate_cnd_3m_bank': {
            'source': 'gsheet',
            'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ0Odb11l33qCVwPS6G2lxrUpfQ5DWXnGw8HFu6uUV_OUx7b-yBIQItN12TwLRTq3Bx3-fBe-pU86ve/pub?gid=483123093&single=true&output=csv",
            'col_date': 'date',
            'col_val': 'IR3TIB01CAM156N'.lower(),
            'lag_days': 10,
        },

        # Interest Rates: 3-Month or 90-Day Rates and Yields: Interbank Rates: Total for United Kingdom
        # https://fred.stlouisfed.org/series/IR3TIB01GBM156N
        'rate_gbp_3m_bank': {
            'source': 'gsheet',
            'link': "https://docs.google.com/spreadsheets/d/e/2PACX-1vTtQSIEiSK1swVM_oElodv6YsjzojdxwfXZm-hDx68DQD6V3HjtuOYpHb4KUQC5uWDFoe9t09-Mibex/pub?gid=1120780951&single=true&output=csv",
            'col_date': 'date',
            'col_val': 'IR3TIB01GBM156N'.lower(),
            'lag_days': 10,
        },

    },

    "features": {
        "D1": [
            "SMA10","MACD","ROC2","MOM4","RSI10","BB20","CCI20","PSAR",
        ],
        "W1": [],
    },


    "features_val_map": {
        "SMA10": ["val"], 
        "MACD": ["macd", "macdsignal", "macdhist"],
        "ROC2": ["val"],   
        "MOM4": ["val"],    
        "RSI10": ["val"],
        "BB20": ["upper","lower","mid"],
        "CCI20": ["val"],
        "ULTOSC": ["val"],
        "CHOP": ["val"],
        "DX14": ["val"],
        "PHASE": ["val"],
        "CRSI": ["val"],
        "PSAR": ["val"],
    },

    
}

















## Consolidator Settings
consolidator_settings = {
    "D1": {
        "timeframe_minutes": 24 * 60,
        "indicators": [
            "SMA10","MACD","ROC2","MOM4","RSI10","BB20","CCI20",
            "ATR10","ATR14","ATR21",
            "PSAR",
            "SMA100",
            "SMA200",
        ],
        "window": 5,
        "window_multiplier_dict": {
            "forex": 1,   
            "forex_futures": 1,   
        },  
    },

    "W1": {
        "timeframe_minutes": 7 * 24 * 60,
        "indicators": [],
        "window": 5,
        "window_multiplier_dict": {
            "forex": 1,  
            "forex_futures": 1,   
        },  
    },

}



## Indicators Settings
indicator_settings = {
    "SMA5": {
        "type": "SMA",
        "lookback": 5,
        "field": "Close",
        "window": 3,
    }, 
    "SMA10": {
        "type": "SMA",
        "lookback": 10,
        "field": "Close",
        "window": 3,
    }, 
    "SMA20": {
        "type": "SMA",
        "lookback": 20,
        "field": "Close",
        "window": 3,
    }, 
    "SMA50": {
        "type": "SMA",
        "lookback": 50,
        "field": "Close",
        "window": 3,
    }, 
    "SMA100": {
        "type": "SMA",
        "lookback": 100,
        "field": "Close",
        "window": 3,
    }, 
    "SMA200": {
        "type": "SMA",
        "lookback": 200,
        "field": "Close",
        "window": 3,
    }, 
    "MACD": {
        "type": "MACD",
        "window": 3,
    },
    "ROC2": {
        "type": "ROC",
        "lookback": 2,
        "field": "Close",
        "window": 3,
    }, 
    "MOM4": {
        "type": "MOM",
        "lookback": 2,
        "field": "Close",
        "window": 3,
    }, 
    "RSI10": {
        "type": "RSI",
        "lookback": 10,
        "ma_type": "Simple",
        "field": "Close",
        "window": 3,
    },
    "BB20": {
        "type": "BOLL",
        "lookback": 20,
        "ma_type": "Simple",
        "std": 2,
        "field": "Close",
        "window": 3,
    },
    "CCI20": {
        "type": "CCI",
        "lookback": 20,
        "field": "Close",
        "window": 3,
    }, 

    "ATR10": {
        "type": "ATR",
        "lookback": 10,
        "field": "Close",
        "window": 3,
    },
    "ATR14": {
        "type": "ATR",
        "lookback": 14,
        "field": "Close",
        "window": 3,
    },
    "ATR21": {
        "type": "ATR",
        "lookback": 21,
        "field": "Close",
        "window": 3,
    },
    "ULTOSC": {
        "type": "ULTOSC",
        "window": 3,
    },
    "CHOP": {
        "type": "CHOP",
        "lookback": 52,
        "window": 3,
    },
    "DX14": {
        "type": "DX",
        "lookback": 14,
        "window": 3,
    },
    "PHASE": {
        "type": "PHASE",
        "lookback": 15,
        "window": 3,
    },
    "CRSI": {
        "type": "CRSI",
        "rsi_len": 15,
        "rsi_field": "Close",
        "rsi_window": 21,
        "window": 3,
    },
    "PSAR": {
        "type": "PSAR",
        "window": 3,
    },
}

signal_settings = {
    "FxLstm_Both_EURUSD": {
        "lstm_ticker": "EURUSD",  
        "valid_tickers": ["EURUSD"],
        "active_timeframe": "D1",
        "prediction_direction_map_dict": {
            0: -1,
            1: 0,
            2: 1,
        },
        "pred_type": 'both',
        "exit_wait_period": 0,
        "risk_pct": 0.0125,
        "enter_long_trades": True,
        "enter_short_trades": True,
        "use_sma_filter": True,
        "sma_filter_lookback": 100,
        "atrLength": 21,
        "longStopMultiplier": 0.1,
        "shortStopMultiplier": 0.2,
        "longRiskRewardMultiplier": 3,
        "shortRiskRewardMultiplier": 3,
        "useTralingStop": False,    
        "trailStopSize": 0.1,
        "use_movement_thres_for_stops": False,
        "movement_thres": 0.002,
        "use_prediction_direction_to_exit": False,
    },

    "FxLstm_Both_EURUSD_F": {
        "lstm_ticker": "EURUSD_F",  
        "valid_tickers": ["EURUSD_F"],
        "active_timeframe": "D1",
        "prediction_direction_map_dict": {
            0: -1,
            1: 0,
            2: 1,
        },
        "pred_type": 'both',
        "exit_wait_period": 0,
        "risk_pct": 0.0125,
        "enter_long_trades": True,
        "enter_short_trades": True,
        "use_sma_filter": True,
        "sma_filter_lookback": 100,
        "atrLength": 21,
        "longStopMultiplier": 0.1,
        "shortStopMultiplier": 0.2,
        "longRiskRewardMultiplier": 3,
        "shortRiskRewardMultiplier": 3,
        "useTralingStop": False,    
        "trailStopSize": 0.1,
        "use_movement_thres_for_stops": False,
        "movement_thres": 0.002,
        "use_prediction_direction_to_exit": False,
    },

    "FxLstm_Both_EURUSD_Trail": {
        "lstm_ticker": "EURUSD",  
        "valid_tickers": ["EURUSD"],
        "active_timeframe": "D1",
        "prediction_direction_map_dict": {
            0: -1,
            1: 0,
            2: 1,
        },
        "pred_type": 'both',
        "exit_wait_period": 0,
        "risk_pct": 0.01,
        "enter_long_trades": True,
        "enter_short_trades": True,
        "use_sma_filter": True,
        "sma_filter_lookback": 100,
        "atrLength": 21,
        "longStopMultiplier": 0.1,
        "shortStopMultiplier": 0.2,
        "longRiskRewardMultiplier": 3,
        "shortRiskRewardMultiplier": 3,
        "useTralingStop": True,    
        "trailStopSize": 0.1,
        "use_movement_thres_for_stops": False,
        "movement_thres": 0.002,
        "use_prediction_direction_to_exit": False,
    },

    "FxLstm_Hybrid_EURUSD": {
        "lstm_ticker": "EURUSD",  
        "valid_tickers": ["EURUSD"],
        "active_timeframe": "D1",
        "prediction_direction_map_dict": {
            0: -1,
            1: 0,
            2: 1,
        },
        "pred_type": 'hybrid',
        "exit_wait_period": 0,
        "risk_pct": 0.0125,
        "enter_long_trades": True,
        "enter_short_trades": True,
        "use_sma_filter": True,
        "sma_filter_lookback": 100,
        "atrLength": 10,
        "longStopMultiplier": 0.5,
        "shortStopMultiplier": 0.2,
        "longRiskRewardMultiplier": 3,
        "shortRiskRewardMultiplier": 3,
        "useTralingStop": False,    
        "trailStopSize": 5.0,
        "use_movement_thres_for_stops": False,
        "movement_thres": 0.002,
        "use_prediction_direction_to_exit": False,
    },

    "FxLstm_Hybrid_EURUSD_Trail": {
        "lstm_ticker": "EURUSD",  
        "valid_tickers": ["EURUSD"],
        "active_timeframe": "D1",
        "prediction_direction_map_dict": {
            0: -1,
            1: 0,
            2: 1,
        },
        "pred_type": 'hybrid',
        "exit_wait_period": 0,
        "risk_pct": 0.01,
        "enter_long_trades": True,
        "enter_short_trades": True,
        "use_sma_filter": True,
        "sma_filter_lookback": 100,
        "atrLength": 10,
        "longStopMultiplier": 0.5,
        "shortStopMultiplier": 0.2,
        "longRiskRewardMultiplier": 3,
        "shortRiskRewardMultiplier": 3,
        "useTralingStop": True,    
        "trailStopSize": 5.0,
        "use_movement_thres_for_stops": False,
        "movement_thres": 0.002,
        "use_prediction_direction_to_exit": False,
    },

    "FxLstm_Both_USDJPY": {
        "lstm_ticker": "USDJPY",  
        "valid_tickers": ["USDJPY"],
        "active_timeframe": "D1",
        "prediction_direction_map_dict": {
            0: -1,
            1: 0,
            2: 1,
        },
        "pred_type": 'both',
        "exit_wait_period": 0,
        "risk_pct": 0.0075,
        "enter_long_trades": True,
        "enter_short_trades": True,
        "use_sma_filter": True,
        "sma_filter_lookback": 100,
        "atrLength": 10,
        "longStopMultiplier": 1.0,
        "shortStopMultiplier": 0.5,
        "longRiskRewardMultiplier": 3,
        "shortRiskRewardMultiplier": 3,
        "useTralingStop": False,    
        "trailStopSize": 4.0,
        "use_movement_thres_for_stops": False,
        "movement_thres": 0.002,
        "use_prediction_direction_to_exit": False,
    },

    "FxLstm_Both_USDJPY_Trail": {
        "lstm_ticker": "USDJPY",  
        "valid_tickers": ["USDJPY"],
        "active_timeframe": "D1",
        "prediction_direction_map_dict": {
            0: -1,
            1: 0,
            2: 1,
        },
        "pred_type": 'both',
        "exit_wait_period": 0,
        "risk_pct": 0.005,
        "enter_long_trades": True,
        "enter_short_trades": True,
        "use_sma_filter": True,
        "sma_filter_lookback": 100,
        "atrLength": 10,
        "longStopMultiplier": 1.0,
        "shortStopMultiplier": 0.5,
        "longRiskRewardMultiplier": 3,
        "shortRiskRewardMultiplier": 3,
        "useTralingStop": True,    
        "trailStopSize": 4.0,
        "use_movement_thres_for_stops": False,
        "movement_thres": 0.002,
        "use_prediction_direction_to_exit": False,
    },

    "FxLstm_Hybrid_GBPUSD": {
        "lstm_ticker": "GBPUSD",  
        "valid_tickers": ["GBPUSD"],
        "active_timeframe": "D1",
        "prediction_direction_map_dict": {
            0: -1,
            1: 0,
            2: 1,
        },
        "pred_type": 'hybrid',
        "exit_wait_period": 0,
        "risk_pct": 0.0075,
        "enter_long_trades": True,
        "enter_short_trades": True,
        "use_sma_filter": True,
        "sma_filter_lookback": 100,
        "atrLength": 10,
        "longStopMultiplier": 0.40,
        "shortStopMultiplier": 0.75,
        "longRiskRewardMultiplier": 3,
        "shortRiskRewardMultiplier": 3,
        "useTralingStop": False,    
        "trailStopSize": 0.5,
        "use_movement_thres_for_stops": False,
        "movement_thres": 0.002,
        "use_prediction_direction_to_exit": False,
    },

    "FxLstm_Hybrid_GBPUSD_Trail": {
        "lstm_ticker": "GBPUSD",  
        "valid_tickers": ["GBPUSD"],
        "active_timeframe": "D1",
        "prediction_direction_map_dict": {
            0: -1,
            1: 0,
            2: 1,
        },
        "pred_type": 'hybrid',
        "exit_wait_period": 0,
        "risk_pct": 0.005,
        "enter_long_trades": True,
        "enter_short_trades": True,
        "use_sma_filter": True,
        "sma_filter_lookback": 100,
        "atrLength": 10,
        "longStopMultiplier": 0.40,
        "shortStopMultiplier": 0.75,
        "longRiskRewardMultiplier": 3,
        "shortRiskRewardMultiplier": 3,
        "useTralingStop": True,    
        "trailStopSize": 0.5,
        "use_movement_thres_for_stops": False,
        "movement_thres": 0.002,
        "use_prediction_direction_to_exit": False,
    },

    "FxLstm_Both_AUDUSD": {
        "lstm_ticker": "AUDUSD",  
        "valid_tickers": ["AUDUSD"],
        "active_timeframe": "D1",
        "prediction_direction_map_dict": {
            0: -1,
            1: 0,
            2: 1,
        },
        "pred_type": 'both',
        "exit_wait_period": 0,
        "risk_pct": 0.0075,
        "enter_long_trades": True,
        "enter_short_trades": True,
        "use_sma_filter": True,
        "sma_filter_lookback": 200,
        "atrLength": 21,
        "longStopMultiplier": 0.75,
        "shortStopMultiplier": 0.25,
        "longRiskRewardMultiplier": 3,
        "shortRiskRewardMultiplier": 3,
        "useTralingStop": False,    
        "trailStopSize": 0.2,
        "use_movement_thres_for_stops": False,
        "movement_thres": 0.002,
        "use_prediction_direction_to_exit": False,
    },

    "FxLstm_Both_AUDUSD_Trail": {
        "lstm_ticker": "AUDUSD",  
        "valid_tickers": ["AUDUSD"],
        "active_timeframe": "D1",
        "prediction_direction_map_dict": {
            0: -1,
            1: 0,
            2: 1,
        },
        "pred_type": 'both',
        "exit_wait_period": 0,
        "risk_pct": 0.005,
        "enter_long_trades": True,
        "enter_short_trades": True,
        "use_sma_filter": True,
        "sma_filter_lookback": 200,
        "atrLength": 21,
        "longStopMultiplier": 0.75,
        "shortStopMultiplier": 0.25,
        "longRiskRewardMultiplier": 3,
        "shortRiskRewardMultiplier": 3,
        "useTralingStop": True,    
        "trailStopSize": 0.2,
        "use_movement_thres_for_stops": False,
        "movement_thres": 0.002,
        "use_prediction_direction_to_exit": False,
    },

    "FxLstm_Hybrid_AUDUSD": {
        "lstm_ticker": "AUDUSD",  
        "valid_tickers": ["AUDUSD"],
        "active_timeframe": "D1",
        "prediction_direction_map_dict": {
            0: -1,
            1: 0,
            2: 1,
        },
        "pred_type": 'hybrid',
        "exit_wait_period": 0,
        "risk_pct": 0.0075,
        "enter_long_trades": True,
        "enter_short_trades": True,
        "use_sma_filter": True,
        "sma_filter_lookback": 200,
        "atrLength": 21,
        "longStopMultiplier": 0.5,
        "shortStopMultiplier": 1.0,
        "longRiskRewardMultiplier": 3,
        "shortRiskRewardMultiplier": 3,
        "useTralingStop": False,    
        "trailStopSize": 0.5,
        "use_movement_thres_for_stops": False,
        "movement_thres": 0.002,
        "use_prediction_direction_to_exit": False,
    },

    "FxLstm_Hybrid_AUDUSD_Trail": {
        "lstm_ticker": "AUDUSD",  
        "valid_tickers": ["AUDUSD"],
        "active_timeframe": "D1",
        "prediction_direction_map_dict": {
            0: -1,
            1: 0,
            2: 1,
        },
        "pred_type": 'hybrid',
        "exit_wait_period": 0,
        "risk_pct": 0.005,
        "enter_long_trades": True,
        "enter_short_trades": True,
        "use_sma_filter": True,
        "sma_filter_lookback": 200,
        "atrLength": 21,
        "longStopMultiplier": 0.5,
        "shortStopMultiplier": 1.0,
        "longRiskRewardMultiplier": 3,
        "shortRiskRewardMultiplier": 3,
        "useTralingStop": True,    
        "trailStopSize": 0.5,
        "use_movement_thres_for_stops": False,
        "movement_thres": 0.002,
        "use_prediction_direction_to_exit": False,
    },

}





model_settings = {
    "col_date": ['datetime'],
    "col_price": 'close_D1',
    "col_price_cur": 'price',
    "col_target": 'target',
    "start_year": 2013,
    "prediction_lookforward_days": 1,
    "max_window_size": 100,
    "scaled_tickers": ["USDJPY","USDJPY_F"],

    "inflation_map_dict": {
        'inflation_usa': 'cpi_usa',
        'inflation_eur': 'cpi_eur',
        'inflation_deu': 'cpi_deu',
        'inflation_gbr': 'cpi_gbr',
        'inflation_chf': 'cpi_chf',
        'inflation_jpn': 'cpi_jpn',
        'inflation_aus': 'cpi_aus',
    },

    "cols_data": [
        'datetime', 'close_D1', 'price', 
        'spy', 'dax', 
        'dff',
        'cpi_usa', 'cpi_eur', 'cpi_deu', 'cpi_gbr','cpi_chf','cpi_jpn','cpi_aus',
        'rate_eur_3m_bank',
        'rate_deu_3m_bank',
        'rate_chf_3m_bank',
        'rate_jpn_3m_bank',
        'rate_aus_3m_bank',
        'rate_cnd_3m_bank',
        'rate_gbp_3m_bank',
        'D1-SMA10-val', 'D1-MACD-macd', 'D1-MACD-macdsignal', 'D1-MACD-macdhist', 'D1-ROC2-val',
        'D1-MOM4-val', 'D1-RSI10-val', 'D1-BB20-upper', 'D1-BB20-mid', 'D1-BB20-lower', 'D1-CCI20-val',
        'D1-PSAR-val',
    ],

    'col_fundamental': [
        'spy','dax',
        'dff',
        'cpi_usa','cpi_eur','cpi_deu','cpi_gbr','cpi_chf','cpi_jpn','cpi_aus',
        'rate_eur_3m_bank',
        'rate_deu_3m_bank',
        'rate_chf_3m_bank',
        'rate_jpn_3m_bank',
        'rate_aus_3m_bank',
        'rate_cnd_3m_bank',
        'rate_gbp_3m_bank',
    ],

    'col_technical': [
        'D1-SMA10-val',
        'D1-MACD-macd',
        'D1-MACD-macdsignal',
        'D1-MACD-macdhist',
        'D1-ROC2-val',
        'D1-MOM4-val',
        'D1-RSI10-val',
        'D1-BB20-upper',
        'D1-BB20-mid', 
        'D1-BB20-lower',
        'D1-CCI20-val',
        'D1-PSAR-val',
    ],

    "model_settings_both": {
        "use_gru_model": True,
        "use_dual_lstm": False,
        "epochs": 1,
        "hidden_size": 50,
        "window_size": 5,
        "thres_multiplier": 3,
        "use_early_stop": False,
        "learning_rate": 0.0005,
        "batch_size": 8,
        "use_weighted_sampler": False,
        "volatility_type": 'thres_auto_v1',
        "valid_lookback_months": 12,
        "train_lookback_months": 48,

        "col_feature_dict": {
            "EURUSD": [
                'dff',
                'inflation_usa_d30',
                'inflation_eur_d30',
                'inflation_deu_d30',
                'rate_eur_3m_bank',
                'rate_deu_3m_bank',
                'D1-SMA10-val',
                'D1-MACD-macd',
                'D1-MACD-macdsignal',
                'D1-MACD-macdhist',
                'D1-ROC2-val',
                'D1-MOM4-val',
                'D1-RSI10-val',
                'D1-BB20-upper',
                'D1-BB20-mid',
                'D1-BB20-lower',
                'D1-CCI20-val',
                'D1-PSAR-val',
            ],

            "EURUSD_F": [
                'dff',
                'inflation_usa_d30',
                'inflation_eur_d30',
                'inflation_deu_d30',
                'rate_eur_3m_bank',
                'rate_deu_3m_bank',
                'D1-SMA10-val',
                'D1-MACD-macd',
                'D1-MACD-macdsignal',
                'D1-MACD-macdhist',
                'D1-ROC2-val',
                'D1-MOM4-val',
                'D1-RSI10-val',
                'D1-BB20-upper',
                'D1-BB20-mid',
                'D1-BB20-lower',
                'D1-CCI20-val',
                'D1-PSAR-val',
            ],

            "USDJPY": [
                'dff',
                'inflation_usa_d30',
                'inflation_jpn_d30',
                'rate_jpn_3m_bank',
                'D1-SMA10-val',
                'D1-MACD-macd',
                'D1-MACD-macdsignal',
                'D1-MACD-macdhist',
                'D1-ROC2-val',
                'D1-MOM4-val',
                'D1-RSI10-val',
                'D1-BB20-upper',
                'D1-BB20-mid',
                'D1-BB20-lower',
                'D1-CCI20-val',
                'D1-PSAR-val',
            ],

            "GBPUSD": [
                'dff',
                'D1-SMA10-val',
                'D1-MACD-macd',
                'D1-MACD-macdsignal',
                'D1-MACD-macdhist',
                'D1-ROC2-val',
                'D1-MOM4-val',
                'D1-RSI10-val',
                'D1-BB20-upper',
                'D1-BB20-mid',
                'D1-BB20-lower',
                'D1-CCI20-val',
            ],

            "AUDUSD": [
                'dff',
                'D1-SMA10-val',
                'D1-MACD-macd',
                'D1-MACD-macdsignal',
                'D1-MACD-macdhist',
                'D1-ROC2-val',
                'D1-MOM4-val',
                'D1-RSI10-val',
                'D1-BB20-upper',
                'D1-BB20-mid',
                'D1-BB20-lower',
                'D1-CCI20-val',
                'D1-PSAR-val',
            ],

        },
    },



    "model_settings_hybrid": {
        "use_gru_model": True,
        "use_dual_lstm": False,
        "epochs": 1,
        "hidden_size": 50,
        "window_size": 20,
        "thres_multiplier": 3,
        "learning_rate": 0.001,
        "batch_size": 8,
        "use_weighted_sampler": False,
        "volatility_type": 'thres_auto_v1',
        "valid_lookback_months": 12,
        "train_lookback_months": 48,

        "col_feature_fundamental_dict": {
            "EURUSD": [
                'dff',
                'inflation_usa_d1',
                'inflation_eur_d1',
                'inflation_deu_d1',
                'rate_eur_3m_bank',
                'rate_deu_3m_bank',
            ],

            "EURUSD_F": [
                'dff',
                'inflation_usa_d1',
                'inflation_eur_d1',
                'inflation_deu_d1',
                'rate_eur_3m_bank',
                'rate_deu_3m_bank',
            ],

            "USDJPY": [
                'dff',
                'rate_jpn_3m_bank',
            ],

            "GBPUSD": [
                'dff',
                'inflation_usa_d1',
                'inflation_gbr_d1',
                'rate_gbp_3m_bank',
            ],

            "AUDUSD": [
                'dff',
                'inflation_usa_d1',
                'inflation_aus_d1',
                'rate_aus_3m_bank',
            ],           
        },

        "col_feature_technical_dict": {
            "EURUSD": [
                'D1-SMA10-val',
                'D1-MACD-macd',
                'D1-MACD-macdsignal',
                'D1-MACD-macdhist',
                'D1-ROC2-val',
                'D1-MOM4-val',
                'D1-RSI10-val',
                'D1-BB20-upper',
                'D1-BB20-mid',
                'D1-BB20-lower',
                'D1-CCI20-val',
            ],

            "EURUSD_F": [
                'D1-SMA10-val',
                'D1-MACD-macd',
                'D1-MACD-macdsignal',
                'D1-MACD-macdhist',
                'D1-ROC2-val',
                'D1-MOM4-val',
                'D1-RSI10-val',
                'D1-BB20-upper',
                'D1-BB20-mid',
                'D1-BB20-lower',
                'D1-CCI20-val',
            ],

            "USDJPY": [
                'D1-SMA10-val',
                'D1-MACD-macd',
                'D1-MACD-macdsignal',
                'D1-MACD-macdhist',
                'D1-ROC2-val',
                'D1-MOM4-val',
                'D1-RSI10-val',
                'D1-BB20-upper',
                'D1-BB20-mid',
                'D1-BB20-lower',
                'D1-CCI20-val',
            ],

            "GBPUSD": [
                'D1-SMA10-val',
                'D1-MACD-macd',
                'D1-MACD-macdsignal',
                'D1-MACD-macdhist',
                'D1-ROC2-val',
                'D1-MOM4-val',
                'D1-RSI10-val',
                'D1-BB20-upper',
                'D1-BB20-mid',
                'D1-BB20-lower',
                'D1-CCI20-val',
            ],

            "AUDUSD": [
                'D1-SMA10-val',
                'D1-MACD-macd',
                'D1-MACD-macdsignal',
                'D1-MACD-macdhist',
                'D1-ROC2-val',
                'D1-MOM4-val',
                'D1-RSI10-val',
                'D1-BB20-upper',
                'D1-BB20-mid',
                'D1-BB20-lower',
                'D1-CCI20-val',
            ],

        },

    },


}



