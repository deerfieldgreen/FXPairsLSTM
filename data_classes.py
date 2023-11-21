from AlgorithmImports import *

import numpy as np
from datetime import datetime, timedelta
from collections import deque

from indicator_classes import (
    CustomPhase,
    CustomCRSI,
    CustomPivot,
    CustomChoppinessIndex,
    CustomDX,
    CustomMACD,
    CustomULTOSC,
)


class SymbolData:
    def __init__(
        self,
        algorithm,
        symbol,
        ticker,
        general_setting,
        consolidator_settings,
        indicator_settings,
    ):
        self.symbol = symbol
        self.ticker = ticker
        self.algorithm = algorithm
        self.general_setting = general_setting
        self.consolidator_settings = consolidator_settings
        self.indicator_settings = indicator_settings
        self.to_plot = algorithm.to_plot

        if general_setting["tickers"][ticker]["type"] in ["forex", "cfd","forex_futures"]:
            self.consolidator_type = "quote"
        elif general_setting["tickers"][ticker]["type"] in ["equity", "index"]:
            self.consolidator_type = "trade"

        self.consolidators = {}
        for timeframe in self.general_setting["consolidator_timeframes"]:
            self.consolidators[timeframe] = DataConsolidator(
                algorithm,
                symbol,
                ticker,
                general_setting,
                consolidator_settings[timeframe],
                indicator_settings,
                self.consolidator_type,
            )

    @property
    def IsReady(self):
        # All consolidators are ready
        is_ready = (
            np.prod([self.consolidators[_t].IsReady for _t in self.general_setting["consolidator_timeframes"]]) == 1
        )
        return is_ready


class DataConsolidator:
    def __init__(
        self,
        algorithm,
        symbol,
        ticker,
        general_setting,
        consolidator_setting,
        indicator_settings,
        consolidator_type,
    ):
        self.symbol = symbol
        self.ticker = ticker
        self.algorithm = algorithm
        self.general_setting = general_setting
        self.consolidator_setting = consolidator_setting
        self.indicator_settings = indicator_settings
        self.consolidator_type = consolidator_type
        self.to_plot = algorithm.to_plot
        self.indicators = {}

        self.ticker_type = self.general_setting["tickers"][self.ticker]["type"]
        self.window_multiplier = self.consolidator_setting["window_multiplier_dict"][self.ticker_type]
        self.window_length = int(self.consolidator_setting["window"] * self.window_multiplier)
        self.time = deque(maxlen=self.window_length)
        self.open = deque(maxlen=self.window_length)
        self.high = deque(maxlen=self.window_length)
        self.low = deque(maxlen=self.window_length)
        self.close = deque(maxlen=self.window_length)
        self.returns = deque(maxlen=self.window_length)

        if "window_large" in self.consolidator_setting:
            self.window_length_large = int(self.consolidator_setting["window_large"] * self.window_multiplier)
            self.time_large = deque(maxlen=self.window_length_large)           
            self.close_large = deque(maxlen=self.window_length_large)    

        self.BarCount = 0

        if self.consolidator_type == "quote":
            if self.consolidator_setting["timeframe_minutes"] in [5, 15, 30, 60]:
                self.Con = QuoteBarConsolidator(
                    TimeSpan.FromMinutes(self.consolidator_setting["timeframe_minutes"])
                )
            elif self.consolidator_setting["timeframe_minutes"] in [4 * 60]:
                self.Con = QuoteBarConsolidator(self.H4Timer)
            elif self.consolidator_setting["timeframe_minutes"] in [24 * 60]:
                self.Con = QuoteBarConsolidator(self.D1Timer)
            elif self.consolidator_setting["timeframe_minutes"] in [7 * 24 * 60]:
                self.Con = QuoteBarConsolidator(self.W1Timer)

        elif self.consolidator_type == "trade":
            if self.consolidator_setting["timeframe_minutes"] in [5, 15, 30, 60]:
                self.Con = TradeBarConsolidator(
                    TimeSpan.FromMinutes(self.consolidator_setting["timeframe_minutes"])
                )
            elif self.consolidator_setting["timeframe_minutes"] in [4 * 60]:
                self.Con = TradeBarConsolidator(self.H4Timer)
            elif self.consolidator_setting["timeframe_minutes"] in [24 * 60]:
                self.Con = TradeBarConsolidator(self.D1Timer)
            elif self.consolidator_setting["timeframe_minutes"] in [7 * 24 * 60]:
                self.Con = TradeBarConsolidator(self.W1Timer)

        self.Con.DataConsolidated += self.ConHandler
        algorithm.SubscriptionManager.AddConsolidator(symbol, self.Con)

        for _indicator in self.consolidator_setting["indicators"]:
            self.indicators[_indicator] = self.get_indicator(symbol, _indicator)

    def ConHandler(self, sender, bar):
        self.BarCount += 1
        self.time.appendleft(bar.Time)
        self.open.appendleft(bar.Open)
        self.high.appendleft(bar.High)
        self.low.appendleft(bar.Low)
        self.close.appendleft(bar.Close)

        if "window_large" in self.consolidator_setting:
            self.time_large.appendleft(bar.Time)
            self.close_large.appendleft(bar.Close)

        if len(self.close) > 1:
            self.returns.appendleft((self.close[0] / self.close[1]) - 1)
        else:
            self.returns.appendleft(0)

        for _indicator in self.consolidator_setting["indicators"]:
            self.update_indicator(bar, _indicator)

    def H4Timer(self, dt):
        start = (dt if dt.hour > 17 else dt - timedelta(1)).date()
        start = datetime.combine(start, datetime.min.time()) + timedelta(hours=17)
        return CalendarInfo(start, timedelta(hours=4))

    def D1Timer(self, dt):
        start = (dt if dt.hour > 17 else dt - timedelta(1)).date()
        start = datetime.combine(start, datetime.min.time()) + timedelta(hours=17)
        return CalendarInfo(start, timedelta(1))

    def W1Timer(self, dt):
        _date = dt.date()
        if _date.weekday() == 6:
            if dt.hour > 17:
                start = _date
            else:
                start = _date - timedelta(days=_date.weekday()) - timedelta(1)
        else:
            start = _date - timedelta(days=_date.weekday()) - timedelta(1)

        start = datetime.combine(start, datetime.min.time()) + timedelta(hours=17)
        return CalendarInfo(start, timedelta(7))

    def get_indicator(self, symbol, name):
        indicator_setting = self.indicator_settings[name]

        indicator_dict = {}
        if indicator_setting["type"] == "EMA":
            indicator_dict["model"] = ExponentialMovingAverage(
                symbol, indicator_setting["lookback"]
            )
            indicator_dict["val"] = deque(maxlen=indicator_setting["window"])

        elif indicator_setting["type"] == "SMA":
            indicator_dict["model"] = SimpleMovingAverage(
                symbol, indicator_setting["lookback"]
            )
            indicator_dict["val"] = deque(maxlen=indicator_setting["window"])

        elif indicator_setting["type"] == "ROC":
            indicator_dict["model"] = RateOfChange(
                symbol, indicator_setting["lookback"]
            )
            indicator_dict["val"] = deque(maxlen=indicator_setting["window"])

        elif indicator_setting["type"] == "MOM":
            indicator_dict["model"] = Momentum(
                symbol, indicator_setting["lookback"]
            )
            indicator_dict["val"] = deque(maxlen=indicator_setting["window"])

        elif indicator_setting["type"] == "CCI":
            indicator_dict["model"] = CommodityChannelIndex(
                symbol, indicator_setting["lookback"]
            )
            indicator_dict["val"] = deque(maxlen=indicator_setting["window"])

        elif indicator_setting["type"] == "RSI":
            if indicator_setting["ma_type"] == "Exponential":
                indicator_dict["model"] = RelativeStrengthIndex(
                    symbol, indicator_setting["lookback"], MovingAverageType.Exponential
                )
            elif indicator_setting["ma_type"] == "Simple":
                indicator_dict["model"] = RelativeStrengthIndex(
                    symbol, indicator_setting["lookback"], MovingAverageType.Simple
                )
            indicator_dict["val"] = deque(maxlen=indicator_setting["window"])

        elif indicator_setting["type"] == "ATR":
            indicator_dict["model"] = AverageTrueRange(
                symbol, indicator_setting["lookback"]
            )
            indicator_dict["val"] = deque(maxlen=indicator_setting["window"])

        elif indicator_setting["type"] == "PSAR":
            indicator_dict["model"] = ParabolicStopAndReverse(symbol, 0.02, 0.02, 0.2)
            indicator_dict["val"] = deque(maxlen=indicator_setting["window"])

        elif indicator_setting["type"] == "BOLL":
            if indicator_setting["ma_type"] == "Exponential":
                indicator_dict["model"] = BollingerBands(
                    name,
                    indicator_setting["lookback"],
                    indicator_setting["std"],
                    MovingAverageType.Exponential,
                )
            elif indicator_setting["ma_type"] == "Simple":
                indicator_dict["model"] = BollingerBands(
                    name,
                    indicator_setting["lookback"],
                    indicator_setting["std"],
                    MovingAverageType.Simple,
                )

            indicator_dict["upper"] = deque(maxlen=indicator_setting["window"])
            indicator_dict["lower"] = deque(maxlen=indicator_setting["window"])
            indicator_dict["mid"] = deque(maxlen=indicator_setting["window"])

        elif indicator_setting["type"] == "PHASE":
            indicator_dict["model"] = CustomPhase(name, indicator_setting["lookback"])
            indicator_dict["val"] = deque(maxlen=indicator_setting["window"])

        elif indicator_setting["type"] == "CRSI":
            indicator_dict["model"] = CustomCRSI(
                name,
                indicator_setting["rsi_len"],
                indicator_setting["rsi_field"],
                indicator_setting["rsi_window"],
            )
            indicator_dict["val"] = deque(maxlen=indicator_setting["window"])

        elif indicator_setting["type"] == "PIVOT":
            indicator_dict["model"] = CustomPivot(name, indicator_setting["period"])

        elif indicator_setting["type"] == "CHOP":
            indicator_dict["model"] = CustomChoppinessIndex(
                name, indicator_setting["lookback"]
            )
            indicator_dict["val"] = deque(maxlen=indicator_setting["window"])

        elif indicator_setting["type"] == "DX":
            indicator_dict["model"] = CustomDX(name, indicator_setting["lookback"])
            indicator_dict["val"] = deque(maxlen=indicator_setting["window"])

        elif indicator_setting["type"] == "MACD":
            indicator_dict["model"] = CustomMACD(name)
            indicator_dict["macd"] = deque(maxlen=indicator_setting["window"])
            indicator_dict["macdsignal"] = deque(maxlen=indicator_setting["window"])
            indicator_dict["macdhist"] = deque(maxlen=indicator_setting["window"])

        elif indicator_setting["type"] == "ULTOSC":
            indicator_dict["model"] = CustomULTOSC(name)
            indicator_dict["val"] = deque(maxlen=indicator_setting["window"])

        return indicator_dict

    def update_indicator(self, bar, name):
        def get_update_val(bar, indicator_setting):
            if indicator_setting["field"] == "Close":
                val = bar.Close
            elif indicator_setting["field"] == "High":
                val = bar.High
            elif indicator_setting["field"] == "Low":
                val = bar.Low
            elif indicator_setting["field"] == "Open":
                val = bar.Open
            return val

        indicator_setting = self.indicator_settings[name]

        if indicator_setting["type"] in ["EMA","SMA","ROC","MOM","RSI"]:
            self.indicators[name]["model"].Update(
                bar.EndTime, get_update_val(bar, indicator_setting)
            )
            self.indicators[name]["val"].appendleft(
                self.indicators[name]["model"].Current.Value
            )

        elif indicator_setting["type"] in ["ATR","PSAR","CCI"]:
            self.indicators[name]["model"].Update(bar)
            self.indicators[name]["val"].appendleft(
                self.indicators[name]["model"].Current.Value
            )

        elif indicator_setting["type"] in ["PHASE", "CRSI", "CHOP", "DX", "ULTOSC"]:
            self.indicators[name]["model"].Update(bar)
            self.indicators[name]["val"].appendleft(
                self.indicators[name]["model"].Value
            )

        elif indicator_setting["type"] in ["PIVOT"]:
            self.indicators[name]["model"].Update(bar)

        elif indicator_setting["type"] in ["BOLL"]:
            self.indicators[name]["model"].Update(
                bar.EndTime, get_update_val(bar, indicator_setting)
            )
            self.indicators[name]["upper"].appendleft(
                self.indicators[name]["model"].UpperBand.Current.Value
            )
            self.indicators[name]["lower"].appendleft(
                self.indicators[name]["model"].LowerBand.Current.Value
            )
            self.indicators[name]["mid"].appendleft(
                self.indicators[name]["model"].MiddleBand.Current.Value
            )

        elif indicator_setting["type"] in ["MACD"]:
            self.indicators[name]["model"].Update(bar)
            self.indicators[name]["macd"].appendleft(
                self.indicators[name]["model"].macd
            )
            self.indicators[name]["macdsignal"].appendleft(
                self.indicators[name]["model"].macdsignal
            )
            self.indicators[name]["macdhist"].appendleft(
                self.indicators[name]["model"].macdhist
            )

    @property
    def IsReady(self):
        # All indicators are ready
        is_ready = (np.prod([self.indicators[_i]["model"].IsReady for _i in self.indicators]) == 1)

        # self.close is fully populated
        is_ready = is_ready and (len(self.close) == self.window_length)

        if "window_large" in self.consolidator_setting:
            # self.close_large is fully populated
            is_ready = is_ready and (len(self.close_large) == self.window_length_large)

        return is_ready




class MarketHours:
    def __init__(self, algorithm, symbol):
        self.hours = algorithm.Securities[symbol].Exchange.Hours
        self.CurrentOpen = self.hours.GetNextMarketOpen(algorithm.Time, False)
        self.CurrentClose = self.hours.GetNextMarketClose(self.CurrentOpen, False)
        self.NextOpen = self.hours.GetNextMarketOpen(self.CurrentClose, False)

    def Update(self):
        self.CurrentOpen = self.NextOpen
        self.CurrentClose = self.hours.GetNextMarketClose(self.CurrentOpen, False)
        self.NextOpen = self.hours.GetNextMarketOpen(self.CurrentClose, False)


