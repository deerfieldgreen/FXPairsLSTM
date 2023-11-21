from AlgorithmImports import *

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque


from utils import (
    getFxPositionSize,
)


class FxLstmSignal:
    def __init__(
        self,
        algorithm,
        symbol,
        ticker,
        general_setting,
        signal_setting,
    ):

        # General Initializations
        self.algorithm = algorithm
        self.symbol = symbol
        self.ticker = ticker
        self.general_setting = general_setting
        self.signal_setting = signal_setting

        self.prediction_direction_map_dict = self.signal_setting['prediction_direction_map_dict']
        self.valid_tickers = signal_setting["valid_tickers"]
        self.enter_long_trades = signal_setting["enter_long_trades"]
        self.enter_short_trades = signal_setting["enter_short_trades"]

        self.prediction_direction = 0
        self.use_exit_reason = self.signal_setting["useTralingStop"]
        self.direction = 0
        self.quantity = 0
        self.entryBarCount = 0
        self.allocation_multiplier = 1
        self.entryPrice = None
        self.stopPrice = None
        self.stopDistance = None
        self.targetPrice = None
        self.targetDistance = None
        self.inTrade = False
        self.lookForExit = False
        self.trailingStop = False


    def update_prediction_direction(self, prediction):
        self.prediction_direction = self.prediction_direction_map_dict[prediction]

    def enter(self, symbolData, price, market_time):
        if self.ticker not in self.valid_tickers:
            return False

        has_enter = False

        if self.inTrade:
            return False

        active_consolidator = symbolData.consolidators[self.signal_setting["active_timeframe"]]
        ATR = active_consolidator.indicators[f"ATR{self.signal_setting['atrLength']}"]

        SMA = active_consolidator.indicators[f"SMA{self.signal_setting['sma_filter_lookback']}"]
        # SMA_fast = active_consolidator.indicators[f"SMA{self.signal_setting['sma_filter_lookback_fast']}"]
        # SMA_slow = active_consolidator.indicators[f"SMA{self.signal_setting['sma_filter_lookback_slow']}"]

        if self.signal_setting["use_sma_filter"]:
            is_long_trend = price > SMA["val"][0]
            is_short_trend = price < SMA["val"][0]
            # is_long_trend = SMA_fast["val"][0] > SMA_slow["val"][0]
            # is_short_trend = SMA_fast["val"][0] < SMA_slow["val"][0]
        else:
            is_long_trend = True
            is_short_trend = True

        if self.enter_long_trades and (self.prediction_direction > 0) and is_long_trend:
            has_enter = True
            self.inTrade = True
            self.direction = 1

            if self.signal_setting["use_movement_thres_for_stops"]:
                stopSize = self.signal_setting["movement_thres"] * self.signal_setting["longStopMultiplier"]
                longStopPrice = active_consolidator.close[0] - stopSize
                longStopDistance = active_consolidator.close[0] - longStopPrice
            else:
                stopSize = ATR["val"][0] * self.signal_setting["longStopMultiplier"]
                longPriceSource = min(active_consolidator.low[0], active_consolidator.low[1])
                longStopPrice = longPriceSource - stopSize
                longStopDistance = active_consolidator.close[0] - longStopPrice
            self.targetPrice = active_consolidator.close[0] + longStopDistance * self.signal_setting["longRiskRewardMultiplier"]
            self.targetDistance = self.targetPrice - active_consolidator.close[0]

            self.quantity = self.direction * getFxPositionSize(
                longStopDistance,
                self.signal_setting["risk_pct"],
                self.algorithm,
                self.symbol,
            )

            self.entryPrice = price
            self.stopPrice = longStopPrice
            self.stopDistance = longStopDistance
            self.entryBarCount = active_consolidator.BarCount
            self.trailingStop = False
            self.lookForExit = False


        if self.enter_short_trades and (self.prediction_direction < 0) and is_short_trend:
            has_enter = True
            self.inTrade = True
            self.direction = -1

            if self.signal_setting["use_movement_thres_for_stops"]:
                stopSize = self.signal_setting["movement_thres"] * self.signal_setting["shortStopMultiplier"]
                shortStopPrice = active_consolidator.close[0] + stopSize
                shortStopDistance = shortStopPrice - active_consolidator.close[0]
            else:
                stopSize = ATR["val"][0] * self.signal_setting["shortStopMultiplier"]
                shortPriceSource = max(active_consolidator.high[0], active_consolidator.high[1])
                shortStopPrice = shortPriceSource + stopSize
                shortStopDistance = shortStopPrice - active_consolidator.close[0]
            self.targetPrice = active_consolidator.close[0] - shortStopDistance * self.signal_setting["shortRiskRewardMultiplier"]
            self.targetDistance = active_consolidator.close[0] - self.targetPrice

            self.quantity = self.direction * getFxPositionSize(
                shortStopDistance,
                self.signal_setting["risk_pct"],
                self.algorithm,
                self.symbol,
            )

            self.entryPrice = price
            self.stopPrice = shortStopPrice
            self.stopDistance = shortStopDistance
            self.entryBarCount = active_consolidator.BarCount
            self.trailingStop = False
            self.lookForExit = False

        return has_enter


    def check_exit(self, symbolData, price, market_time):

        if not self.inTrade:
            return False

        to_exit = False

        active_consolidator = symbolData.consolidators[self.signal_setting["active_timeframe"]]
        can_exit = (active_consolidator.BarCount - self.entryBarCount) > self.signal_setting["exit_wait_period"]
        ATR = active_consolidator.indicators[f"ATR{self.signal_setting['atrLength']}"]

        if (self.direction > 0):
            if (price <= self.stopPrice) and (not self.trailingStop):
                to_exit = True
            if (price >= self.targetPrice) and (not self.use_exit_reason):
                to_exit = True
            if self.signal_setting["use_prediction_direction_to_exit"] and (self.prediction_direction < 0):
                to_exit = True     

        if (self.direction < 0):
            if (price >= self.stopPrice) and (not self.trailingStop):
                to_exit = True
            if (price <= self.targetPrice) and (not self.use_exit_reason):
                to_exit = True
            if self.signal_setting["use_prediction_direction_to_exit"] and (self.prediction_direction > 0):
                to_exit = True     

        if to_exit and can_exit:
            return to_exit

        if (self.direction > 0):
            if (price >= self.targetPrice) and self.use_exit_reason:
                self.lookForExit = True

        if  (self.direction > 0):
            if self.signal_setting["useTralingStop"] and self.lookForExit:
                trail = active_consolidator.close[0] - ATR["val"][0] * self.signal_setting["trailStopSize"]
                if trail > self.stopPrice:
                    self.stopPrice = trail
                    self.trailingStop = True


        if (self.direction > 0):
            if (self.signal_setting["useTralingStop"] and self.lookForExit and (price <= self.stopPrice)):
                to_exit = True

        if (self.direction < 0):
            if (price <= self.targetPrice) and self.use_exit_reason:
                self.lookForExit = True


        if  (self.direction < 0):
            if self.signal_setting["useTralingStop"] and self.lookForExit:
                trail = active_consolidator.close[0] + ATR["val"][0] * self.signal_setting["trailStopSize"]
                if trail < self.stopPrice:
                    self.stopPrice = trail
                    self.trailingStop = True

        if (self.direction < 0):
            if (self.signal_setting["useTralingStop"] and self.lookForExit and (price >= self.stopPrice)):
                to_exit = True

        to_exit = to_exit and can_exit

        return to_exit


    def update_exit(self):
        self.direction = 0
        self.quantity = 0
        self.entryBarCount = 0
        self.allocation_multiplier = 1
        self.entryPrice = None
        self.stopPrice = None
        self.stopDistance = None
        self.targetPrice = None
        self.targetDistance = None
        self.inTrade = False
        self.lookForExit = False
        self.trailingStop = False




