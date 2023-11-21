from AlgorithmImports import *

import numpy as np
from datetime import datetime, timedelta
from collections import deque
import talib as ta


class CustomSimpleMovingAverage:
    def __init__(self, name, period):
        self.Name = name
        self.Time = datetime.min
        self.Value = 0
        self.IsReady = False
        self.queue = deque(maxlen=period)

    def __repr__(self):
        return "{0} -> IsReady: {1}. Time: {2}. Value: {3}".format(
            self.Name, self.IsReady, self.Time, self.Value
        )

    # Update method is mandatory
    def Update(self, EndTime, Val):
        self.queue.appendleft(Val)
        count = len(self.queue)
        self.Time = EndTime
        self.Value = sum(self.queue) / count
        self.IsReady = count == self.queue.maxlen

    def Undo(self):
        del self.queue[0]


class CustomPhase:
    def __init__(self, name, period):
        self.Name = name
        self.Time = datetime.min
        self.Value = 0
        self.IsReady = False
        self.period = period
        self.close = deque(maxlen=period)
        self.high = deque(maxlen=period)
        self.low = deque(maxlen=period)

    def __repr__(self):
        return "{0} -> IsReady: {1}. Time: {2}. Value: {3}".format(
            self.Name, self.IsReady, self.Time, self.Value
        )

    def Update(self, bar):
        self.close.appendleft(bar.Close)
        self.high.appendleft(bar.High)
        self.low.appendleft(bar.Low)
        self.Time = bar.EndTime

        RealPart = 0.0
        ImagPart = 0.0
        _count = min([len(self.close), len(self.high), len(self.low)])
        for J in range(_count):
            Weight = (
                self.close[J] + self.close[J] + self.high[J] + self.low[J]
            ) * 10000
            if self.period != 0:
                RealPart = RealPart + np.cos(90 * J / self.period) * Weight * 2
                ImagPart = (
                    (ImagPart + np.sin(90 * J / self.period) * Weight)
                    + (ImagPart + np.sin(180 * J / self.period) * Weight)
                ) / 2
        Phase = ((np.arctan(ImagPart / RealPart)) - 0.685) * 100
        self.Value = Phase
        self.IsReady = _count == self.period


class CustomCRSI:
    def __init__(self, name, rsi_len, rsi_field, rsi_window):
        self.Name = name
        self.Time = datetime.min
        self.Value = 0
        self.IsReady = False
        self.rsi_len = rsi_len
        self.rsi_field = rsi_field
        self.rsi_window = rsi_window
        self.RSI = RelativeStrengthIndex(
            f"{name}-RSI", rsi_len, MovingAverageType.Exponential
        )
        self.RSIval = deque(maxlen=rsi_window)
        self.CRSIval = deque(maxlen=3)

    def __repr__(self):
        return "{0} -> IsReady: {1}. Time: {2}. Value: {3}".format(
            self.Name, self.IsReady, self.Time, self.Value
        )

    def Update(self, bar):
        self.Time = bar.EndTime
        self.RSI.Update(bar.EndTime, self.get_update_val(bar))
        self.RSIval.appendleft(self.RSI.Current.Value)

        vibration = 10
        torque = 0.618 / (vibration + 1)
        phasingLag = (vibration - 1) / 0.618
        if len(self.RSIval) > int(phasingLag):
            if len(self.CRSIval) > 1:
                crsi1 = self.CRSIval[1]
            else:
                crsi1 = 0
            self.Value = (
                torque * (2 * self.RSIval[0] - self.RSIval[int(phasingLag)])
                + (1 - torque) * crsi1
            )
        else:
            self.Value = 0

        self.CRSIval.appendleft(self.Value)
        self.IsReady = (
            self.RSI.IsReady
            and (len(self.RSIval) == self.rsi_window)
            and (len(self.CRSIval) == 3)
        )

    def get_update_val(self, bar):
        if self.rsi_field == "Close":
            val = bar.Close
        elif self.rsi_field == "High":
            val = bar.High
        elif self.rsi_field == "Low":
            val = bar.Low
        elif self.rsi_field == "Open":
            val = bar.Open
        return val


class CustomPivot:
    def __init__(self, name, period):
        self.Name = name
        self.Time = datetime.min
        self.Value = 0
        self.IsReady = False
        self.period = period
        self.close = deque(maxlen=period)
        self.high = deque(maxlen=period)
        self.low = deque(maxlen=period)
        self.arr_time = deque(maxlen=period)
        self.p = deque(maxlen=period)
        self.r1 = deque(maxlen=period)
        self.s1 = deque(maxlen=period)
        # self.r2 = deque(maxlen=period)
        # self.s2 = deque(maxlen=period)
        # self.r3 = deque(maxlen=period)
        # self.s3 = deque(maxlen=period)
        # self.r4 = deque(maxlen=period)
        # self.s4 = deque(maxlen=period)
        # self.r5 = deque(maxlen=period)
        # self.s5 = deque(maxlen=period)

    def __repr__(self):
        return "{0} -> IsReady: {1}. Time: {2}. Value: {3}".format(
            self.Name, self.IsReady, self.Time, self.Value
        )

    def Update(self, bar):
        self.close.appendleft(bar.Close)
        self.high.appendleft(bar.High)
        self.low.appendleft(bar.Low)
        self.Time = bar.EndTime
        self.arr_time.appendleft(bar.EndTime)
        pivotX_Median = (self.high[0] + self.low[0] + self.close[0]) / 3
        self.Value = pivotX_Median
        self.p.appendleft(pivotX_Median)
        self.r1.appendleft(pivotX_Median * 2 - self.low[0])
        self.s1.appendleft(pivotX_Median * 2 - self.high[0])
        # self.r2.appendleft(pivotX_Median + 1 * (self.high[0] - self.low[0]))
        # self.s2.appendleft(pivotX_Median - 1 * (self.high[0] - self.low[0]))
        # self.r3.appendleft(pivotX_Median * 2 + (self.high[0] - 2 * self.low[0]))
        # self.s3.appendleft(pivotX_Median * 2 - (2 * self.high[0] - self.low[0]))
        # self.r4.appendleft(pivotX_Median * 3 + (self.high[0] - 3 * self.low[0]))
        # self.s4.appendleft(pivotX_Median * 3 - (3 * self.high[0] - self.low[0]))
        # self.r5.appendleft(pivotX_Median * 4 + (self.high[0] - 4 * self.low[0]))
        # self.s5.appendleft(pivotX_Median * 4 - (4 * self.high[0] - self.low[0]))
        self.IsReady = len(self.p) == self.period


class CustomChoppinessIndex:
    def __init__(self, name, period):
        self.Name = name
        self.Time = datetime.min
        self.Value = 0
        self.IsReady = False
        self.period = period

        self.high = deque(maxlen=period)
        self.low = deque(maxlen=period)
        self.ATR = AverageTrueRange(f"{name}-RSI", 1)
        self.ATRval = deque(maxlen=period)

    def __repr__(self):
        return "{0} -> IsReady: {1}. Time: {2}. Value: {3}".format(
            self.Name, self.IsReady, self.Time, self.Value
        )

    def Update(self, bar):
        self.high.appendleft(bar.High)
        self.low.appendleft(bar.Low)
        self.Time = bar.EndTime
        self.ATR.Update(bar)
        self.ATRval.appendleft(self.ATR.Current.Value)
        if (max(self.high) - min(self.low)) == 0:
            self.Value = 0
        else:
            self.Value = (
                100
                * np.log10(sum(self.ATRval) / (max(self.high) - min(self.low)))
                / np.log10(self.period)
            )
        self.IsReady = self.ATR.IsReady and (len(self.ATRval) == self.period)


class CustomDX:
    def __init__(self, name, period):
        self.Name = name
        self.Time = datetime.min
        self.Value = 0
        self.IsReady = False
        self.period = period
        self.window_len = period * 2 + 25
        self.high = deque(maxlen=self.window_len)
        self.low = deque(maxlen=self.window_len)
        self.close = deque(maxlen=self.window_len)

    def __repr__(self):
        return "{0} -> IsReady: {1}. Time: {2}. Value: {3}".format(
            self.Name, self.IsReady, self.Time, self.Value
        )

    def Update(self, bar):
        self.high.appendleft(bar.High)
        self.low.appendleft(bar.Low)
        self.close.appendleft(bar.Close)
        self.Time = bar.EndTime
        self.IsReady = len(self.close) == (self.window_len)

        if self.IsReady:
            ta_out = ta.DX(
                np.array(self.high),
                np.array(self.low),
                np.array(self.close),
                timeperiod=self.period,
            )
            self.Value = ta_out[-1]
        else:
            self.Value = 0


class CustomMACD:
    def __init__(self, name):
        self.Name = name
        self.Time = datetime.min
        self.IsReady = False
        self.window_len = 100
        self.close = deque(maxlen=self.window_len)
        self.macd = 0
        self.macdsignal = 0
        self.macdhist = 0

    def __repr__(self):
        return "{0} -> IsReady: {1}. Time: {2}. Value: {3}".format(
            self.Name, self.IsReady, self.Time, self.Value
        )

    def Update(self, bar):
        self.close.appendleft(bar.Close)
        self.Time = bar.EndTime
        self.IsReady = len(self.close) == (self.window_len)

        if self.IsReady:
            macd, macdsignal, macdhist = ta.MACD(
                np.array(self.close), fastperiod=12, slowperiod=26, signalperiod=9
            )
            self.macd = macd[-1]
            self.macdsignal = macdsignal[-1]
            self.macdhist = macdhist[-1]
        else:
            self.macd = 0
            self.macdsignal = 0
            self.macdhist = 0


class CustomULTOSC:
    def __init__(self, name):
        self.Name = name
        self.Time = datetime.min
        self.Value = 0
        self.IsReady = False
        self.window_len = 100
        self.high = deque(maxlen=self.window_len)
        self.low = deque(maxlen=self.window_len)
        self.close = deque(maxlen=self.window_len)

    def __repr__(self):
        return "{0} -> IsReady: {1}. Time: {2}. Value: {3}".format(
            self.Name, self.IsReady, self.Time, self.Value
        )

    def Update(self, bar):
        self.high.appendleft(bar.High)
        self.low.appendleft(bar.Low)
        self.close.appendleft(bar.Close)
        self.Time = bar.EndTime
        self.IsReady = len(self.close) == (self.window_len)

        if self.IsReady:
            ta_out = ta.ULTOSC(
                np.array(self.high),
                np.array(self.low),
                np.array(self.close),
                timeperiod1=7,
                timeperiod2=14,
                timeperiod3=28,
            )
            self.Value = ta_out[-1]
        else:
            self.Value = 0
