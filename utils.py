from AlgorithmImports import *
import scipy
import math



def getFxPositionSize(stopPoints, riskPct, algorithm, symbol):
    Balance = algorithm.Portfolio.TotalPortfolioValue
    LotSize = algorithm.Securities[symbol].SymbolProperties.LotSize
    conversionRate = algorithm.Securities[symbol].QuoteCurrency.ConversionRate
    pointValue = LotSize * conversionRate
    units = int(np.ceil((Balance * riskPct) / (stopPoints * pointValue)))

    return units