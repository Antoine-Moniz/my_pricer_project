import datetime as dt
from typing import List, Tuple

class Market:
    """
    Classe représentant les données de marché nécessaires au pricing.

    Attributs :
        S0 (float) : prix spot du sous-jacent
        r (float) : taux sans risque
        vol (float) : volatilité
        dividends (list[tuple[datetime, float]]) : liste de dividendes (date, montant)
    """

    def __init__(self, S0: float, r: float, vol: float,
                 dividends: List[Tuple[dt.datetime, float]] = None) -> None:
        if S0 <= 0:
            raise ValueError("Le prix du sous-jacent doit être positif")
        if vol < 0:
            raise ValueError("La volatilité doit être positive")
        self.S0 = S0
        self.r = r
        self.vol = vol
        self.dividends = dividends if dividends is not None else []

    def get_dividend_on_period(self, start: dt.datetime, end: dt.datetime) -> float:
        """
        Retourne le dividende total payé si une ex-div date est comprise entre start et end.
        """
        return sum(amount for (date, amount) in self.dividends if start < date <= end)
