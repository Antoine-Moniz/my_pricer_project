import datetime as dt

class Option:
    """
    Classe représentant une option simple (call ou put, européenne ou américaine).

    Attributs :
        K (float) : prix d'exercice
        maturity (datetime) : date de maturité
        call_put (str) : "call" ou "put"
        style (str) : "european" ou "american"
    """

    def __init__(self, K: float, maturity: dt.datetime, call_put: str, style: str) -> None:
        if call_put not in ["call", "put"]:
            raise ValueError("call_put doit être 'call' ou 'put'")
        if style not in ["european", "american"]:
            raise ValueError("style doit être 'european' ou 'american'")
        self.K = K
        self.maturity = maturity
        self.call_put = call_put
        self.style = style

    def payoff(self, S: float) -> float:
        """Retourne le payoff de l’option pour un prix du sous-jacent S"""
        if self.call_put == "call":
            return max(S - self.K, 0.0)
        return max(self.K - S, 0.0)

    def is_american(self) -> bool:
        """Renvoie True si l’option est américaine"""
        return self.style == "american"


