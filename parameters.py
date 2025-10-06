import datetime as dt

class PricerParameters:
    """
    Classe regroupant les paramètres de calcul du modèle.

    Attributs :
        pricing_date (datetime) : date de valorisation
        nb_steps (int) : nombre d’étapes dans l’arbre
        pruning (bool) : activer/désactiver le pruning
        p_min (float) : probabilité minimale en cas de pruning
    """

    def __init__(self, pricing_date: dt.datetime, nb_steps: int, pruning: bool = True, p_min: float = 1e-9) -> None:
        self.pricing_date = pricing_date
        self.nb_steps = nb_steps
        self.pruning = pruning
        self.p_min = p_min


