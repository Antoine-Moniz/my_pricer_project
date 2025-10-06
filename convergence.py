import numpy as np
from functools import lru_cache
import hashlib
from black_sholes import BlackScholes
from plot import TreePlotter


class ConvergenceStudy:
    """Étude de convergence et comparaison prix/pentes entre Tree et Black–Scholes."""

    def __init__(self, bs: BlackScholes):
        self.bs = bs
        self._cache = {}  # Cache local pour éviter les recalculs
    
    def _get_cache_key(self, nb_steps_max: int) -> str:
        """Génère une clé de cache unique basée sur les paramètres."""
        params_str = f"{self.bs.option.K}_{self.bs.option.call_put}_{self.bs.option.style}_{self.bs.market.S0}_{self.bs.market.r}_{self.bs.market.vol}_{nb_steps_max}"
        return hashlib.md5(params_str.encode()).hexdigest()[:16]
    
    def _get_cached_convergence(self, nb_steps_max: int):
        """Récupère les données de convergence du cache si disponibles."""
        cache_key = self._get_cache_key(nb_steps_max)
        return self._cache.get(cache_key), cache_key

    def study_convergence(self, nb_steps_max: int = 50):
        """Étudie la convergence (Tree → BS) avec cache pour éviter les recalculs."""
        # Vérifier le cache d'abord
        cached_data, cache_key = self._get_cached_convergence(nb_steps_max)
        
        if cached_data is not None:
            diffs, steps = cached_data
        else:
            # Calcul et mise en cache
            diffs, steps = self.bs.convergence_values(nb_steps_max)
            self._cache[cache_key] = (diffs, steps)
        
        # Affichage du graphique optimisé
        TreePlotter.plot_convergence(diffs, steps, self.bs.option.call_put)
        return diffs

    def study_prices_and_slopes(self, K_min: int, K_max: int):
        """Étudie prix + pentes avec cache."""
        # Générer une clé de cache pour cette étude
        cache_key = f"prices_slopes_{K_min}_{K_max}_{self._get_cache_key(50)}"
        
        if cache_key in self._cache:
            strikes, BS_prices, Tree_prices, BS_slopes, Tree_slopes, diffs = self._cache[cache_key]
        else:
            # Calcul et mise en cache
            strikes, BS_prices, Tree_prices, BS_slopes, Tree_slopes, diffs = self.bs.prices_and_slopes_values(K_min, K_max)
            self._cache[cache_key] = (strikes, BS_prices, Tree_prices, BS_slopes, Tree_slopes, diffs)
        
        # Affichage du graphique optimisé
        TreePlotter.plot_prices_and_slopes(strikes, BS_prices, Tree_prices, BS_slopes, Tree_slopes, diffs)
        return strikes, BS_prices, Tree_prices, BS_slopes, Tree_slopes, diffs
