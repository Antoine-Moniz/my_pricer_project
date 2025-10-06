# greeks.py
import copy
import numpy as np
from scipy.stats import norm
from black_sholes import BlackScholes
from trinomial import TrinomialTree
from parameters import PricerParameters
from market import Market
from option import Option
from monte_carlo import MonteCarloPricer
from typing import Callable, Union, Tuple
from functools import lru_cache
# Numba retiré temporairement pour éviter les erreurs


class ParameterBundle:
    """Bundle de paramètres pour optimiser les calculs de Greeks."""
    def __init__(self, params: PricerParameters, market: Market, option: Option):
        # Marché
        self.S0: float = market.S0
        self.r: float = market.r
        self.vol: float = market.vol
        self.dividends = market.dividends
        
        # Option
        self.K: float = option.K
        self.maturity = option.maturity
        self.call_put: str = option.call_put
        self.style: str = option.style
        
        # Pricing
        self.pricing_date = params.pricing_date
        self.nb_steps: int = params.nb_steps
        self.pruning: bool = params.pruning
        self.p_min: float = params.p_min


class FiniteDifferenceCalculator:
    """Calculateur optimisé de différences finies avec cache pour les Greeks."""
    
    def __init__(self, pricing_func: Callable[[ParameterBundle], float], bundle: ParameterBundle):
        self.pricing_func = pricing_func
        self.bundle = bundle
        self._cache = {}  # Cache pour éviter les recalculs
        self._base_price = None  # Prix de base mis en cache
    
    def _get_base_price(self) -> float:
        """Obtient le prix de base avec mise en cache."""
        if self._base_price is None:
            self._base_price = self.pricing_func(self.bundle)
        return self._base_price
    
    def _get_cached_price(self, bundle: ParameterBundle, param_name: str, shift: float) -> float:
        """Obtient un prix avec mise en cache."""
        cache_key = f"{param_name}_{shift}"
        if cache_key not in self._cache:
            self._cache[cache_key] = self.pricing_func(bundle)
        return self._cache[cache_key]
    
    def first_derivative(self, param_name: str, shift: float, for_gamma: bool = False) -> Union[float, Tuple[float, float]]:
        """Calcule la dérivée première par différences finies avec optimisations."""
        bundle_up = copy.deepcopy(self.bundle)
        bundle_down = copy.deepcopy(self.bundle)
        
        setattr(bundle_up, param_name, getattr(self.bundle, param_name) + shift)
        setattr(bundle_down, param_name, getattr(self.bundle, param_name) - shift)
        
        if for_gamma:
            # Pour gamma, on utilise le prix de base mis en cache
            base_price = self._get_base_price()
            price_up = self._get_cached_price(bundle_up, param_name, shift)
            price_down = self._get_cached_price(bundle_down, param_name, -shift)
            
            delta_up = (price_up - base_price) / shift
            delta_down = (base_price - price_down) / shift
            return delta_up, delta_down
        
        price_up = self._get_cached_price(bundle_up, param_name, shift)
        price_down = self._get_cached_price(bundle_down, param_name, -shift)
        return (price_up - price_down) / (2 * shift)


# Fonction sans Numba pour éviter les problèmes d'import
def tree_pricing_function_cached(s0: float, r: float, vol: float, k: float, 
                                maturity_days: int, pricing_days: int, call_put: str, 
                                style: str, nb_steps: int, pruning: bool, p_min: float) -> float:
    """Fonction de pricing mise en cache pour éviter les recalculs identiques."""
    import datetime as dt
    
    pricing_date = dt.datetime.fromordinal(pricing_days)
    maturity = dt.datetime.fromordinal(maturity_days)
    
    params = PricerParameters(pricing_date, nb_steps, pruning, p_min)
    market = Market(s0, r, vol, None)  # Pas de dividendes pour la cache
    option = Option(k, maturity, call_put, style)
    tree = TrinomialTree(params, market, option)
    return tree.price_backward()


# Cache LRU appliqué après la définition
tree_pricing_function_cached = lru_cache(maxsize=128)(tree_pricing_function_cached)


def finite_difference_numba(price_up: float, price_down: float, shift: float) -> float:
    """Calcul des différences finies (optimisation Numba retirée temporairement)."""
    return (price_up - price_down) / (2.0 * shift)


def gamma_calculation_numba(price_up: float, price_base: float, price_down: float, shift: float) -> float:
    """Calcul du gamma (optimisation Numba retirée temporairement)."""
    delta_up = (price_up - price_base) / shift
    delta_down = (price_base - price_down) / shift
    return (delta_up - delta_down) / shift


def tree_pricing_function(bundle: ParameterBundle) -> float:
    """Fonction de pricing optimisée pour l'arbre trinomial."""
    # Utilise la version cachée si pas de dividendes
    if not bundle.dividends:
        return tree_pricing_function_cached(
            bundle.S0, bundle.r, bundle.vol, bundle.K,
            bundle.maturity.toordinal(), bundle.pricing_date.toordinal(),
            bundle.call_put, bundle.style, bundle.nb_steps, bundle.pruning, bundle.p_min
        )
    
    # Version normale avec dividendes
    params = PricerParameters(bundle.pricing_date, bundle.nb_steps, bundle.pruning, bundle.p_min)
    market = Market(bundle.S0, bundle.r, bundle.vol, bundle.dividends)
    option = Option(bundle.K, bundle.maturity, bundle.call_put, bundle.style)
    tree = TrinomialTree(params, market, option)
    return tree.price_backward()


# Fonction sans Numba pour éviter les problèmes d'import
def mc_pricing_function_cached(s0: float, r: float, vol: float, k: float,
                              maturity_days: int, pricing_days: int, call_put: str,
                              n_paths: int, seed: int) -> float:
    """Fonction de pricing Monte Carlo mise en cache."""
    import datetime as dt
    
    pricing_date = dt.datetime.fromordinal(pricing_days)
    maturity = dt.datetime.fromordinal(maturity_days)
    
    params = PricerParameters(pricing_date, 50)  # nb_steps pas important pour MC
    market = Market(s0, r, vol, None)
    option = Option(k, maturity, call_put, "european")
    mc = MonteCarloPricer(params, market, option, n_paths, seed)
    return mc.pricing()


# Cache LRU appliqué après la définition
mc_pricing_function_cached = lru_cache(maxsize=64)(mc_pricing_function_cached)


def mc_pricing_function(bundle: ParameterBundle, n_paths: int = 10000, seed: int = 42) -> float:
    """Fonction de pricing Monte Carlo optimisée."""
    # Utilise la version cachée si pas de dividendes
    if not bundle.dividends:
        return mc_pricing_function_cached(
            bundle.S0, bundle.r, bundle.vol, bundle.K,
            bundle.maturity.toordinal(), bundle.pricing_date.toordinal(),
            bundle.call_put, n_paths, seed
        )
    
    # Version normale avec dividendes
    params = PricerParameters(bundle.pricing_date, bundle.nb_steps, bundle.pruning, bundle.p_min)
    market = Market(bundle.S0, bundle.r, bundle.vol, bundle.dividends)
    option = Option(bundle.K, bundle.maturity, bundle.call_put, bundle.style)
    mc = MonteCarloPricer(params, market, option, n_paths, seed)
    return mc.pricing()



class GreeksBS:
    """Greeks analytiques Black-Scholes optimisés (sans Numba pour éviter les erreurs)."""
    def __init__(self, bs: BlackScholes):
        self.bs = bs
        self.d1, self.d2 = bs._d1_d2()

    def delta(self) -> float:
        if self.bs.option.call_put == "call":
            return norm.cdf(self.d1)
        return -norm.cdf(-self.d1)

    def gamma(self) -> float:
        return norm.pdf(self.d1) / (self.bs.market.S0 * self.bs.market.vol * np.sqrt(self.bs.T))

    def vega(self) -> float:
        return self.bs.market.S0 * norm.pdf(self.d1) * np.sqrt(self.bs.T) / 100

    def theta(self) -> float:
        S0, K, r, sigma, T = self.bs.market.S0, self.bs.option.K, self.bs.market.r, self.bs.market.vol, self.bs.T
        first = -(S0 * norm.pdf(self.d1) * sigma) / (2 * np.sqrt(T))
        if self.bs.option.call_put == "call":
            second = -r * K * np.exp(-r * T) * norm.cdf(self.d2)
        else:
            second = r * K * np.exp(-r * T) * norm.cdf(-self.d2)
        return (first + second) / 365

    def rho(self) -> float:
        K, r, T = self.bs.option.K, self.bs.market.r, self.bs.T
        if self.bs.option.call_put == "call":
            return (K * T * np.exp(-r * T) * norm.cdf(self.d2)) / 100
        else:
            return -(K * T * np.exp(-r * T) * norm.cdf(-self.d2)) / 100


class GreeksTree:
    """Grecques optimisées par arbre trinomial (différences finies)."""
    def __init__(self, params: PricerParameters, market: Market, option: Option):
        self.bundle = ParameterBundle(params, market, option)
        self.calculator = FiniteDifferenceCalculator(tree_pricing_function, self.bundle)

    def delta(self, h: float = 1.0) -> float:
        return self.calculator.first_derivative("S0", h)

    def gamma(self, h: float = 1.0) -> float:
        delta_up, delta_down = self.calculator.first_derivative("S0", h, for_gamma=True)
        return (delta_up - delta_down) / h

    def vega(self, h: float = 0.01) -> float:
        return self.calculator.first_derivative("vol", h) / 100

    def rho(self, h: float = 0.01) -> float:
        return self.calculator.first_derivative("r", h) / 100

    def theta(self, h_days: int = 1) -> float:
        """Approximation en décalant la date de pricing."""
        bundle_shifted = copy.deepcopy(self.bundle)
        bundle_shifted.pricing_date += np.timedelta64(h_days, 'D')
        
        price_shifted = tree_pricing_function(bundle_shifted)
        price_base = tree_pricing_function(self.bundle)
        return (price_shifted - price_base) / h_days


class GreeksMonteCarlo:
    """Greeks optimisés par Monte Carlo (différences finies)."""

    def __init__(self, params, market, option, n_paths: int = 10000, seed: int = 42):
        self.bundle = ParameterBundle(params, market, option)
        self.n_paths = n_paths
        self.seed = seed
        
        # Fonction de pricing avec paramètres fixés
        def mc_pricing_with_params(bundle: ParameterBundle) -> float:
            return mc_pricing_function(bundle, self.n_paths, self.seed)
        
        self.calculator = FiniteDifferenceCalculator(mc_pricing_with_params, self.bundle)

    def delta(self, h: float = 1.0) -> float:
        return self.calculator.first_derivative("S0", h)

    def gamma(self, h: float = 1.0) -> float:
        delta_up, delta_down = self.calculator.first_derivative("S0", h, for_gamma=True)
        return (delta_up - delta_down) / h

    def vega(self, h: float = 0.01) -> float:
        return self.calculator.first_derivative("vol", h) / 100

    def rho(self, h: float = 0.01) -> float:
        return self.calculator.first_derivative("r", h) / 100

    def theta(self, h_days: int = 1) -> float:
        """Décale la date de valorisation de h jours."""
        bundle_shifted = copy.deepcopy(self.bundle)
        bundle_shifted.pricing_date += np.timedelta64(h_days, "D")
        
        price_shifted = mc_pricing_function(bundle_shifted, self.n_paths, self.seed)
        price_base = mc_pricing_function(self.bundle, self.n_paths, self.seed)
        return (price_shifted - price_base) / h_days
