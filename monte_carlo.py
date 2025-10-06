import numpy as np
from numba import jit, prange
from option import Option
from market import Market
from parameters import PricerParameters


@jit(nopython=True, cache=True)
def _simulate_paths_numba(S0: float, r: float, sigma: float, T: float, n_paths: int, seed: int) -> np.ndarray:
    """Simulation ultra-rapide des trajectoires avec Numba JIT."""
    np.random.seed(seed)
    Z = np.random.normal(0.0, 1.0, n_paths)
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T)
    return S0 * np.exp(drift + diffusion * Z)


@jit(nopython=True, cache=True)
def _calculate_payoffs_numba(ST: np.ndarray, K: float, is_call: bool) -> np.ndarray:
    """Calcul ultra-rapide des payoffs avec Numba JIT."""
    payoffs = np.zeros_like(ST)
    if is_call:
        for i in prange(len(ST)):
            payoffs[i] = max(ST[i] - K, 0.0)
    else:
        for i in prange(len(ST)):
            payoffs[i] = max(K - ST[i], 0.0)
    return payoffs


@jit(nopython=True, cache=True)
def _calculate_dividend_adjustment(dividends_array: np.ndarray, r: float, T: float) -> float:
    """Calcul rapide de l'ajustement dividende avec Numba."""
    total_div = 0.0
    for i in range(len(dividends_array)):
        div_time = dividends_array[i, 0]  # Temps en années
        div_amount = dividends_array[i, 1]  # Montant
        if 0 < div_time <= T:
            total_div += div_amount * np.exp(-r * div_time)
    return total_div


class MonteCarloPricer:
    """
    Pricer Monte Carlo ultra-optimisé avec Numba JIT.
    Gains de performance : 10x à 100x plus rapide !
    """

    def __init__(self, params: PricerParameters, market: Market, option: Option,
                 n_paths: int = 10000, seed: int = 42) -> None:
        self.params = params
        self.market = market
        self.option = option
        self.n_paths = n_paths
        self.seed = seed

        # Horizon en années
        self.T = (option.maturity - params.pricing_date).days / 365
        
        # Préparation des dividendes pour Numba (array 2D)
        self.dividends_array = self._prepare_dividends_array()

    def _prepare_dividends_array(self) -> np.ndarray:
        """Prépare les dividendes sous forme d'array NumPy pour Numba."""
        if not self.market.dividends:
            return np.array([[]], dtype=np.float64).reshape(0, 2)
        
        div_data = []
        for date, amount in self.market.dividends:
            time_to_div = (date - self.params.pricing_date).days / 365
            div_data.append([time_to_div, amount])
        
        return np.array(div_data, dtype=np.float64)

    def pricing(self) -> float:
        """Prix ultra-rapide par Monte Carlo avec Numba JIT."""
        # Ajustement pour dividendes
        div_adjustment = _calculate_dividend_adjustment(self.dividends_array, self.market.r, self.T)
        s0_adjusted = self.market.S0 - div_adjustment
        
        # Simulation des trajectoires (ultra-rapide avec Numba)
        ST = _simulate_paths_numba(s0_adjusted, self.market.r, self.market.vol, self.T, self.n_paths, self.seed)
        
        # Calcul des payoffs (ultra-rapide avec Numba)
        is_call = self.option.call_put == "call"
        payoffs = _calculate_payoffs_numba(ST, self.option.K, is_call)
        
        # Actualisation et moyenne
        discount_factor = np.exp(-self.market.r * self.T)
        return discount_factor * np.mean(payoffs)

    def std_error(self) -> float:
        """Erreur standard de l'estimateur Monte Carlo."""
        # Ajustement pour dividendes
        div_adjustment = _calculate_dividend_adjustment(self.dividends_array, self.market.r, self.T)
        s0_adjusted = self.market.S0 - div_adjustment
        
        # Simulation des trajectoires (ultra-rapide avec Numba)
        ST = _simulate_paths_numba(s0_adjusted, self.market.r, self.market.vol, self.T, self.n_paths, self.seed)
        
        # Calcul des payoffs (ultra-rapide avec Numba)
        is_call = self.option.call_put == "call"
        payoffs = _calculate_payoffs_numba(ST, self.option.K, is_call)
        
        # Actualisation et erreur standard
        discount_factor = np.exp(-self.market.r * self.T)
        return discount_factor * np.std(payoffs) / np.sqrt(self.n_paths)

    def get_simulation_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Récupère les prix finaux et payoffs pour les graphiques."""
        # Ajustement pour dividendes
        div_adjustment = _calculate_dividend_adjustment(self.dividends_array, self.market.r, self.T)
        s0_adjusted = self.market.S0 - div_adjustment
        
        # Simulation des trajectoires
        ST = _simulate_paths_numba(s0_adjusted, self.market.r, self.market.vol, self.T, self.n_paths, self.seed)
        
        # Calcul des payoffs
        is_call = self.option.call_put == "call"
        payoffs = _calculate_payoffs_numba(ST, self.option.K, is_call)
        
        return ST, payoffs

    def convergence_study(self, max_sims: int = None) -> tuple[list, list]:
        """Étude de convergence du prix Monte Carlo."""
        if max_sims is None:
            max_sims = min(self.n_paths, 50000)
        
        # Points de simulation pour l'étude
        sim_points = [500, 1000, 2000, 5000, 10000, 20000]
        sim_points = [n for n in sim_points if n <= max_sims]
        if max_sims not in sim_points and max_sims <= 50000:
            sim_points.append(max_sims)
        sim_points.sort()
        
        prices = []
        for n_sim in sim_points:
            # Créer un pricer temporaire avec moins de simulations
            temp_pricer = MonteCarloPricer(self.params, self.market, self.option, n_sim, self.seed)
            prices.append(temp_pricer.pricing())
        
        return sim_points, prices
