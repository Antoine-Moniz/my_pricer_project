from option import Option
from market import Market
from parameters import PricerParameters
from trinomial import TrinomialTree

import numpy as np
from scipy.stats import norm
from typing import Tuple


class BlackScholes:
    """
    Pricer Black-Scholes pour options européennes (sans dividendes discrets).
    Ne gère que les calculs.
    """

    def __init__(self, params: PricerParameters, market: Market, option: Option) -> None:
        # ✅ Bloque seulement si dividendes effectivement présents ou option américaine
        if option.is_american() or (market.dividends is not None and len(market.dividends) > 0):
            raise TypeError("Option européenne sans dividende discret requise.")
        self.params, self.market, self.option = params, market, option
        self.T = (option.maturity - params.pricing_date).days / 365

    def _d1_d2(self) -> Tuple[float, float]:
        """Calcule d1 et d2 en fonction du strike courant."""
        S0, K, r, sigma, T = self.market.S0, self.option.K, self.market.r, self.market.vol, self.T
        sqrtT = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / sqrtT
        return d1, d1 - sqrtT

    def pricing(self) -> float:
        """Formule fermée Black-Scholes (call/put européen)."""
        S0, K, r, T = self.market.S0, self.option.K, self.market.r, self.T
        d1, d2 = self._d1_d2()
        disc = np.exp(-r * T)
        if self.option.call_put == "call":
            return S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
        return K * disc * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    def _get_convergence_steps(self, nb_steps_max: int) -> np.ndarray:
        """Génère un échantillonnage intelligent des steps pour convergence."""
        if nb_steps_max <= 50:
            steps = np.arange(1, nb_steps_max + 1)
        elif nb_steps_max <= 200:
            steps = np.concatenate([
                np.arange(1, 21),
                np.arange(25, nb_steps_max + 1, 5)
            ])
        else:
            steps = np.unique(np.concatenate([
                np.arange(1, 21),
                np.arange(25, 101, 5),
                np.arange(110, 201, 10),
                np.logspace(np.log10(220), np.log10(nb_steps_max), 15).astype(int)
            ]))
        return steps[steps <= nb_steps_max]

    def _compute_convergence_diffs(self, steps: np.ndarray, bs_price: float) -> np.ndarray:
        """Calcule les différences de convergence pour les steps donnés."""
        return np.array([
            (TrinomialTree(
                PricerParameters(self.params.pricing_date, n, self.params.pruning, self.params.p_min),
                self.market, self.option
            ).price_backward() - bs_price) * n
            for n in steps
        ])

    def convergence_values(self, nb_steps_max: int):
        """Calcule les valeurs de convergence avec échantillonnage intelligent."""
        bs_price = self.pricing()
        steps = self._get_convergence_steps(nb_steps_max)
        diffs = self._compute_convergence_diffs(steps, bs_price)
        return diffs, steps

    def _optimize_strikes_range(self, K_min: int, K_max: int) -> np.ndarray:
        """Optimise la plage de strikes pour limiter les calculs."""
        total_strikes = K_max - K_min + 1
        
        if total_strikes <= 50:
            return np.arange(K_min, K_max + 1)
        
        max_points = 50
        step = max(1, total_strikes // max_points)
        strikes = np.arange(K_min, K_max + 1, step)
        if strikes[-1] != K_max:
            strikes = np.append(strikes, K_max)
        
        return strikes

    def _compute_prices_for_strikes(self, strikes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calcule les prix BS et Tree pour chaque strike."""
        bs_prices, tree_prices = [], []
        original_k = self.option.K
        
        try:
            for k in strikes:
                self.option.K = k
                bs_prices.append(self.pricing())
                tree_prices.append(TrinomialTree(self.params, self.market, self.option).price_backward())
        finally:
            self.option.K = original_k
        
        return np.array(bs_prices), np.array(tree_prices)

    def prices_and_slopes_values(self, K_min: int, K_max: int):
        """Calcule les valeurs prix + pentes avec optimisation automatique."""
        strikes = self._optimize_strikes_range(K_min, K_max)
        bs_prices, tree_prices = self._compute_prices_for_strikes(strikes)
        
        diffs = tree_prices - bs_prices
        bs_slopes = np.diff(bs_prices) / np.diff(strikes)
        tree_slopes = np.diff(tree_prices) / np.diff(strikes)
        
        return strikes, bs_prices, tree_prices, bs_slopes, tree_slopes, diffs

    def _compute_delta_gamma_sensitivities(self, s_range: np.ndarray) -> Tuple[list, list]:
        """Calcule Delta et Gamma pour une plage de prix spot."""
        original_s0 = self.market.S0
        deltas, gammas = [], []
        
        try:
            for s in s_range:
                self.market.S0 = s
                d1, d2 = self._d1_d2()
                if self.option.call_put == "call":
                    deltas.append(norm.cdf(d1))
                else:
                    deltas.append(-norm.cdf(-d1))
                gammas.append(norm.pdf(d1) / (s * self.market.vol * np.sqrt(self.T)))
        finally:
            self.market.S0 = original_s0
        
        return deltas, gammas

    def _compute_vega_sensitivities(self, vol_range: np.ndarray) -> list:
        """Calcule Vega pour une plage de volatilités."""
        original_vol = self.market.vol
        vegas = []
        
        try:
            for vol in vol_range:
                self.market.vol = vol
                d1, d2 = self._d1_d2()
                vegas.append(self.market.S0 * norm.pdf(d1) * np.sqrt(self.T))
        finally:
            self.market.vol = original_vol
        
        return vegas

    def _compute_time_sensitivities(self, time_range: np.ndarray) -> list:
        """Calcule les prix pour une plage de temps (Theta)."""
        original_t = self.T
        prices_time = []
        
        try:
            for t in time_range:
                self.T = t
                prices_time.append(self.pricing())
        finally:
            self.T = original_t
        
        return prices_time

    def sensitivity_analysis(self, param_range=0.3, n_points=50):
        """Étude de sensibilité aux paramètres pour graphiques."""
        original_s0, original_vol = self.market.S0, self.market.vol
        
        s_range = np.linspace(original_s0 * (1 - param_range), original_s0 * (1 + param_range), n_points)
        vol_range = np.linspace(max(0.01, original_vol * (1 - param_range)), original_vol * (1 + param_range), n_points)
        time_range = np.linspace(0.01, self.T, n_points)
        
        deltas, gammas = self._compute_delta_gamma_sensitivities(s_range)
        vegas = self._compute_vega_sensitivities(vol_range)
        prices_time = self._compute_time_sensitivities(time_range)
        
        return {
            'S_range': s_range, 'deltas': deltas, 'gammas': gammas,
            'vol_range': vol_range, 'vegas': vegas,
            'time_range': time_range, 'prices_time': prices_time
        }

    def price_surface(self, strike_range=0.4, time_range_days=365, n_points=30):
        """Étude de surface de prix (Strike vs Temps)."""
        original_K = self.option.K
        original_T = self.T
        
        # Plages
        K_range = np.linspace(original_K * (1 - strike_range), original_K * (1 + strike_range), n_points)
        T_range = np.linspace(0.01, time_range_days/365, n_points)
        
        # Surface
        prices = np.zeros((len(T_range), len(K_range)))
        for i, T in enumerate(T_range):
            self.T = T
            for j, K in enumerate(K_range):
                self.option.K = K
                prices[i, j] = self.pricing()
        
        # Restaurer
        self.option.K = original_K
        self.T = original_T
        
        return K_range, T_range, prices

    def compare_with_monte_carlo(self, n_paths=10000, seed=42):
        """Comparaison avec Monte Carlo."""
        from monte_carlo import MonteCarloPricer
        
        bs_price = self.pricing()
        mc = MonteCarloPricer(self.params, self.market, self.option, n_paths, seed)
        mc_price = mc.pricing()
        mc_stderr = mc.std_error()
        
        return {
            'bs_price': bs_price,
            'mc_price': mc_price,
            'mc_stderr': mc_stderr,
            'difference': abs(bs_price - mc_price),
            'relative_error': abs(bs_price - mc_price) / bs_price * 100
        }
