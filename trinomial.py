# trinomial.py - Version corrigée basée sur la référence

from market import Market
from option import Option
from parameters import PricerParameters
from typing import Optional, Tuple
import datetime as dt
import numpy as np
 
# Optional Numba support: provide a safe no-op `njit` decorator when Numba
# is not available so the module can be imported in environments without it.
try:
    from numba import njit  # type: ignore
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
    def njit(*args, **kwargs):
        # Support both @njit and @njit()
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def _decorator(func):
            return func
        return _decorator


@njit
def _compute_transition_probs(next_mid_price: float, expected: float, variance: float, alpha: float):
    """Compute transition probabilities (pure numeric) so it can be jitted by Numba.

    Returns (prob_up, prob_mid, prob_down)
    """
    prob_down = ((next_mid_price ** (-2) * (variance + expected ** 2) - 1 -
                  (alpha + 1) * (next_mid_price ** (-1) * expected - 1))
                 / ((1 - alpha) * (alpha ** (-2) - 1)))

    prob_up = ((next_mid_price ** (-1) * expected - 1) -
               ((alpha ** (-1) - 1) * prob_down)) / (alpha - 1)

    prob_mid = 1.0 - prob_up - prob_down
    return prob_up, prob_mid, prob_down


@njit
def _compute_forward_and_variance(spot: float, r: float, delta_t: float, vol: float, dividend_adjustment: float):
    """Compute forward (expected) and variance from numeric inputs (Numba-friendly)."""
    pure_forward = spot * np.exp(r * delta_t)
    expected = pure_forward - dividend_adjustment
    variance = (spot ** 2 * 
                np.exp(2 * r * delta_t) * 
                (np.exp(vol ** 2 * delta_t) - 1))
    return expected, variance


@njit
def _propagate_reach_probs_numeric(parent_reach: float, prob_up: float, prob_mid: float, prob_down: float):
    """Compute increments for child reach probabilities."""
    return parent_reach * prob_up, parent_reach * prob_mid, parent_reach * prob_down


@njit
def _should_build_branches(pruning: bool, reach_prob: float, p_min: float) -> int:
    """Decide if branches should be built: returns 1 if yes, 0 if no."""
    if not pruning:
        return 1
    return 1 if reach_prob > p_min else 0


class TreeNode:
    """Noeud de l'arbre trinomial basé sur la référence."""
    
    def __init__(self, spot: float, tree: 'TrinomialTree', date: dt.datetime) -> None:
        self.spot: float = spot
        self.tree: 'TrinomialTree' = tree
        self.date: dt.datetime = date
        self.opt_val: Optional[float] = None
        self.prob_up: float = 0
        self.prob_mid: float = 0
        self.prob_down: float = 0
        self.reach_prob: float = 0
        self.next_up: Optional['TreeNode'] = None
        self.next_mid: Optional['TreeNode'] = None
        self.next_down: Optional['TreeNode'] = None
        self.upper_node: Optional['TreeNode'] = None
        self.lower_node: Optional['TreeNode'] = None
        self.early_exercise: bool = False

    def forward_and_variance(self) -> Tuple[float, float]:
        """Calcule la valeur forward et la variance avec gestion des dividendes."""
        # Calcul de la valeur forward de base
        # Ajustement pour les dividendes sur la période (non-numba)
        # utiliser un delta_t flottant pour éviter l'arrondi qui peut déplacer la période
        next_date = self.date + dt.timedelta(days=self.tree.delta_t * 365)
        dividend_adjustment = self.tree.market.get_dividend_on_period(self.date, next_date)

        # Appel au helper numérique jitable
        expected, variance = _compute_forward_and_variance(float(self.spot), float(self.tree.market.r),
                                                           float(self.tree.delta_t), float(self.tree.market.vol),
                                                           float(dividend_adjustment))
        return expected, variance

    def set_transition_probs(self) -> None:
        """Calcule les probabilités."""
        if self.tree.params.pruning and self.reach_prob < self.tree.params.p_min:
            self.prob_up, self.prob_mid, self.prob_down = 0, 1, 0
            return

        expected, variance = self.forward_and_variance()
        next_mid_price = float(self.next_mid.spot)

        # Use the pure numeric helper; this can be jitted by Numba for speed
        prob_up, prob_mid, prob_down = _compute_transition_probs(next_mid_price, expected, variance, float(self.tree.alpha))

        self.prob_up = prob_up
        self.prob_mid = prob_mid
        self.prob_down = prob_down

        if (self.prob_up < 0 or self.prob_mid < 0 or self.prob_down < 0):
            print(f"Probabilité négative à la date {self.date}")

    def propagate_reach_probs(self) -> None:
        """Propage les probabilités d'atteindre les noeuds."""
        # Try to use numeric helper when values are pure floats
        try:
            inc_up, inc_mid, inc_down = _propagate_reach_probs_numeric(self.reach_prob, self.prob_up, self.prob_mid, self.prob_down)
            if self.next_up is not None:
                self.next_up.reach_prob += inc_up
            if self.next_mid is not None:
                self.next_mid.reach_prob += inc_mid
            if self.next_down is not None:
                self.next_down.reach_prob += inc_down
            return
        except Exception:
            # Fallback simple loop
            probabilities = [self.prob_up, self.prob_mid, self.prob_down]
            nodes = [self.next_up, self.next_mid, self.next_down]
            for prob, node in zip(probabilities, nodes):
                if node is not None:
                    node.reach_prob += self.reach_prob * prob

    def compute_next_mid(self) -> 'TreeNode':
        """Crée le prochain noeud du milieu."""
        if self.next_mid is None:
            expected = self.forward_and_variance()[0]
            next_date = self.date + dt.timedelta(days=self.tree.delta_t * 365)
            if isinstance(self, TrunkNode):
                self.next_mid = TrunkNode(expected, self.tree, next_date, self)
            else:
                self.next_mid = TreeNode(expected, self.tree, next_date)
        # Si nous sommes en période de dividende, ajuster le noeud du milieu pour
        # préserver la recombinaison (on cherche le candidat approprié)
        if self.dividend_period():
            n_candidate = None
            if self.lower_node is not None and getattr(self.lower_node, 'next_up', None) is not None:
                n_candidate = self.lower_node.next_up
            elif self.upper_node is not None and getattr(self.upper_node, 'next_down', None) is not None:
                n_candidate = self.upper_node.next_down
            if n_candidate is not None:
                self.find_next_mid(n_candidate)
        return self.next_mid

    def dividend_period(self) -> bool:
        """True si un dividende est payé entre ce noeud et le suivant."""
        # Récupère toutes les dates d'ex-div du marché (liste de tuples (date, amount))
        try:
            ex_dates = [d for (d, a) in self.tree.market.dividends]
        except Exception:
            ex_dates = []
        if not ex_dates:
            return False
        next_date = self.date + dt.timedelta(days=self.tree.delta_t * 365)
        for d in ex_dates:
            if self.date < d <= next_date:
                return True
        return False

    def find_next_mid(self, n_candidate: 'TreeNode') -> 'TreeNode':
        """Ajuste self.next_mid pour qu'il pointe vers le noeud candidat correct lors d'un dividende.

        On déplace le candidat vers le haut ou vers le bas tant que la condition
        de recombinaison n'est pas satisfaite.
        """
        forward = self.forward_and_variance()[0]
        # Monter le candidat tant que forward >= moyenne(candidate, candidate*alpha)
        while forward >= (n_candidate.spot + n_candidate.spot * self.tree.alpha) / 2:
            n_candidate = n_candidate.compute_up_node()
        # Descendre le candidat tant que forward <= moyenne(candidate, candidate/alpha)
        while forward <= (n_candidate.spot + n_candidate.spot / self.tree.alpha) / 2:
            n_candidate = n_candidate.compute_down_node()
        self.next_mid = n_candidate
        return n_candidate

    def compute_up_node(self) -> 'TreeNode':
        """Crée le noeud supérieur."""
        if self.upper_node is None:
            self.upper_node = TreeNode(self.spot * self.tree.alpha, self.tree, self.date)
            self.upper_node.lower_node = self
        return self.upper_node

    def compute_down_node(self) -> 'TreeNode':
        """Crée le noeud inférieur."""
        if self.lower_node is None:
            self.lower_node = TreeNode(self.spot / self.tree.alpha, self.tree, self.date)
            self.lower_node.upper_node = self
        return self.lower_node

    def price(self) -> float:
        """Calcule le prix récursivement."""
        if self.opt_val is not None:
            return self.opt_val
        
        if self.next_mid is None:
            self.opt_val = self.tree.option.payoff(self.spot)
            return self.opt_val
        
        price_up = self.next_up.price() if self.next_up else 0
        price_down = self.next_down.price() if self.next_down else 0
        price_mid = self.next_mid.price()
        
        price = (price_up * self.prob_up + price_mid * self.prob_mid + 
                price_down * self.prob_down) * self.tree.df
        
        if self.tree.option.is_american():
            intrinsic = self.tree.option.payoff(self.spot)
            if intrinsic > price:
                price = intrinsic
                self.early_exercise = True
        
        self.opt_val = price
        return price


class TrunkNode(TreeNode):
    """Noeud du tronc avec connexion arrière."""
    
    def __init__(self, spot: float, tree: 'TrinomialTree', date: dt.datetime, prev_node: Optional['TrunkNode'] = None):
        super().__init__(spot, tree, date)
        self.prev_node = prev_node


class TrinomialTree:
    """Arbre trinomial corrigé basé sur la référence."""
    
    def __init__(self, params: PricerParameters, market: Market, option: Option) -> None:
        self.params = params
        self.market = market
        self.option = option
        self.delta_t: float = (option.maturity - params.pricing_date).days / (params.nb_steps * 365)
        self.df: float = np.exp(-market.r * self.delta_t)
        self.alpha: float = np.exp(market.vol * np.sqrt(3 * self.delta_t))
        
    # Remarque: on gère les dividendes localement par période dans les noeuds.
    # On n'applique pas d'ajustement global du spot ici pour éviter le double comptage.

    def _calculate_total_dividend_adjustment(self) -> float:
        """Calcule la valeur présente de tous les dividendes futurs."""
        total_pv_dividends = 0.0
        maturity_time = (self.option.maturity - self.params.pricing_date).days / 365
        
        for div_date, div_amount in self.market.dividends:
            time_to_dividend = (div_date - self.params.pricing_date).days / 365
            # Ne considérer que les dividendes avant maturité
            if 0 < time_to_dividend <= maturity_time:
                pv_dividend = div_amount * np.exp(-self.market.r * time_to_dividend)
                total_pv_dividends += pv_dividend
        
        return total_pv_dividends

    def build_triplet(self, current_node: TreeNode) -> TreeNode:
        """Construit le triplet de noeuds suivants."""
        current_node.next_mid = current_node.compute_next_mid()
        # Si pas de pruning ou probabilité suffisante
        if _should_build_branches(self.params.pruning, current_node.reach_prob, self.params.p_min):
            current_node.next_up = current_node.next_mid.compute_up_node()
            current_node.next_down = current_node.next_mid.compute_down_node()
        
        current_node.set_transition_probs()
        current_node.propagate_reach_probs()
        return current_node

    def _build_upward_nodes(self, parent_trunk: TrunkNode) -> None:
        """Construit les nœuds vers le haut."""
        parent_node = parent_trunk
        while parent_node.upper_node is not None:
            parent_node = parent_node.upper_node
            if _should_build_branches(self.params.pruning, parent_node.lower_node.reach_prob, self.params.p_min):
                parent_node.next_mid = parent_node.lower_node.next_up
                self.build_triplet(parent_node)
            else:
                self.build_triplet(parent_node)
                # Connexions pour pruning
                parent_node.next_mid.lower_node = parent_node.lower_node.next_mid
                parent_node.lower_node.next_mid.upper_node = parent_node.next_mid

    def _build_downward_nodes(self, parent_trunk: TrunkNode) -> None:
        """Construit les nœuds vers le bas."""
        parent_node = parent_trunk
        while parent_node.lower_node is not None:
            parent_node = parent_node.lower_node
            if _should_build_branches(self.params.pruning, parent_node.upper_node.reach_prob, self.params.p_min):
                parent_node.next_mid = parent_node.upper_node.next_down
                self.build_triplet(parent_node)
            else:
                self.build_triplet(parent_node)
                # Connexions pour pruning
                parent_node.next_mid.upper_node = parent_node.upper_node.next_mid
                parent_node.upper_node.next_mid.lower_node = parent_node.next_mid

    def build_column(self, current_trunk: TrunkNode) -> TrunkNode:
        """Construit une colonne de l'arbre."""
        parent_trunk = self.build_triplet(current_trunk)
        
        self._build_upward_nodes(parent_trunk)
        self._build_downward_nodes(parent_trunk)
        
        return parent_trunk.next_mid

    def build_tree(self) -> None:
        """Construit l'arbre complet avec ajustement pour dividendes."""
        if self.params.pruning and self.params.p_min is None:
            raise ValueError("Pruning nécessite p_min")
        
        # On utilise le spot du marché en racine; les dividendes sont gérés localement
        self.root = TrunkNode(self.market.S0, self, self.params.pricing_date)
        self.root.reach_prob = 1.0
        current_trunk = self.root

        for _ in range(self.params.nb_steps):
            current_trunk = self.build_column(current_trunk)

        self.last = current_trunk

    def price_backward(self) -> float:
        """Pricing backward optimisé."""
        self.build_tree()
        current_trunk = self.last
        
        while current_trunk.date > self.params.pricing_date:
            current_node = current_trunk
            current_node.price()

            # Vers le haut
            while current_node.upper_node is not None:
                current_node = current_node.upper_node
                current_node.price()

            # Vers le bas
            current_node = current_trunk
            while current_node.lower_node is not None:
                current_node = current_node.lower_node
                current_node.price()

            current_trunk = current_trunk.prev_node
        
        current_trunk.price()
        return current_trunk.opt_val

    def recursive_pricing(self) -> float:
        """Pricing récursif avec protection."""
        try:
            self.build_tree()
            return self.root.price()
        except RecursionError:
            raise RuntimeError("Profondeur de récursion atteinte : utilisez price_backward pour un grand nombre de pas.")