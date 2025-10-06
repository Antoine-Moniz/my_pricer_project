# trinomial.py - Version corrigée basée sur la référence

from market import Market
from option import Option
from parameters import PricerParameters
from typing import Optional, Tuple
import datetime as dt
import numpy as np


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
        pure_forward: float = self.spot * np.exp(self.tree.market.r * self.tree.delta_t)
        
        # Ajustement pour les dividendes sur la période
        next_date = self.date + dt.timedelta(days=int(self.tree.delta_t * 365))
        dividend_adjustment = self.tree.market.get_dividend_on_period(self.date, next_date)
        
        # Forward ajusté pour les dividendes
        expected: float = pure_forward - dividend_adjustment
        
        # Variance (inchangée pour la simplicité)
        variance: float = (self.spot ** 2 * 
                          np.exp(2 * self.tree.market.r * self.tree.delta_t) * 
                          (np.exp(self.tree.market.vol ** 2 * self.tree.delta_t) - 1))
        
        return expected, variance

    def set_transition_probs(self) -> None:
        """Calcule les probabilités (référence exacte)."""
        if self.tree.params.pruning and self.reach_prob < self.tree.params.p_min:
            self.prob_up, self.prob_mid, self.prob_down = 0, 1, 0
            return

        expected, variance = self.forward_and_variance()
        next_mid_price = self.next_mid.spot
        
        self.prob_down = ((next_mid_price ** (-2) * (variance + expected ** 2) - 1 - 
                          (self.tree.alpha + 1) * (next_mid_price ** (-1) * expected - 1)) 
                         / ((1 - self.tree.alpha) * (self.tree.alpha ** (-2) - 1)))
        
        self.prob_up = ((next_mid_price ** (-1) * expected - 1) - 
                       ((self.tree.alpha ** (-1) - 1) * self.prob_down)) / (self.tree.alpha - 1)
        
        self.prob_mid = 1.0 - self.prob_up - self.prob_down

        if (self.prob_up < 0 or self.prob_mid < 0 or self.prob_down < 0):
            print(f"Probabilité négative à la date {self.date}")

    def propagate_reach_probs(self) -> None:
        """Propage les probabilités d'atteindre les noeuds."""
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
        return self.next_mid

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
        
        # Calcul de l'ajustement total pour les dividendes
        self.dividend_adjustment = self._calculate_total_dividend_adjustment()

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
        if not self.params.pruning or current_node.reach_prob > self.params.p_min:
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
            if not self.params.pruning or parent_node.lower_node.reach_prob > self.params.p_min:
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
            if not self.params.pruning or parent_node.upper_node.reach_prob > self.params.p_min:
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
        
        # Spot ajusté pour les dividendes (comme dans Monte Carlo)
        adjusted_spot = self.market.S0 - self.dividend_adjustment
        
        self.root = TrunkNode(adjusted_spot, self, self.params.pricing_date)
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