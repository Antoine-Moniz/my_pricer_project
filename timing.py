import time
from trinomial import TrinomialTree


class TimingStudy:
    """Étude des temps d’exécution entre backward et recursive."""

    def __init__(self, params, market, option):
        self.params = params
        self.market = market
        self.option = option

    def measure_time(self, method: str):
        """Mesure le temps pour une méthode donnée."""
        tree = TrinomialTree(self.params, self.market, self.option)
        start = time.time()
        if method == "backward":
            tree.price_backward()
        else:
            tree.recursive_pricing()
        return time.time() - start

    def run(self, max_steps=100):
        """Mesure les temps des deux méthodes pour plusieurs steps."""
        times_backward, times_recursive = [], []
        for n in range(1, max_steps + 1):
            self.params.nb_steps = n
            times_backward.append(self.measure_time("backward"))
            times_recursive.append(self.measure_time("recursive"))
        return times_backward, times_recursive
