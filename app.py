import streamlit as st
import datetime as dt
import numpy as np

from option import Option
from market import Market
from parameters import PricerParameters
from black_sholes import BlackScholes
from trinomial import TrinomialTree
from monte_carlo import MonteCarloPricer
from greeks import GreeksBS, GreeksTree, GreeksMonteCarlo
from convergence import ConvergenceStudy
from timing import TimingStudy
from plot import TreePlotter, TreeExporter, MonteCarloPlotter, BlackScholesPlotter


class OptionPricerApp:
    """Application Streamlit pour pricer des options (Black-Scholes, Trinomial Tree, Monte Carlo)."""

    def __init__(self):
        self.inputs = {}

    # =========================
    # Sidebar
    # =========================
    def _sidebar_base(self):
        pricing_date = st.sidebar.date_input("Date de pricing", dt.date(2025, 1, 1))
        maturity = st.sidebar.date_input("Maturité", dt.date(2026, 1, 1))
        S0 = st.sidebar.number_input("Spot (S0)", value=100.0)
        K = st.sidebar.number_input("Strike (K)", value=100.0)
        r = st.sidebar.number_input("Taux sans risque (r)", value=0.05, format="%.4f")
        vol = st.sidebar.number_input("Volatilité (σ)", value=0.2, format="%.4f")
        return pricing_date, maturity, S0, K, r, vol

    def _sidebar_dividends(self):
        st.sidebar.subheader("📤 Dividendes")
        nb_div = st.sidebar.number_input("Nombre de dividendes", min_value=0, max_value=5, value=0)
        dividends = []
        for i in range(int(nb_div)):
            d_date = st.sidebar.date_input(f"Date ex-div {i+1}", dt.date(2025, 6, 1))
            d_montant = st.sidebar.number_input(f"Montant dividende {i+1}", value=0.0, key=f"div_{i}")
            dividends.append((dt.datetime.combine(d_date, dt.time()), d_montant))
        return dividends

    def _sidebar_tree(self):
        st.sidebar.subheader("🌳 Arbre Trinomial - Options avancées")
        mode = st.sidebar.radio("Choix du critère", ["Nombre de pas", "Erreur max tolérée"])

        if mode == "Nombre de pas":
            # MODIFICATION DEMANDÉE : number_input au lieu de slider, valeur par défaut 500
            steps = st.sidebar.number_input("Nombre de pas (Tree)", min_value=1, max_value=10000, value=500, step=1)
            max_gap = None
        else:
            max_gap = st.sidebar.number_input("Erreur max tolérée", value=0.01, format="%.4f")
            steps = None

        call_put = st.sidebar.radio("Type d'option", ["call", "put"])
        style = st.sidebar.radio("Style", ["european", "american"])
        use_pruning = st.sidebar.checkbox("Activer pruning", value=True)
        p_min = st.sidebar.number_input("Probabilité minimale (p_min)", value=1e-9, format="%.1e")
        tree_method = st.sidebar.radio("Méthode de pricing Tree", ["backward", "recursive"])

        return steps, max_gap, call_put, style, use_pruning, p_min, tree_method

    def _sidebar_montecarlo(self):
        st.sidebar.subheader("🎲 Monte Carlo - Options avancées")
        n_paths = st.sidebar.number_input("Nombre de simulations", min_value=1000, max_value=200000, value=10000, step=1000)
        seed = st.sidebar.number_input("Graine aléatoire", value=42, step=1)
        return n_paths, seed

    def _sidebar_shifts(self):
        st.sidebar.subheader("⚖️ Shifts Greeks")
        use_custom = st.sidebar.checkbox("Personnaliser les shifts", value=False)

        if use_custom:
            delta_gamma_shift = st.sidebar.number_input("Shift Δ/Γ (spot)", value=1.0, step=0.5)
            vega_shift = st.sidebar.number_input("Shift Vega (volatilité)", value=0.01, format="%.4f")
            rho_shift = st.sidebar.number_input("Shift Rho (taux)", value=0.01, format="%.4f")
            theta_shift = st.sidebar.number_input("Shift Theta (jours)", value=1, step=1)
        else:
            delta_gamma_shift, vega_shift, rho_shift, theta_shift = 1.0, 0.01, 0.01, 1

        return dict(delta_gamma=delta_gamma_shift, vega=vega_shift, rho=rho_shift, theta=theta_shift)

    def _get_monte_carlo_params(self, selected_method):
        """Récupère les paramètres Monte Carlo si nécessaire."""
        if selected_method == "Monte Carlo":
            return self._sidebar_montecarlo()
        return 10000, 42  # Valeurs par défaut

    def _build_inputs_dict(self, pricing_date, maturity, S0, K, r, vol, dividends, 
                          steps, max_gap, call_put, style, use_pruning, p_min, 
                          tree_method, n_paths, seed):
        """Construit le dictionnaire des inputs."""
        return {
            'pricing_date': pricing_date, 'maturity': maturity, 'S0': S0, 'K': K, 
            'r': r, 'vol': vol, 'dividends': dividends, 'steps': steps, 
            'max_gap': max_gap, 'call_put': call_put, 'style': style,
            'use_pruning': use_pruning, 'p_min': p_min, 'tree_method': tree_method,
            'n_paths': n_paths, 'seed': seed
        }

    def _show_black_scholes_info(self):
        """Affiche les informations sur Black-Scholes."""
        st.sidebar.info(
            "ℹ️ **Rappel Black-Scholes** :\n\n"
            "- S'applique uniquement aux **options européennes**\n"
            "- Pas de **dividendes discrets** (seul un dividende continu est gérable)\n"
            "- Pas applicable aux **options américaines**\n\n"
            "👉 Sinon, utilisez l'arbre trinomial ou Monte Carlo."
        )

    def sidebar_inputs(self, selected_method=None):
        """Configure la sidebar avec tous les paramètres nécessaires."""
        st.sidebar.header("⚙️ Paramètres")
        
        # Collecte des paramètres de base
        pricing_date, maturity, S0, K, r, vol = self._sidebar_base()
        dividends = self._sidebar_dividends()
        steps, max_gap, call_put, style, use_pruning, p_min, tree_method = self._sidebar_tree()
        n_paths, seed = self._get_monte_carlo_params(selected_method)
        
        # Construction du dictionnaire des inputs
        self.inputs = self._build_inputs_dict(
            pricing_date, maturity, S0, K, r, vol, dividends,
            steps, max_gap, call_put, style, use_pruning, p_min,
            tree_method, n_paths, seed
        )
        
        # Affichage des informations
        self._show_black_scholes_info()

    # =========================
    # Objets
    # =========================
    def _create_datetime_objects(self):
        """Crée les objets datetime pour pricing et maturité."""
        pricing_datetime = dt.datetime.combine(self.inputs["pricing_date"], dt.time())
        maturity_datetime = dt.datetime.combine(self.inputs["maturity"], dt.time())
        return pricing_datetime, maturity_datetime

    def _create_market_object(self):
        """Crée l'objet Market avec les paramètres saisis."""
        return Market(self.inputs["S0"], self.inputs["r"], self.inputs["vol"], 
                     dividends=self.inputs["dividends"])

    def _create_option_object(self, maturity_datetime):
        """Crée l'objet Option avec les paramètres saisis."""
        return Option(self.inputs["K"], maturity_datetime, 
                     self.inputs["call_put"], self.inputs["style"])

    def _calculate_steps_from_error(self):
        """Calcule le nombre de pas basé sur l'erreur tolérée."""
        max_gap = self.inputs["max_gap"]
        if max_gap <= 0.001:
            return 500
        elif max_gap <= 0.005:
            return 200
        elif max_gap <= 0.01:
            return 100
        else:
            return 50

    def _determine_steps(self):
        """Détermine le nombre de pas à utiliser."""
        steps = self.inputs["steps"]
        if steps is None and self.inputs["max_gap"] is not None:
            steps = self._calculate_steps_from_error()
            st.sidebar.write(f"📊 Nombre de pas calculé automatiquement : {steps}")
        return steps

    def _create_parameters_object(self, pricing_datetime, steps):
        """Crée l'objet PricerParameters."""
        return PricerParameters(pricing_datetime, steps,
                               pruning=self.inputs["use_pruning"], 
                               p_min=self.inputs["p_min"])

    def build_objects(self):
        """Construit les objets Market, Option et PricerParameters."""
        pricing_datetime, maturity_datetime = self._create_datetime_objects()
        market = self._create_market_object()
        option = self._create_option_object(maturity_datetime)
        steps = self._determine_steps()
        params = self._create_parameters_object(pricing_datetime, steps)
        return market, option, params

    # =========================
    # Pricing Section
    # =========================
    def _handle_black_scholes_pricing(self, market, option, params):
        """Gère le pricing Black-Scholes avec validation."""
        if option.is_american() or (market.dividends is not None and len(market.dividends) > 0):
            st.warning("⚠️ Black-Scholes n'est défini que pour les options européennes **sans dividendes discrets** et **non américaines**.\n\n"
                       "👉 Dans ce cas, ni le prix ni les Greeks BS ne sont affichés.\n"
                       "➡️ Utilisez plutôt l'arbre trinomial ou Monte Carlo.")
            return
        
        try:
            bs = BlackScholes(params, market, option)
            price = bs.pricing()
            st.success(f"Prix Black-Scholes : {price:.4f}")
            self.display_greeks_bs(bs)
        except Exception as e:
            st.error(f"Erreur : {e} ⚠️ Black-Scholes ne supporte pas les dividendes discrets ou options américaines.")

    def _handle_trinomial_pricing(self, market, option, params):
        """Gère le pricing par arbre trinomial."""
        shifts = self._sidebar_shifts()
        self.inputs["shifts"] = shifts
        tree = TrinomialTree(params, market, option)
        
        price = (tree.price_backward() if self.inputs["tree_method"] == "backward" 
                else tree.recursive_pricing())
        
        st.success(f"Prix Trinomial Tree ({self.inputs['tree_method']}) : {price:.4f}")
        self.display_greeks_tree(params, market, option)
        
        if st.button("📤 Générer et exporter l'arbre en Excel"):
            self.export_tree(tree)

    def _display_monte_carlo_info(self, option):
        """Affiche les messages informatifs pour Monte Carlo."""
        if option.is_american():
            st.info("ℹ️ **Option américaine détectée** : Monte Carlo calcule le prix comme une option européenne.\n\n"
                   "🚀 **Extension future possible** : Implémentation de la méthode Longstaff-Schwartz pour "
                   "gérer l'exercice optimal des options américaines en Monte Carlo.")
        else:
            st.info("✅ **Option européenne** : Monte Carlo est parfaitement adapté à ce type d'option.\n\n"
                   "🔮 **Extension possible** : Implémentation future de la méthode Longstaff-Schwartz "
                   "pour supporter les options américaines en Monte Carlo.")

    def _handle_monte_carlo_pricing(self, market, option, params):
        """Gère le pricing Monte Carlo."""
        mc = MonteCarloPricer(params, market, option, self.inputs["n_paths"], self.inputs["seed"])
        price = mc.pricing()
        stderr = mc.std_error()
        st.success(f"Prix Monte Carlo : {price:.4f} (± {stderr:.4f})")
        self._display_monte_carlo_info(option)
        # Afficher les Greeks calculés par Monte Carlo
        try:
            self.display_greeks_montecarlo(params, market, option)
        except Exception as e:
            # Ne pas casser l'UI si le calcul des Greeks MC échoue
            st.warning(f"Impossible d'afficher les Greeks Monte Carlo : {e}")

    def pricing_section_with_method(self, market, option, params, method):
        """Version de pricing_section qui utilise une méthode déjà sélectionnée"""
        if method == "Black-Scholes":
            self._handle_black_scholes_pricing(market, option, params)
        elif method == "Trinomial Tree":
            self._handle_trinomial_pricing(market, option, params)
        else:  # Monte Carlo
            self._handle_monte_carlo_pricing(market, option, params)

    # =========================
    # Greeks
    # =========================
    def display_greeks_bs(self, bs: BlackScholes):
        g = GreeksBS(bs)
        st.subheader("Greeks (BS)")
        st.write(f"Delta : {g.delta():.4f}")
        st.write(f"Gamma : {g.gamma():.4f}")
        st.write(f"Vega  : {g.vega():.4f}")
        st.write(f"Theta : {g.theta():.4f}")
        st.write(f"Rho   : {g.rho():.4f}")

    def display_greeks_tree(self, params, market, option):
        g = GreeksTree(params, market, option)
        h = self.inputs.get("shifts", dict(delta_gamma=1.0, vega=0.01, rho=0.01, theta=1))
        st.subheader("Greeks (Tree)")
        st.write(f"Delta : {g.delta(h['delta_gamma']):.4f}")
        st.write(f"Gamma : {g.gamma(h['delta_gamma']):.4f}")
        st.write(f"Vega  : {g.vega(h['vega']):.4f}")
        st.write(f"Theta : {g.theta(h['theta']):.4f}")
        st.write(f"Rho   : {g.rho(h['rho']):.4f}")

    def display_greeks_montecarlo(self, params, market, option):
        """Affiche les Greeks calculés par Monte Carlo (différences finies)."""
        # Utilise la configuration d'inputs (n_paths, seed) pour l'instance MC
        n_paths = int(self.inputs.get("n_paths", 10000))
        seed = int(self.inputs.get("seed", 42))

        g = GreeksMonteCarlo(params, market, option, n_paths=n_paths, seed=seed)
        h = self.inputs.get("shifts", dict(delta_gamma=1.0, vega=0.01, rho=0.01, theta=1))
        st.subheader("Greeks (Monte Carlo)")
        st.write(f"Delta : {g.delta(h['delta_gamma']):.4f}")
        st.write(f"Gamma : {g.gamma(h['delta_gamma']):.4f}")
        st.write(f"Vega  : {g.vega(h['vega']):.4f}")
        st.write(f"Theta : {g.theta(h['theta']):.4f}")
        st.write(f"Rho   : {g.rho(h['rho']):.4f}")

    # =========================
    # Export
    # =========================
    def export_tree(self, tree):
        file_path, df_preview = TreeExporter.save_to_excel(tree, ["spot", "valorisation"], filename="arbre_option")
        with open(file_path, "rb") as f:
            st.download_button(
                label="⬇️ Télécharger le fichier Excel",
                data=f,
                file_name="arbre_option.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        st.subheader("Aperçu visuel des 5 premiers niveaux de l'arbre")
        TreePlotter.plot_tree(tree, max_steps=5, show_values=True)

    # =========================
    # Études
    # =========================
    def _monte_carlo_studies(self, params, market, option):
        """Affiche les études spécialisées pour Monte Carlo."""
        st.header("🎲 Analyses Monte Carlo")
        
        if st.checkbox("📈 Convergence du prix Monte Carlo"):
            mc = MonteCarloPricer(params, market, option, self.inputs["n_paths"], self.inputs["seed"])
            MonteCarloPlotter.plot_convergence(mc, self.inputs)
            
        if st.checkbox("📈 Distribution des prix finaux"):
            mc = MonteCarloPricer(params, market, option, self.inputs["n_paths"], self.inputs["seed"])
            MonteCarloPlotter.plot_price_distribution(mc)
            
        if st.checkbox("🎯 Distribution des payoffs"):
            mc = MonteCarloPricer(params, market, option, self.inputs["n_paths"], self.inputs["seed"])
            MonteCarloPlotter.plot_payoff_distribution(mc)

    def _black_scholes_studies(self, params, market, option):
        """Affiche les études spécialisées pour Black-Scholes."""
        st.header("📈 Analyses Black-Scholes")
        
        if st.checkbox("📈 Sensibilité aux paramètres (Greeks)"):
            bs = BlackScholes(params, market, option)
            BlackScholesPlotter.plot_sensitivity_analysis(bs)
            
        if st.checkbox("🎨 Surface de prix (3D)"):
            bs = BlackScholes(params, market, option)
            BlackScholesPlotter.plot_price_surface(bs)
            
        if st.checkbox("📉 Volatilité implicite vs réalisée"):
            bs = BlackScholes(params, market, option)
            BlackScholesPlotter.plot_volatility_analysis(bs, market, option)
            
        if st.checkbox("🎲 Comparaison Monte Carlo"):
            bs = BlackScholes(params, market, option)
            BlackScholesPlotter.plot_monte_carlo_comparison(bs, self.inputs)

    def _check_bs_compatibility(self, option, market):
        """Vérifie si Black-Scholes est compatible avec l'option."""
        return not (option.is_american() or (market.dividends and len(market.dividends) > 0))

    def _show_bs_warning(self):
        """Affiche l'avertissement pour Black-Scholes."""
        st.warning("⚠️ Étude indisponible : Black-Scholes n'est défini que pour les options "
                  "européennes **sans dividendes discrets** et **non américaines**.")

    def _convergence_study(self, params, market, option):
        """Effectue l'étude de convergence Tree → BS."""
        if not self._check_bs_compatibility(option, market):
            self._show_bs_warning()
            return
            
        try:
            bs = BlackScholes(params, market, option)
            conv = ConvergenceStudy(bs)
            conv.study_convergence(nb_steps_max=params.nb_steps)
        except Exception as e:
            st.error(f"Étude de convergence impossible ⚠️\n\nRaison : {e}")

    def _prices_slopes_study(self, params, market, option):
        """Effectue l'étude prix & pentes BS vs Tree."""
        if not self._check_bs_compatibility(option, market):
            self._show_bs_warning()
            return
            
        try:
            bs = BlackScholes(params, market, option)
            conv = ConvergenceStudy(bs)
            k_min, k_max = int(self.inputs["K"] * 0.8), int(self.inputs["K"] * 1.2)
            conv.study_prices_and_slopes(k_min, k_max)
        except Exception as e:
            st.error(f"Étude BS vs Tree impossible ⚠️\n\nRaison : {e}")

    def _timing_study(self, params, market, option):
        """Effectue l'étude des temps d'exécution."""
        timing = TimingStudy(params, market, option)
        times_b, times_r = timing.run(max_steps=50)
        TreePlotter.plot_execution_times(times_b, "Backward")
        TreePlotter.plot_execution_times(times_r, "Recursive")

    def _trinomial_studies(self, params, market, option):
        """Affiche les études spécialisées pour Trinomial Tree."""
        st.header("🔎 Études supplémentaires")

        if st.checkbox("Étudier la convergence (Tree → BS)"):
            self._convergence_study(params, market, option)

        if st.checkbox("Étudier prix & pentes (BS vs Tree)"):
            self._prices_slopes_study(params, market, option)

        if st.checkbox("Étudier temps d'exécution (Backward vs Recursive)"):
            self._timing_study(params, market, option)

    def studies_section(self, params, market, option, method):
        """Affiche les études selon la méthode sélectionnée."""
        if method == "Monte Carlo":
            self._monte_carlo_studies(params, market, option)
        elif method == "Black-Scholes":
            self._black_scholes_studies(params, market, option)
        elif method == "Trinomial Tree":
            self._trinomial_studies(params, market, option)

    # =========================
    # Run
    # =========================
    def run(self):
        st.title("📈 Option Pricer - Black-Scholes, Trinomial Tree & Monte Carlo")
        
        # Première passe pour obtenir la méthode sélectionnée
        method = st.radio("Méthode de pricing", ["Black-Scholes", "Trinomial Tree", "Monte Carlo"])
        
        # Sidebar avec options conditionnelles
        self.sidebar_inputs(method)
        market, option, params = self.build_objects()
        
        # Pricing avec la méthode déjà sélectionnée
        self.pricing_section_with_method(market, option, params, method)
        self.studies_section(params, market, option, method)


def main():
    OptionPricerApp().run()


if __name__ == "__main__":
    main()