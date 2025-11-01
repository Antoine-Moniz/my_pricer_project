# **Option Pricer - Projet Finance Quantitative**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

> **Application complète de pricing d'options avec interface interactive Streamlit**  
> Implémentation de Black-Scholes, Arbres Trinomiaux et Monte Carlo avec analyses de convergence et visualisations avancées.

---

## 📋 **Table des Matières**

- [ Aperçu](#-aperçu)
- [ Fonctionnalités](#-fonctionnalités)
- [ Installation](#-installation)
- [ Utilisation](#-utilisation)
- [ Architecture](#-architecture)
- [ Méthodes de Pricing](#-méthodes-de-pricing)
- [ Visualisations](#-visualisations)
- [ Analyses Avancées](#-analyses-avancées)
- [ Tests et Validation](#-tests-et-validation)
- [ Contribution](#-contribution)
- [ Licence](#-licence)

---

##  **Aperçu**

Cette application implémente les principales méthodes de pricing d'options en finance quantitative :

- ** Théorie** : Modèle de Black-Scholes-Merton
- ** Numérique** : Arbres trinomiaux avec pruning intelligent
- ** Simulation** : Monte Carlo avec techniques de réduction de variance
- ** Interface** : Application Streamlit interactive et intuitive
- ** Performance** : Optimisations algorithmiques et calculs parallélisés

###  **Objectifs du Projet**

1. **Pédagogique** : Illustration des concepts de finance quantitative
2. **Pratique** : Outil de pricing professionnel avec interface moderne
3. **Recherche** : Plateforme d'analyse et de comparaison des méthodes
4. **Performance** : Implémentation optimisée pour des calculs rapides

---

##  **Fonctionnalités**

###  **Méthodes de Pricing**
- **Black-Scholes** : Solution analytique exacte
- **Arbre Trinomial** : Méthode numérique avec pruning
- **Monte Carlo** : Simulation avec antithetic variates

###  **Types d'Options**
- **Européennes** : Call et Put
- **Américaines** : Exercice anticipé optimal
- **Gestion des dividendes** : Ajustement automatique

###  **Visualisations Interactives**
- **Arbres de pricing** : Visualisation 3D interactive
- **Convergence** : Analyse de la précision vs nombre de pas
- **Surfaces de prix** : Visualisation 3D en fonction de S et T
- **Distributions** : Histogrammes des simulations Monte Carlo
- **Grecques** : Sensibilités et profils de risque

###  **Analyses de Performance**
- **Timing** : Comparaison des temps d'exécution
- **Convergence** : Études de convergence automatisées
- **Optimisations** : Échantillonnage intelligent adaptatif

###  **Exportation**
- **Excel** : Export complet des arbres et résultats
- **Graphiques** : PNG/PDF 
- **Données** : CSV pour analyses externes

---

##  **Installation**

###  **Prérequis**
- Python 3.8+
- Git


###  **Dépendances Principales**

```txt
streamlit>=1.28.0
numpy>=1.21.0
pandas>=1.3.0
plotly>=5.0.0
openpyxl>=3.0.9
scipy>=1.7.0
```

---

### Installation rapide

Installez les dépendances listées dans `requirements.txt` :

```bash
pip install -r requirements.txt
```


###  **Interface Utilisateur**

#### **Sidebar - Paramètres**
- ** Marché** : Spot, taux, volatilité
- ** Option** : Strike, maturité, type
- ** Dividendes** : Dates et montants
- ** Méthodes** : Configuration des algorithmes

#### **Main Panel - Résultats**
- ** Prix** : Affichage des prix calculés
- ** Graphiques** : Visualisations interactives
- ** Analyses** : Convergence et performance
- ** Export** : Téléchargement des résultats

---

##  **Architecture**

###  **Structure du Projet**

```
my_pricer_project1/
├──  main.py              # Point d'entrée principal
├──  app.py               # Interface Streamlit principale
├──  black_sholes.py      # Modèle Black-Scholes-Merton
├──  trinomial.py         # Arbre trinomial avec pruning
├──  monte_carlo.py       # Simulation Monte Carlo
├──  option.py            # Définition des options
├──  market.py            # Paramètres de marché
├──  parameters.py        # Paramètres de calcul
├──  greeks.py            # Calcul des sensibilités
├──  convergence.py       # Études de convergence
├──  timing.py            # Analyses de performance
├──  plot.py              # Graphiques et visualisations
├──  README.md            # Documentation du projet
├──  exports/             # Dossier pour fichiers Excel exportés
└──  __pycache__/         # Cache Python (généré automatiquement)
```

###  **Design Pattern**

- **Modularité** : Chaque méthode dans un module séparé
- **Extensibilité** : Interface commune pour toutes les méthodes
- **Performance** : Optimisations ciblées par méthode
- **Maintenabilité** : Code documenté et testé

---

##  **Méthodes de Pricing**

###  **Black-Scholes**

**Formule analytique exacte pour les options européennes**

```python
# Call européen
C = S₀ * N(d₁) - K * e^(-rT) * N(d₂)

# Put européen  
P = K * e^(-rT) * N(-d₂) - S₀ * N(-d₁)
```

**Avantages** :
- ✅ Solution exacte et instantanée
- ✅ Grecques analytiques
- ✅ Base théorique solide

**Limitations** :
- ❌ Uniquement options européennes
- ❌ Hypothèses restrictives

###  **Arbre Trinomial**

**Méthode numérique avec 3 branches par nœud**

```python
# Facteurs de montée/descente
α = exp(σ * √(3 * Δt))
up = S * α
down = S / α
mid = S * exp(r * Δt)  # Forward
```

**Avantages** :
- ✅ Options américaines
- ✅ Dividendes discrets
- ✅ Flexibilité maximale
- ✅ Pruning pour performance

**Caractéristiques** :
-  Convergence O(Δt²)
-  Optimisations algorithmiques
-  Précision contrôlable

###  **Monte Carlo**

**Simulation de trajectoires stochastiques**

```python
# Simulation GBM
dS = S * (r * dt + σ * √dt * Z)
S_T = S₀ * exp((r - σ²/2) * T + σ * √T * Z)
```
---

##  **Visualisations**

###  **Graphiques Disponibles**

1. ** Arbres de Pricing**
   - Visualisation interactive des nœuds
   - Navigation par niveau
   - Affichage des probabilités

2. ** Convergence**
   - Erreur vs nombre de pas
   - Comparaison des méthodes
   - Analyse de performance

3. ** Surfaces 3D**
   - Prix en fonction de (S, T)
   - Volatilité implicite
   - Grecques

4. ** Distributions**
   - Histogrammes Monte Carlo
   - Statistiques descriptives
   - Intervalles de confiance

5. ** Performance**
   - Temps d'exécution
   - Profils de complexité
   - Comparaisons algorithmiques
---


###  **Sensibilité (Grecques)**

- **Δ (Delta)** : Sensibilité au prix du sous-jacent
- **Γ (Gamma)** : Convexité
- **Θ (Theta)** : Décroissance temporelle
- **ν (Vega)** : Sensibilité à la volatilité
- **ρ (Rho)** : Sensibilité aux taux

---


###  **Validation**

1. **Convergence** : Vérification vs Black-Scholes
2. **Cohérence** : Put-Call parity
3. **Limites** : Comportements asymptotiques
4. **Performance** : Benchmarks de vitesse

###  **Métriques de Qualité**

- **Précision** : Erreur < 0.01% pour options européennes
- **Performance** : < 1s pour arbres 1000 pas
- **Couverture** : Tests > 90%
- **Documentation** : Docstrings complètes

---


###  **Améliorations Suggérées**

- [ ] Options asiatiques
- [ ] Modèles à volatilité stochastique
- [ ] Méthodes différences finies
- [ ] Interface API REST
- [ ] Optimisation GPU

---

##  **Références**

1. **Hull, J.** - *Options, Futures, and Other Derivatives*
2. **Wilmott, P.** - *Paul Wilmott Introduces Quantitative Finance*
3. **Glasserman, P.** - *Monte Carlo Methods in Financial Engineering*
4. **Shreve, S.** - *Stochastic Calculus for Finance*

