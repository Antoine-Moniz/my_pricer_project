# main.py
import subprocess
import sys
import os

if __name__ == "__main__":
    # Chemin vers app.py (même dossier que main.py)
    app_path = os.path.join(os.path.dirname(__file__), "app.py")

    # Lancer streamlit avec le même interpréteur Python
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


#ce qu il reste a faire mettre sur github et faire un readme et la gestion des pacakages 📊 Échantillonnage intelligent: 61 calculs au lieu de 500
#⚡ Gain de temps estimé: 8.2x plus rapide


# pour aller a 2 millions de pas on construit que le tronc horizontal
#ensuite on construit les branches verticales  une a une 
# et on calcule les prix et proba avec les methodes qu on a deja si possible et on suprimme la branche vertival d avant
# a chaque fois on a que 2 branches verticales a stocker et l'horizontal

