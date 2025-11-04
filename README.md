# Projet Data Science — Investissement immobilier en Île-de-France

## 1. Objectif
Développer un tableau de bord interactif d’aide à la décision pour l’investissement locatif en Île-de-France, à partir des données DVF et de sources complémentaires (loyers, référentiels géographiques). L’application permet de filtrer, analyser et comparer prix au m², surfaces et rendements par commune, département et zone.

---

## 2. Contenu du dépôt

```
Projet-Data-science-Investissement-immobilier/
├─ data/
│  ├─ raw/      # Données brutes (DVF, loyers, référentiels)
│  └─ clean/    # Données nettoyées et enrichies
├─ notebooks/
│     ├─ data_cleaner_advanced.py       # Pipeline de nettoyage
│     ├─ dashboard_app.ipynb           # Application  (GUI)
├─ requirements.txt               # Dépendances
└─ README.md
```

---

## 3. Persona (configuration par défaut)

```python
PERSONA = {
    'nom': 'Manager IT',
    'age': 40,
    'salaire_annuel': 70000,
    'apport': 50000,
    'capacite_emprunt': 150000,
    'budget_max': 200000,
    'surface_min': 15,
    'surface_max': 65,
    'cible_locataire': 'Étudiants / Jeunes actifs',
    'objectif_rendement_net': 4.5,
    'risque': 'Faible',
    'temps_disponible': 'Limité'
}
```

Ce persona pilote les valeurs initiales de plusieurs filtres (surface, budget, objectif de rendement) et sert de base aux recommandations.

---
## 4. Données utilisées (faire attention juste a les renommé comme ci dessous)

| Source | Description | Emplacement | Format | Lien |
|---|---|---|---|---|
| Agglo_2024 (Aires d’attraction des villes - INSEE) | Classification des communes par agglomération / aire urbaine | `data/raw/` | XLSX | [Seule donnée qui sera incluse car le lien utilisé pour la récuperer ne fonctionne plus]() |
| base_code_postaux | Base officielle des codes postaux et communes | `data/raw/` | CSV | [https://www.data.gouv.fr/api/1/datasets/r/008a2dda-2c60-4b63-b910-998f6f818089]() |
| pred-app12-mef-dhup_2024 | Données de loyers observés (DHUP / MEF) | `data/raw/` | CSV | [https://static.data.gouv.fr/resources/carte-des-loyers-indicateurs-de-loyers-dannonce-par-commune-en-2024/20241205-153048/pred-app12-mef-dhup.csv ]() |
| DVF_2025 | Transactions immobilières 2025 (DGFiP / DVF+) | `data/raw/` | CSV / Parquet | [https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20251018-234902/valeursfoncieres-2025-s1.txt.zip]() |

---

### Script de nettoyage : `data_cleaner_advanced.py`

Ce script est appelé lors de l'éxecution du dashboard, il:
- harmonise les noms de colonnes et les types de données ;
- calcule `prix_m2 = valeur_fonciere / surface_reelle_bati` ;
- calcule le rendement :
  - `rendement_brut = (loyer_annuel / valeur_fonciere) * 100`
  - `rendement_net = (loyer_annuel * (1 - charges/100) / valeur_fonciere) * 100`
- élimine les valeurs aberrantes (IQR et/ou quantiles 1%–99%) ;
- écrit une table consolidée et propre dans `data/clean/`.

---


## 5. Installation

### 5.1. Environnement

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 5.2. Dépendances principales
- pandas, numpy
- matplotlib
- mplcursors (info-bulles au survol)
- tkinter (GUI)
- ipywidgets (version notebook)

---

## 6. Lancer l’application

Ouvrir `dashboard_app.ipynb` dans Jupyter et exécuter toutes les cellules.  


---

## 7. Widgets et fonctionnalités (détails)

L’interface est organisée en six onglets. Tous les graphiques intègrent des info-bulles au survol grâce à `mplcursors`.

### 7.1. Vue d’ensemble
- Indicateurs (cards) : nombre de transactions, prix/m² médian, rendement net médian, surface médiane.
- Histogramme prix/m² : distribution avec ligne de médiane, clipping 1–99 % pour lisser les extrêmes.
- Filtres synchronisés : appliqués depuis la barre latérale (surface, budget, prix/m², zone, département, années, élimination d’outliers, rendement minimum).

**Fonctionnement interne**  
- Les indicateurs sont calculés sur le DataFrame filtré.  
- L’histogramme est calculé avec 40 classes ; une ligne verticale marque la médiane.  
- Les info-bulles d’histogramme affichent la borne du bin et le compte.

### 7.2. Top Communes
- Objectif : identifier les meilleures communes selon le rendement net médian.
- Contrôle Top N : Spinbox intégré à l’onglet (valeur par défaut 15, modifiable de 5 à 100).
- Tableau : rang, commune, code postal, nombre de ventes, surface médiane, prix médian, rendement net médian.
- Défilement : scrollbar verticale.

**Fonctionnement interne**  
- Groupby sur `['nom_commune', 'code_postal']` avec agrégations : `count`, `median` (prix_m2, surface, valeur_fonciere, rendement_net).  
- Tri décroissant sur `rdt_net`, puis `head(Top N)`.  
- Rendu via `ttk.Treeview` avec scrollbar.

### 7.3. Analyse Prix
- Nuage de points : surface vs prix/m², coloré par rendement net lorsqu’il est disponible.
- Histogramme prix/m² : seconde visualisation latérale avec médiane et info-bulles par bin.
- Info-bulles du scatter : commune (CP), surface, prix/m², rendement net si disponible.

**Fonctionnement interne**  
- Échantillonnage jusqu’à 3 000 points si dataset volumineux.  
- Colorbar activée si `rendement_net` présent.  
- Info-bulles basées sur l’index du point survolé.

### 7.4. Rendement
- Histogramme rendements nets : ligne de médiane et rappel de l’objectif de rendement du persona.
- Rendement par zone : barres horizontales Paris / Petite Couronne / Grande Couronne.
- Info-bulles : détail par bin (histogramme) et par barre (zones).

**Fonctionnement interne**  
- Capping visuel des rendements à 12 % pour stabiliser l’échelle.  
- Groupby sur `zone_geo` pour le graphe horizontal ; ordre décroissant.

### 7.5. Carte
- Agrégations par département : prix/m² médian, rendement net médian (ou nombre de transactions si rendement manquant).
- Deux barres horizontales : prix/m² médian et rendement net médian par département (top 8).
- Info-bulles : valeur exacte au survol.

**Fonctionnement interne**  
- Création d’un `code_departement` via les deux premiers chiffres du code postal si manquant.  
- Tri sur rendement net médian si disponible, sinon sur le volume.  
- Visualisation en barres horizontales pour lisibilité.

### 7.6. Recommandations Persona
- Récapitulatif profil : budget, surface cible, objectif de rendement, profil de risque.
- Statistiques de la sélection : médianes de surface, prix/m², prix total ; part des biens dans le budget ; part atteignant l’objectif de rendement.
- Zone défilante : section scrollable pour de longues descriptions.

**Fonctionnement interne**  
- Calcul du pourcentage dans le budget et de la part atteignant l’objectif.  
- Rendu au format texte avec style “card”.

---

## 8. Barre latérale (filtres)

- Presets : boutons rapides (Mon profil, Budget Max, Petit Budget, Haut Rendement).
- Surface (m²) : min/max.
- Budget total (k€) : min/max (apport et budget max du persona comme bornes par défaut).
- Prix/m² (€) : min/max.
- Localisation : zone (Paris, Petite Couronne, Grande Couronne) et département.
- Rentabilité : loyer €/m², charges en %, rendement net minimum.
- Options : période (années), suppression des outliers (IQR × 2).
- Actions : Appliquer, Reset, Exporter (CSV des données filtrées).

**Confort d’usage**  
- Scrollbar fiable et support de la molette (Windows/Linux/macOS).

---

## 9. Calculs clés

- Loyer annuel = `loyer_m2` (slider) × `surface_reelle_bati` × 12  
- Loyer net = `loyer annuel × (1 − charges %)`  
- Rendement brut = `(loyer annuel / valeur_fonciere) × 100`  
- Rendement net = `(loyer net / valeur_fonciere) × 100`  
- Nettoyage des extrêmes : IQR et clipping quantiles pour `prix_m2`.

---

## 10. Bonnes pratiques et performances

- Agrégations effectuées après filtrage pour éviter les surcharges mémoire.
- Échantillonnage sur certains nuages de points (> 3 000) pour préserver la fluidité.
- Limitation à 40 classes pour les histogrammes.
- Clipping 1–99 % pour atténuer l’impact des outliers sur les axes.


---

## 11. Dépannage

- Erreur « No group keys passed » : se produit si un groupby est appelé sur un DataFrame vide. Vérifier que les filtres ne vident pas complètement la sélection.  
- Message « données non disponibles » dans un onglet : certaines colonnes requises (ex. `rendement_net`) peuvent manquer selon les filtres ou les sources. Relancer un nettoyage complet via `data_cleaner_advanced.py`.  
- Si `tkinter` manque : installer le paquet système adéquat (ex. `sudo apt-get install python3-tk` sur Debian/Ubuntu).

---

## 12. Git — commandes utiles

Initialisation locale et push :

```bash
git init
git remote add origin https://github.com/kenzi0228/Projet-Data-science-Investissement-immobilier.git
git add .
git commit -m "Ajout du dashboard et du notebook v2"
git branch -M main
git push -u origin main
```

---

## 14. Licence et auteur

- Licence : MIT (si applicable).  
- Auteurs : Mohamed Kenzi Lali, Aina Ralijaona, Victor Lin (ECE Paris, ING4). Projet académique Data Science.
