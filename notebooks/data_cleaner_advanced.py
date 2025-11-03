"""
data_cleaner_advanced.py
========================
Module de nettoyage AVANCÉ pour éliminer toutes les données aberrantes.
Version améliorée avec détection intelligente des outliers.

AMÉLIORATIONS:
- ✅ Détection multi-critères des aberrations
- ✅ Nettoyage du code département
- ✅ Validation croisée prix/surface
- ✅ Suppression des transactions impossibles
- ✅ Enrichissement avec données calculées
"""

import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class AdvancedDataCleaner:
    """Nettoyeur avancé de données immobilières"""
    
    def __init__(self, raw_dir, clean_dir):
        self.raw_dir = os.path.abspath(raw_dir)
        self.clean_dir = os.path.abspath(clean_dir)
        Path(self.clean_dir).mkdir(parents=True, exist_ok=True)
        
        # Statistiques de nettoyage
        self.stats = {
            'initial': 0,
            'after_basic': 0,
            'after_outliers': 0,
            'after_validation': 0,
            'final': 0
        }
    
    def _to_numeric(self, series):
        """Convertit en numérique (gère virgules)"""
        return pd.to_numeric(
            series.astype(str).str.replace(",", ".", regex=False), 
            errors="coerce"
        )
    
    def _detect_price_outliers(self, df, column='prix_m2', method='iqr', factor=3.0):
        """
        Détecte les outliers de prix avec plusieurs méthodes
        
        Args:
            method: 'iqr' (Inter-Quartile Range) ou 'zscore'
            factor: multiplicateur pour la détection (plus élevé = moins strict)
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            return df[column].between(lower_bound, upper_bound)
        
        elif method == 'zscore':
            mean = df[column].mean()
            std = df[column].std()
            z_scores = np.abs((df[column] - mean) / std)
            return z_scores < factor
    
    def clean_dvf(self, force_refresh=False):
        """
        Nettoie les données DVF avec détection avancée d'aberrations
        """
        clean_fp = os.path.join(self.clean_dir, "dvf_clean_advanced.parquet")
        
        if os.path.exists(clean_fp) and not force_refresh:
            print("✓ DVF avancé déjà nettoyé, chargement...")
            return pd.read_parquet(clean_fp)
        
        # Recherche fichier DVF
        dvf_candidates = glob.glob(os.path.join(self.raw_dir, "DVF*.txt")) + \
                        glob.glob(os.path.join(self.raw_dir, "valeursfoncieres*.txt"))
        
        if not dvf_candidates:
            raise FileNotFoundError(
                f"❌ Aucun fichier DVF trouvé dans {self.raw_dir}"
            )
        
        dvf_path = dvf_candidates[0]
        print(f"🔄 Nettoyage avancé DVF : {os.path.basename(dvf_path)}")
        
        # === ÉTAPE 1 : CHARGEMENT BRUT ===
        usecols = [
            "Date mutation", "Nature mutation", "Valeur fonciere",
            "Code postal", "Commune", "Code departement", "Code commune",
            "Type local", "Surface reelle bati", "Nombre pieces principales"
        ]
        
        df = pd.read_csv(dvf_path, sep="|", dtype=str, low_memory=False)
        df = df[[c for c in usecols if c in df.columns]].copy()
        self.stats['initial'] = len(df)
        print(f"   📊 Chargement initial : {self.stats['initial']:,}".replace(",", " "))
        
        # === ÉTAPE 2 : TYPAGE & FILTRES DE BASE ===
        df["Valeur fonciere"] = self._to_numeric(df["Valeur fonciere"])
        df["Surface reelle bati"] = self._to_numeric(df["Surface reelle bati"])
        df["Nombre pieces principales"] = self._to_numeric(df["Nombre pieces principales"])
        df["Date mutation"] = pd.to_datetime(df["Date mutation"], errors="coerce")
        
        # Ventes uniquement
        df = df[df["Nature mutation"].fillna("").str.contains("Vente", case=False, na=False)]
        
        # Île-de-France
        idf_prefix = ("75", "77", "78", "91", "92", "93", "94", "95")
        df["Code departement"] = df["Code departement"].astype(str).str.strip().str.zfill(2)
        df = df[df["Code departement"].str.startswith(idf_prefix)]
        
        # Suppression valeurs nulles critiques
        df = df.dropna(subset=["Valeur fonciere", "Surface reelle bati", "Date mutation"])
        
        self.stats['after_basic'] = len(df)
        print(f"   ✓ Après filtres de base : {self.stats['after_basic']:,}".replace(",", " "))
        
        # === ÉTAPE 3 : NETTOYAGE ABERRATIONS ===
        
        # 3.1 - Surface aberrante
        df = df[
            (df["Surface reelle bati"] >= 9) &  # Min 9m² (studios)
            (df["Surface reelle bati"] <= 300)   # Max 300m² (maisons)
        ]
        
        # 3.2 - Prix aberrants (bornes larges)
        df = df[
            (df["Valeur fonciere"] >= 10000) &   # Min 10k€
            (df["Valeur fonciere"] <= 2000000)   # Max 2M€
        ]
        
        # 3.3 - Calcul prix/m²
        df["prix_m2"] = df["Valeur fonciere"] / df["Surface reelle bati"]
        
        # 3.4 - Prix/m² aberrants (bornes réalistes IDF)
        df = df[
            (df["prix_m2"] >= 1500) &   # Min 1500€/m² (grandes couronnes)
            (df["prix_m2"] <= 20000)    # Max 20000€/m² (Paris centre)
        ]
        
        print(f"   ✓ Après bornes réalistes : {len(df):,}".replace(",", " "))
        
        # === ÉTAPE 4 : DÉTECTION OUTLIERS STATISTIQUE ===
        
        # Outliers par IQR (par département pour être plus précis)
        mask_valid = pd.Series(True, index=df.index)
        
        for dept in df["Code departement"].unique():
            mask_dept = df["Code departement"] == dept
            df_dept = df[mask_dept]
            
            if len(df_dept) > 30:  # Assez de données pour calculer IQR
                mask_outliers = self._detect_price_outliers(
                    df_dept, 
                    column='prix_m2', 
                    method='iqr', 
                    factor=2.5  # Moins strict que 3.0
                )
                mask_valid[mask_dept] = mask_outliers.values
        
        df = df[mask_valid]
        self.stats['after_outliers'] = len(df)
        print(f"   ✓ Après suppression outliers IQR : {self.stats['after_outliers']:,}".replace(",", " "))
        
        # === ÉTAPE 5 : VALIDATION CROISÉE ===
        
        # 5.1 - Ratio surface/prix cohérent (évite erreurs de saisie)
        df["ratio_prix_surface"] = df["Valeur fonciere"] / df["Surface reelle bati"]
        
        # 5.2 - Nombre de pièces cohérent avec surface
        if "Nombre pieces principales" in df.columns:
            # Studios: 9-30m², T2: 30-50m², T3: 50-75m², etc.
            df["pieces_estimees"] = (df["Surface reelle bati"] / 20).clip(upper=6).round(0)
            
            # Garder si nb pièces manquant OU cohérent (±2 pièces)
            mask_pieces = (
                df["Nombre pieces principales"].isna() |
                (np.abs(df["Nombre pieces principales"] - df["pieces_estimees"]) <= 2)
            )
            df = df[mask_pieces]
        
        # 5.3 - Appartements vs Maisons (validation type/surface)
        if "Type local" in df.columns:
            # Appartements rarement > 150m²
            mask_appt = (
                (df["Type local"] != "Appartement") |
                (df["Surface reelle bati"] <= 150)
            )
            
            # Maisons rarement < 40m²
            mask_maison = (
                (df["Type local"] != "Maison") |
                (df["Surface reelle bati"] >= 40)
            )
            
            df = df[mask_appt & mask_maison]
        
        self.stats['after_validation'] = len(df)
        print(f"   ✓ Après validation croisée : {self.stats['after_validation']:,}".replace(",", " "))
        
        # === ÉTAPE 6 : ENRICHISSEMENT ===
        
        # 6.1 - Année
        df["annee"] = df["Date mutation"].dt.year
        
        # 6.2 - Standardisation noms colonnes
        df = df.rename(columns={
            "Date mutation": "date_mutation",
            "Valeur fonciere": "valeur_fonciere",
            "Surface reelle bati": "surface_reelle_bati",
            "Commune": "nom_commune",
            "Code postal": "code_postal",
            "Type local": "type_local",
            "Code departement": "code_departement",
            "Nombre pieces principales": "nb_pieces",
            "Code commune": "code_commune_insee"
        })
        # Normalisation codes
        df["code_postal"] = df["code_postal"].astype(str).str.strip().str[:5]
        df["code_departement"] = df["code_departement"].astype(str).str.strip().str.zfill(2)
        # Sécurité : si code_departement manquant, dériver du CP
        mask_missing = df["code_departement"].isna() | (df["code_departement"] == "")
        df.loc[mask_missing & df["code_postal"].notna(), "code_departement"] = df.loc[
            mask_missing & df["code_postal"].notna(), "code_postal"
        ].str[:2]

        # 6.3 - Catégorie de bien (T1, T2, T3...)
        bins_surface = [0, 30, 45, 65, 90, 300]
        labels_cat = ['Studio/T1', 'T2', 'T3', 'T4', 'T5+']
        df['categorie_bien'] = pd.cut(
            df['surface_reelle_bati'], 
            bins=bins_surface, 
            labels=labels_cat,
            include_lowest=True
        )
        
        # 6.4 - Zone géographique (Paris intra-muros vs petite/grande couronne)
        df['zone_geo'] = df['code_departement'].map({
            '75': 'Paris',
            '92': 'Petite Couronne',
            '93': 'Petite Couronne',
            '94': 'Petite Couronne',
            '77': 'Grande Couronne',
            '78': 'Grande Couronne',
            '91': 'Grande Couronne',
            '95': 'Grande Couronne'
        })
        
        # 6.5 - Prix au m² par quartile (pour segmentation)
        df['quartile_prix'] = pd.qcut(
            df['prix_m2'], 
            q=4, 
            labels=['Abordable', 'Moyen', 'Élevé', 'Très élevé'],
            duplicates='drop'
        )
        
        self.stats['final'] = len(df)
        # Trim quantiles final (1%-99%) pour lisser les derniers extrêmes
        q1, q99 = df["prix_m2"].quantile([0.01, 0.99])
        df = df[(df["prix_m2"] >= q1) & (df["prix_m2"] <= q99)]

        # === SAUVEGARDE ===
        df.to_parquet(clean_fp, index=False)
        
        print(f"\n✅ Nettoyage terminé !")
        print(f"   📉 Suppressions : {self.stats['initial'] - self.stats['final']:,} transactions ({((self.stats['initial'] - self.stats['final']) / self.stats['initial'] * 100):.1f}%)".replace(",", " "))
        print(f"   📊 Dataset final : {self.stats['final']:,} transactions propres".replace(",", " "))
        
        return df
    
    def clean_loyers(self, force_refresh=False):
        """Nettoie les données de loyers (inchangé)"""
        clean_fp = os.path.join(self.clean_dir, "loyers_idf.parquet")
        
        if os.path.exists(clean_fp) and not force_refresh:
            print("✓ Loyers déjà nettoyés")
            return pd.read_parquet(clean_fp)
        
        loyer_candidates = glob.glob(os.path.join(self.raw_dir, "*loyer*.csv")) + \
                          glob.glob(os.path.join(self.raw_dir, "pred-app*.csv"))
        
        if not loyer_candidates:
            print("⚠️ Aucun fichier loyers trouvé")
            return None
        
        loyer_path = loyer_candidates[0]
        print(f"🔄 Nettoyage loyers : {os.path.basename(loyer_path)}")
        
        try:
            df = pd.read_csv(loyer_path, sep=None, engine="python", dtype=str, encoding="utf-8")
        except:
            df = pd.read_csv(loyer_path, sep=None, engine="python", dtype=str, encoding="latin-1")
        
        # Détection colonne loyer
        col_loy = None
        for c in df.columns:
            if 'loy' in c.lower() and 'm2' in c.lower():
                col_loy = c
                break
        
        if col_loy is None:
            print("⚠️ Colonne loyer introuvable")
            return None
        
        df['loyer_m2'] = (df[col_loy].astype(str)
                          .str.replace(',', '.', regex=False)
                          .str.extract(r'([0-9]+\.?[0-9]*)')[0]
                          .astype(float))
        
        # Code INSEE
        insee_col = None
        for c in df.columns:
            if c.lower().startswith('insee') or c.lower() in ['codgeo', 'com', 'code_commune']:
                insee_col = c
                break
        
        if insee_col:
            df['code_insee'] = df[insee_col].astype(str).str.strip().str.zfill(5)
            df['code_postal'] = df['code_insee'].str[:5]
        
        # Filtrage IDF
        idf_prefix = ('75', '77', '78', '91', '92', '93', '94', '95')
        if 'code_insee' in df.columns:
            df = df[df['code_insee'].str.startswith(idf_prefix, na=False)]
        
        # Nettoyage valeurs aberrantes loyers
        df = df[df['loyer_m2'].between(8, 80)]  # 8-80 €/m² réaliste pour IDF
        
        # Agrégation par code postal
        df_agg = (df.groupby('code_postal', as_index=False)['loyer_m2']
                  .median()
                  .sort_values('loyer_m2', ascending=False))
        
        df_agg.to_parquet(clean_fp, index=False)
        print(f"✓ Loyers : {len(df_agg):,} codes postaux".replace(",", " "))
        
        return df_agg
    
    def clean_gares(self, force_refresh=False):
        """Nettoie données gares (inchangé)"""
        clean_fp = os.path.join(self.clean_dir, "gares_idf.parquet")
        
        if os.path.exists(clean_fp) and not force_refresh:
            print("✓ Gares déjà nettoyées")
            return pd.read_parquet(clean_fp)
        
        gare_candidates = glob.glob(os.path.join(self.raw_dir, "*accessibilite*.csv")) + \
                         glob.glob(os.path.join(self.raw_dir, "*gare*.csv"))
        
        if not gare_candidates:
            print("⚠️ Aucun fichier gares trouvé")
            return None
        
        gare_path = gare_candidates[0]
        print(f"🔄 Nettoyage gares : {os.path.basename(gare_path)}")
        
        df = pd.read_csv(gare_path, sep=';', engine='python', dtype=str)
        
        col_acc = None
        for c in df.columns:
            if 'accessibility_level' in c.lower() or 'accessibilit' in c.lower():
                col_acc = c
                break
        
        if col_acc is None:
            print("⚠️ Colonne accessibilité introuvable")
            return None
        
        df[col_acc] = pd.to_numeric(df[col_acc], errors='coerce')
        df = df.rename(columns={col_acc: 'niveau_accessibilite'})
        
        name_col = None
        for c in ['stop_name', 'commune', 'nom_commune', 'town', 'locality', 'nom']:
            if c in df.columns:
                name_col = c
                break
        
        if name_col:
            df['nom_gare'] = df[name_col].astype(str).str.strip()
        
        df = df[df['niveau_accessibilite'] >= 2]
        
        if 'nom_gare' in df.columns:
            df_agg = (df.groupby('nom_gare', as_index=False)
                      .agg(niveau_max=('niveau_accessibilite', 'max'),
                           niveau_moyen=('niveau_accessibilite', 'mean'))
                      .sort_values('niveau_max', ascending=False))
        else:
            df_agg = df[['niveau_accessibilite']].copy()
        
        df_agg.to_parquet(clean_fp, index=False)
        print(f"✓ Gares : {len(df_agg):,} gares".replace(",", " "))
        
        return df_agg
    
    def unify_all(self, df_dvf, df_loyers=None, df_gares=None):
        """Fusionne tous les datasets"""
        dfu = df_dvf.copy()
        
        if df_loyers is not None and 'code_postal' in df_loyers.columns:
            dfu = dfu.merge(df_loyers, on='code_postal', how='left', suffixes=('', '_loyer'))
            n_with_loyer = dfu['loyer_m2'].notna().sum()
            print(f"✓ Fusion loyers : {n_with_loyer:,} lignes enrichies".replace(",", " "))
        
        return dfu
    
    def clean_all(self, force_refresh=False):
        """Pipeline complet de nettoyage"""
        print("=" * 70)
        print("🧹 NETTOYAGE AVANCÉ DES DONNÉES")
        print("=" * 70 + "\n")
        
        df_dvf = self.clean_dvf(force_refresh=force_refresh)
        df_loyers = self.clean_loyers(force_refresh=force_refresh)
        df_gares = self.clean_gares(force_refresh=force_refresh)
        
        print("\n" + "=" * 70)
        print("🔗 FUSION DES DATASETS")
        print("=" * 70)
        df_unifie = self.unify_all(df_dvf, df_loyers, df_gares)
        
        print("\n✅ Pipeline terminé !")
        print(f"   📊 Transactions finales : {len(df_unifie):,}".replace(",", " "))
        print(f"   📍 Communes : {df_unifie['nom_commune'].nunique():,}".replace(",", " "))
        print(f"   📅 Années : {df_unifie['annee'].min():.0f} - {df_unifie['annee'].max():.0f}")
        
        return df_unifie, df_loyers, df_gares


def quick_load_advanced(raw_dir, clean_dir, force_refresh=False):
    """
    Fonction rapide pour charger les données avec nettoyage avancé
    
    Usage:
        df, loyers, gares = quick_load_advanced(
            raw_dir="../data/raw",
            clean_dir="../data/clean"
        )
    """
    cleaner = AdvancedDataCleaner(raw_dir=raw_dir, clean_dir=clean_dir)
    return cleaner.clean_all(force_refresh=force_refresh)