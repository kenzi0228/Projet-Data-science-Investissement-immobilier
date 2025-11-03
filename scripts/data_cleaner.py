"""
data_cleaner.py
===============
Module de nettoyage automatisé pour les données DVF, loyers IDF et accessibilité gares.
Utilisé par le dashboard d'investissement locatif.
"""

import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class DataCleaner:
    """Classe principale pour nettoyer et préparer les données d'investissement immobilier."""
    
    def __init__(self, raw_dir="../data/raw", clean_dir="../data/clean"):
        """
        Initialise les chemins et crée les dossiers si nécessaire.
        
        Args:
            raw_dir: Dossier contenant les données brutes
            clean_dir: Dossier de sortie pour les données nettoyées
        """
        self.raw_dir = os.path.abspath(raw_dir)
        self.clean_dir = os.path.abspath(clean_dir)
        Path(self.clean_dir).mkdir(parents=True, exist_ok=True)
    
    # ==================== DVF (Demandes de Valeurs Foncières) ====================
    
    def _to_numeric(self, series):
        """Convertit une série en numérique (gère les virgules)."""
        return pd.to_numeric(
            series.astype(str).str.replace(",", ".", regex=False), 
            errors="coerce"
        )
    
    def clean_dvf(self, force_refresh=False):
        """
        Nettoie les données DVF (ventes immobilières Île-de-France).
        
        Args:
            force_refresh: Si True, recharge même si le fichier clean existe
            
        Returns:
            DataFrame nettoyé avec colonnes standardisées
        """
        clean_fp = os.path.join(self.clean_dir, "dvf_clean.parquet")
        
        # Si déjà nettoyé et pas de refresh forcé
        if os.path.exists(clean_fp) and not force_refresh:
            print("✓ DVF déjà nettoyé, chargement...")
            return pd.read_parquet(clean_fp)
        
        # Recherche du fichier DVF
        dvf_candidates = glob.glob(os.path.join(self.raw_dir, "DVF*.txt")) + \
                        glob.glob(os.path.join(self.raw_dir, "valeursfoncieres*.txt"))
        
        if not dvf_candidates:
            raise FileNotFoundError(
                f"Aucun fichier DVF trouvé dans {self.raw_dir}. "
                "Téléchargez-le depuis data.gouv.fr"
            )
        
        dvf_path = dvf_candidates[0]
        print(f"🔄 Nettoyage DVF depuis : {os.path.basename(dvf_path)}")
        
        # Colonnes utiles
        usecols = [
            "Date mutation", "Nature mutation", "Valeur fonciere",
            "Code postal", "Commune", "Code departement", "Code commune",
            "Type local", "Surface reelle bati", "Nombre pieces principales"
        ]
        
        # Chargement
        df = pd.read_csv(dvf_path, sep="|", dtype=str, low_memory=False)
        df = df[[c for c in usecols if c in df.columns]].copy()
        
        # Typage
        df["Valeur fonciere"] = self._to_numeric(df["Valeur fonciere"])
        df["Surface reelle bati"] = self._to_numeric(df["Surface reelle bati"])
        df["Nombre pieces principales"] = self._to_numeric(df["Nombre pieces principales"])
        df["Date mutation"] = pd.to_datetime(df["Date mutation"], errors="coerce")
        
        # Filtres : ventes uniquement
        df = df[df["Nature mutation"].fillna("").str.contains("Vente", case=False, na=False)]
        
        # Filtres : Île-de-France uniquement
        idf_prefix = ("75", "77", "78", "91", "92", "93", "94", "95")
        df = df[df["Code departement"].astype(str).str.startswith(idf_prefix)]
        
        # Calcul prix/m²
        df["prix_m2"] = df["Valeur fonciere"] / df["Surface reelle bati"]
        
        # Nettoyage aberrations
        df = df[
            (df["Surface reelle bati"] > 8) & 
            (df["Valeur fonciere"] > 1000) &
            (df["prix_m2"].between(100, 30000))
        ]
        
        # Année
        df["annee"] = df["Date mutation"].dt.year
        
        # Standardisation noms colonnes
        df = df.rename(columns={
            "Date mutation": "date_mutation",
            "Valeur fonciere": "valeur_fonciere",
            "Surface reelle bati": "surface_reelle_bati",
            "Commune": "nom_commune",
            "Code postal": "code_postal",
            "Type local": "type_local",
            "Code departement": "code_departement",
            "Nombre pieces principales": "nb_pieces"
        })
        
        # Sauvegarde
        df.to_parquet(clean_fp, index=False)
        print(f"✓ DVF nettoyé : {len(df):,} transactions IDF sauvegardées".replace(",", " "))
        
        return df
    
    # ==================== LOYERS IDF ====================
    
    def clean_loyers(self, force_refresh=False):
        """
        Nettoie les données de loyers prédits (ÎDF).
        
        Args:
            force_refresh: Si True, recharge même si le fichier clean existe
            
        Returns:
            DataFrame avec loyer/m² par commune/code postal
        """
        clean_fp = os.path.join(self.clean_dir, "loyers_idf.parquet")
        
        if os.path.exists(clean_fp) and not force_refresh:
            print("✓ Loyers déjà nettoyés, chargement...")
            return pd.read_parquet(clean_fp)
        
        # Recherche fichier
        loyer_candidates = glob.glob(os.path.join(self.raw_dir, "*loyer*.csv")) + \
                          glob.glob(os.path.join(self.raw_dir, "pred-app*.csv"))
        
        if not loyer_candidates:
            print("⚠️ Aucun fichier loyers trouvé, skip...")
            return None
        
        loyer_path = loyer_candidates[0]
        print(f"🔄 Nettoyage loyers depuis : {os.path.basename(loyer_path)}")
        
        # Chargement flexible (encoding)
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
            for c in df.columns:
                if 'loypred' in c.lower() or 'loy' in c.lower():
                    col_loy = c
                    break
        
        if col_loy is None:
            print("⚠️ Colonne loyer introuvable, skip...")
            return None
        
        # Extraction numérique loyer
        df['loyer_m2'] = (df[col_loy].astype(str)
                          .str.replace(',', '.', regex=False)
                          .str.extract(r'([0-9]+\.?[0-9]*)')[0]
                          .astype(float))
        
        # Détection code INSEE/commune
        insee_col = None
        for c in df.columns:
            if c.lower().startswith('insee') or c.lower() in ['codgeo', 'com', 'code_commune']:
                insee_col = c
                break
        
        if insee_col:
            df['code_insee'] = df[insee_col].astype(str).str.strip().str.zfill(5)
        
        # Filtrage ÎDF
        idf_prefix = ('75', '77', '78', '91', '92', '93', '94', '95')
        if 'code_insee' in df.columns:
            df = df[df['code_insee'].str.startswith(idf_prefix, na=False)].copy()
        
        # Détection nom commune
        comm_col = None
        for c in ['commune', 'nom_commune', 'NOM_COM', 'LIBGEO', 'LIBELLE_COMMUNE']:
            if c in df.columns:
                comm_col = c
                break
        
        if comm_col:
            df['nom_commune'] = df[comm_col].astype(str).str.strip().str.title()
        
        # Code postal depuis INSEE (5 premiers caractères)
        if 'code_insee' in df.columns:
            df['code_postal'] = df['code_insee'].str[:5]
        
        # Nettoyage valeurs aberrantes
        df = df[df['loyer_m2'].between(5, 100)].copy()
        
        # Agrégation par code postal (médiane)
        cols_agg = ['code_postal']
        if 'nom_commune' in df.columns:
            cols_agg.append('nom_commune')
        
        df_agg = (df.groupby(cols_agg, as_index=False)['loyer_m2']
                  .median()
                  .sort_values('loyer_m2', ascending=False))
        
        # Sauvegarde
        df_agg.to_parquet(clean_fp, index=False)
        print(f"✓ Loyers nettoyés : {len(df_agg):,} communes/CP IDF".replace(",", " "))
        
        return df_agg
    
    # ==================== ACCESSIBILITÉ GARES ====================
    
    def clean_gares(self, force_refresh=False):
        """
        Nettoie les données d'accessibilité en gare.
        
        Args:
            force_refresh: Si True, recharge même si le fichier clean existe
            
        Returns:
            DataFrame avec score accessibilité par gare/commune
        """
        clean_fp = os.path.join(self.clean_dir, "gares_idf.parquet")
        
        if os.path.exists(clean_fp) and not force_refresh:
            print("✓ Gares déjà nettoyées, chargement...")
            return pd.read_parquet(clean_fp)
        
        # Recherche fichier
        gare_candidates = glob.glob(os.path.join(self.raw_dir, "*accessibilite*.csv")) + \
                         glob.glob(os.path.join(self.raw_dir, "*gare*.csv"))
        
        if not gare_candidates:
            print("⚠️ Aucun fichier gares trouvé, skip...")
            return None
        
        gare_path = gare_candidates[0]
        print(f"🔄 Nettoyage gares depuis : {os.path.basename(gare_path)}")
        
        # Chargement
        df = pd.read_csv(gare_path, sep=';', engine='python', dtype=str)
        
        # Détection colonne accessibilité
        col_acc = None
        for c in df.columns:
            if 'accessibility_level' in c.lower() or 'accessibilit' in c.lower():
                col_acc = c
                break
        
        if col_acc is None:
            print("⚠️ Colonne accessibilité introuvable, skip...")
            return None
        
        df[col_acc] = pd.to_numeric(df[col_acc], errors='coerce')
        df = df.rename(columns={col_acc: 'niveau_accessibilite'})
        
        # Détection nom gare/commune
        name_col = None
        for c in ['stop_name', 'commune', 'nom_commune', 'town', 'locality', 'nom']:
            if c in df.columns:
                name_col = c
                break
        
        if name_col:
            df['nom_gare'] = df[name_col].astype(str).str.strip()
        
        # Simuler code postal depuis nom (simplification - idéalement géolocalisation)
        # Pour l'instant on garde juste les gares IDF identifiables
        
        # Filtrage niveau >= 2 (gares relativement accessibles)
        df = df[df['niveau_accessibilite'] >= 2].copy()
        
        # Agrégation simple par nom de gare
        if 'nom_gare' in df.columns:
            df_agg = (df.groupby('nom_gare', as_index=False)
                      .agg(niveau_max=('niveau_accessibilite', 'max'),
                           niveau_moyen=('niveau_accessibilite', 'mean'))
                      .sort_values('niveau_max', ascending=False))
        else:
            df_agg = df[['niveau_accessibilite']].copy()
        
        # Sauvegarde
        df_agg.to_parquet(clean_fp, index=False)
        print(f"✓ Gares nettoyées : {len(df_agg):,} gares accessibles IDF".replace(",", " "))
        
        return df_agg
    
    # ==================== FUSION COMPLÈTE ====================
    
    def unify_all(self, df_dvf, df_loyers=None, df_gares=None):
        """
        Fusionne DVF avec loyers et gares via code_postal/commune.
        
        Args:
            df_dvf: DataFrame DVF nettoyé
            df_loyers: DataFrame loyers (optionnel)
            df_gares: DataFrame gares (optionnel)
            
        Returns:
            DataFrame unifié avec toutes les colonnes disponibles
        """
        dfu = df_dvf.copy()
        
        # Fusion loyers
        if df_loyers is not None and 'code_postal' in df_loyers.columns:
            dfu = dfu.merge(df_loyers, on='code_postal', how='left', suffixes=('', '_loyer'))
            print(f"✓ Fusion loyers : {dfu['loyer_m2'].notna().sum():,} lignes enrichies".replace(",", " "))
        
        # Fusion gares (simplifiée - nécessite géolocalisation réelle)
        # Pour l'instant on skip cette partie car besoin de coordonnées GPS
        
        return dfu
    
    # ==================== FONCTION PRINCIPALE ====================
    
    def clean_all(self, force_refresh=False):
        """
        Nettoie toutes les sources de données et les fusionne.
        
        Args:
            force_refresh: Si True, force le recalcul même si fichiers propres existent
            
        Returns:
            tuple: (df_dvf_unifié, df_loyers, df_gares)
        """
        print("=" * 60)
        print("🧹 NETTOYAGE AUTOMATIQUE DES DONNÉES")
        print("=" * 60)
        
        # 1) DVF
        df_dvf = self.clean_dvf(force_refresh=force_refresh)
        
        # 2) Loyers
        df_loyers = self.clean_loyers(force_refresh=force_refresh)
        
        # 3) Gares
        df_gares = self.clean_gares(force_refresh=force_refresh)
        
        # 4) Fusion
        print("\n" + "=" * 60)
        print("🔗 FUSION DES DATASETS")
        print("=" * 60)
        df_unifie = self.unify_all(df_dvf, df_loyers, df_gares)
        
        print("\n✅ Nettoyage terminé !")
        print(f"   Dataset final : {len(df_unifie):,} transactions".replace(",", " "))
        print(f"   Colonnes : {', '.join(df_unifie.columns.tolist()[:8])}...")
        
        return df_unifie, df_loyers, df_gares


# ==================== FONCTION RAPIDE ====================

def quick_load(raw_dir="../data/raw", clean_dir="../data/clean", force_refresh=False):
    """
    Fonction rapide pour charger toutes les données nettoyées en une ligne.
    
    Usage:
        df_unifie, df_loyers, df_gares = quick_load()
    
    Args:
        raw_dir: Dossier données brutes
        clean_dir: Dossier données propres
        force_refresh: Forcer le recalcul
        
    Returns:
        tuple: (DataFrame unifié, DataFrame loyers, DataFrame gares)
    """
    cleaner = DataCleaner(raw_dir=raw_dir, clean_dir=clean_dir)
    return cleaner.clean_all(force_refresh=force_refresh)