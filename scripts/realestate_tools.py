
import os, re, pandas as pd, numpy as np, unicodedata

def _strip_accents(s):
    if not isinstance(s, str): return s
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def _norm_str_ser(ser):
    s = ser.astype(str).str.strip()
    s = s.apply(_strip_accents)
    s = s.str.replace(r'\s+', ' ', regex=True)
    return s

def _coerce_num_fr(ser):
    s = ser.astype(str).str.replace('\u00A0','', regex=False)  # nbsp
    s = s.str.replace(' ', '', regex=False).str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce')

def _zfill_safe(ser, width):
    s = ser.astype(str).str.strip()
    mask_alnum = s.str.contains('[A-Za-z]', na=False)
    s = np.where(mask_alnum, s, s.str.zfill(width))
    return pd.Series(s, index=ser.index, dtype='object')

def _pick_col(df, candidates, required=False):
    for c in candidates:
        if c in df.columns: return c
        for col in df.columns:
            base = _strip_accents(str(col)).lower()
            if base == _strip_accents(str(c)).lower():
                return col
    if required:
        raise KeyError(f"Colonne manquante parmi {candidates}")
    return None

def harmonise_dvf(df):
    df = df.copy()
    c_commune  = _pick_col(df, ['nom_commune','Commune','commune'], required=False) or 'nom_commune'
    c_cp       = _pick_col(df, ['code_postal','cp','Code postal'], required=False) or 'code_postal'
    c_insee    = _pick_col(df, ['code_commune_insee','code_insee','insee'], required=False) or 'code_commune_insee'
    c_annee    = _pick_col(df, ['annee','year'], required=False) or 'annee'
    c_surf     = _pick_col(df, ['surface_reelle_bati','surface','surface_bati'], required=False) or 'surface_reelle_bati'
    c_val      = _pick_col(df, ['valeur_fonciere','valeur','prix_total'], required=False) or 'valeur_fonciere'
    c_p2       = _pick_col(df, ['prix_m2','prix_m^2','price_m2'], required=False) or 'prix_m2'

    for col in [c_commune, c_cp, c_insee, c_annee, c_surf, c_val, c_p2]:
        if col not in df.columns:
            df[col] = np.nan

    df = df.rename(columns={
        c_commune: 'nom_commune',
        c_cp: 'code_postal',
        c_insee: 'code_commune_insee',
        c_annee: 'annee',
        c_surf: 'surface_reelle_bati',
        c_val: 'valeur_fonciere',
        c_p2: 'prix_m2'
    })

    df['nom_commune'] = _norm_str_ser(df['nom_commune'])
    df['code_postal'] = _zfill_safe(df['code_postal'], 5)
    df['code_commune_insee'] = _zfill_safe(df['code_commune_insee'], 5)

    if not pd.api.types.is_numeric_dtype(df['prix_m2']):
        df['prix_m2'] = _coerce_num_fr(df['prix_m2'])
    if not pd.api.types.is_numeric_dtype(df['surface_reelle_bati']):
        df['surface_reelle_bati'] = _coerce_num_fr(df['surface_reelle_bati'])
    if not pd.api.types.is_numeric_dtype(df['valeur_fonciere']):
        df['valeur_fonciere'] = _coerce_num_fr(df['valeur_fonciere'])

    df = df.dropna(subset=['prix_m2','surface_reelle_bati','valeur_fonciere','nom_commune'])
    df = df[(df['surface_reelle_bati'] > 8) & (df['valeur_fonciere'] > 1000) & (df['prix_m2'].between(100, 30000))]

    return df

def harmonise_loyers(loyers_df):
    if loyers_df is None: return None
    df = loyers_df.copy()
    c_cp   = _pick_col(df, ['code_postal','cp','Code postal'], required=True)
    c_lm2  = _pick_col(df, ['loyer_m2','loyer_metre_carre','rent_m2','lm2'], required=False)
    if c_lm2 is None:
        cands = [c for c in df.columns if 'loyer' in _strip_accents(c).lower() and 'm2' in _strip_accents(c).lower()]
        c_lm2 = cands[0] if cands else None
    if c_lm2 is None:
        raise KeyError("Impossible de trouver la colonne loyer_m2 (ou équivalent) dans loyers_df.")

    df = df.rename(columns={c_cp:'code_postal', c_lm2:'loyer_m2'})
    df['code_postal'] = _zfill_safe(df['code_postal'], 5)
    if not pd.api.types.is_numeric_dtype(df['loyer_m2']):
        df['loyer_m2'] = _coerce_num_fr(df['loyer_m2'])
    df = df.dropna(subset=['code_postal','loyer_m2'])
    return df[['code_postal','loyer_m2']].copy()

def harmonise_gares(gares_df):
    if gares_df is None: return None
    df = gares_df.copy()
    c_cp  = _pick_col(df, ['code_postal','cp','Code postal'], required=True)
    c_tg  = _pick_col(df, ['temps_gare_min','temps_gare','time_gare_min','min_gare'], required=False)
    c_nb  = _pick_col(df, ['nb_gares_15min','gares_15min','count_gares_15m'], required=False)

    rename_map = {c_cp:'code_postal'}
    if c_tg: rename_map[c_tg] = 'temps_gare_min'
    if c_nb: rename_map[c_nb] = 'nb_gares_15min'
    df = df.rename(columns=rename_map)

    df['code_postal'] = _zfill_safe(df['code_postal'], 5)
    if 'temps_gare_min' in df and not pd.api.types.is_numeric_dtype(df['temps_gare_min']):
        df['temps_gare_min'] = _coerce_num_fr(df['temps_gare_min'])
    if 'nb_gares_15min' in df and not pd.api.types.is_numeric_dtype(df['nb_gares_15min']):
        df['nb_gares_15min'] = _coerce_num_fr(df['nb_gares_15min'])
    return df

def attach_loyers(dvf_df, loyers_df):
    if loyers_df is None: return dvf_df.copy()
    j = dvf_df.merge(loyers_df, on='code_postal', how='left')
    return j

def attach_gares(dvf_df, gares_df):
    if gares_df is None: return dvf_df.copy()
    j = dvf_df.merge(gares_df, on='code_postal', how='left')
    return j

def unify_all(dvf_df, loyers_df=None, gares_df=None):
    base = harmonise_dvf(dvf_df)
    L = harmonise_loyers(loyers_df) if loyers_df is not None else None
    G = harmonise_gares(gares_df)   if gares_df is not None else None
    out = attach_loyers(base, L) if L is not None else base
    out = attach_gares(out, G)   if G is not None else out
    return out
