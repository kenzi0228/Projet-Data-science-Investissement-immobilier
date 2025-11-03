
"""
Dashboard Immobilier — Application Tkinter complète (version unifiée et robuste)

Fonctionnalités clés :
- Barre latérale de filtres (Années, Type de bien, Commune, bornes Surface/Budget/Prix/m²)
- Calcul du rendement brut/net avec loyer au m² et % charges paramétrables
- Nettoyage/trim des valeurs aberrantes (quantiles 1%–99%)
- 4 vues :
  1) Aperçu (overview) : histogramme prix/m² + scatter Surface vs Valeur foncière
  2) Prix : médiane prix/m² par groupe (zone, département, type) + histogramme
  3) Top communes : top N communes par rendement net + table
  4) Table : aperçu tabulaire filtré

Pré-requis :
- pandas, numpy, matplotlib, tkinter

NB : Cette application se veut un remplacement "drop-in" si vous ne pouvez pas patcher
votre ancien fichier. Elle conserve l'esprit et les fonctionnalités que vous avez décrites.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------------------------------------------------------------
# Chargement de données (adaptez aux chemins de votre projet)
# -------------------------------------------------------------

def _find_raw():
    for d in [Path("./data"), Path("../data"), Path("/mnt/data"), Path(".")]:
        if d.exists():
            for p in d.rglob("*"):
                n = p.name.lower()
                if p.is_file() and (("dvf" in n or "valeurs_foncieres" in n) and p.suffix.lower() in [".csv",".txt"]):
                    return p
    return None

def _guess_sep(fp: Path):
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        h=f.readline()
    return ";" if h.count(";")>=h.count(",") else ","

def load_dataset():
    clean = Path("./data_clean/dvf_clean.parquet")
    if clean.exists():
        return pd.read_parquet(clean)
    alt = Path("/mnt/data/dvf_clean.parquet")
    if alt.exists():
        return pd.read_parquet(alt)

    raw = _find_raw()
    if raw is None:
        return pd.DataFrame()
    sep = _guess_sep(raw)
    dtype_hint = {
        "valeur_fonciere":"float64","surface_reelle_bati":"float64","nombre_pieces_principales":"float64",
        "code_postal":"string","code_commune":"string","nom_commune":"string","type_local":"string","annee":"Int64"
    }
    df = pd.read_csv(raw, sep=sep, dtype=dtype_hint, low_memory=False, encoding="utf-8", na_values=["","NA","NaN"])
    if "valeur_fonciere" in df and "surface_reelle_bati" in df:
        vf = pd.to_numeric(df["valeur_fonciere"], errors="coerce")
        sr = pd.to_numeric(df["surface_reelle_bati"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            df["prix_m2"] = vf / sr
    # Normalisation département / zone
    if "code_departement" not in df.columns and "code_postal" in df.columns:
        df["code_departement"] = df["code_postal"].astype(str).str[:2]
    if "zone_geo" not in df.columns and "code_departement" in df.columns:
        map_zone = {
            "75": "Paris",
            "92": "Petite Couronne", "93": "Petite Couronne", "94": "Petite Couronne",
            "77": "Grande Couronne", "78": "Grande Couronne", "91": "Grande Couronne", "95": "Grande Couronne",
        }
        df["zone_geo"] = df["code_departement"].map(map_zone)
    return df


# -------------------------------------------------------------
# Application
# -------------------------------------------------------------

class DashboardApp(tk.Tk):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.title("Dashboard Immobilier — App Unifiée")
        self.geometry("1360x780")
        self.configure(bg="#0f1115")

        self.colors = {
            "bg": "#0f1115",
            "fg": "#e6e6e6",
            "muted": "#a7a7a7",
            "card": "#171a21",
            "accent": "#4aa3ff",
            "secondary": "#2b2f3a",
            "danger": "#ff6b6b",
            "success": "#27ae60",
        }

        self.df_unifie = df.copy()
        self.current_filtered_data = pd.DataFrame()

        self._build_layout()
        self._init_filters()
        self._apply_filters()

    # ------------------ Utils ------------------
    @staticmethod
    def _clip_q(series, qlow=0.01, qhigh=0.99):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return s
        lo, hi = s.quantile([qlow, qhigh])
        return s.clip(lower=lo, upper=hi)

    def _compute_yields(self, d: pd.DataFrame) -> pd.DataFrame:
        if d is None or d.empty:
            return d
        d = d.copy()
        for col in ["valeur_fonciere", "surface_reelle_bati", "prix_m2"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors='coerce')

        loyer_m2 = float(self.w_loyer.get())
        charges_pct = float(self.w_charges.get()) / 100.0

        loyer_annuel = loyer_m2 * d["surface_reelle_bati"] * 12.0
        loyer_net = loyer_annuel * (1.0 - charges_pct)

        with np.errstate(divide="ignore", invalid="ignore"):
            d["rendement_brut"] = (loyer_annuel / d["valeur_fonciere"]) * 100.0
            d["rendement_net"]  = (loyer_net / d["valeur_fonciere"]) * 100.0

        d.replace([np.inf, -np.inf], np.nan, inplace=True)
        return d

    # ------------------ UI ------------------
    def _build_layout(self):
        # Sidebar + main
        self.sidebar = tk.Frame(self, bg=self.colors["card"], width=320)
        self.sidebar.pack(side="left", fill="y")

        self.main = tk.Frame(self, bg=self.colors["bg"])
        self.main.pack(side="right", fill="both", expand=True)

        # Tabs
        self.nb = ttk.Notebook(self.main)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background=self.colors["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=self.colors["secondary"], foreground=self.colors["fg"])
        style.map("TNotebook.Tab", background=[("selected", self.colors["card"])])

        self.tab_overview = tk.Frame(self.nb, bg=self.colors["bg"])
        self.tab_prix = tk.Frame(self.nb, bg=self.colors["bg"])
        self.tab_top = tk.Frame(self.nb, bg=self.colors["bg"])
        self.tab_table = tk.Frame(self.nb, bg=self.colors["bg"])

        self.nb.add(self.tab_overview, text="Aperçu")
        self.nb.add(self.tab_prix, text="Prix")
        self.nb.add(self.tab_top, text="Top communes")
        self.nb.add(self.tab_table, text="Table")

        self.nb.pack(fill="both", expand=True, padx=10, pady=10)

    def _add_slider(self, parent, label, from_, to, default):
        fr = tk.Frame(parent, bg=self.colors["card"])
        fr.pack(fill="x", padx=15, pady=6)
        tk.Label(fr, text=label, bg=self.colors["card"], fg=self.colors["fg"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        var = tk.DoubleVar(value=default)
        sc = tk.Scale(fr, from_=from_, to=to, orient="horizontal",
                      bg=self.colors["card"], fg=self.colors["fg"],
                      highlightthickness=0, troughcolor=self.colors["secondary"],
                      resolution=0.1, variable=var, length=240)
        sc.pack(anchor="w")
        return var

    def _add_range_slider(self, parent, label, min_v, max_v, default):
        frame = tk.Frame(parent, bg=self.colors['card'])
        frame.pack(fill='x', padx=15, pady=6)

        tk.Label(frame, text=label, bg=self.colors['card'], fg=self.colors['fg'],
                 font=('Segoe UI', 9, 'bold')).pack(anchor='w')

        var_min = tk.IntVar(value=default[0])
        var_max = tk.IntVar(value=default[1])

        row = tk.Frame(frame, bg=self.colors['card'])
        row.pack(fill='x')

        tk.Label(row, text="Min", bg=self.colors['card'], fg=self.colors['fg']).pack(side='left', padx=(0,6))
        s_min = tk.Scale(row, from_=min_v, to=max_v, orient='horizontal',
                         variable=var_min, bg=self.colors['card'], fg=self.colors['fg'],
                         highlightthickness=0, troughcolor=self.colors['secondary'],
                         length=110, command=lambda _=None: var_min.set(min(var_min.get(), var_max.get())))
        s_min.pack(side='left')

        tk.Label(row, text="  ", bg=self.colors['card']).pack(side='left')

        tk.Label(row, text="Max", bg=self.colors['card'], fg=self.colors['fg']).pack(side='left', padx=(6,6))
        s_max = tk.Scale(row, from_=min_v, to=max_v, orient='horizontal',
                         variable=var_max, bg=self.colors['card'], fg=self.colors['fg'],
                         highlightthickness=0, troughcolor=self.colors['secondary'],
                         length=110, command=lambda _=None: var_max.set(max(var_min.get(), var_max.get())))
        s_max.pack(side='left')

        return {'min': var_min, 'max': var_max}

    def _init_filters(self):
        tk.Label(self.sidebar, text="Filtres", bg=self.colors["card"], fg=self.colors["fg"],
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=15, pady=(10,6))

        df = self.df_unifie
        years = sorted([int(x) for x in df["annee"].dropna().unique().tolist()]) if "annee" in df.columns else []
        yr_min, yr_max = (years[0], years[-1]) if years else (2000, 2025)

        # Années
        self.range_year = self._add_range_slider(self.sidebar, "Années", yr_min, yr_max, (yr_min, yr_max))

        # Type local
        types = sorted(df["type_local"].dropna().unique().tolist()) if "type_local" in df.columns else []
        fr_type = tk.Frame(self.sidebar, bg=self.colors["card"]); fr_type.pack(fill="x", padx=15, pady=6)
        tk.Label(fr_type, text="Type de bien", bg=self.colors["card"], fg=self.colors["fg"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.cb_type = ttk.Combobox(fr_type, values=["(Tous)"] + types, state="readonly")
        self.cb_type.current(0)
        self.cb_type.pack(fill="x")

        # Commune
        communes = sorted(df["nom_commune"].dropna().unique().tolist()) if "nom_commune" in df.columns else []
        fr_com = tk.Frame(self.sidebar, bg=self.colors["card"]); fr_com.pack(fill="x", padx=15, pady=6)
        tk.Label(fr_com, text="Commune", bg=self.colors["card"], fg=self.colors["fg"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.cb_commune = ttk.Combobox(fr_com, values=["(Aucune)"] + communes, state="readonly")
        self.cb_commune.current(0)
        self.cb_commune.pack(fill="x")

        # Loyer & Charges
        self.w_loyer = self._add_slider(self.sidebar, "Loyer €/m²/mois", 5, 40, 22.0)
        self.w_charges = self._add_slider(self.sidebar, "Charges %", 0, 40, 25.0)

        # Bornes
        self.bound_surface = self._add_range_slider(self.sidebar, "Surface (m²)", 5, 300, (10, 120))
        self.bound_budget  = self._add_range_slider(self.sidebar, "Budget (€)", 20000, 1500000, (80000, 500000))
        self.bound_px2     = self._add_range_slider(self.sidebar, "Prix/m² (€)", 300, 20000, (1000, 12000))

        # Boutons
        fr_btn = tk.Frame(self.sidebar, bg=self.colors["card"]); fr_btn.pack(fill="x", padx=15, pady=10)
        tk.Button(fr_btn, text="Appliquer", command=self._apply_filters).pack(side="left", padx=4)
        tk.Button(fr_btn, text="Réinitialiser", command=self._reset_filters).pack(side="left", padx=4)

    # ------------------ Filtrage ------------------
    def _reset_filters(self):
        df = self.df_unifie
        years = sorted([int(x) for x in df["annee"].dropna().unique().tolist()]) if "annee" in df.columns else []
        yr_min, yr_max = (years[0], years[-1]) if years else (2000, 2025)
        self.range_year["min"].set(yr_min); self.range_year["max"].set(yr_max)
        self.cb_type.current(0)
        self.cb_commune.current(0)
        self.w_loyer.set(22.0)
        self.w_charges.set(25.0)
        self.bound_surface["min"].set(10); self.bound_surface["max"].set(120)
        self.bound_budget["min"].set(80000); self.bound_budget["max"].set(500000)
        self.bound_px2["min"].set(1000); self.bound_px2["max"].set(12000)
        self._apply_filters()

    def _apply_filters(self):
        d = self.df_unifie.copy()

        # Fallback géo
        if "code_departement" not in d.columns and "code_postal" in d.columns:
            d["code_departement"] = d["code_postal"].astype(str).str[:2]
        if "zone_geo" not in d.columns and "code_departement" in d.columns:
            _map = {"75":"Paris","92":"Petite Couronne","93":"Petite Couronne","94":"Petite Couronne",
                    "77":"Grande Couronne","78":"Grande Couronne","91":"Grande Couronne","95":"Grande Couronne"}
            d["zone_geo"] = d["code_departement"].map(_map)

        # Filtres
        if "annee" in d.columns:
            y0, y1 = self.range_year["min"].get(), self.range_year["max"].get()
            d = d[d["annee"].fillna(-1).astype("Int64").between(y0, y1)]

        sel_type = self.cb_type.get()
        if sel_type and sel_type != "(Tous)" and "type_local" in d.columns:
            d = d[d["type_local"] == sel_type]

        sel_com = self.cb_commune.get()
        if sel_com and sel_com != "(Aucune)" and "nom_commune" in d.columns:
            d = d[d["nom_commune"].fillna("").str.lower() == sel_com.lower()]

        # Surface / Budget / Prix/m²
        if "surface_reelle_bati" in d.columns:
            s0, s1 = self.bound_surface["min"].get(), self.bound_surface["max"].get()
            d = d[(d["surface_reelle_bati"] >= s0) & (d["surface_reelle_bati"] <= s1)]
        if "valeur_fonciere" in d.columns:
            b0, b1 = self.bound_budget["min"].get(), self.bound_budget["max"].get()
            d = d[(d["valeur_fonciere"] >= b0) & (d["valeur_fonciere"] <= b1)]
        if "prix_m2" in d.columns:
            p0, p1 = self.bound_px2["min"].get(), self.bound_px2["max"].get()
            d = d[(d["prix_m2"] >= p0) & (d["prix_m2"] <= p1)]

        # Recalcul rendement
        d = self._compute_yields(d)

        # Trim final (stabilité des graphes)
        if "prix_m2" in d.columns and len(d) > 50:
            lo, hi = d["prix_m2"].quantile([0.01, 0.99])
            d = d[(d["prix_m2"] >= lo) & (d["prix_m2"] <= hi)]

        self.current_filtered_data = d
        self._render_overview(self.tab_overview)
        self._render_prix(self.tab_prix)
        self._render_top_communes(self.tab_top)
        self._render_table(self.tab_table)

    # ------------------ Vues ------------------
    def _render_overview(self, parent):
        for w in parent.winfo_children():
            w.destroy()

        d = self.current_filtered_data
        if d is None or len(d) == 0:
            tk.Label(parent, text="Aucune donnée à afficher.",
                     bg=self.colors["bg"], fg=self.colors["fg"],
                     font=("Segoe UI", 12)).pack(expand=True, fill="both", padx=10, pady=10)
            return

        fig = plt.Figure(figsize=(10, 5), dpi=100)
        fig.patch.set_facecolor(self.colors["bg"])

        # Hist prix/m²
        ax1 = fig.add_subplot(121)
        ax1.set_facecolor(self.colors["card"])
        if "prix_m2" in d.columns:
            p = self._clip_q(d["prix_m2"], 0.01, 0.99)
            if len(p) > 0:
                ax1.hist(p, bins=40, edgecolor="white")
                ax1.axvline(p.median(), linestyle="--", linewidth=2, color=self.colors["danger"],
                            label=f"Médiane: {p.median():,.0f} €".replace(",", " "))
                ax1.legend()
            ax1.set_title("Distribution prix/m² (1–99%)", color=self.colors["fg"])
            ax1.set_xlabel("€/m²", color=self.colors["fg"])
            ax1.set_ylabel("Fréquence", color=self.colors["fg"])
            ax1.tick_params(colors=self.colors["fg"])
            ax1.grid(alpha=0.2)

        # Scatter Surface vs Valeur foncière
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor(self.colors["card"])
        if {"surface_reelle_bati", "valeur_fonciere"} <= set(d.columns):
            x = pd.to_numeric(d["surface_reelle_bati"], errors="coerce")
            y = pd.to_numeric(d["valeur_fonciere"], errors="coerce")
            x = self._clip_q(x, 0.01, 0.99)
            y = self._clip_q(y, 0.01, 0.99)
            idx = x.index.intersection(y.index)
            if len(idx) > 0:
                sam = np.random.default_rng(42).choice(idx, size=min(3000, len(idx)), replace=False)
                ax2.scatter(x.loc[sam], y.loc[sam], s=6)
            ax2.set_title("Surface vs Valeur foncière (échantillon, 1–99%)", color=self.colors["fg"])
            ax2.set_xlabel("Surface (m²)", color=self.colors["fg"])
            ax2.set_ylabel("Valeur (€)", color=self.colors["fg"])
            ax2.tick_params(colors=self.colors["fg"])
            ax2.grid(alpha=0.2)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _render_prix(self, parent):
        for w in parent.winfo_children():
            w.destroy()

        d = self.current_filtered_data
        if d is None or len(d) == 0:
            tk.Label(parent, text="Aucune donnée à afficher.",
                     bg=self.colors["bg"], fg=self.colors["fg"],
                     font=("Segoe UI", 12)).pack(expand=True, fill="both", padx=10, pady=10)
            return

        # fallback géo
        if "code_departement" not in d.columns and "code_postal" in d.columns:
            d = d.copy()
            d["code_departement"] = d["code_postal"].astype(str).str[:2]
        if "zone_geo" not in d.columns and "code_departement" in d.columns:
            map_zone = {"75":"Paris","92":"Petite Couronne","93":"Petite Couronne","94":"Petite Couronne",
                        "77":"Grande Couronne","78":"Grande Couronne","91":"Grande Couronne","95":"Grande Couronne"}
            d = d.copy()
            d["zone_geo"] = d["code_departement"].map(map_zone)

        fig = plt.Figure(figsize=(10, 5), dpi=100)
        fig.patch.set_facecolor(self.colors["bg"])

        # Choix groupement
        group_col = None
        for c in ["zone_geo", "code_departement", "type_local"]:
            if c in d.columns and d[c].notna().any():
                group_col = c
                break

        # Médiane par groupe
        ax1 = fig.add_subplot(121)
        ax1.set_facecolor(self.colors["card"])
        if "prix_m2" in d.columns:
            if group_col is not None:
                g = d[[group_col, "prix_m2"]].copy()
                g["prix_m2"] = pd.to_numeric(g["prix_m2"], errors="coerce")
                g = g.dropna()
                if not g.empty:
                    med = g.groupby(group_col)["prix_m2"].median().sort_values(ascending=False).head(12)
                    med.plot(kind="bar", ax=ax1)
                    ax1.set_title(f"Médiane prix/m² par {group_col} (Top 12)", color=self.colors["fg"])
                else:
                    ax1.text(0.5, 0.5, "Données insuffisantes", ha="center", va="center", color=self.colors["fg"])
            else:
                med = pd.to_numeric(d["prix_m2"], errors="coerce").dropna().median()
                ax1.bar(["Global"], [med])
                ax1.set_title("Médiane prix/m² (globale)", color=self.colors["fg"])
            ax1.set_ylabel("€/m²", color=self.colors["fg"])
            ax1.tick_params(colors=self.colors["fg"])
            ax1.grid(axis="y", alpha=0.2)

        # Histogramme
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor(self.colors["card"])
        if "prix_m2" in d.columns:
            p = self._clip_q(d["prix_m2"], 0.01, 0.99)
            if len(p) > 0:
                ax2.hist(p, bins=40)
                ax2.axvline(p.median(), linestyle="--", linewidth=2, color=self.colors["danger"],
                            label=f"Médiane: {p.median():,.0f} €".replace(",", " "))
                ax2.legend()
            ax2.set_title("Distribution des prix/m² (1–99%)", color=self.colors["fg"])
            ax2.set_xlabel("Prix/m² (€)", color=self.colors["fg"])
            ax2.set_ylabel("Fréquence", color=self.colors["fg"])
            ax2.tick_params(colors=self.colors["fg"])
            ax2.grid(alpha=0.2)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _render_top_communes(self, parent):
        for w in parent.winfo_children():
            w.destroy()

        d = self.current_filtered_data
        if d is None or len(d) == 0:
            tk.Label(parent, text="Aucune donnée à afficher.",
                     bg=self.colors["bg"], fg=self.colors["fg"],
                     font=("Segoe UI", 12)).pack(expand=True, fill="both", padx=10, pady=10)
            return

        fr = tk.Frame(parent, bg=self.colors["bg"]); fr.pack(fill="both", expand=True, padx=10, pady=10)

        # Calcul top communes par rendement_net
        if {"nom_commune","rendement_net"} <= set(d.columns):
            g = d.groupby("nom_commune", dropna=True)["rendement_net"].median().sort_values(ascending=False).head(20)
            # Tableau simple
            tree = ttk.Treeview(fr, columns=("Commune", "Rendement (%)"), show="headings", height=12)
            tree.heading("Commune", text="Commune"); tree.heading("Rendement (%)", text="Rendement (%)")
            tree.column("Commune", width=220); tree.column("Rendement (%)", width=120, anchor="e")
            for name, val in g.items():
                tree.insert("", "end", values=(name, f"{val:,.2f}".replace(",", " ")))
            tree.pack(side="left", fill="y")
            # Bar chart
            fig = plt.Figure(figsize=(7, 5), dpi=100); fig.patch.set_facecolor(self.colors["bg"])
            ax = fig.add_subplot(111); ax.set_facecolor(self.colors["card"])
            g.plot(kind="barh", ax=ax)
            ax.invert_yaxis()
            ax.set_title("Top 20 communes par rendement net (médiane)", color=self.colors["fg"])
            ax.tick_params(colors=self.colors["fg"])
            for spine in ax.spines.values():
                spine.set_edgecolor(self.colors["secondary"])
            canvas = FigureCanvasTkAgg(fig, master=fr); canvas.draw()
            canvas.get_tk_widget().pack(side="right", fill="both", expand=True)
        else:
            tk.Label(parent, text="Colonnes nécessaires manquantes (nom_commune, rendement_net).",
                     bg=self.colors["bg"], fg=self.colors["muted"]).pack(anchor="w", padx=10, pady=10)

    def _render_table(self, parent):
        for w in parent.winfo_children():
            w.destroy()
        d = self.current_filtered_data
        fr = tk.Frame(parent, bg=self.colors["bg"]); fr.pack(fill="both", expand=True, padx=10, pady=10)
        if d is None or len(d) == 0:
            tk.Label(fr, text="Aucune donnée.", bg=self.colors["bg"], fg=self.colors["fg"]).pack()
            return
        # Table view
        cols = list(d.columns)
        tree = ttk.Treeview(fr, columns=cols, show="headings")
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, anchor="w")
        # Insert first 300 rows for performance
        for _, row in d.head(300).iterrows():
            tree.insert("", "end", values=[row.get(c, "") for c in cols])
        tree.pack(fill="both", expand=True)

# -------------------------------------------------------------
# Exécution
# -------------------------------------------------------------

def main():
    df = load_dataset()
    app = DashboardApp(df)
    app.mainloop()

if __name__ == "__main__":
    main()
