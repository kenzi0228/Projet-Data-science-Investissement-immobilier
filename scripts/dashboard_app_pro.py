"""
dashboard_app_pro.py
====================
Application professionnelle complète pour investissement locatif IDF.

PERSONA INTÉGRÉ:
- Manager IT, 40 ans, 70k€/an
- Budget: 50k€ d'apport
- Cible: Studio/T1/T2 pour étudiants/jeunes actifs
- Objectif: Rentabilité nette + faible risque

USAGE:
    python dashboard_app_pro.py
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

warnings.filterwarnings("ignore")

# Import module nettoyage avancé
try:
    import data_cleaner_advanced as dca
except ImportError:
    print("❌ Erreur: data_cleaner_advanced.py introuvable")
    print("   Assurez-vous que le fichier est dans le même dossier")
    sys.exit(1)


# === CONFIGURATION PERSONA ===
PERSONA = {
    'nom': 'Manager IT',
    'age': 40,
    'salaire_annuel': 70000,
    'apport': 50000,
    'capacite_emprunt': 150000,  # ~200k€ budget total max
    'budget_max': 200000,
    'surface_min': 15,
    'surface_max': 65,  # Studio à petit T3
    'cible_locataire': 'Étudiants / Jeunes actifs',
    'objectif_rendement_net': 4.5,  # % minimum
    'risque': 'Faible',
    'temps_disponible': 'Limité'
}


class DashboardAppPro:
    """Application dashboard professionnelle"""
    
    def __init__(self, root):
        self.root = root
        self.root.title(f"🏠 Dashboard Investissement - {PERSONA['nom']}")
        self.root.geometry("1700x950")
        
        # Variables
        self.df_unifie = None
        self.df_loyers = None
        self.df_gares = None
        self.current_filtered_data = None
        
        # Thème
        self.theme = "dark"
        self.colors = self._get_colors()
        
        # Configuration style
        self._configure_style()
        
        # Construction interface
        self._build_ui()
        
        # Chargement données
        self.root.after(100, self._load_data)
    
    def _get_colors(self):
        """Schéma de couleurs"""
        if self.theme == "dark":
            return {
                'bg': '#0f172a',
                'fg': '#e2e8f0',
                'accent': '#3b82f6',
                'secondary': '#1e293b',
                'card': '#1e293b',
                'success': '#10b981',
                'warning': '#f59e0b',
                'danger': '#ef4444',
                'text_muted': '#94a3b8',
                'border': '#334155'
            }
        else:
            return {
                'bg': '#f8fafc',
                'fg': '#1e293b',
                'accent': '#3b82f6',
                'secondary': '#e2e8f0',
                'card': '#ffffff',
                'success': '#10b981',
                'warning': '#f59e0b',
                'danger': '#ef4444',
                'text_muted': '#64748b',
                'border': '#cbd5e1'
            }
    
    def _configure_style(self):
        """Configure le style ttk"""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('.', background=self.colors['bg'],
                       foreground=self.colors['fg'])
        
        style.configure('Card.TFrame', background=self.colors['card'])
        
        style.configure('Title.TLabel',
                       background=self.colors['bg'],
                       foreground=self.colors['accent'],
                       font=('Segoe UI', 18, 'bold'))
        
        style.configure('Subtitle.TLabel',
                       background=self.colors['bg'],
                       foreground=self.colors['text_muted'],
                       font=('Segoe UI', 10))
        
        style.configure('TNotebook', background=self.colors['bg'], borderwidth=0)
        style.configure('TNotebook.Tab',
                       background=self.colors['secondary'],
                       foreground=self.colors['fg'],
                       padding=[15, 8],
                       font=('Segoe UI', 9, 'bold'))
        
        style.map('TNotebook.Tab',
                 background=[('selected', self.colors['card'])],
                 foreground=[('selected', self.colors['accent'])])
    
    def _build_ui(self):
        """Construit l'interface"""
        self.root.configure(bg=self.colors['bg'])
        
        # Header
        header = tk.Frame(self.root, bg=self.colors['bg'], height=70)
        header.pack(fill='x', padx=20, pady=(15, 5))
        
        title = ttk.Label(header, 
                         text=f"🏠 Dashboard Investissement - {PERSONA['nom']}",
                         style='Title.TLabel')
        title.pack(anchor='w')
        
        subtitle = ttk.Label(header,
                            text=f"Budget: {PERSONA['budget_max']:,}€ | Cible: {PERSONA['cible_locataire']} | Objectif rendement: {PERSONA['objectif_rendement_net']}%".replace(",", " "),
                            style='Subtitle.TLabel')
        subtitle.pack(anchor='w')
        
        # Layout principal
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Sidebar + Content
        self._build_sidebar(main_container)
        self._build_content(main_container)
        
        # Footer
        footer = tk.Frame(self.root, bg=self.colors['secondary'], height=35)
        footer.pack(fill='x', side='bottom')
        
        self.status_label = ttk.Label(footer, text="⏳ Chargement...",
                                     style='Subtitle.TLabel')
        self.status_label.pack(side='left', padx=20, pady=8)
        
        self.count_label = ttk.Label(footer, text="",
                                     style='Subtitle.TLabel')
        self.count_label.pack(side='right', padx=20, pady=8)
    
    def _build_sidebar(self, parent):
        """Construit le panneau de filtres"""
        sidebar = tk.Frame(parent, bg=self.colors['card'], width=360)
        sidebar.pack(side='left', fill='y', padx=(0, 15))
        sidebar.pack_propagate(False)
        
        # Canvas scrollable
        canvas = tk.Canvas(sidebar, bg=self.colors['card'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar, orient='vertical', command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=self.colors['card'])
        
        scrollable.bind("<Configure>",
                       lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable, anchor='nw', width=340)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # === TITRE ===
        tk.Label(scrollable, text="🎛️ FILTRES",
                bg=self.colors['accent'], fg='white',
                font=('Segoe UI', 12, 'bold'), pady=10).pack(fill='x')
        
        # === PRESETS PERSONA ===
        self._add_section(scrollable, "👤 Profil Investisseur")
        
        preset_frame = tk.Frame(scrollable, bg=self.colors['card'])
        preset_frame.pack(fill='x', padx=15, pady=5)
        
        presets_persona = [
            ("🎯 Mon Profil", (PERSONA['surface_min'], PERSONA['surface_max'], 
                             PERSONA['apport'], PERSONA['budget_max'])),
            ("💼 Budget Max", (15, 65, 50000, PERSONA['budget_max'])),
            ("💰 Petit Budget", (15, 35, 30000, 100000)),
            ("🏆 Haut Rendement", (20, 50, 40000, 150000))
        ]
        
        for i, (label, values) in enumerate(presets_persona):
            btn = tk.Button(preset_frame, text=label,
                          bg=[self.colors['accent'], self.colors['success'], 
                              self.colors['warning'], self.colors['danger']][i],
                          fg='white', font=('Segoe UI', 8, 'bold'),
                          relief='flat', cursor='hand2',
                          command=lambda v=values: self._apply_preset(*v))
            btn.pack(fill='x', pady=2)
        
        # === SURFACE & BUDGET ===
        self._add_section(scrollable, "🏠 Surface & Prix")
        
        self.w_surface = self._add_range_slider(scrollable, "Surface (m²)", 
                                                10, 200, 
                                                (PERSONA['surface_min'], PERSONA['surface_max']))
        
        self.w_prix_total = self._add_range_slider(scrollable, "Budget (k€)",
                                                   30, 500,
                                                   (PERSONA['apport']//1000, PERSONA['budget_max']//1000))
        
        self.w_prix_m2 = self._add_range_slider(scrollable, "Prix/m² (€)",
                                                1500, 20000, (3000, 12000))
        
        # === LOCALISATION ===
        self._add_section(scrollable, "🗺️ Localisation")
        
        self.w_zone = self._add_dropdown(scrollable, "Zone",
                                        ['(Toutes)', 'Paris', 'Petite Couronne', 'Grande Couronne'])
        
        self.w_departement = self._add_dropdown(scrollable, "Département",
                                               ['(Tous)', '75', '77', '78', '91', '92', '93', '94', '95'])
        
        # === RENTABILITÉ ===
        self._add_section(scrollable, "💰 Rentabilité")
        
        self.w_loyer = self._add_slider(scrollable, "Loyer €/m²", 10, 50, 22)
        self.w_charges = self._add_slider(scrollable, "Charges (%)", 15, 40, 25)
        self.w_rdt_min = self._add_slider(scrollable, "Rendement min (%)",
                                         0, 10, PERSONA['objectif_rendement_net'])
        
        # === OPTIONS ===
        self._add_section(scrollable, "⚙️ Options")
        
        self.w_annees = self._add_year_range(scrollable)
        self.w_outliers = self._add_checkbox(scrollable, "Supprimer outliers")
        self.w_top_n = self._add_slider(scrollable, "Top N résultats", 5, 30, 15)
        
        # === ACTIONS ===
        self._add_section(scrollable, "🎬 Actions")
        
        action_frame = tk.Frame(scrollable, bg=self.colors['card'])
        action_frame.pack(fill='x', padx=15, pady=10)
        
        tk.Button(action_frame, text="✅ Appliquer",
                 bg=self.colors['success'], fg='white',
                 font=('Segoe UI', 10, 'bold'), relief='flat',
                 cursor='hand2', height=2,
                 command=self._apply_filters).pack(fill='x', pady=2)
        
        tk.Button(action_frame, text="🔄 Reset",
                 bg=self.colors['warning'], fg='white',
                 font=('Segoe UI', 10, 'bold'), relief='flat',
                 cursor='hand2', height=2,
                 command=self._reset_filters).pack(fill='x', pady=2)
        
        tk.Button(action_frame, text="📥 Exporter",
                 bg=self.colors['accent'], fg='white',
                 font=('Segoe UI', 10, 'bold'), relief='flat',
                 cursor='hand2', height=2,
                 command=self._export_data).pack(fill='x', pady=2)
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
    
    def _add_section(self, parent, text):
        """Ajoute un en-tête de section"""
        tk.Label(parent, text=text,
                bg=self.colors['secondary'], fg=self.colors['accent'],
                font=('Segoe UI', 9, 'bold'),
                pady=8, padx=10, anchor='w').pack(fill='x', padx=10, pady=(12, 3))
    
    def _add_range_slider(self, parent, label, min_v, max_v, default):
        frame = tk.Frame(parent, bg=self.colors['card'])
        frame.pack(fill='x', padx=15, pady=3)

        tk.Label(frame, text=label, bg=self.colors['card'], fg=self.colors['fg'],
                font=('Segoe UI', 8)).pack(anchor='w')

        var_min = tk.IntVar(value=default[0])
        var_max = tk.IntVar(value=default[1])

        row = tk.Frame(frame, bg=self.colors['card'])
        row.pack(fill='x')

        # Slider MIN
        tk.Label(row, text="Min", bg=self.colors['card'], fg=self.colors['fg']).pack(side='left', padx=(0,6))
        s_min = tk.Scale(row, from_=min_v, to=max_v, orient='horizontal',
                        variable=var_min, bg=self.colors['card'], fg=self.colors['fg'],
                        highlightthickness=0, troughcolor=self.colors['secondary'],
                        length=120, command=lambda _=None: var_min.set(min(var_min.get(), var_max.get())))
        s_min.pack(side='left')

        tk.Label(row, text="  ", bg=self.colors['card']).pack(side='left')

        # Slider MAX
        tk.Label(row, text="Max", bg=self.colors['card'], fg=self.colors['fg']).pack(side='left', padx=(6,6))
        s_max = tk.Scale(row, from_=min_v, to=max_v, orient='horizontal',
                        variable=var_max, bg=self.colors['card'], fg=self.colors['fg'],
                        highlightthickness=0, troughcolor=self.colors['secondary'],
                        length=120, command=lambda _=None: var_max.set(max(var_min.get(), var_max.get())))
        s_max.pack(side='left')

        return {'min': var_min, 'max': var_max}

    frame = tk.Frame(parent, bg=self.colors['card'])
    frame.pack(fill='x', padx=15, pady=3)

    tk.Label(frame, text=label, bg=self.colors['card'], fg=self.colors['fg'],
             font=('Segoe UI', 8)).pack(anchor='w')

    var_min = tk.IntVar(value=default[0])
    var_max = tk.IntVar(value=default[1])

    row = tk.Frame(frame, bg=self.colors['card'])
    row.pack(fill='x')

    # Slider MIN
    tk.Label(row, text="Min", bg=self.colors['card'], fg=self.colors['fg']).pack(side='left', padx=(0,6))
    s_min = tk.Scale(row, from_=min_v, to=max_v, orient='horizontal',
                     variable=var_min, bg=self.colors['card'], fg=self.colors['fg'],
                     highlightthickness=0, troughcolor=self.colors['secondary'],
                     length=120, command=lambda _=None: var_min.set(min(var_min.get(), var_max.get())))
    s_min.pack(side='left')

    # Espace
    tk.Label(row, text="  ", bg=self.colors['card']).pack(side='left')

    # Slider MAX
    tk.Label(row, text="Max", bg=self.colors['card'], fg=self.colors['fg']).pack(side='left', padx=(6,6))
    s_max = tk.Scale(row, from_=min_v, to=max_v, orient='horizontal',
                     variable=var_max, bg=self.colors['card'], fg=self.colors['fg'],
                     highlightthickness=0, troughcolor=self.colors['secondary'],
                     length=120, command=lambda _=None: var_max.set(max(var_min.get(), var_max.get())))
    s_max.pack(side='left')

    return {'min': var_min, 'max': var_max}
    def _compute_yields(self, d: pd.DataFrame) -> pd.DataFrame:
    if d is None or d.empty:
        return d
    # Nettoyage de base
    d = d.copy()
    for col in ["valeur_fonciere", "surface_reelle_bati", "prix_m2"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors='coerce')

    # Loyer net selon sliders actuels
    loyer_m2 = float(self.w_loyer.get()) if hasattr(self, "w_loyer") else 22.0
    charges_pct = float(self.w_charges.get())/100 if hasattr(self, "w_charges") else 0.25

    loyer_annuel = loyer_m2 * d["surface_reelle_bati"] * 12
    loyer_net = loyer_annuel * (1 - charges_pct)

    d["rendement_brut"] = (loyer_annuel / d["valeur_fonciere"]) * 100
    d["rendement_net"]  = (loyer_net / d["valeur_fonciere"]) * 100

    # Nettoyage inf/NaN
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    return d


    def _add_slider(self, parent, label, min_v, max_v, default):
        """Slider simple"""
        frame = tk.Frame(parent, bg=self.colors['card'])
        frame.pack(fill='x', padx=15, pady=3)
        
        tk.Label(frame, text=label, bg=self.colors['card'], fg=self.colors['fg'],
                font=('Segoe UI', 8)).pack(anchor='w')
        
        var = tk.DoubleVar(value=default)
        
        tk.Scale(frame, from_=min_v, to=max_v, resolution=0.5,
                orient='horizontal', variable=var, bg=self.colors['card'],
                fg=self.colors['fg'], highlightthickness=0,
                troughcolor=self.colors['secondary'], length=280).pack()
        
        return var
    
    def _add_dropdown(self, parent, label, options):
        """Menu déroulant"""
        frame = tk.Frame(parent, bg=self.colors['card'])
        frame.pack(fill='x', padx=15, pady=3)
        
        tk.Label(frame, text=label, bg=self.colors['card'], fg=self.colors['fg'],
                font=('Segoe UI', 8)).pack(anchor='w')
        
        var = tk.StringVar(value=options[0])
        ttk.Combobox(frame, textvariable=var, values=options,
                    state='readonly', width=30).pack(fill='x')
        
        return var
    
    def _add_checkbox(self, parent, label):
        """Checkbox"""
        var = tk.BooleanVar(value=True)
        tk.Checkbutton(parent, text=label, variable=var,
                      bg=self.colors['card'], fg=self.colors['fg'],
                      selectcolor=self.colors['secondary'],
                      font=('Segoe UI', 8)).pack(anchor='w', padx=15, pady=2)
        return var
    
    def _add_year_range(self, parent):
        """Sélecteur d'années"""
        frame = tk.Frame(parent, bg=self.colors['card'])
        frame.pack(fill='x', padx=15, pady=3)
        
        tk.Label(frame, text="Années", bg=self.colors['card'], fg=self.colors['fg'],
                font=('Segoe UI', 8)).pack(anchor='w')
        
        years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
        var_min = tk.IntVar(value=years[0])
        var_max = tk.IntVar(value=years[-1])
        
        year_frame = tk.Frame(frame, bg=self.colors['card'])
        year_frame.pack(fill='x')
        
        tk.Spinbox(year_frame, from_=years[0], to=years[-1],
                  textvariable=var_min, width=6).pack(side='left', padx=2)
        tk.Label(year_frame, text="→", bg=self.colors['card'],
                fg=self.colors['fg']).pack(side='left', padx=3)
        tk.Spinbox(year_frame, from_=years[0], to=years[-1],
                  textvariable=var_max, width=6).pack(side='left', padx=2)
        
        return {'min': var_min, 'max': var_max}
    
    def _apply_preset(self, surf_min, surf_max, budget_min, budget_max):
        """Applique un preset"""
        self.w_surface['min'].set(surf_min)
        self.w_surface['max'].set(surf_max)
        self.w_prix_total['min'].set(budget_min // 1000)
        self.w_prix_total['max'].set(budget_max // 1000)
    
    def _build_content(self, parent):
        """Zone de contenu"""
        content = tk.Frame(parent, bg=self.colors['bg'])
        content.pack(side='left', fill='both', expand=True)
        
        # Onglets
        self.notebook = ttk.Notebook(content)
        self.notebook.pack(fill='both', expand=True)
        
        self.tabs = {}
        tab_names = [
            "📊 Vue d'ensemble",
            "🏆 Top Communes",
            "💰 Analyse Prix",
            "🎯 Rendement",
            "🗺️ Carte",
            "📈 Recommandations Persona"
        ]
        
        for name in tab_names:
            frame = ttk.Frame(self.notebook, style='Card.TFrame')
            self.notebook.add(frame, text=name)
            self.tabs[name] = frame
    
    def _load_data(self):
        """Charge les données avec nettoyage avancé"""
        try:
            self.status_label.config(text="⏳ Chargement et nettoyage avancé...")
            self.root.update()
            
            # CHEMIN CORRIGÉ
            self.df_unifie, self.df_loyers, self.df_gares = dca.quick_load_advanced(
                raw_dir="../Projet-Data-science-Investissement-immobilier/data/raw",
                clean_dir="../Projet-Data-science-Investissement-immobilier/data/clean",
                force_refresh=False
            )
            
            self.current_filtered_data = self.df_unifie.copy()
            self.current_filtered_data = self._compute_yields(self.current_filtered_data)

            # Affichage widgets
            self._render_all_widgets()
            
            self.status_label.config(
                text=f"✅ {len(self.df_unifie):,} transactions chargées et nettoyées".replace(",", " ")
            )
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur chargement :\n{str(e)}")
            self.status_label.config(text="❌ Erreur de chargement")
    
    def _apply_filters(self):
        """Applique les filtres"""
        if self.df_unifie is None:
            return
        
        try:
            self.status_label.config(text="⏳ Filtrage...")
            self.root.update()
            
            d = self.df_unifie.copy()
            # Sécurise colonnes attendues
            if 'code_departement' not in d.columns and 'code_postal' in d.columns:
                d['code_departement'] = d['code_postal'].astype(str).str[:2]

            if 'zone_geo' not in d.columns and 'code_departement' in d.columns:
                map_zone = {
                    '75': 'Paris','92': 'Petite Couronne','93': 'Petite Couronne','94': 'Petite Couronne',
                    '77': 'Grande Couronne','78': 'Grande Couronne','91': 'Grande Couronne','95': 'Grande Couronne'
                }
                d['zone_geo'] = d['code_departement'].map(map_zone)

            n_init = len(d)
            
            # Surface
            d = d[d['surface_reelle_bati'].between(
                self.w_surface['min'].get(), self.w_surface['max'].get()
            )]
            
            # Prix total (k€ → €)
            d = d[d['valeur_fonciere'].between(
                self.w_prix_total['min'].get() * 1000,
                self.w_prix_total['max'].get() * 1000
            )]
            
            # Prix/m²
            d = d[d['prix_m2'].between(
                self.w_prix_m2['min'].get(), self.w_prix_m2['max'].get()
            )]
            
            # Années
            d = d[(d['annee'] >= self.w_annees['min'].get()) &
                  (d['annee'] <= self.w_annees['max'].get())]
            
            # Zone géographique
            if self.w_zone.get() != '(Toutes)':
                d = d[d['zone_geo'] == self.w_zone.get()]
            
            # Département
            if self.w_departement.get() != '(Tous)':
                d = d[d['code_departement'] == self.w_departement.get()]
            
            # Outliers
            if self.w_outliers.get() and len(d) > 50:
                Q1 = d['prix_m2'].quantile(0.25)
                Q3 = d['prix_m2'].quantile(0.75)
                IQR = Q3 - Q1
                d = d[d['prix_m2'].between(Q1 - 2*IQR, Q3 + 2*IQR)]
            
            # Calcul rendement
            loyer_annuel = self.w_loyer.get() * d['surface_reelle_bati'] * 12
            charges_pct = self.w_charges.get() / 100
            loyer_net = loyer_annuel * (1 - charges_pct)
            # Recalcule rendement sur le sous-ensemble et selon sliders courants
            d = self._compute_yields(d)

            d['rendement_net'] = (loyer_net / d['valeur_fonciere']) * 100
            d['rendement_brut'] = (loyer_annuel / d['valeur_fonciere']) * 100
            
            # Filtre rendement min
            d = d[d['rendement_net'] >= self.w_rdt_min.get()]
            # Trim quantiles (limite les pics extrêmes résiduels)
            if 'prix_m2' in d.columns and len(d) > 50:
                lo, hi = d['prix_m2'].quantile([0.01, 0.99])
                d = d[(d['prix_m2'] >= lo) & (d['prix_m2'] <= hi)]

            self.current_filtered_data = d
            n_final = len(d)
            
            self._render_all_widgets()
            
            self.status_label.config(text="✅ Filtres appliqués")
            self.count_label.config(
                text=f"📊 {n_final:,} / {n_init:,} ({n_final/n_init*100:.1f}%)".replace(",", " ")
            )
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur filtrage :\n{str(e)}")
    
    def _reset_filters(self):
        """Réinitialise"""
        self.w_surface['min'].set(PERSONA['surface_min'])
        self.w_surface['max'].set(PERSONA['surface_max'])
        self.w_prix_total['min'].set(PERSONA['apport'] // 1000)
        self.w_prix_total['max'].set(PERSONA['budget_max'] // 1000)
        self.w_prix_m2['min'].set(3000)
        self.w_prix_m2['max'].set(12000)
        self.w_zone.set('(Toutes)')
        self.w_departement.set('(Tous)')
        self.w_loyer.set(22.0)
        self.w_charges.set(25.0)
        self.w_rdt_min.set(PERSONA['objectif_rendement_net'])
        self.w_outliers.set(True)
        self.w_top_n.set(15)
        
        self.status_label.config(text="🔄 Filtres réinitialisés")
        self._apply_filters()
    
    def _export_data(self):
        """Exporte CSV"""
        if self.current_filtered_data is None or len(self.current_filtered_data) == 0:
            messagebox.showwarning("Attention", "Aucune donnée à exporter")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            initialfile="investissement_persona.csv"
        )
        
        if filepath:
            try:
                self.current_filtered_data.to_csv(filepath, index=False, encoding='utf-8-sig')
                messagebox.showinfo("Succès",
                    f"✅ {len(self.current_filtered_data):,} transactions exportées".replace(",", " "))
            except Exception as e:
                messagebox.showerror("Erreur", str(e))
    
    def _render_all_widgets(self):
        """Affiche tous les widgets"""
        if self.current_filtered_data is None:
            return
        
        self._render_overview(self.tabs["📊 Vue d'ensemble"])
        self._render_top_communes(self.tabs["🏆 Top Communes"])
        self._render_prix(self.tabs["💰 Analyse Prix"])
        self._render_rendement(self.tabs["🎯 Rendement"])
        self._render_carte(self.tabs["🗺️ Carte"])
        self._render_recommandations_persona(self.tabs["📈 Recommandations Persona"])
    
    def _render_overview(self, parent):
        """Vue d'ensemble avec KPIs"""
        for w in parent.winfo_children():
            w.destroy()
        
        d = self.current_filtered_data
        
        # KPIs
        kpi_frame = tk.Frame(parent, bg=self.colors['card'])
        kpi_frame.pack(fill='x', padx=20, pady=15)
        
        kpis = [
            ("Transactions", f"{len(d):,}".replace(",", " "), self.colors['accent']),
            ("Prix/m² médian", f"{d['prix_m2'].median():,.0f} €".replace(",", " "), self.colors['success']),
            ("Rendement net", f"{d['rendement_net'].median():.2f}%" if 'rendement_net' in d.columns else "N/A", self.colors['warning']),
            ("Surface médiane", f"{d['surface_reelle_bati'].median():.0f} m²", self.colors['danger'])
        ]
        
        for label, value, color in kpis:
            card = tk.Frame(kpi_frame, bg=color, relief='flat')
            card.pack(side='left', fill='both', expand=True, padx=8, pady=8)
            
            tk.Label(card, text=label, bg=color, fg='white',
                    font=('Segoe UI', 9)).pack(pady=(12, 2))
            tk.Label(card, text=value, bg=color, fg='white',
                    font=('Segoe UI', 20, 'bold')).pack(pady=(0, 12))
        
        # Graphique
        fig = Figure(figsize=(13, 4.5), facecolor=self.colors['card'])
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors['card'])
        ax.set_xlabel('Surface (m²)', color=self.colors['fg'])
        ax.set_ylabel('Fréquence', color=self.colors['fg'])
        p = _clip_q(d['prix_m2'], 0.01, 0.99)
        ax.hist(p, bins=40, color=self.colors['accent'], alpha=0.7, edgecolor='white')
        ax.set_title("Distribution prix/m² (1–99%)")

        ax.tick_params(colors=self.colors['fg'])
        ax.grid(alpha=0.2)
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=10)
   

    def _render_top_communes(self, parent):
        """Top communes avec rendement"""
        for w in parent.winfo_children():
            w.destroy()
        
        d = self.current_filtered_data
        top_n = int(self.w_top_n.get())
        
        if 'rendement_net' not in d.columns:
            tk.Label(parent, text="⚠️ Données de rendement non disponibles",
                    bg=self.colors['card'], fg=self.colors['text_muted'],
                    font=('Segoe UI', 12)).pack(expand=True)
            return
        
        top = (d.groupby(['nom_commune', 'code_postal'], as_index=False)
               .agg(nb=('prix_m2', 'count'),
                    prix_m2_med=('prix_m2', 'median'),
                    surf_med=('surface_reelle_bati', 'median'),
                    prix_med=('valeur_fonciere', 'median'),
                    rdt_net=('rendement_net', 'median'))
               .sort_values('rdt_net', ascending=False)
               .head(top_n))
        
        # Titre
        tk.Label(parent, text=f"🏆 Top {top_n} communes (rendement net médian)",
                bg=self.colors['card'], fg=self.colors['accent'],
                font=('Segoe UI', 14, 'bold')).pack(pady=(15, 10))
        
        # Tableau scrollable
        table_frame = tk.Frame(parent, bg=self.colors['card'])
        table_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Colonnes
        columns = ['Rang', 'Commune', 'CP', 'Nb', 'Surf (m²)', 'Prix méd (k€)', 'Rdt net (%)']
        
        # Création tableau
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=top_n)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='center')
        
        # Remplissage
        for idx, row in top.iterrows():
            tree.insert('', 'end', values=(
                idx + 1,
                row['nom_commune'],
                row['code_postal'],
                int(row['nb']),
                f"{row['surf_med']:.0f}",
                f"{row['prix_med']/1000:.0f}",
                f"{row['rdt_net']:.2f}"
            ), tags=('row',))
        
        tree.pack(fill='both', expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=tree.yview)
        scrollbar.pack(side='right', fill='y')
        tree.configure(yscrollcommand=scrollbar.set)
    
    def _render_prix(self, parent):
        """Analyse des prix"""
        for w in parent.winfo_children():
            w.destroy()
        
        d = self.current_filtered_data
        ds = d.dropna(subset=['surface_reelle_bati', 'prix_m2'])
        ds = ds.sample(min(3000, len(ds)), random_state=42)
        
        fig = Figure(figsize=(13, 6), facecolor=self.colors['card'])
        
        # Scatter
        ax1 = fig.add_subplot(121)
        ax1.set_facecolor(self.colors['card'])
        
        if 'rendement_net' in ds.columns:
            scatter = ax1.scatter(ds['surface_reelle_bati'], ds['prix_m2'],
                                c=ds['rendement_net'], cmap='RdYlGn',
                                s=30, alpha=0.6, edgecolors='white', linewidth=0.3)
            plt.colorbar(scatter, ax=ax1, label='Rendement net (%)')
        else:
            ax1.scatter(ds['surface_reelle_bati'], ds['prix_m2'],
                       s=30, alpha=0.6, c=self.colors['danger'],
                       edgecolors='white', linewidth=0.3)
        
        ax1.set_xlabel('Surface (m²)', color=self.colors['fg'])
        ax1.set_ylabel('Prix/m² (€)', color=self.colors['fg'])
        ax1.set_title('Prix/m² vs Surface', color=self.colors['fg'], fontweight='bold')
        ax1.tick_params(colors=self.colors['fg'])
        ax1.grid(alpha=0.2)
        
        # Histogramme prix
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor(self.colors['card'])
        
        p = _clip_q(df_zone['prix_m2'], 0.01, 0.99)
        axZ.hist(p, bins=40)
        ax2.axvline(d['prix_m2'].median(), color=self.colors['danger'],
                   linestyle='--', linewidth=2,
                   label=f"Médiane: {d['prix_m2'].median():,.0f} €".replace(",", " "))
        ax2.set_xlabel('Prix/m² (€)', color=self.colors['fg'])
        ax2.set_ylabel('Fréquence', color=self.colors['fg'])
        ax2.set_title('Distribution des prix/m²', color=self.colors['fg'], fontweight='bold')
        ax2.tick_params(colors=self.colors['fg'])
        ax2.legend(facecolor=self.colors['secondary'], labelcolor=self.colors['fg'])
        ax2.grid(alpha=0.2)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=20)
    
    def _render_rendement(self, parent):
        """Analyse rendement"""
        for w in parent.winfo_children():
            w.destroy()
        
        d = self.current_filtered_data
        
        if 'rendement_net' not in d.columns or not d['rendement_net'].notna().any():
            tk.Label(parent, text="⚠️ Données rendement indisponibles",
                    bg=self.colors['card'], fg=self.colors['text_muted'],
                    font=('Segoe UI', 12)).pack(expand=True)
            return
        
        fig = Figure(figsize=(13, 6), facecolor=self.colors['card'])
        
        # Histogramme rendement
        ax1 = fig.add_subplot(121)
        ax1.set_facecolor(self.colors['card'])
        
        rdt = d['rendement_net'].clip(upper=12)
        ax1.hist(rdt.dropna(), bins=40,
                color=self.colors['success'], alpha=0.7, edgecolor='white')
        ax1.axvline(rdt.median(), color=self.colors['danger'],
                   linestyle='--', linewidth=2,
                   label=f"Médiane: {rdt.median():.2f}%")
        ax1.axvline(PERSONA['objectif_rendement_net'], color=self.colors['warning'],
                   linestyle=':', linewidth=2,
                   label=f"Objectif: {PERSONA['objectif_rendement_net']}%")
        ax1.set_xlabel('Rendement net (%)', color=self.colors['fg'])
        ax1.set_ylabel('Fréquence', color=self.colors['fg'])
        ax1.set_title('Distribution des rendements nets', color=self.colors['fg'], fontweight='bold')
        ax1.tick_params(colors=self.colors['fg'])
        ax1.legend(facecolor=self.colors['secondary'], labelcolor=self.colors['fg'])
        ax1.grid(alpha=0.2)
        
        # Rendement par zone
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor(self.colors['card'])
        
        if 'zone_geo' in d.columns:
            zone_rdt = (d.groupby('zone_geo')['rendement_net']
                       .median()
                       .sort_values(ascending=False))
            
            ax2.barh(zone_rdt.index, zone_rdt.values,
                    color=[self.colors['accent'], self.colors['success'], self.colors['warning']],
                    alpha=0.8, edgecolor='white')
            ax2.set_xlabel('Rendement net médian (%)', color=self.colors['fg'])
            ax2.set_title('Rendement par zone', color=self.colors['fg'], fontweight='bold')
            ax2.tick_params(colors=self.colors['fg'])
            ax2.invert_yaxis()
            ax2.grid(alpha=0.2, axis='x')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=20)
    
    def _render_carte(self, parent):
        """Carte départements"""
        for w in parent.winfo_children():
            w.destroy()
        
        d = self.current_filtered_data
        
        # CORRECTION : Vérifier code_departement
        if 'code_departement' not in d.columns or d['code_departement'].isna().all():
            tk.Label(parent, text="⚠️ Code département non disponible dans les données",
                    bg=self.colors['card'], fg=self.colors['text_muted'],
                    font=('Segoe UI', 12)).pack(expand=True)
            return
        
        dept_stats = (d.groupby('code_departement')
                      .agg(nb=('prix_m2', 'count'),
                           prix_med=('prix_m2', 'median'),
                           rdt_med=('rendement_net', 'median') if 'rendement_net' in d.columns else ('prix_m2', 'count'))
                      .sort_values('rdt_med' if 'rendement_net' in d.columns else 'nb', ascending=False)
                      .head(8))
        
        if len(dept_stats) == 0:
            tk.Label(parent, text="⚠️ Aucune donnée à afficher",
                    bg=self.colors['card'], fg=self.colors['text_muted'],
                    font=('Segoe UI', 12)).pack(expand=True)
            return
        
        fig = Figure(figsize=(14, 6), facecolor=self.colors['card'])
        
        # Prix médian
        ax1 = fig.add_subplot(121)
        ax1.set_facecolor(self.colors['card'])
        ax1.barh(dept_stats.index.astype(str), dept_stats['prix_med'],
                color=self.colors['accent'], alpha=0.8, edgecolor='white')
        ax1.set_xlabel('Prix/m² médian (€)', color=self.colors['fg'])
        ax1.set_title('Prix médian par département', color=self.colors['fg'], fontweight='bold')
        ax1.tick_params(colors=self.colors['fg'])
        ax1.invert_yaxis()
        ax1.grid(alpha=0.2, axis='x')
        
        # Rendement ou Volume
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor(self.colors['card'])
        
        if 'rendement_net' in d.columns:
            ax2.barh(dept_stats.index.astype(str), dept_stats['rdt_med'],
                    color=self.colors['success'], alpha=0.8, edgecolor='white')
            ax2.set_xlabel('Rendement net médian (%)', color=self.colors['fg'])
            ax2.set_title('Rendement par département', color=self.colors['fg'], fontweight='bold')
        else:
            ax2.barh(dept_stats.index.astype(str), dept_stats['nb'],
                    color=self.colors['success'], alpha=0.8, edgecolor='white')
            ax2.set_xlabel('Nombre de transactions', color=self.colors['fg'])
            ax2.set_title('Volume par département', color=self.colors['fg'], fontweight='bold')
        
        ax2.tick_params(colors=self.colors['fg'])
        ax2.invert_yaxis()
        ax2.grid(alpha=0.2, axis='x')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=20)
    
    def _render_recommandations_persona(self, parent):
        """Recommandations personnalisées selon le persona"""
        for w in parent.winfo_children():
            w.destroy()
        
        d = self.current_filtered_data
        
        # Frame scrollable
        canvas = tk.Canvas(parent, bg=self.colors['card'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=self.colors['card'])
        
        scrollable.bind("<Configure>",
                       lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # === TITRE ===
        tk.Label(scrollable, text=f"📈 RECOMMANDATIONS POUR {PERSONA['nom'].upper()}",
                bg=self.colors['accent'], fg='white',
                font=('Segoe UI', 16, 'bold'),
                pady=15).pack(fill='x')
        
        # === RAPPEL PROFIL ===
        profil_frame = tk.Frame(scrollable, bg=self.colors['secondary'],
                               relief='flat', bd=2)
        profil_frame.pack(fill='x', padx=30, pady=15)
        
        tk.Label(profil_frame, text="👤 VOTRE PROFIL",
                bg=self.colors['secondary'], fg=self.colors['accent'],
                font=('Segoe UI', 12, 'bold')).pack(anchor='w', padx=15, pady=(10, 5))
        
        profil_text = f"""
• Budget disponible : {PERSONA['apport']:,} € (apport) + capacité d'emprunt
• Budget max conseillé : {PERSONA['budget_max']:,} €
• Surface ciblée : {PERSONA['surface_min']}-{PERSONA['surface_max']} m² (Studio à T3)
• Locataires cibles : {PERSONA['cible_locataire']}
• Objectif rendement net : ≥ {PERSONA['objectif_rendement_net']}%
• Profil de risque : {PERSONA['risque']}
        """.replace(",", " ")
        
        tk.Label(profil_frame, text=profil_text,
                bg=self.colors['secondary'], fg=self.colors['fg'],
                font=('Consolas', 9), justify='left').pack(anchor='w', padx=15, pady=(0, 10))
        
        # === STATISTIQUES SÉLECTION ===
        stats_frame = tk.Frame(scrollable, bg=self.colors['card'])
        stats_frame.pack(fill='x', padx=30, pady=10)
        
        tk.Label(stats_frame, text="📊 STATISTIQUES DE VOTRE SÉLECTION",
                bg=self.colors['card'], fg=self.colors['accent'],
                font=('Segoe UI', 12, 'bold')).pack(anchor='w', pady=(0, 8))
        
        # Calculs
        prix_med = d['prix_m2'].median()
        surf_med = d['surface_reelle_bati'].median()
        prix_total_med = d['valeur_fonciere'].median()
        
        if 'rendement_net' in d.columns and d['rendement_net'].notna().any():
            rdt_med = d['rendement_net'].median()
        else:
            rdt_med = None
        
        # Biens dans budget
        dans_budget = d[d['valeur_fonciere'] <= PERSONA['budget_max']]
        pct_budget = (len(dans_budget) / len(d) * 100) if len(d) > 0 else 0
        
        # Rendement objectif
        if rdt_med:
            objectif_ok = d[d['rendement_net'] >= PERSONA['objectif_rendement_net']]
            pct_objectif = (len(objectif_ok) / len(d) * 100) if len(d) > 0 else 0
        else:
            pct_objectif = 0
        
        stats_text = f"""
Surface médiane : {surf_med:.0f} m²
Prix/m² médian : {prix_med:,.0f} €
Prix total médian : {prix_total_med:,.0f} €
{'Rendement net médian : ' + f'{rdt_med:.2f}%' if rdt_med else 'Rendement : Non calculé'}

Biens dans votre budget (≤ {PERSONA['budget_max']:,}€) : {len(dans_budget):,} ({pct_budget:.1f}%)
{'Biens atteignant objectif rendement : ' + f'{len(objectif_ok):,} ({pct_objectif:.1f}%)' if rdt_med else ''}
        """.replace(",", " ")
        
        tk.Label(stats_frame, text=stats_text,
                bg=self.colors['card'], fg=self.colors['fg'],
                font=('Consolas', 10), justify='left').pack(anchor='w', padx=15)
        
        # === ÉVALUATION ===
        eval_frame = tk.Frame(scrollable, bg=self.colors['card'])
        eval_frame.pack(fill='x', padx=30, pady=10)
        
        tk.Label(eval_frame, text="🎯 ÉVALUATION POUR VOTRE PROFIL",
                bg=self.colors['card'], fg=self.colors['accent'],
                font=('Segoe UI', 12, 'bold')).pack(anchor='w', pady=(0, 8))
        
        # Évaluation intelligente
        evaluations = []
        
        if rdt_med:
            if rdt_med >= PERSONA['objectif_rendement_net'] + 1:
                evaluations.append(("✅ EXCELLENT", 
                    f"Rendement médian de {rdt_med:.2f}% > objectif ({PERSONA['objectif_rendement_net']}%)",
                    self.colors['success']))
            elif rdt_med >= PERSONA['objectif_rendement_net']:
                evaluations.append(("✔️ BON", 
                    f"Rendement médian de {rdt_med:.2f}% atteint l'objectif",
                    self.colors['accent']))
            else:
                evaluations.append(("⚠️ MODÉRÉ",
                    f"Rendement de {rdt_med:.2f}% sous l'objectif ({PERSONA['objectif_rendement_net']}%)",
                    self.colors['warning']))
        
        if prix_total_med <= PERSONA['budget_max']:
            evaluations.append(("💰 ACCESSIBLE",
                f"Prix médian {prix_total_med/1000:.0f}k€ dans votre budget",
                self.colors['success']))
        else:
            evaluations.append(("💸 TENDU",
                f"Prix médian {prix_total_med/1000:.0f}k€ > budget {PERSONA['budget_max']/1000:.0f}k€",
                self.colors['warning']))
        
        if surf_med >= PERSONA['surface_min'] and surf_med <= PERSONA['surface_max']:
            evaluations.append(("📐 IDÉAL",
                f"Surface médiane {surf_med:.0f}m² correspond au profil",
                self.colors['success']))
        
        if pct_budget >= 50:
            evaluations.append(("🎯 LARGE CHOIX",
                f"{pct_budget:.0f}% des biens dans votre budget",
                self.colors['success']))
        elif pct_budget >= 25:
            evaluations.append(("⚖️ CHOIX MODÉRÉ",
                f"{pct_budget:.0f}% des biens dans votre budget",
                self.colors['accent']))
        else:
            evaluations.append(("⚠️ CHOIX LIMITÉ",
                f"Seulement {pct_budget:.0f}% dans budget - Ajustez filtres",
                self.colors['warning']))
        
        for titre, desc, color in evaluations:
            eval_card = tk.Frame(eval_frame, bg=color, relief='flat')
            eval_card.pack(fill='x', pady=5)
            
            tk.Label(eval_card, text=titre,
                    bg=color, fg='white',
                    font=('Segoe UI', 11, 'bold')).pack(anchor='w', padx=15, pady=(8, 2))
            tk.Label(eval_card, text=desc,
                    bg=color, fg='white',
                    font=('Segoe UI', 9)).pack(anchor='w', padx=15, pady=(0, 8))
        
        # === TOP 5 RECOMMANDATIONS ===
        top_frame = tk.Frame(scrollable, bg=self.colors['card'])
        top_frame.pack(fill='x', padx=30, pady=15)
        
        tk.Label(top_frame, text="🏆 TOP 5 RECOMMANDATIONS POUR VOUS",
                bg=self.colors['card'], fg=self.colors['accent'],
                font=('Segoe UI', 12, 'bold')).pack(anchor='w', pady=(0, 10))
        
        # Filtrer selon le persona
        recommandations = d[
            (d['valeur_fonciere'] <= PERSONA['budget_max']) &
            (d['surface_reelle_bati'].between(PERSONA['surface_min'], PERSONA['surface_max']))
        ]
        
        if 'rendement_net' in recommandations.columns:
            recommandations = recommandations[
                recommandations['rendement_net'] >= PERSONA['objectif_rendement_net']
            ]
            
            top5 = (recommandations.groupby(['nom_commune', 'code_postal'], as_index=False)
                   .agg(nb=('prix_m2', 'count'),
                        prix_med=('valeur_fonciere', 'median'),
                        surf_med=('surface_reelle_bati', 'median'),
                        rdt_med=('rendement_net', 'median'))
                   .sort_values('rdt_med', ascending=False)
                   .head(5))
            
            if len(top5) > 0:
                for idx, row in top5.iterrows():
                    reco_card = tk.Frame(top_frame, bg=self.colors['secondary'], relief='flat')
                    reco_card.pack(fill='x', pady=5)
                    
                    titre = f"#{idx+1} {row['nom_commune']} ({row['code_postal']})"
                    details = f"   Prix: {row['prix_med']/1000:.0f}k€ | Surface: {row['surf_med']:.0f}m² | Rendement: {row['rdt_med']:.2f}% | {int(row['nb'])} biens"
                    
                    tk.Label(reco_card, text=titre,
                            bg=self.colors['secondary'], fg=self.colors['accent'],
                            font=('Segoe UI', 10, 'bold')).pack(anchor='w', padx=15, pady=(8, 2))
                    tk.Label(reco_card, text=details,
                            bg=self.colors['secondary'], fg=self.colors['fg'],
                            font=('Consolas', 8)).pack(anchor='w', padx=15, pady=(0, 8))
            else:
                tk.Label(top_frame, text="⚠️ Aucun bien ne correspond à tous vos critères.\nAssouplissez les filtres.",
                        bg=self.colors['card'], fg=self.colors['text_muted'],
                        font=('Segoe UI', 10)).pack(pady=20)
        
        # === CONSEILS FINAUX ===
        conseils_frame = tk.Frame(scrollable, bg=self.colors['warning'], relief='flat')
        conseils_frame.pack(fill='x', padx=30, pady=15)
        
        tk.Label(conseils_frame, text="💡 CONSEILS PERSONNALISÉS",
                bg=self.colors['warning'], fg='white',
                font=('Segoe UI', 11, 'bold')).pack(anchor='w', padx=15, pady=(10, 5))
        
        conseils = [
            f"✓ Ciblez zones universitaires pour {PERSONA['cible_locataire']}",
            f"✓ Privilégiez proximité RER/métro (rotation locataires)",
            f"✓ Prévoyez {self.w_charges.get():.0f}% charges (vacance/travaux/gestion)",
            f"✓ Demande forte pour {PERSONA['surface_min']}-{PERSONA['surface_max']}m² auprès cible",
            "✓ Visitez 3-5 biens top avant de décider",
            "✓ Vérifiez copropriété (charges, travaux prévus)",
            "✓ Consultez plans Grand Paris Express (valorisation future)"
        ]
        
        for conseil in conseils:
            tk.Label(conseils_frame, text=conseil,
                    bg=self.colors['warning'], fg='white',
                    font=('Segoe UI', 9), justify='left').pack(anchor='w', padx=15, pady=2)
        
        tk.Label(conseils_frame, text="",
                bg=self.colors['warning']).pack(pady=5)
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')


def main():
    """Point d'entrée"""
    root = tk.Tk()
    
    # Centrer fenêtre
    root.update_idletasks()
    width = 1700
    height = 950
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    app = DashboardAppPro(root)
    root.mainloop()


if __name__ == "__main__":
    main()