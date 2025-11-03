
import os, glob, re
from pathlib import Path
import pandas as pd
import numpy as np

import ipywidgets as W
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML

# ----------------------- THEME -----------------------
THEME_CSS_LIGHT = """
<style>
:root {
  --bg:#f9fafb;
  --card:#ffffff;
  --border:#e5e7eb;
  --text:#111827;
  --accent:#2563eb;
  --muted:#6b7280;
}
div.app {
  background: var(--bg);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica Neue, Arial;
  color: var(--text);
}
.app h2 { color: var(--accent); margin: 0 0 6px 0; font-weight: 700; }
.app p  { color: var(--muted); margin: 0 0 12px 0; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 12px; }
.kpi-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.kpi { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px; text-align: center; }
.kpi .label { color: var(--muted); font-size: 12px; }
.kpi .value { font-size: 22px; font-weight: 700; color: var(--accent); }
.tbltitle { font-weight: 600; margin: 8px 0 6px 0; }
.note { color: var(--muted); font-size: 12px; margin-top: 6px; }
.btnbar { display: flex; gap: 8px; flex-wrap: wrap; margin: 6px 0 8px 0; }
</style>
"""

THEME_CSS_DARK = """
<style>
:root{
  --bg:#0f172a;
  --panel:#111827;
  --card:#131a2a;
  --muted:#9ca3af;
  --text:#e5e7eb;
  --border: rgba(255,255,255,.06);
  --accent:#22d3ee;
}
div.app{ background:linear-gradient(135deg,#0f172a 0%,#0b1220 100%); color:var(--text); font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,Helvetica Neue,Arial;}
.app h2{ color:var(--accent); margin:0 0 6px 0; font-weight:700;}
.app p{ color:var(--muted); margin:0 0 12px 0;}
.card{ background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:12px;}
.kpi-grid{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px;}
.kpi{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:12px; text-align:center;}
.kpi .label{ color:var(--muted); font-size:12px;}
.kpi .value{ font-size:22px; font-weight:700; color:var(--accent);}
.tbltitle{ font-weight:600; margin:8px 0 6px 0;}
.note{ color:var(--muted); font-size:12px; margin-top:6px;}
.btnbar { display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 8px 0;}
</style>
"""

def inject_theme(theme='light', title="Analyse DVF — Île-de-France", subtitle="Repérage des meilleures zones d'investissement"):
    css = THEME_CSS_LIGHT if theme=='light' else THEME_CSS_DARK
    display(HTML(css + f'<div class="app"><h2>{title}</h2><p>{subtitle}</p></div>'))

def kpi_card(label, value_html):
    return f'<div class="kpi"><div class="label">{label}</div><div class="value">{value_html}</div></div>'

def fmt_int(x):
    try: return f"{int(x):,}".replace(",", " ")
    except: return "–"

# ----------------------- DATA -----------------------
def autodetect_clean_fp(clean_dir):
    default_fp = os.path.join(clean_dir, "dvf_clean.parquet")
    if os.path.exists(default_fp): return default_fp
    candidates = sorted(
        glob.glob(os.path.join(clean_dir, "*.parquet")),
        key=os.path.getmtime, reverse=True
    )
    dvf = [p for p in candidates if "dvf" in os.path.basename(p).lower()]
    if dvf: return dvf[0]
    if candidates: return candidates[0]
    raise FileNotFoundError("Aucun parquet trouvé dans le dossier clean/.")

def load_df(clean_fp=None, clean_dir=None):
    if clean_fp is None:
        if clean_dir is None:
            raise ValueError("Fournir clean_fp ou clean_dir à load_df.")
        clean_fp = autodetect_clean_fp(clean_dir)
    df = pd.read_parquet(clean_fp).copy()
    for c in ["prix_m2","surface_reelle_bati","valeur_fonciere","annee","nom_commune","code_postal"]:
        if c not in df.columns: df[c] = np.nan
    return df

# ----------------------- FILTERS -----------------------
def iqr_bounds(s, k=2.0):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k*iqr, q3 + k*iqr

def apply_filters(d, widgets):
    d = d.copy()
    smin, smax = widgets["w_surface"].value
    d = d[d["surface_reelle_bati"].between(smin, smax)]
    if isinstance(widgets["w_year"], W.SelectionRangeSlider):
        y0, y1 = widgets["w_year"].value
        d = d[(d["annee"]>=y0) & (d["annee"]<=y1)]
    if widgets["w_commune"].value != "(Toutes)":
        d = d[d["nom_commune"] == widgets["w_commune"].value]
    if widgets["w_iqr"].value and len(d) >= 50 and d["prix_m2"].notna().any():
        lo, hi = iqr_bounds(d["prix_m2"], k=2.0)
        d = d[d["prix_m2"].between(lo, hi)]
    loyer_m2 = widgets["w_loyer"].value
    d = d.assign(revenu_annuel = loyer_m2 * d["surface_reelle_bati"] * 12.0)
    d = d.assign(yield_brut    = d["revenu_annuel"] / d["valeur_fonciere"])
    return d

# ----------------------- CONTROLS -----------------------
def build_controls(df, topn=10):
    w_surface = W.IntRangeSlider(value=(20, 80), min=10, max=200, step=1,
                                 description="Surface (m²)", continuous_update=False,
                                 style={'description_width':'110px'}, layout=W.Layout(width='100%'))
    w_loyer = W.FloatSlider(value=22.0, min=5.0, max=45.0, step=0.5,
                            readout_format=".1f", description="Loyer €/m²",
                            continuous_update=False, style={'description_width':'110px'},
                            layout=W.Layout(width='100%'))
    w_topn = W.IntSlider(value=topn, min=5, max=30, step=1, description="Top N",
                         continuous_update=False, style={'description_width':'110px'},
                         layout=W.Layout(width='100%'))
    w_iqr = W.ToggleButtons(options=[('IQR on', True), ('IQR off', False)],
                            value=True, description="Outliers", style={'description_width':'110px'})
    w_hist = W.Checkbox(value=True, description="Histogramme prix/m²")

    w_year = (W.SelectionRangeSlider(
                options=sorted(df["annee"].dropna().astype(int).unique()),
                index=(0, max(0, len(df["annee"].dropna().unique())-1)),
                description="Années", continuous_update=False,
                layout=W.Layout(width='100%'), style={'description_width':'110px'}
              ) if df["annee"].notna().any() else W.Label("Années : n/a"))

    communes = ["(Toutes)"] + sorted(df["nom_commune"].dropna().astype(str).unique().tolist())
    w_commune = W.Dropdown(options=communes, value="(Toutes)", description="Commune",
                           layout=W.Layout(width='100%'), style={'description_width':'110px'})

    # Presets
    btn_studio = W.Button(description="Studios (15–30 m²)")
    btn_t2     = W.Button(description="T2 (30–45 m²)")
    btn_t3     = W.Button(description="T3 (45–65 m²)")
    def _set(r): w_surface.value = r
    btn_studio.on_click(lambda _: _set((15,30)))
    btn_t2.on_click(lambda _: _set((30,45)))
    btn_t3.on_click(lambda _: _set((45,65)))
    bar_presets = W.HBox([btn_studio, btn_t2, btn_t3], layout=W.Layout(justify_content="flex-start"))

    w_btn_export = W.Button(description="Exporter CSV", tooltip="Exporte le dataset filtré (clean/).")
    w_msg = W.HTML("")

    widgets = dict(
        w_surface=w_surface, w_loyer=w_loyer, w_topn=w_topn, w_iqr=w_iqr, w_hist=w_hist,
        w_year=w_year, w_commune=w_commune, bar_presets=bar_presets,
        w_btn_export=w_btn_export, w_msg=w_msg
    )
    return widgets

# ----------------------- RENDER -----------------------
def _render_overview(df, widgets, out_overview):
    with out_overview:
        clear_output(wait=True)
        d = apply_filters(df, widgets)
        nb  = len(d)
        p50 = float(d["prix_m2"].median()) if d["prix_m2"].notna().any() else np.nan
        ymd = float(d["yield_brut"].median()) if d["yield_brut"].notna().any() else np.nan
        s50 = float(d["surface_reelle_bati"].median()) if d["surface_reelle_bati"].notna().any() else np.nan
        kpis = ''.join([
            kpi_card("Transactions", fmt_int(nb)),
            kpi_card("Prix/m² médiane", f"{p50:,.0f} €".replace(",", " ") if p50==p50 else "–"),
            kpi_card("Rendement brut médian", f"{ymd*100:,.1f} %".replace(",", " ") if ymd==ymd else "–"),
        ])
        display(HTML(f'<div class="card"><div class="kpi-grid">{kpis}</div>'
                     f'<div class="note">Réglez les filtres à gauche. Export CSV disponible.</div></div>'))
        if nb:
            attract = "intéressant" if (ymd or 0) > 0.055 else "modéré"
            insight = (
                f"<p>Sur cette sélection (médiane ~{int(s50) if s50==s50 else '–'} m²), "
                f"le prix médian ~<b>{int(p50):,} €/m²</b> et le rendement brut médian "
                f"<b>{(ymd*100):.1f}%</b> — {attract} pour un premier investissement.</p>"
            ).replace(",", " ")
            display(HTML(f'<div class="card">{insight}</div>'))

def _render_table(df, widgets, out_table):
    with out_table:
        clear_output(wait=True)
        d = apply_filters(df, widgets)
        if len(d):
            top = (d.groupby(["nom_commune","code_postal"], as_index=False)["yield_brut"]
                     .median().sort_values("yield_brut", ascending=False).head(widgets["w_topn"].value))
            top["yield_brut_%"] = (top["yield_brut"]*100).round(2)
            to_show = top[["nom_commune","code_postal","yield_brut_%"]].rename(
                columns={"nom_commune":"Commune","code_postal":"CP","yield_brut_%":"Yield brut (%)"}
            )
            html = to_show.style.hide(axis="index") \
                               .set_table_styles([
                                   {'selector':'th','props':'text-align:left; padding:6px 8px;'},
                                   {'selector':'td','props':'padding:6px 8px;'}
                               ]) \
                               .bar(subset=["Yield brut (%)"],
                                    vmin=to_show["Yield brut (%)"].min(),
                                    vmax=to_show["Yield brut (%)"].max()) \
                               .to_html()
            display(HTML('<div class="tbltitle">Top communes (médiane du rendement)</div>'))
            display(HTML(f'<div class="card">{html}</div>'))
        else:
            display(HTML('<div class="card">Aucune donnée avec ces filtres.</div>'))

def _render_plots(df, widgets, out_plot1, out_plot2):
    try:
        import plotly.express as px
        with out_plot1:
            clear_output(wait=True)
            d = apply_filters(df, widgets)
            if len(d):
                ds = d.sample(min(4000, len(d)), random_state=0)
                fig = px.scatter(ds, x="surface_reelle_bati", y="prix_m2",
                                 hover_data=["nom_commune","code_postal","annee"],
                                 trendline="lowess", height=420, template="plotly_white")
                fig.update_layout(margin=dict(l=10,r=10,t=10,b=10))
                fig.update_xaxes(title="Surface (m²)"); fig.update_yaxes(title="Prix/m² (€)")
                fig.show()
            else:
                display(HTML('<div class="card">—</div>'))
        with out_plot2:
            clear_output(wait=True)
            d = apply_filters(df, widgets)
            if len(d) and widgets["w_hist"].value and d["prix_m2"].notna().any():
                cut = d["prix_m2"].clip(upper=d["prix_m2"].quantile(0.99))
                fig = px.histogram(cut, x=cut, nbins=45, height=380, template="plotly_white")
                fig.update_layout(margin=dict(l=10,r=10,t=10,b=10))
                fig.update_xaxes(title="Prix/m² (€)"); fig.update_yaxes(title="Fréquence")
                fig.show()
            else:
                display(HTML('<div class="card">—</div>'))
    except Exception:
        with out_plot1:
            clear_output(wait=True)
            d = apply_filters(df, widgets)
            if len(d):
                ds = d.sample(min(4000, len(d)), random_state=0)
                plt.figure(figsize=(7,4.4))
                plt.scatter(ds["surface_reelle_bati"], ds["prix_m2"], s=12, alpha=.6)
                plt.title("Prix/m² vs Surface (échantillon)")
                plt.xlabel("Surface (m²)"); plt.ylabel("Prix/m² (€)")
                plt.grid(alpha=.2); plt.tight_layout(); plt.show()
            else:
                display(HTML('<div class="card">—</div>'))
        with out_plot2:
            clear_output(wait=True)
            d = apply_filters(df, widgets)
            if len(d) and widgets["w_hist"].value and d["prix_m2"].notna().any():
                cut = d["prix_m2"].clip(upper=d["prix_m2"].quantile(0.99))
                plt.figure(figsize=(7,4))
                plt.hist(cut, bins=45)
                plt.title("Distribution du prix/m² (99e centile)")
                plt.xlabel("Prix/m² (€)"); plt.ylabel("Fréquence")
                plt.grid(alpha=.2); plt.tight_layout(); plt.show()
            else:
                display(HTML('<div class="card">—</div>'))

def boot_dashboard(df, theme='light', topn=10, title="Analyse DVF — Île-de-France", subtitle="Repérage simple des meilleures zones d'investissement"):
    inject_theme(theme=theme, title=title, subtitle=subtitle)

    widgets = build_controls(df, topn=topn)
    # panels
    out_overview = W.Output()
    out_table    = W.Output()
    out_plot1    = W.Output()
    out_plot2    = W.Output()

    # layout
    controls_panel = W.VBox([
        W.HTML('<div class="card"><b>Paramètres d’analyse</b></div>'),
        W.VBox([
            widgets["bar_presets"],
            widgets["w_surface"], widgets["w_loyer"], widgets["w_topn"], widgets["w_iqr"],
            widgets["w_hist"], widgets["w_year"], widgets["w_commune"],
            W.HBox([widgets["w_btn_export"], widgets["w_msg"]])
        ], layout=W.Layout(padding="0 6px 8px 6px"))
    ])

    tabs = W.Tab(children=[
        W.VBox([out_overview]),
        W.VBox([out_table]),
        W.VBox([out_plot1]),
        W.VBox([out_plot2]),
    ])
    tabs.set_title(0, "Vue d’ensemble")
    tabs.set_title(1, "Communes")
    tabs.set_title(2, "Dispersion")
    tabs.set_title(3, "Distribution")

    display(W.HBox([controls_panel, tabs], layout=W.Layout(width="100%")))

    # observers
    def render(_=None):
        _render_overview(df, widgets, out_overview)
        _render_table(df, widgets, out_table)
        _render_plots(df, widgets, out_plot1, out_plot2)

    for w in ("w_surface","w_loyer","w_topn","w_iqr","w_hist","w_commune"):
        widgets[w].observe(render, "value")
    if isinstance(widgets["w_year"], W.SelectionRangeSlider):
        widgets["w_year"].observe(render, "value")

    def on_export_clicked(_):
        d = apply_filters(df, widgets)
        clean_dir = os.path.join(os.path.abspath(os.path.join("..")), "data", "clean")
        Path(clean_dir).mkdir(parents=True, exist_ok=True)
        out_fp = os.path.join(clean_dir, "dvf_filtre_export.csv")
        d.to_csv(out_fp, index=False)
        widgets["w_msg"].value = f'<span style="color:#2563eb">Exporté : {out_fp}</span>'
    widgets["w_btn_export"].on_click(on_export_clicked)

    # initial render
    render()

    # return handle
    return dict(widgets=widgets, tabs=tabs,
                outs=dict(overview=out_overview, table=out_table, plot1=out_plot1, plot2=out_plot2),
                render=render)

def get_controls(boot): return boot["widgets"]

def add_tab(boot, title, output_widget=None):
    if output_widget is None:
        output_widget = W.Output()
    tabs = boot["tabs"]
    children = list(tabs.children) + [W.VBox([output_widget])]
    tabs.children = tuple(children)
    tabs.set_title(len(children)-1, title)
    return output_widget
