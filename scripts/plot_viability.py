"""Gera os SVGs de docs/09-analise-graficos.md a partir das saidas do sim_viability.

    .venv/bin/python scripts/plot_viability.py --in /tmp/viab --out docs/diagramas

Le os .npz (series temporais) e .json (resumos) produzidos por
scripts/sim_viability.py e emite um SVG por figura. Sem dependencia de estado:
cada figura declara o que exige e e pulada com aviso se os dados faltarem.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# paleta: uma cor por "modelo de atuador", estavel entre figuras
C = {
    "urdf":       "#1d4e6b",
    "derated":    "#0e7a5f",
    "catalog":    "#9a6b0a",
    "catalog_tn": "#a8301f",
    "zero":       "#a8301f",
    "grid":       "#d5dce4",
    "ink":        "#171c23",
    "muted":      "#5a6875",
}
LABEL = {
    "urdf": "teto URDF (treino)",
    "derated": "teto derated (firmware)",
    "catalog": "pico do catalogo",
    "catalog_tn": "catalogo + curva T-N",
    "urdf_tn": "URDF + T-N",
    "derated_tn": "derated + T-N",
}


def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor("white")
    ax.grid(True, color=C["grid"], linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["grid"])
    ax.tick_params(colors=C["muted"], labelsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=C["ink"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=C["ink"])
    if title:
        ax.set_title(title, fontsize=10.5, color=C["ink"], loc="left", pad=10)


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


# ---------------------------------------------------------------- figura 1
def fig_torque_vs_catalog(d, out_dir):
    """Demanda de torque por junta contra os tres tetos."""
    names = [str(n) for n in d["joint_names"]]
    peak = np.abs(d["tau"]).max(axis=0)
    leg = [i for i, n in enumerate(names) if any(
        k in n for k in ("Hip", "Knee", "Ankle", "Waist"))]
    y = np.arange(len(leg))

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.barh(y, peak[leg], height=0.62, color=C["urdf"], alpha=0.85,
            label="demanda medida (pico)")
    ax.scatter(d["effort_used"][leg], y, marker="|", s=260, linewidths=2.2,
               color=C["muted"], label=LABEL["urdf"], zorder=4)
    ax.scatter(d["derated"][leg], y, marker="|", s=260, linewidths=2.2,
               color=C["derated"], label=LABEL["derated"], zorder=5)
    ax.scatter(d["catalog_peak_tau"][leg], y, marker="|", s=260, linewidths=2.6,
               color=C["catalog"], label=LABEL["catalog"], zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i].replace("_", " ") for i in leg], fontsize=8)
    ax.invert_yaxis()
    _style(ax, "torque [Nm]", "",
           "Fig. 1 — demanda de torque vs. os tres tetos (vx = 1,0 m/s)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _save(fig, out_dir, "viab-01-torque-vs-catalogo.svg")


# ---------------------------------------------------------------- figura 2
def fig_speed_vs_catalog(d, out_dir):
    """Velocidade de junta contra rated e peak do catalogo."""
    names = [str(n) for n in d["joint_names"]]
    vpk = np.abs(d["dof_vel"]).max(axis=0)
    leg = [i for i, n in enumerate(names) if any(
        k in n for k in ("Hip", "Knee", "Ankle", "Waist"))]
    ratio = vpk[leg] / d["catalog_peak_vel"][leg]
    order = np.argsort(-ratio)
    idx = [leg[i] for i in order]
    y = np.arange(len(idx))

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    cols = [C["zero"] if vpk[i] > d["catalog_peak_vel"][i] else C["urdf"] for i in idx]
    ax.barh(y, vpk[idx], height=0.62, color=cols, alpha=0.85,
            label="velocidade medida (pico)")
    ax.scatter(d["catalog_rated_vel"][idx], y, marker="|", s=260, linewidths=2.2,
               color=C["derated"], label="rated speed (torque cheio ate aqui)", zorder=5)
    ax.scatter(d["catalog_peak_vel"][idx], y, marker="|", s=260, linewidths=2.6,
               color=C["catalog"], label="peak speed (torque = 0)", zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i].replace("_", " ") for i in idx], fontsize=8)
    ax.invert_yaxis()
    _style(ax, "velocidade de junta [rad/s]", "",
           "Fig. 2 — velocidade de junta vs. envelope do catalogo (vx = 1,0 m/s)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _save(fig, out_dir, "viab-02-velocidade-vs-catalogo.svg")


# ---------------------------------------------------------------- figura 3
def fig_observability(in_dir, out_dir):
    """A comparacao de variavel unica: 60 s vs ~1 s."""
    pairs = [("m_urdf_vx1.0.npz", "estado verdadeiro", C["urdf"]),
             ("z_urdf_vx1.0.npz", "root_h e root_vel zerados", C["zero"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for path, lab, col in pairs:
        f = os.path.join(in_dir, path)
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True)
        axes[0].plot(d["t"], d["root_h"], color=col, linewidth=1.5, label=lab)
        axes[1].plot(d["t"], d["speed"], color=col, linewidth=1.5, label=lab)
    axes[1].axhline(1.0, color=C["muted"], linestyle="--", linewidth=1.0,
                    label="comando (1,0 m/s)")
    _style(axes[0], "tempo [s]", "altura do tronco [m]",
           "Fig. 3a — altura do tronco")
    _style(axes[1], "tempo [s]", "velocidade horizontal [m/s]",
           "Fig. 3b — velocidade vs. comando")
    axes[0].set_xlim(0, 8); axes[1].set_xlim(0, 8)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].legend(fontsize=8, frameon=False)
    _save(fig, out_dir, "viab-03-observabilidade.svg")


# ---------------------------------------------------------------- figura 4
def fig_sweep(in_dir, out_dir):
    """Varredura de velocidade: sobrevivencia e rastreamento."""
    series = {}
    for f in sorted(glob.glob(os.path.join(in_dir, "s_*.json"))):
        j = json.load(open(f))
        key = "catalog_tn" if j["tn"] else j["effort"]
        series.setdefault(key, []).append(j)
    if not series:
        print("  (sem dados de varredura)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for key, rows in sorted(series.items()):
        rows.sort(key=lambda r: r["vx"])
        vx = [r["vx"] for r in rows]
        surv = [r["elapsed_s"] for r in rows]
        trk = [r["tracking_pct"] for r in rows]
        col = C.get(key, C["muted"])
        axes[0].plot(vx, surv, "o-", color=col, linewidth=1.6, markersize=4,
                     label=LABEL.get(key, key))
        axes[1].plot(vx, trk, "o-", color=col, linewidth=1.6, markersize=4,
                     label=LABEL.get(key, key))
        for r in rows:
            if r["fell"]:
                axes[0].plot(r["vx"], r["elapsed_s"], "x", color=col,
                             markersize=9, markeredgewidth=2)
    axes[0].axhline(60, color=C["muted"], linestyle="--", linewidth=1.0)
    axes[0].text(0.32, 57.5, "orcamento de 60 s (x = caiu)", fontsize=7.5,
                 color=C["muted"])
    axes[1].axhline(100, color=C["muted"], linestyle="--", linewidth=1.0)
    _style(axes[0], "vx comandado [m/s]", "sobrevivencia [s]",
           "Fig. 4a — sobrevivencia vs. velocidade")
    _style(axes[1], "vx comandado [m/s]", "rastreamento [%]",
           "Fig. 4b — rastreamento vs. velocidade")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    axes[1].legend(fontsize=8, frameon=False)
    _save(fig, out_dir, "viab-04-varredura-velocidade.svg")


# ---------------------------------------------------------------- figura 5
def fig_actuator_models(in_dir, out_dir):
    """Sobrevivencia por modelo de atuador, nas duas velocidades."""
    models, vxs = [], ["0.5", "1.0"]
    for eff in ("urdf", "derated", "catalog"):
        for suf in ("", "_tn"):
            key = eff + suf
            got = [os.path.join(in_dir, f"m_{key}_vx{v}.json") for v in vxs]
            if all(os.path.exists(g) for g in got):
                models.append((key, [json.load(open(g)) for g in got]))
    if not models:
        print("  (sem dados de modelos)")
        return

    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for k, (v, hatch) in enumerate(zip(vxs, (None, "///"))):
        vals = [m[1][k]["elapsed_s"] for m in models]
        fell = [m[1][k]["fell"] for m in models]
        cols = [C["zero"] if f else C["derated"] for f in fell]
        ax.bar(x + (k - 0.5) * w, vals, w, color=cols, alpha=0.88,
               hatch=hatch, edgecolor="white", linewidth=0.8,
               label=f"vx = {v} m/s")
        for xi, val, f in zip(x + (k - 0.5) * w, vals, fell):
            ax.text(xi, val + 1.0, f"{val:.0f}" + ("" if not f else " ✕"),
                    ha="center", fontsize=7.5, color=C["ink"])
    ax.axhline(60, color=C["muted"], linestyle="--", linewidth=1.0)
    short = {"urdf": "URDF\n(treino)", "urdf_tn": "URDF\n+ T-N",
             "derated": "derated\n(firmware)", "derated_tn": "derated\n+ T-N",
             "catalog": "pico do\ncatalogo", "catalog_tn": "catalogo\n+ T-N"}
    ax.set_xticks(x)
    ax.set_xticklabels([short.get(m[0], m[0]) for m in models], fontsize=8.5)
    ax.set_ylim(0, 78)
    _style(ax, "", "sobrevivencia [s]",
           "Fig. 5 — sobrevivencia por modelo de atuador (verde = 60 s, vermelho = caiu)")
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.02))
    _save(fig, out_dir, "viab-05-modelos-atuador.svg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="/tmp/viab")
    ap.add_argument("--out", dest="out_dir", default="docs/diagramas")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    print("gerando SVGs:")

    base = os.path.join(a.in_dir, "m_urdf_vx1.0.npz")
    if os.path.exists(base):
        d = np.load(base, allow_pickle=True)
        fig_torque_vs_catalog(d, a.out_dir)
        fig_speed_vs_catalog(d, a.out_dir)
    else:
        print(f"  (falta {base}; figuras 1-2 puladas)")
    fig_observability(a.in_dir, a.out_dir)
    fig_sweep(a.in_dir, a.out_dir)
    fig_actuator_models(a.in_dir, a.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
