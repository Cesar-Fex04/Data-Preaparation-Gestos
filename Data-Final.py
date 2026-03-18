"""
=============================================================
  DATA PREPARATION - RECONOCIMIENTO DE GESTOS EN TABLETA
  Inteligencia Artificial  —  Script FINAL (limpieza DTW)
=============================================================
  NOVEDADES vs versión anterior:
  • PASO 3.5 — DETECCIÓN DE MUESTRAS BASURA con DTW
    (Dynamic Time Warping — tolera variaciones de velocidad
     y número de puntos, compara la FORMA completa)

    Estrategia en 2 rondas:
    ─────────────────────────────────────────────────────
    RONDA 1 — Filtro rápido de ángulo global:
      Descarta muestras cuyo ángulo global se aleja más de
      ANGULO_TOLERANCIA_DEG del ángulo esperado del gesto.
      Elimina gestos completamente invertidos/erróneos.

    RONDA 2 — Filtro DTW (forma completa):
      1. Normaliza cada trayectoria (interp → N puntos,
         centrar en origen, escalar por longitud total).
      2. Calcula la mediana punto a punto de las muestras
         que pasaron la ronda 1 → trayectoria de referencia.
      3. Computa distancia DTW 2D de cada muestra vs ref.
      4. Descarta muestras con dist > media + K_STD * std.
         (K_STD=2.0 conservador; bajar a 1.5 si persiste ruido)
    ─────────────────────────────────────────────────────
    Todas las eliminadas quedan en basura_detectada.csv
    con su motivo y distancia DTW.

  FEATURES (42 por muestra):
  • Spline cúbica a N=20 → 19 Δx + 19 Δy + 4 globales

  INSTALACIÓN REQUERIDA:
    pip install dtaidistance

  USO:
    python data_final.py

  SALIDA:
  • dataset_gestos_final.csv   ← PRODUCTO PRINCIPAL
  • basura_detectada.csv       ← registro de eliminadas
  • resultados_cv_final.csv    ← precisión por fold/clase
  • graficas_final/            ← 11 imágenes (01-11)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from collections import defaultdict
from scipy.interpolate import splprep, splev, interp1d
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

try:
    from dtaidistance import dtw as dtw_lib
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("  [AVISO] dtaidistance no encontrado.")
    print("          Instala con: pip install dtaidistance")
    print("          Se usara distancia euclidiana como fallback.\n")

# ─── CONFIGURACIÓN ────────────────────────────────────────────────
DATA_DIR     = Path("users_01_to_10")
OUTPUT_CSV   = "dataset_gestos_final.csv"
BASURA_CSV   = "basura_detectada.csv"
RESULTS_CSV  = "resultados_cv_final.csv"
GRAFICAS_DIR = Path("graficas_final")
N_RESAMPLE   = 20       # puntos spline para features
N_DTW        = 50       # puntos para comparacion DTW
N_FOLDS      = 5
VAL_SIZE     = 0.20
RANDOM_STATE = 42
N_TREES      = 100

# Ronda 1: tolerancia de angulo global (grados) - amplia, solo descarta obvios
ANGULO_TOLERANCIA_DEG = 70

# Ronda 2: cuantas desviaciones estandar se permite sobre la media DTW
# 2.0 = conservador (solo outliers claros)
# 1.5 = mas agresivo (si aun queda ruido, bajar a 1.5)
K_STD = 2.0

# Angulo esperado por gesto (grados, arctan2(dy, dx))
GESTO_ANGULO_IDEAL = {
    1: -135,   # Diagonal NW
    2:  -45,   # Diagonal NE
    3:  135,   # Diagonal SW
    4:   45,   # Diagonal SE
    5:  -90,   # Vertical arriba
    6:  180,   # Horizontal izquierda
    7:    0,   # Horizontal derecha
    8:   90,   # Vertical abajo
}

DIRECCIONES = {
    1: "Diagonal NW", 2: "Diagonal NE",
    3: "Diagonal SW", 4: "Diagonal SE",
    5: "Vertical U",  6: "Horizontal L",
    7: "Horizontal R", 8: "Vertical D"
}

GRAFICAS_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════
#  UTILIDADES DTW
# ════════════════════════════════════════════════════════════════

def resample_traj(x, y, n_points=N_DTW):
    t     = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, n_points)
    fx    = interp1d(t, x, kind="linear")
    fy    = interp1d(t, y, kind="linear")
    return np.column_stack([fx(t_new), fy(t_new)])

def normalize_traj(traj):
    """Centra en origen y escala por longitud total."""
    traj = traj - traj[0]
    total_len = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    if total_len > 1e-9:
        traj = traj / total_len
    return traj

def dtw_distance_2d(a, b):
    if DTW_AVAILABLE:
        dx = dtw_lib.distance(a[:, 0].astype(np.float64),
                              b[:, 0].astype(np.float64))
        dy = dtw_lib.distance(a[:, 1].astype(np.float64),
                              b[:, 1].astype(np.float64))
        return float(np.sqrt(dx**2 + dy**2))
    else:
        # fallback euclidiano punto a punto
        return float(np.sqrt(np.sum((a - b)**2)))

def compute_reference(trajs_norm):
    """Mediana punto a punto - robusta ante outliers residuales."""
    stacked = np.stack(trajs_norm, axis=0)
    return np.median(stacked, axis=0)

def angulo_diff(a_deg, b_deg):
    d = (a_deg - b_deg + 180) % 360 - 180
    return abs(d)


# ════════════════════════════════════════════════════════════════
#  PASO 1 — CARGA DE DATOS
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("PASO 1: Cargando archivos...")
print("=" * 65)

records_raw = []
for user_dir in sorted(DATA_DIR.iterdir()):
    if not user_dir.is_dir():
        continue
    user_id = int(user_dir.name.split("_")[1])
    for csv_file in sorted(user_dir.glob("*.csv")):
        parts      = csv_file.stem.split("_")
        gesture_id = int(parts[1])
        sample_id  = int(parts[3])
        df         = pd.read_csv(csv_file)
        records_raw.append({
            "user_id"   : user_id,
            "gesture_id": gesture_id,
            "sample_id" : sample_id,
            "file"      : str(csv_file),
            "n_points"  : len(df),
            "df"        : df
        })

print(f"  Total archivos: {len(records_raw)}")
df_inv = pd.DataFrame([{k: v for k, v in r.items() if k != "df"}
                        for r in records_raw])
pivot = df_inv.pivot_table(
    index="user_id", columns="gesture_id",
    values="sample_id", aggfunc="count", fill_value=0)
pivot.columns = [f"G{c}" for c in pivot.columns]
print("\n  Muestras por usuario/gesto (original):")
print(pivot.to_string())


# ════════════════════════════════════════════════════════════════
#  PASO 2 — LONGITUDES
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 2: Longitudes...")
print("=" * 65)
len_stats = df_inv.groupby("gesture_id")["n_points"].agg(["min","max","mean"]).round(1)
len_stats.index = [f"G{i}" for i in len_stats.index]
print(len_stats.to_string())
promedio_global = df_inv["n_points"].mean()
print(f"\n  Promedio global: {promedio_global:.1f} pts  |  N_RESAMPLE={N_RESAMPLE}  |  N_DTW={N_DTW}")


# ════════════════════════════════════════════════════════════════
#  PASO 3 — DUPLICADOS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 3: Duplicados...")
print("=" * 65)
seen_hashes = {}
indices_dup = set()
for i, r in enumerate(records_raw):
    key = tuple(r["df"]["x"].round(4).tolist()) + tuple(r["df"]["y"].round(4).tolist())
    h   = hash(key)
    if h in seen_hashes:
        indices_dup.add(i)
    else:
        seen_hashes[h] = i
print(f"  Duplicados: {len(indices_dup)}")


# ════════════════════════════════════════════════════════════════
#  PASO 3.5 — DETECCION DE BASURA (DTW 2 rondas)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 3.5: Limpieza con DTW (2 rondas)...")
print("=" * 65)

indices_basura = {}
basura_rows    = []

por_gesto = defaultdict(list)
for i, r in enumerate(records_raw):
    if i in indices_dup or r["n_points"] < 3:
        continue
    por_gesto[r["gesture_id"]].append(i)

for gid in range(1, 9):
    indices_g = por_gesto[gid]
    ideal     = GESTO_ANGULO_IDEAL[gid]
    print(f"\n  G{gid} ({DIRECCIONES[gid]}) — {len(indices_g)} muestras candidatas")

    # --- RONDA 1: angulo global ---
    candidatos_r2 = []
    for i in indices_g:
        r    = records_raw[i]
        x    = r["df"]["x"].values.astype(float)
        y    = r["df"]["y"].values.astype(float)
        dx_g = float(x[-1] - x[0])
        dy_g = float(y[-1] - y[0])
        ang  = float(np.degrees(np.arctan2(dy_g, dx_g)))
        diff = angulo_diff(ang, ideal)
        if gid == 6:
            diff = min(diff, angulo_diff(ang, -180))
        if diff > ANGULO_TOLERANCIA_DEG:
            motivo = (f"R1-angulo: {ang:.1f} vs ideal {ideal} "
                      f"(diff={diff:.1f} > {ANGULO_TOLERANCIA_DEG})")
            indices_basura[i] = motivo
        else:
            candidatos_r2.append(i)

    r1_elim = len(indices_g) - len(candidatos_r2)
    print(f"    R1 angulo:  -{r1_elim} eliminadas, {len(candidatos_r2)} pasan a R2")

    if len(candidatos_r2) < 3:
        print(f"    [!] Muy pocas muestras, omitiendo R2")
        continue

    # --- RONDA 2: DTW forma completa ---
    trajs_norm = []
    for i in candidatos_r2:
        r  = records_raw[i]
        x  = r["df"]["x"].values.astype(float)
        y  = r["df"]["y"].values.astype(float)
        tr = resample_traj(x, y, N_DTW)
        tr = normalize_traj(tr)
        trajs_norm.append(tr)

    referencia = compute_reference(trajs_norm)
    distancias = np.array([dtw_distance_2d(tr, referencia) for tr in trajs_norm])

    media_d = distancias.mean()
    std_d   = distancias.std()
    umbral  = media_d + K_STD * std_d
    print(f"    R2 DTW:     media={media_d:.4f}  std={std_d:.4f}  "
          f"umbral={umbral:.4f}  (K_STD={K_STD})")

    r2_elim = 0
    for i, dist in zip(candidatos_r2, distancias):
        if dist > umbral:
            motivo = (f"R2-DTW: dist={dist:.4f} > umbral={umbral:.4f}")
            indices_basura[i] = motivo
            r2_elim += 1

    print(f"    R2 DTW:     -{r2_elim} eliminadas, "
          f"{len(candidatos_r2)-r2_elim} validas")

# Consolidar
for idx, motivo in indices_basura.items():
    r = records_raw[idx]
    basura_rows.append({
        "user_id"   : r["user_id"],
        "gesture_id": r["gesture_id"],
        "sample_id" : r["sample_id"],
        "n_points"  : r["n_points"],
        "archivo"   : r["file"],
        "motivo"    : motivo
    })

df_basura = pd.DataFrame(basura_rows)
df_basura.to_csv(BASURA_CSV, index=False)

total_dup  = len(indices_dup)
total_bsra = len(indices_basura)
r1_total   = sum(1 for m in basura_rows if "R1" in m["motivo"])
r2_total   = sum(1 for m in basura_rows if "R2" in m["motivo"])

print(f"\n  {'─'*50}")
print(f"  Duplicados             : {total_dup}")
print(f"  Basura R1 (angulo)     : {r1_total}")
print(f"  Basura R2 (DTW)        : {r2_total}")
print(f"  Total eliminadas       : {total_dup + total_bsra}")
print(f"  Muestras validas       : {len(records_raw) - total_dup - total_bsra}")
print(f"  Registro: {BASURA_CSV}")


# ════════════════════════════════════════════════════════════════
#  PASO 4 — VISUALIZACIONES
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 4: Visualizaciones...")
print("=" * 65)

indices_excluir = indices_dup | set(indices_basura.keys())
idx_limpios = {g: [] for g in range(1, 9)}
idx_sucios  = {g: [] for g in range(1, 9)}
for i, r in enumerate(records_raw):
    g = r["gesture_id"]
    (idx_sucios if i in indices_excluir else idx_limpios)[g].append(i)

colores_u = cm.tab10(np.linspace(0, 1, 10))
colores_b = cm.Blues(np.linspace(0.35, 0.9, 5))

# 01 - Trayectorias user_01
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Gestos - user_01", fontsize=12, fontweight="bold")
for gidx, g in enumerate(range(1, 9)):
    ax    = axes[gidx // 4][gidx % 4]
    files = sorted((DATA_DIR / "user_01").glob(f"gesture_0{g}_sample_*.csv"))
    for k, f in enumerate(files[:5]):
        df = pd.read_csv(f)
        ax.plot(df["x"], df["y"], color=colores_b[k], alpha=0.8, linewidth=1.8)
        ax.plot(df["x"].iloc[0],  df["y"].iloc[0],  "go", markersize=5)
        ax.plot(df["x"].iloc[-1], df["y"].iloc[-1], "r^", markersize=5)
    ax.set_title(f"G{g} - {DIRECCIONES[g]}", fontweight="bold", fontsize=9)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3); ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(GRAFICAS_DIR / "01_trayectorias_user01.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Guardada: 01_trayectorias_user01.png")

# 02 - Antes (basura en rojo)
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("ANTES de limpieza  (rojo = eliminada)", fontsize=12, fontweight="bold")
for gidx, g in enumerate(range(1, 9)):
    ax = axes[gidx // 4][gidx % 4]
    for i, r in enumerate(records_raw):
        if r["gesture_id"] != g:
            continue
        es_b = i in indices_excluir
        ax.plot(r["df"]["x"], r["df"]["y"],
                color="red" if es_b else colores_u[r["user_id"]-1],
                alpha=0.7 if es_b else 0.20,
                linewidth=1.8 if es_b else 0.9,
                zorder=5 if es_b else 1)
    ax.set_title(f"G{g} - {DIRECCIONES[g]}", fontweight="bold", fontsize=8)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3); ax.set_aspect("equal")
axes[0][0].legend(handles=[
    Line2D([0],[0], color="gray", alpha=0.5, label="Valida"),
    Line2D([0],[0], color="red",  alpha=0.8, label="Eliminada")], fontsize=7)
plt.tight_layout()
plt.savefig(GRAFICAS_DIR / "02_antes_limpieza.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Guardada: 02_antes_limpieza.png")

# 03 - Despues (solo validas)
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("DESPUES de limpieza DTW (solo muestras validas)",
             fontsize=12, fontweight="bold")
for gidx, g in enumerate(range(1, 9)):
    ax = axes[gidx // 4][gidx % 4]
    for i in idx_limpios[g]:
        r = records_raw[i]
        ax.plot(r["df"]["x"], r["df"]["y"],
                color=colores_u[r["user_id"]-1], alpha=0.3, linewidth=0.9)
    ax.set_title(f"G{g} - {DIRECCIONES[g]}\n"
                 f"({len(idx_limpios[g])} validas, {len(idx_sucios[g])} eliminadas)",
                 fontweight="bold", fontsize=8)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3); ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(GRAFICAS_DIR / "03_despues_limpieza_DTW.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Guardada: 03_despues_limpieza_DTW.png")

# 04 - Validas vs eliminadas
fig, ax = plt.subplots(figsize=(11, 5))
x_pos   = np.arange(8)
n_valid = [len(idx_limpios[g]) for g in range(1, 9)]
n_elim  = [len(idx_sucios[g])  for g in range(1, 9)]
b1 = ax.bar(x_pos-0.2, n_valid, 0.38, label="Validas",    color="#2E75B6", alpha=0.85)
b2 = ax.bar(x_pos+0.2, n_elim,  0.38, label="Eliminadas", color="#C00000", alpha=0.85)
for b in b1:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
            str(int(b.get_height())), ha="center", fontsize=9)
for b in b2:
    if b.get_height() > 0:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                str(int(b.get_height())), ha="center", fontsize=9, color="#C00000")
ax.set_xticks(x_pos)
ax.set_xticklabels([f"G{g}\n{DIRECCIONES[g]}" for g in range(1,9)], fontsize=7.5)
ax.set_ylabel("N muestras")
ax.set_title("Muestras validas vs eliminadas (DTW)", fontweight="bold", fontsize=11)
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
ax.text(0.98, 0.97, f"Eliminadas: {sum(n_elim)}\nValidas: {sum(n_valid)}",
    ha="right", va="top", transform=ax.transAxes, fontsize=9,
    bbox=dict(boxstyle="round", facecolor="#FFF2CC", alpha=0.9))
plt.tight_layout()
plt.savefig(GRAFICAS_DIR / "04_validas_vs_eliminadas.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Guardada: 04_validas_vs_eliminadas.png")

# 05 - Distribucion final
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Distribucion dataset limpio", fontsize=13, fontweight="bold")
c_g = pd.Series({g: len(idx_limpios[g]) for g in range(1, 9)})
c_u = pd.Series({
    u: sum(1 for i in range(len(records_raw))
           if i not in indices_excluir and records_raw[i]["user_id"] == u)
    for u in range(1, 11)})
axes[0].bar([f"G{i}" for i in c_g.index], c_g.values, color="steelblue", edgecolor="white")
axes[0].set_title("Por Gesto"); axes[0].set_ylabel("Cantidad")
for i, v in enumerate(c_g.values):
    axes[0].text(i, v+0.5, str(v), ha="center", fontsize=9)
axes[1].bar([f"U{i}" for i in c_u.index], c_u.values, color="coral", edgecolor="white")
axes[1].set_title("Por Usuario"); axes[1].set_ylabel("Cantidad")
for i, v in enumerate(c_u.values):
    axes[1].text(i, v+0.5, str(v), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(GRAFICAS_DIR / "05_distribucion_limpia.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Guardada: 05_distribucion_limpia.png")

# 06 - Boxplot longitudes
fig, ax = plt.subplots(figsize=(12, 5))
data_box = [[records_raw[i]["n_points"] for i in idx_limpios[g]] for g in range(1, 9)]
ax.boxplot(data_box, tick_labels=[f"G{g}" for g in range(1,9)],
           patch_artist=True, boxprops=dict(facecolor="lightblue"))
ax.axhline(promedio_global, color="red",   linestyle="--", linewidth=1.5,
           label=f"Promedio global={promedio_global:.1f}")
ax.axhline(N_RESAMPLE,      color="green", linestyle="--", linewidth=1.5,
           label=f"N_RESAMPLE={N_RESAMPLE}")
ax.set_title("Longitud de trayectorias (limpio)", fontsize=11, fontweight="bold")
ax.set_xlabel("Gesto"); ax.set_ylabel("Puntos"); ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(GRAFICAS_DIR / "06_longitud_boxplot.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Guardada: 06_longitud_boxplot.png")


# ════════════════════════════════════════════════════════════════
#  PASO 5 — EXTRACCION DE FEATURES
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"PASO 5: Extraccion de features (N={N_RESAMPLE})...")
print("=" * 65)

def extract_features(df_sample, n_points=N_RESAMPLE):
    x = df_sample["x"].values.astype(float)
    y = df_sample["y"].values.astype(float)
    if len(x) >= 4:
        try:
            tck, u   = splprep([x, y], s=0, k=3)
            u_new    = np.linspace(0, 1, n_points)
            x_r, y_r = splev(u_new, tck)
        except Exception:
            t = np.linspace(0, 1, len(x)); t_n = np.linspace(0, 1, n_points)
            x_r = np.interp(t_n, t, x);   y_r  = np.interp(t_n, t, y)
    else:
        t = np.linspace(0, 1, len(x)); t_n = np.linspace(0, 1, n_points)
        x_r = np.interp(t_n, t, x);   y_r  = np.interp(t_n, t, y)
    dx_local = np.diff(x_r)
    dy_local = np.diff(y_r)
    dx_g   = float(x[-1] - x[0])
    dy_g   = float(y[-1] - y[0])
    angle  = float(np.arctan2(dy_g, dx_g))
    length = float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)))
    return list(dx_local) + list(dy_local) + [dx_g, dy_g, angle, length]

rows = []
for i, r in enumerate(records_raw):
    if i in indices_excluir:
        continue
    if r["n_points"] < 3:
        continue
    rows.append([r["user_id"], r["gesture_id"]] + extract_features(r["df"]))

N_DELTA   = N_RESAMPLE - 1
feat_cols = ([f"dx_{i}" for i in range(N_DELTA)] +
             [f"dy_{i}" for i in range(N_DELTA)] +
             ["dx_global", "dy_global", "angle", "total_length"])
df_full   = pd.DataFrame(rows, columns=["user_id", "gesture_label"] + feat_cols)

print(f"  Dataset limpio: {df_full.shape[0]} filas x {df_full.shape[1]} columnas")
print(f"  Muestras por gesto:")
print(df_full["gesture_label"].value_counts().sort_index().to_string())


# ════════════════════════════════════════════════════════════════
#  PASO 6 — DIVISION DE DATOS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 6: Division 80/20 + Cross Validation...")
print("=" * 65)

X = df_full[feat_cols].values
y = df_full["gesture_label"].values

indices_all       = np.arange(len(df_full))
idx_work, idx_val = train_test_split(
    indices_all, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y)
X_work_arr = X[idx_work]; y_work_arr = y[idx_work]
X_val      = X[idx_val];  y_val      = y[idx_val]

print(f"  Total: {len(df_full)}  |  80% trabajo: {len(idx_work)}  |  20% val: {len(idx_val)}")

skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_number = np.zeros(len(df_full), dtype=int)
for fold, (tr_idx, te_idx) in enumerate(skf.split(X_work_arr, y_work_arr), 1):
    fold_number[idx_work[te_idx]] = fold

df_full["fold"]  = fold_number
df_full["split"] = "validation"
df_full.loc[df_full["fold"] > 0, "split"] = "test"


# ════════════════════════════════════════════════════════════════
#  PASO 7 — CROSS VALIDATION
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"PASO 7: Cross Validation Random Forest ({N_TREES} arboles)...")
print("=" * 65)

gestos_list    = list(range(1, 9))
resultados     = []
fold_data_plot = []

print(f"\n  {'Fold':<6} {'Total':>8}   " + "   ".join([f"G{g}" for g in gestos_list]))
print("  " + "-" * 72)

for fold, (tr_idx, te_idx) in enumerate(skf.split(X_work_arr, y_work_arr), 1):
    X_tr = X_work_arr[tr_idx]; y_tr = y_work_arr[tr_idx]
    X_te = X_work_arr[te_idx]; y_te = y_work_arr[te_idx]
    clf  = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred    = clf.predict(X_te)
    acc_total = accuracy_score(y_te, y_pred)
    acc_g     = {g: accuracy_score(y_te[y_te==g], y_pred[y_te==g])
                 if (y_te==g).sum() > 0 else 0.0 for g in gestos_list}
    resultados.append({
        "fold": fold, "acc_total": acc_total,
        "n_train": len(y_tr), "n_test": len(y_te),
        **{f"acc_G{g}": acc_g[g] for g in gestos_list}
    })
    fold_data_plot.append({
        "fold": fold,
        "train_counts": [int((y_tr==g).sum()) for g in gestos_list],
        "test_counts" : [int((y_te==g).sum()) for g in gestos_list],
        "acc_total"   : acc_total,
        "acc_gestos"  : [acc_g[g] for g in gestos_list]
    })
    print(f"  Fold {fold}  {acc_total*100:>6.1f}%   " +
          "   ".join([f"{acc_g[g]*100:>4.0f}%" for g in gestos_list]))

df_res    = pd.DataFrame(resultados)
media_row = df_res[[f"acc_G{g}" for g in gestos_list] + ["acc_total"]].mean()

print("  " + "-" * 72)
print(f"  Media  {media_row['acc_total']*100:>6.1f}%   " +
      "   ".join([f"{media_row[f'acc_G{g}']*100:>4.0f}%" for g in gestos_list]))

print(f"\n  Entrenando modelo final (80% completo)...")
clf_final = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
clf_final.fit(X_work_arr, y_work_arr)
y_pred_val = clf_final.predict(X_val)
acc_val    = accuracy_score(y_val, y_pred_val)
acc_val_g  = {g: accuracy_score(y_val[y_val==g], y_pred_val[y_val==g])
              if (y_val==g).sum() > 0 else 0.0 for g in gestos_list}

print(f"\n  {'='*65}")
print(f"  PRUEBA FINAL — 20% validacion intocable")
print(f"  Precision total : {acc_val*100:.1f}%")
print(f"  Por gesto: " + "   ".join([f"G{g}:{acc_val_g[g]*100:.0f}%" for g in gestos_list]))
print(f"  {'='*65}")
print(f"\n  {'SUPERA el 90% minimo' if acc_val >= 0.90 else 'NO alcanza 90% — revisar K_STD'}")

df_res.to_csv(RESULTS_CSV, index=False)
print(f"  Resultados CV: {RESULTS_CSV}")


# ════════════════════════════════════════════════════════════════
#  PASO 8 — GUARDAR DATASET FINAL
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 8: Guardando dataset...")
print("=" * 65)
df_full.to_csv(OUTPUT_CSV, index=False)
print(f"  {OUTPUT_CSV}  —  {df_full.shape[0]} filas x {df_full.shape[1]} cols")


# ════════════════════════════════════════════════════════════════
#  PASO 9 — GRAFICAS DE RESULTADOS CV
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 9: Graficas de resultados...")
print("=" * 65)

# 07 - Estructura 3 capas
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")
ax.set_title("Division de datos — 3 capas (limpieza DTW)", fontsize=12, fontweight="bold")
ax.add_patch(mpatches.FancyBboxPatch((0.3,7.3),13.2,1.2,
    boxstyle="round,pad=0.1",facecolor="#DDDDDD",edgecolor="#555",linewidth=2))
ax.text(7, 8.05, f"DATASET LIMPIO — {len(df_full)} muestras (100%)",
    ha="center", fontweight="bold", fontsize=10)
for xp in [3.5, 11]:
    ax.annotate("", xy=(xp,7.1), xytext=(xp,7.3),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
ax.add_patch(mpatches.FancyBboxPatch((0.3,5.6),9.2,1.2,
    boxstyle="round,pad=0.1",facecolor="#2E75B6",edgecolor="#1F4E79",linewidth=2,alpha=0.85))
ax.text(4.9, 6.25, f"80% Trabajo — {len(idx_work)} muestras",
    ha="center", color="white", fontweight="bold", fontsize=10)
ax.add_patch(mpatches.FancyBboxPatch((9.7,5.6),3.7,1.2,
    boxstyle="round,pad=0.1",facecolor="#C00000",edgecolor="#900000",linewidth=2,alpha=0.85))
ax.text(11.55, 6.25, f"20% VALIDACION\n{len(idx_val)} muestras — INTOCABLE",
    ha="center", color="white", fontweight="bold", fontsize=9)
ax.annotate("", xy=(4.9,5.3), xytext=(4.9,5.6),
    arrowprops=dict(arrowstyle="->", color="#1F4E79", lw=2))
ax.text(4.9, 5.1, f"Cross Validation ({N_FOLDS} Folds estratificados)",
    ha="center", fontweight="bold", fontsize=9.5, color="#1F4E79")
fold_colors = ["#2980B9","#27AE60","#E67E22","#8E44AD","#16A085"]
y_f = 4.5
for i in range(N_FOLDS):
    fc   = fold_colors[i]
    n_tr = resultados[i]["n_train"]; n_te = resultados[i]["n_test"]
    w_tr = 7.0 * n_tr / (n_tr + n_te); w_te = 7.0 - w_tr
    ax.add_patch(mpatches.FancyBboxPatch((0.3,y_f),w_tr,0.5,
        boxstyle="round,pad=0.04",facecolor=fc,alpha=0.3,edgecolor=fc,linewidth=1))
    ax.add_patch(mpatches.FancyBboxPatch((0.3+w_tr,y_f),w_te,0.5,
        boxstyle="round,pad=0.04",facecolor=fc,alpha=0.85,edgecolor=fc,linewidth=1))
    ax.text(0.3+w_tr/2,      y_f+0.25, f"Train ({n_tr})",
        ha="center", fontsize=7.5, color=fc, fontweight="bold")
    ax.text(0.3+w_tr+w_te/2, y_f+0.25, f"Test ({n_te})",
        ha="center", fontsize=7.5, color="white", fontweight="bold")
    ax.text(0.0, y_f+0.25, f"F{i+1}", ha="center", fontsize=8.5, color=fc, fontweight="bold")
    acc_f = resultados[i]["acc_total"]
    ax.text(7.6, y_f+0.25, f"{acc_f*100:.1f}%", ha="left", fontsize=9,
            color="#1a7a1a" if acc_f >= 0.90 else "#CC0000", fontweight="bold")
    y_f -= 0.62
ax.text(4.5, 1.5,
    f"Promedio CV = {media_row['acc_total']*100:.1f}%  |  Validacion final = {acc_val*100:.1f}%",
    ha="center", fontsize=9.5, color="#1a7a1a", fontweight="bold",
    bbox=dict(boxstyle="round", facecolor="#D5E8D4", alpha=0.9))
ax.annotate("", xy=(11.55,1.9), xytext=(11.55,5.6),
    arrowprops=dict(arrowstyle="->", color="#C00000", lw=2.5, linestyle="dashed"))
ax.text(11.55, 1.55, f"Final\n{acc_val*100:.1f}%\n(1 vez)",
    ha="center", fontsize=8.5, color="#C00000", fontweight="bold")
plt.tight_layout()
plt.savefig(GRAFICAS_DIR / "07_estructura_division_CV.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Guardada: 07_estructura_division_CV.png")

# 08 - Folds detalle
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Cross Validation — 5 Folds (datos limpios DTW)", fontsize=11, fontweight="bold")
axes_flat = axes.flatten()
for i, fdata in enumerate(fold_data_plot):
    ax    = axes_flat[i]
    x_pos = np.arange(8)
    ax.bar(x_pos-0.22, fdata["train_counts"], 0.4, label="Train", color="#2E75B6", alpha=0.75)
    ax.bar(x_pos+0.22, fdata["test_counts"],  0.4, label="Test",  color="#E67E22", alpha=0.9)
    for j, (acc, n_te) in enumerate(zip(fdata["acc_gestos"], fdata["test_counts"])):
        ax.text(x_pos[j]+0.22, n_te+1, f"{acc*100:.0f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color="#1a7a1a" if acc >= 0.90 else "#CC0000")
    acc_t   = fdata["acc_total"]
    ax.set_title(f"Fold {fdata['fold']}  —  {acc_t*100:.1f}%",
                 fontweight="bold", fontsize=10,
                 color="#1a7a1a" if acc_t >= 0.90 else "#CC0000")
    ax.set_xticks(x_pos); ax.set_xticklabels([f"G{g}" for g in gestos_list])
    ax.set_ylabel("N muestras")
    ax.set_ylim(0, max(max(fdata["train_counts"]) * 1.15, 20))
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3, axis="y")

# Panel resumen (slot 6)
ax = axes_flat[5]; ax.axis("off"); ax.set_facecolor("#F0F7FF")
ax.text(0.5, 0.97, "Tabla resumen por fold",
    ha="center", va="top", fontweight="bold", fontsize=11,
    transform=ax.transAxes, color="#1F4E79")
headers = ["Fold","Total"] + [f"G{g}" for g in gestos_list]
col_xs  = [0.03, 0.14] + [0.14+0.10*(j+1) for j in range(8)]
y_t     = 0.89
ax.add_patch(plt.Rectangle((0.01,y_t-0.055),0.98,0.055,
    facecolor="#2E75B6",transform=ax.transAxes,alpha=0.9))
for cx, h in zip(col_xs, headers):
    ax.text(cx+0.01, y_t-0.025, h, transform=ax.transAxes,
        fontsize=8, color="white", fontweight="bold", va="center")
y_t -= 0.062
for fi, res in enumerate(resultados):
    bg = "#F2F7FB" if fi%2==0 else "white"
    ax.add_patch(plt.Rectangle((0.01,y_t-0.055),0.98,0.055,
        facecolor=bg,transform=ax.transAxes,alpha=0.9))
    vals = [str(res["fold"]), f"{res['acc_total']*100:.1f}%"] + \
           [f"{res[f'acc_G{g}']*100:.0f}%" for g in gestos_list]
    for cx, v in zip(col_xs, vals):
        c = "#1a7a1a" if "%" in v and float(v.replace("%","")) >= 90 else "#333"
        ax.text(cx+0.01, y_t-0.025, v, transform=ax.transAxes, fontsize=7.5, color=c, va="center")
    y_t -= 0.062
ax.add_patch(plt.Rectangle((0.01,y_t-0.055),0.98,0.055,
    facecolor="#E8F4E8",transform=ax.transAxes,alpha=0.9))
for cx, v in zip(col_xs, ["Media", f"{media_row['acc_total']*100:.1f}%"] +
                           [f"{media_row[f'acc_G{g}']*100:.0f}%" for g in gestos_list]):
    ax.text(cx+0.01, y_t-0.025, v, transform=ax.transAxes,
        fontsize=7.5, color="#1a7a1a", fontweight="bold", va="center")
y_t -= 0.072
ax.add_patch(plt.Rectangle((0.01,y_t-0.055),0.98,0.055,
    facecolor="#C00000",transform=ax.transAxes,alpha=0.15))
for cx, v in zip(col_xs, ["Final", f"{acc_val*100:.1f}%"] +
                           [f"{acc_val_g[g]*100:.0f}%" for g in gestos_list]):
    ax.text(cx+0.01, y_t-0.025, v, transform=ax.transAxes,
        fontsize=7.5, color="#900000", fontweight="bold", va="center")
ax.text(0.5, 0.05, f"Promedio CV: {media_row['acc_total']*100:.1f}%  |  Final: {acc_val*100:.1f}%",
    ha="center", va="center", transform=ax.transAxes, fontsize=9.5, fontweight="bold",
    color="#1a7a1a", bbox=dict(boxstyle="round", facecolor="#D5E8D4", alpha=0.9))
plt.tight_layout()
plt.savefig(GRAFICAS_DIR / "08_cross_validation_resultados.png",
            dpi=130, bbox_inches="tight", facecolor="white")
plt.close()
print("  Guardada: 08_cross_validation_resultados.png")

# 09 - Vista previa dataset
cols_show = ["gesture_label","user_id","dx_0","dx_1","dx_2","dy_0","dy_1","dy_2",
             "dx_global","dy_global","angle","total_length","split","fold"]
df_fmt = df_full[cols_show].head(10).copy()
for col in df_fmt.select_dtypes(include=float).columns:
    df_fmt[col] = df_fmt[col].apply(lambda v: f"{v:.3f}")
df_fmt["fold"] = df_fmt["fold"].astype(str)
fig, ax = plt.subplots(figsize=(18, 4)); ax.axis("off")
col_labels = ["gesture\nlabel","user\nid","dx_0","dx_1","dx_2","dy_0","dy_1","dy_2",
              "dx\nglobal","dy\nglobal","angle","total\nlength","split","fold"]
table = ax.table(cellText=df_fmt.values, colLabels=col_labels, cellLoc="center", loc="center")
table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1, 1.9)
for j in range(len(col_labels)):
    table[0,j].set_facecolor("#2E75B6")
    table[0,j].set_text_props(color="white", fontweight="bold")
for i in range(1, 11):
    for j in range(len(col_labels)):
        table[i,j].set_facecolor("#F2F7FB" if i%2==0 else "white")
        if j == 0:
            table[i,j].set_facecolor("#D6E4F0")
            table[i,j].set_text_props(fontweight="bold")
ax.set_title("Vista previa dataset_gestos_final.csv — primeras 10 filas",
             fontsize=10, fontweight="bold", pad=18)
plt.tight_layout()
plt.savefig(GRAFICAS_DIR / "09_captura_dataset.png",
            dpi=130, bbox_inches="tight", facecolor="white")
plt.close()
print("  Guardada: 09_captura_dataset.png")

# 10 - Tabla resultados por clase
fig = plt.figure(figsize=(16, 10)); fig.patch.set_facecolor('white')
fig.text(0.5, 0.97, "Precision por Clase — CV con datos limpios (DTW)",
    ha='center', va='top', fontsize=13, fontweight='bold', color='#1F4E79')
ax_t = fig.add_axes([0.03, 0.42, 0.94, 0.47]); ax_t.axis('off')
fd = []
for r in resultados:
    fd.append([f"Fold {int(r['fold'])}", f"{r['acc_total']*100:.1f}%"] +
               [f"{r[f'acc_G{g}']*100:.0f}%" for g in gestos_list])
fd.append(["Media", f"{media_row['acc_total']*100:.1f}%"] +
           [f"{media_row[f'acc_G{g}']*100:.0f}%" for g in gestos_list])
fd.append(["Val. Final", f"{acc_val*100:.1f}%"] +
           [f"{acc_val_g[g]*100:.0f}%" for g in gestos_list])
clt = ["", "Total"] + [f"G{g}" for g in gestos_list]
tbl = ax_t.table(cellText=fd, colLabels=clt, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.8)
for j in range(len(clt)):
    tbl[0,j].set_facecolor('#1F4E79')
    tbl[0,j].set_text_props(color='white', fontweight='bold', fontsize=10)
for i in range(1, 6):
    tbl[i,0].set_facecolor('#D6E4F0')
    tbl[i,0].set_text_props(fontweight='bold', color='#1F4E79')
    at = resultados[i-1]['acc_total']
    tbl[i,1].set_facecolor('#D5E8D4' if at>=0.90 else '#FFE0E0')
    tbl[i,1].set_text_props(fontweight='bold', color='#1a7a1a' if at>=0.90 else '#CC0000')
    for j, g in enumerate(gestos_list):
        ag = resultados[i-1][f'acc_G{g}']
        bg = '#C8F0C8' if ag>=0.98 else ('#FFF0C0' if ag>=0.90 else '#FFCCCC')
        tbl[i,j+2].set_facecolor(bg)
        tbl[i,j+2].set_text_props(fontweight='bold', color='#1a7a1a' if ag>=0.90 else '#CC0000')
for j in range(len(clt)):
    tbl[6,j].set_facecolor('#2E75B6'); tbl[6,j].set_text_props(color='white', fontweight='bold')
    tbl[7,j].set_facecolor('#7B0000'); tbl[7,j].set_text_props(color='white', fontweight='bold')
ax_b = fig.add_axes([0.05, 0.05, 0.60, 0.32])
x = np.arange(8); width = 0.13
for i, (r, c) in enumerate(zip(resultados, ['#2980B9','#27AE60','#E67E22','#8E44AD','#16A085'])):
    ax_b.bar(x+i*width-width*2, [r[f'acc_G{g}']*100 for g in gestos_list],
             width, label=f"Fold {int(r['fold'])}", color=c, alpha=0.8)
ax_b.plot(x+width/2, [acc_val_g[g]*100 for g in gestos_list],
    'r--o', linewidth=2.5, markersize=7, label='Val. Final', zorder=5)
ax_b.axhline(90, color='red', linestyle=':', linewidth=1.5, alpha=0.6, label='90%')
ax_b.set_xticks(x+width); ax_b.set_xticklabels([f"G{g}" for g in gestos_list], fontsize=9)
ax_b.set_ylabel('Precision (%)'); ax_b.set_ylim(75, 108)
ax_b.set_title('Precision por clase', fontweight='bold')
ax_b.legend(fontsize=8, ncol=3, loc='lower right'); ax_b.grid(True, alpha=0.3, axis='y')
ax_r = fig.add_axes([0.68, 0.05, 0.30, 0.32])
ax_r.axis('off'); ax_r.set_facecolor('#F0F7FF')
ax_r.text(0.5, 0.97, "Resumen Final", ha='center', va='top',
    fontweight='bold', fontsize=12, color='#1F4E79', transform=ax_r.transAxes)
for y_l, (etiq, val, col) in zip([0.82,0.665,0.51,0.355,0.20], [
    ("Promedio CV:",      f"{media_row['acc_total']*100:.1f}%", '#2E75B6'),
    ("Val. Final:",       f"{acc_val*100:.1f}%",               '#7B0000'),
    ("Minimo requerido:", "> 90%",                              '#555555'),
    ("Elim. DTW:",        f"{r2_total} muestras",              '#E67E22'),
    ("Dataset limpio:",   f"{len(df_full)} muestras",          '#27AE60'),
]):
    ax_r.text(0.08, y_l, etiq, fontsize=9.5, color='#333', transform=ax_r.transAxes, va='top')
    ax_r.text(0.93, y_l, val, fontsize=10.5, fontweight='bold',
        color=col, transform=ax_r.transAxes, va='top', ha='right')
ax_r.add_patch(mpatches.FancyBboxPatch((0.05,0.04),0.90,0.12,
    boxstyle="round,pad=0.05", facecolor='#D5E8D4', edgecolor='#27AE60',
    linewidth=2, transform=ax_r.transAxes))
ax_r.text(0.5, 0.10,
    f"{'APROBADO' if acc_val >= 0.90 else 'REVISAR'} — "
    f"{'Supera' if acc_val >= 0.90 else 'No alcanza'} 90%",
    ha='center', va='center', fontsize=9.5, fontweight='bold',
    color='#1a7a1a' if acc_val >= 0.90 else '#CC0000', transform=ax_r.transAxes)
plt.savefig(GRAFICAS_DIR / "10_tabla_resultados_por_clase.png",
            dpi=140, bbox_inches='tight', facecolor='white')
plt.close()
print("  Guardada: 10_tabla_resultados_por_clase.png")

# 11 - Mapa de calor basura
fig, ax = plt.subplots(figsize=(10, 6))
heat = np.zeros((10, 8))
for brow in basura_rows:
    heat[brow["user_id"]-1, brow["gesture_id"]-1] += 1
im = ax.imshow(heat, cmap="Reds", aspect="auto", vmin=0)
ax.set_xticks(range(8)); ax.set_xticklabels([f"G{g}" for g in range(1,9)])
ax.set_yticks(range(10)); ax.set_yticklabels([f"U{u}" for u in range(1,11)])
ax.set_xlabel("Gesto"); ax.set_ylabel("Usuario")
ax.set_title("Muestras eliminadas por usuario y gesto (DTW)", fontweight="bold")
plt.colorbar(im, ax=ax, label="N eliminadas")
for i in range(10):
    for j in range(8):
        v = int(heat[i, j])
        if v > 0:
            ax.text(j, i, str(v), ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if v > 1 else "darkred")
plt.tight_layout()
plt.savefig(GRAFICAS_DIR / "11_mapa_calor_basura.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Guardada: 11_mapa_calor_basura.png")


# ── RESUMEN FINAL ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PIPELINE COMPLETO — LIMPIEZA CON DTW")
print(f"  Producto principal : {OUTPUT_CSV}")
print(f"  Registro de basura : {BASURA_CSV}")
print(f"  Resultados CV      : {RESULTS_CSV}")
print(f"  Graficas           : {GRAFICAS_DIR}/  (11 imagenes)")
print(f"  Muestras originales: {len(records_raw)}")
print(f"  Duplicados         : {total_dup}")
print(f"  Basura R1 (angulo) : {r1_total}")
print(f"  Basura R2 (DTW)    : {r2_total}")
print(f"  Dataset limpio     : {len(df_full)} muestras")
print(f"  Precision CV media : {media_row['acc_total']*100:.1f}%")
print(f"  Precision final    : {acc_val*100:.1f}%")
print(f"  Requisito > 90%    : {'CUMPLIDO' if acc_val >= 0.90 else 'REVISAR K_STD'}")
print("=" * 65)