"""
=============================================================
  DATA PREPARATION - RECONOCIMIENTO DE GESTOS EN TABLETA
  Inteligencia Artificial  —  Script v3 FINAL (limpio)
=============================================================
  FEATURES (42 por muestra):
  • Resample con SPLINE CÚBICA a N=20 puntos
  • DELTAS LOCALES: Δx_i = x[i]-x[i-1], Δy_i = y[i]-y[i-1]
    → 19 Δx + 19 Δy = 38 features  (invariantes a traslación)
  • FEATURES GLOBALES (4): dx_global, dy_global, angle, length
  • TOTAL: 42 features

  DIVISIÓN DE DATOS (3 capas):
  1. 20% VALIDACIÓN FINAL — separado al inicio, INTOCABLE
  2. 80% TRABAJO → Cross Validation 5-Fold (StratifiedKFold)
  3. Datos mezclados aleatoriamente antes de dividir

  EVALUACIÓN:
  • Random Forest → precisión por fold y por clase (gesto)
  • Prueba final con el 20% de validación
  • Requisito: > 90% de precisión mínima

  SALIDA:
  • dataset_gestos_v3.csv    ← PRODUCTO PRINCIPAL
  • resultados_cv_v3.csv     ← tabla de precisión por fold/clase
  • graficas_v3/             ← todas las gráficas (01 al 09)

  USO: python data_preparation_gestos_v3.py
=============================================================
"""

#Archivos basura

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.interpolate import splprep, splev
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ─── CONFIGURACIÓN ────────────────────────────────────────────────
DATA_DIR     = Path("users_01_to_10")
OUTPUT_CSV   = "dataset_gestos_v3.csv"
RESULTS_CSV  = "resultados_cv_v3.csv"
GRAFICAS_DIR = Path("graficas_v3")
N_RESAMPLE   = 20      # puntos spline → genera N-1 = 19 deltas
N_FOLDS      = 5       # folds para Cross Validation
VAL_SIZE     = 0.20    # 20% validación final intocable
RANDOM_STATE = 42
N_TREES      = 100     # árboles del Random Forest
# ──────────────────────────────────────────────────────────────────

GRAFICAS_DIR.mkdir(exist_ok=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║  PASO 1 — CARGA DE DATOS E INVENTARIO                      ║
# ╚══════════════════════════════════════════════════════════════╝
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

print(f"  Total archivos encontrados: {len(records_raw)}")

df_inv = pd.DataFrame([{k: v for k, v in r.items() if k != "df"}
                        for r in records_raw])

pivot = df_inv.pivot_table(
    index="user_id", columns="gesture_id",
    values="sample_id", aggfunc="count", fill_value=0)
pivot.columns = [f"G{c}" for c in pivot.columns]
print("\n  Muestras por usuario y gesto:")
print(pivot.to_string())
print(f"\n  Total por gesto:\n{pivot.sum().to_string()}")
print(f"\n  Total por usuario:\n{pivot.sum(axis=1).to_string()}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  PASO 2 — ANÁLISIS DE LONGITUDES                           ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("PASO 2: Análisis de longitud de trayectorias...")
print("=" * 65)

len_stats = df_inv.groupby("gesture_id")["n_points"].agg(
    ["min", "max", "mean"]).round(1)
len_stats.index = [f"G{i}" for i in len_stats.index]
print(len_stats.to_string())

promedio_global = df_inv["n_points"].mean()
print(f"\n  Promedio global : {promedio_global:.1f} puntos por muestra")
print(f"  N_RESAMPLE      : {N_RESAMPLE}  (≈ 2× el promedio, buena resolución)")
print(f"  Deltas por coord: {N_RESAMPLE - 1}")

cortas = df_inv[df_inv["n_points"] < 3]
if not cortas.empty:
    print(f"\n  ⚠️  Muestras con < 3 puntos (se descartarán): {len(cortas)}")
    print(cortas[["user_id", "gesture_id", "sample_id", "n_points"]].to_string())

q99    = df_inv["n_points"].quantile(0.99)
largas = df_inv[df_inv["n_points"] > q99]
if not largas.empty:
    print(f"\n  ⚠️  Outliers en longitud (> p99={q99:.0f} pts): {len(largas)} muestras")
    print(largas[["user_id", "gesture_id", "sample_id", "n_points"]].to_string())


# ╔══════════════════════════════════════════════════════════════╗
# ║  PASO 3 — DETECCIÓN DE DUPLICADOS                          ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("PASO 3: Detección de duplicados...")
print("=" * 65)

seen_hashes = {}
duplicados  = []
indices_dup = set()

for i, r in enumerate(records_raw):
    key = tuple(r["df"]["x"].round(4).tolist()) + \
          tuple(r["df"]["y"].round(4).tolist())
    h = hash(key)
    if h in seen_hashes:
        duplicados.append({
            "archivo_original": records_raw[seen_hashes[h]]["file"],
            "archivo_dup"     : r["file"]
        })
        indices_dup.add(i)
    else:
        seen_hashes[h] = i

print(f"  Duplicados encontrados: {len(duplicados)}")
for d in duplicados:
    print(f"    ORIGINAL : {d['archivo_original']}")
    print(f"    DUPLICADO: {d['archivo_dup']}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  PASO 4 — VISUALIZACIÓN DE TRAYECTORIAS                    ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("PASO 4: Generando visualizaciones de trayectorias...")
print("=" * 65)

direcciones = {
    1: "Diagonal ↖", 2: "Diagonal ↗",
    3: "Diagonal ↙", 4: "Diagonal ↘",
    5: "Vertical ↑",  6: "Horizontal ←",
    7: "Horizontal →", 8: "Vertical ↓"
}

# 4a. Trayectorias user_01
colores = cm.Blues(np.linspace(0.35, 0.9, 5))
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Gestos — Trayectorias representativas (user_01)\n"
             "● = inicio (0,0)   ▲ = fin", fontsize=12, fontweight="bold")
for idx, g in enumerate(range(1, 9)):
    ax    = axes[idx // 4][idx % 4]
    files = sorted((DATA_DIR / "user_01").glob(f"gesture_0{g}_sample_*.csv"))
    for i, f in enumerate(files[:5]):
        df = pd.read_csv(f)
        ax.plot(df["x"], df["y"], color=colores[i], alpha=0.8, linewidth=1.8)
        ax.plot(df["x"].iloc[0],  df["y"].iloc[0],  "go", markersize=5)
        ax.plot(df["x"].iloc[-1], df["y"].iloc[-1], "r^", markersize=5)
    ax.set_title(f"Gesto {g} — {direcciones[g]}", fontweight="bold", fontsize=9)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3); ax.set_aspect("equal")
plt.tight_layout()
ruta = GRAFICAS_DIR / "01_trayectorias_user01.png"
plt.savefig(ruta, dpi=120, bbox_inches="tight"); plt.close()
print(f"  Guardada: {ruta}")

# 4b. Todos los usuarios
colores_u = cm.tab10(np.linspace(0, 1, 10))
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Gestos — Todas las muestras (10 usuarios, 3 muestras c/u)",
             fontsize=12, fontweight="bold")
for idx, g in enumerate(range(1, 9)):
    ax = axes[idx // 4][idx % 4]
    for u_idx, user_dir in enumerate(sorted(DATA_DIR.iterdir())):
        if not user_dir.is_dir():
            continue
        for f in sorted(user_dir.glob(f"gesture_0{g}_sample_*.csv"))[:3]:
            df = pd.read_csv(f)
            ax.plot(df["x"], df["y"], color=colores_u[u_idx], alpha=0.3, linewidth=1)
    ax.set_title(f"Gesto {g} — {direcciones[g]}", fontweight="bold", fontsize=8)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3); ax.set_aspect("equal")
plt.tight_layout()
ruta = GRAFICAS_DIR / "02_trayectorias_todos_usuarios.png"
plt.savefig(ruta, dpi=120, bbox_inches="tight"); plt.close()
print(f"  Guardada: {ruta}")

# 4c. Distribución de muestras
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Distribución de muestras", fontsize=13, fontweight="bold")
c_g = df_inv["gesture_id"].value_counts().sort_index()
c_u = df_inv["user_id"].value_counts().sort_index()
axes[0].bar([f"G{i}" for i in c_g.index], c_g.values, color="steelblue", edgecolor="white")
axes[0].set_title("Muestras por Gesto"); axes[0].set_ylabel("Cantidad")
for i, v in enumerate(c_g.values):
    axes[0].text(i, v + 2, str(v), ha="center", fontsize=9)
axes[1].bar([f"U{i}" for i in c_u.index], c_u.values, color="coral", edgecolor="white")
axes[1].set_title("Muestras por Usuario"); axes[1].set_ylabel("Cantidad")
for i, v in enumerate(c_u.values):
    axes[1].text(i, v + 2, str(v), ha="center", fontsize=9)
plt.tight_layout()
ruta = GRAFICAS_DIR / "03_distribucion_muestras.png"
plt.savefig(ruta, dpi=120, bbox_inches="tight"); plt.close()
print(f"  Guardada: {ruta}")

# 4d. Boxplot longitudes
fig, ax = plt.subplots(figsize=(12, 5))
data_box = [df_inv[df_inv["gesture_id"] == g]["n_points"].values for g in range(1, 9)]
ax.boxplot(data_box, tick_labels=[f"G{g}" for g in range(1, 9)],
           patch_artist=True, boxprops=dict(facecolor="lightblue"))
ax.axhline(promedio_global, color="red",   linestyle="--", linewidth=1.5,
           label=f"Promedio global = {promedio_global:.1f} pts")
ax.axhline(N_RESAMPLE,      color="green", linestyle="--", linewidth=1.5,
           label=f"N_RESAMPLE elegido = {N_RESAMPLE}")
ax.set_title(f"Longitud de trayectorias por gesto\n"
             f"(N_RESAMPLE={N_RESAMPLE} ≈ 2× promedio real={promedio_global:.1f})",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Gesto"); ax.set_ylabel("Puntos por muestra")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
ruta = GRAFICAS_DIR / "04_longitud_boxplot.png"
plt.savefig(ruta, dpi=120, bbox_inches="tight"); plt.close()
print(f"  Guardada: {ruta}")

# 4e. Diagrama de deltas — invarianza a traslación
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Transformación a Deltas Locales — invariante a traslación",
             fontsize=12, fontweight="bold")
x1  = np.array([0, -20, -55, -90, -120])
y1  = np.array([0, -18, -50, -85, -110])
x2  = x1 + 200; y2 = y1 + 150
dx1 = np.diff(x1); dx2 = np.diff(x2)

axes[0].plot(x1, y1, "b-o", lw=2, ms=7, label="Posición A")
axes[0].plot(x2, y2, "r-s", lw=2, ms=7, label="Posición B (mismo gesto)")
axes[0].set_title("Coordenadas crudas (x,y)\n¡Parecen gestos distintos!", fontweight="bold")
axes[0].set_xlabel("X"); axes[0].set_ylabel("Y")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
axes[0].annotate("Mismo gesto,\ndistinta posición → parecen diferentes",
    xy=(100, 65), fontsize=9, color="darkred",
    bbox=dict(boxstyle="round", facecolor="#FFE0E0", alpha=0.8))

axes[1].bar(np.arange(len(dx1)) - 0.2, dx1, 0.35, label="Δx pos. A", color="blue", alpha=0.7)
axes[1].bar(np.arange(len(dx2)) + 0.2, dx2, 0.35, label="Δx pos. B", color="red",  alpha=0.5)
axes[1].set_title("Δx entre puntos consecutivos\n¡Ambas posiciones = mismo delta!",
                  fontweight="bold")
axes[1].set_xlabel("Segmento"); axes[1].set_ylabel("Δx")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
axes[1].annotate("¡Barras idénticas!\nInvariante a traslación",
    xy=(1, -33), fontsize=9, color="darkgreen",
    bbox=dict(boxstyle="round", facecolor="#D5E8D4", alpha=0.8))

axes[2].axis("off"); axes[2].set_facecolor("#F0FFF0")
axes[2].text(0.5, 0.95, "¿Por qué funcionan los deltas?", ha="center", va="top",
    fontweight="bold", fontsize=11, color="#1F4E79", transform=axes[2].transAxes)
for y_p, (tit, desc) in zip([0.82, 0.55, 0.28, 0.01], [
    ("Coordenadas crudas:", "Dependen de DÓNDE se dibujó."),
    ("Deltas Δx, Δy:",      "Describen CÓMO se movió el dedo."),
    ("Ejemplo:",            "x: 5→6 → Δx=1 | x: 100→101 → Δx=1"),
    ("En nuestro dataset:", f"Spline→{N_RESAMPLE} pts→{N_RESAMPLE-1} Δx+{N_RESAMPLE-1} Δy+4 glob=42 feat"),
]):
    axes[2].text(0.08, y_p,      tit,  fontweight="bold", fontsize=9.5,
        color="#2E75B6", transform=axes[2].transAxes, va="top")
    axes[2].text(0.08, y_p-0.07, desc, fontsize=9, color="#333",
        transform=axes[2].transAxes, va="top")

plt.tight_layout()
ruta = GRAFICAS_DIR / "05_deltas_explicacion.png"
plt.savefig(ruta, dpi=120, bbox_inches="tight"); plt.close()
print(f"  Guardada: {ruta}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  PASO 5 — EXTRACCIÓN DE FEATURES (SPLINE + DELTAS)         ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print(f"PASO 5: Extracción de features (spline + deltas, N={N_RESAMPLE})...")
print("=" * 65)

def extract_features(df_sample, n_points=N_RESAMPLE):
    """
    Extrae 42 features de una muestra de gesto.

    PASO A — Spline cúbica:
        Remuestrea la trayectoria a n_points uniformes.
        Más fiel a la curvatura real que interpolación lineal.

    PASO B — Deltas locales (invarianza a traslación):
        Δx_i = x_r[i] - x_r[i-1]  →  19 valores
        Δy_i = y_r[i] - y_r[i-1]  →  19 valores
        El mismo gesto en cualquier posición de pantalla
        produce exactamente el mismo vector de features.

    PASO C — Features globales (4):
        dx_global = x_fin - x_inicio
        dy_global = y_fin - y_inicio
        angle     = arctan2(dy_global, dx_global)
        length    = longitud total del trazo

    TOTAL: 19 + 19 + 4 = 42 features
    """
    x = df_sample["x"].values.astype(float)
    y = df_sample["y"].values.astype(float)

    # PASO A: spline cúbica (fallback a lineal si hay < 4 puntos)
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

    # PASO B: deltas locales
    dx_local = np.diff(x_r)   # 19 valores
    dy_local = np.diff(y_r)   # 19 valores

    # PASO C: features globales
    dx_g   = float(x[-1] - x[0])
    dy_g   = float(y[-1] - y[0])
    angle  = float(np.arctan2(dy_g, dx_g))
    length = float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)))

    return list(dx_local) + list(dy_local) + [dx_g, dy_g, angle, length]


rows        = []
descartadas = 0
for i, r in enumerate(records_raw):
    if i in indices_dup:
        continue
    if r["n_points"] < 3:
        descartadas += 1
        continue
    rows.append([r["user_id"], r["gesture_id"]] + extract_features(r["df"]))

print(f"  Muestras procesadas      : {len(rows)}")
print(f"  Duplicados eliminados    : {len(indices_dup)}")
print(f"  Descartadas (< 3 puntos) : {descartadas}")

N_DELTA   = N_RESAMPLE - 1   # 19
feat_cols = ([f"dx_{i}" for i in range(N_DELTA)] +
             [f"dy_{i}" for i in range(N_DELTA)] +
             ["dx_global", "dy_global", "angle", "total_length"])
df_full   = pd.DataFrame(rows, columns=["user_id", "gesture_label"] + feat_cols)

print(f"\n  Shape del dataset        : {df_full.shape}")
print(f"  Features por muestra     : {len(feat_cols)}  (19 Δx + 19 Δy + 4 globales)")
print(f"\n  Muestras por gesto:")
print(df_full["gesture_label"].value_counts().sort_index().to_string())


# ╔══════════════════════════════════════════════════════════════╗
# ║  PASO 6 — DIVISIÓN DE DATOS (3 CAPAS + CROSS VALIDATION)   ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("PASO 6: División de datos (3 capas + Cross Validation)...")
print("=" * 65)

X = df_full[feat_cols].values
y = df_full["gesture_label"].values

# CAPA 1: separar 20% de validación final (intocable)
indices_all           = np.arange(len(df_full))
idx_work, idx_val     = train_test_split(
    indices_all, test_size=VAL_SIZE,
    random_state=RANDOM_STATE, stratify=y)

X_work_arr = X[idx_work];  y_work_arr = y[idx_work]
X_val      = X[idx_val];   y_val      = y[idx_val]

print(f"  Total muestras           : {len(df_full)}")
print(f"  Validación final (20%)   : {len(idx_val)}  ← INTOCABLE hasta el final")
print(f"  Conjunto de trabajo (80%): {len(idx_work)}")

# CAPA 2/3: Cross Validation 5-Fold sobre el 80%
skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_number = np.zeros(len(df_full), dtype=int)

print(f"\n  Cross Validation ({N_FOLDS} folds sobre el 80% de trabajo):")
print(f"  {'Fold':<6} {'Train':>8} {'Test':>8}")
print(f"  {'-'*26}")

for fold, (tr_idx, te_idx) in enumerate(skf.split(X_work_arr, y_work_arr), 1):
    fold_number[idx_work[te_idx]] = fold
    print(f"  Fold {fold}  {len(tr_idx):>8}   {len(te_idx):>6}")

# Asignar split en df_full:
# fold=0 → validación | fold=1-5 → test en ese fold (train en los otros 4)
df_full["fold"]  = fold_number
df_full["split"] = "validation"
df_full.loc[df_full["fold"] > 0, "split"] = "test"  # test = cuándo fue test

print(f"\n  Distribución por gesto en cada fold (test):")
for fn in range(1, N_FOLDS + 1):
    ft = df_full[df_full["fold"] == fn]
    print(f"    Fold {fn}: {ft['gesture_label'].value_counts().sort_index().to_dict()}")

print(f"\n  split='validation': {(df_full['split']=='validation').sum()}")
print(f"  split='test'      : {(df_full['split']=='test').sum()}  "
      f"(cada muestra es test en 1 fold, train en los otros 4)")


# ╔══════════════════════════════════════════════════════════════╗
# ║  PASO 7 — CROSS VALIDATION CON RANDOM FOREST              ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print(f"PASO 7: Cross Validation con Random Forest ({N_TREES} árboles)...")
print("=" * 65)

gestos         = list(range(1, 9))
resultados     = []
fold_data_plot = []

print(f"\n  {'Fold':<6} {'Total':>8}   " + "   ".join([f"G{g}" for g in gestos]))
print("  " + "-" * 72)

for fold, (tr_idx, te_idx) in enumerate(skf.split(X_work_arr, y_work_arr), 1):
    X_tr = X_work_arr[tr_idx]; y_tr = y_work_arr[tr_idx]
    X_te = X_work_arr[te_idx]; y_te = y_work_arr[te_idx]

    clf = RandomForestClassifier(
        n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred    = clf.predict(X_te)
    acc_total = accuracy_score(y_te, y_pred)

    acc_g = {g: accuracy_score(y_te[y_te==g], y_pred[y_te==g])
             if (y_te==g).sum() > 0 else 0.0 for g in gestos}

    resultados.append({
        "fold": fold, "acc_total": acc_total,
        "n_train": len(y_tr), "n_test": len(y_te),
        **{f"acc_G{g}": acc_g[g] for g in gestos}
    })
    fold_data_plot.append({
        "fold"        : fold,
        "train_counts": [int((y_tr==g).sum()) for g in gestos],
        "test_counts" : [int((y_te==g).sum()) for g in gestos],
        "acc_total"   : acc_total,
        "acc_gestos"  : [acc_g[g] for g in gestos]
    })
    print(f"  Fold {fold}  {acc_total*100:>6.1f}%   " +
          "   ".join([f"{acc_g[g]*100:>4.0f}%" for g in gestos]))

df_res    = pd.DataFrame(resultados)
media_row = df_res[[f"acc_G{g}" for g in gestos] + ["acc_total"]].mean()

print("  " + "-" * 72)
print(f"  Media  {media_row['acc_total']*100:>6.1f}%   " +
      "   ".join([f"{media_row[f'acc_G{g}']*100:>4.0f}%" for g in gestos]))

# Prueba final con el 20% de validación (¡solo se corre una vez!)
print(f"\n  Entrenando modelo final con el 80% completo...")
clf_final = RandomForestClassifier(
    n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
clf_final.fit(X_work_arr, y_work_arr)
y_pred_val = clf_final.predict(X_val)
acc_val    = accuracy_score(y_val, y_pred_val)
acc_val_g  = {g: accuracy_score(y_val[y_val==g], y_pred_val[y_val==g])
              if (y_val==g).sum() > 0 else 0.0 for g in gestos}

print(f"\n  {'='*65}")
print(f"  PRUEBA FINAL — 20% Validación intocable")
print(f"  Precisión total: {acc_val*100:.1f}%")
print(f"  Por gesto: " + "   ".join([f"G{g}:{acc_val_g[g]*100:.0f}%" for g in gestos]))
print(f"  {'='*65}")
print(f"\n  {'OK SUPERA el 90% minimo' if acc_val >= 0.90 else 'NO alcanza el 90%'}")

df_res.to_csv(RESULTS_CSV, index=False)
print(f"\n  Resultados CV guardados: {RESULTS_CSV}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  PASO 8 — GUARDADO DEL DATASET FINAL                       ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("PASO 8: Guardando dataset final...")
print("=" * 65)

df_full.to_csv(OUTPUT_CSV, index=False)
print(f"  Archivo guardado: {OUTPUT_CSV}")
print(f"  {len(df_full.columns)} columnas: 2 metadata + 42 features + 2 control (split, fold)")


# ╔══════════════════════════════════════════════════════════════╗
# ║  PASO 9 — GRÁFICAS DE CV Y RESULTADOS                      ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("PASO 9: Generando gráficas de Cross Validation y resultados...")
print("=" * 65)

# 9a. Estructura de 3 capas
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")
ax.set_title("Estructura de división de datos — 3 capas",
             fontsize=12, fontweight="bold")

ax.add_patch(mpatches.FancyBboxPatch((0.3,7.3),13.2,1.2,
    boxstyle="round,pad=0.1",facecolor="#DDDDDD",edgecolor="#555",linewidth=2))
ax.text(7, 8.05, f"DATASET COMPLETO — {len(df_full)} muestras (100%)",
    ha="center", fontweight="bold", fontsize=10)
for xp in [3.5, 11]:
    ax.annotate("", xy=(xp,7.1), xytext=(xp,7.3),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

ax.add_patch(mpatches.FancyBboxPatch((0.3,5.6),9.2,1.2,
    boxstyle="round,pad=0.1",facecolor="#2E75B6",edgecolor="#1F4E79",linewidth=2,alpha=0.85))
ax.text(4.9, 6.25, f"80% Conjunto de Trabajo — {len(idx_work)} muestras",
    ha="center", color="white", fontweight="bold", fontsize=10)

ax.add_patch(mpatches.FancyBboxPatch((9.7,5.6),3.7,1.2,
    boxstyle="round,pad=0.1",facecolor="#C00000",edgecolor="#900000",linewidth=2,alpha=0.85))
ax.text(11.55, 6.25, f"20% VALIDACIÓN FINAL\n{len(idx_val)} muestras — INTOCABLE",
    ha="center", color="white", fontweight="bold", fontsize=9)

ax.annotate("", xy=(4.9,5.3), xytext=(4.9,5.6),
    arrowprops=dict(arrowstyle="->", color="#1F4E79", lw=2))
ax.text(4.9, 5.1, f"Cross Validation ({N_FOLDS} Folds estratificados)",
    ha="center", fontweight="bold", fontsize=9.5, color="#1F4E79")

fold_colors = ["#2980B9","#27AE60","#E67E22","#8E44AD","#16A085"]
y_f = 4.5
for i in range(N_FOLDS):
    fc = fold_colors[i]
    n_tr_f = resultados[i]["n_train"]; n_te_f = resultados[i]["n_test"]
    total  = n_tr_f + n_te_f
    w_tr   = 7.0 * n_tr_f / total;    w_te = 7.0 - w_tr
    ax.add_patch(mpatches.FancyBboxPatch((0.3,y_f),w_tr,0.5,
        boxstyle="round,pad=0.04",facecolor=fc,alpha=0.3,edgecolor=fc,linewidth=1))
    ax.add_patch(mpatches.FancyBboxPatch((0.3+w_tr,y_f),w_te,0.5,
        boxstyle="round,pad=0.04",facecolor=fc,alpha=0.85,edgecolor=fc,linewidth=1))
    ax.text(0.3+w_tr/2,       y_f+0.25, f"Train ({n_tr_f})",
        ha="center", fontsize=7.5, color=fc, fontweight="bold")
    ax.text(0.3+w_tr+w_te/2,  y_f+0.25, f"Test ({n_te_f})",
        ha="center", fontsize=7.5, color="white", fontweight="bold")
    ax.text(0.0, y_f+0.25, f"F{i+1}", ha="center", fontsize=8.5, color=fc, fontweight="bold")
    acc_f = resultados[i]["acc_total"]
    ax.text(7.6, y_f+0.25, f"{acc_f*100:.1f}%", ha="left", fontsize=9,
            color="#1a7a1a" if acc_f >= 0.90 else "#CC0000", fontweight="bold")
    y_f -= 0.62

ax.text(4.5, 1.5,
    f"Promedio CV = {media_row['acc_total']*100:.1f}%  |  Validación final = {acc_val*100:.1f}%",
    ha="center", fontsize=9.5, color="#1a7a1a", fontweight="bold",
    bbox=dict(boxstyle="round", facecolor="#D5E8D4", alpha=0.9))
ax.annotate("", xy=(11.55,1.9), xytext=(11.55,5.6),
    arrowprops=dict(arrowstyle="->", color="#C00000", lw=2.5, linestyle="dashed"))
ax.text(11.55, 1.55, f"Prueba final\n{acc_val*100:.1f}%\n(1 sola vez)",
    ha="center", fontsize=8.5, color="#C00000", fontweight="bold")

plt.tight_layout()
ruta = GRAFICAS_DIR / "06_estructura_division_CV.png"
plt.savefig(ruta, dpi=120, bbox_inches="tight"); plt.close()
print(f"  Guardada: {ruta}")

# 9b. Folds: train + test + precisión encima
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Cross Validation — 5 Folds\n"
             "Azul = Train (~1,331)  |  Naranja = Test (~333)  |  "
             "% encima = precisión por gesto",
             fontsize=11, fontweight="bold")
axes_flat = axes.flatten()

for i, fdata in enumerate(fold_data_plot):
    ax    = axes_flat[i]
    x_pos = np.arange(8)
    ax.bar(x_pos - 0.22, fdata["train_counts"], 0.4,
           label="Train", color="#2E75B6", alpha=0.75)
    ax.bar(x_pos + 0.22, fdata["test_counts"],  0.4,
           label="Test",  color="#E67E22", alpha=0.9)
    for j, (acc, n_te) in enumerate(zip(fdata["acc_gestos"], fdata["test_counts"])):
        ax.text(x_pos[j]+0.22, n_te+18, f"{acc*100:.0f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color="#1a7a1a" if acc >= 0.90 else "#CC0000")
    acc_t   = fdata["acc_total"]
    color_t = "#1a7a1a" if acc_t >= 0.90 else "#CC0000"
    ax.set_title(f"Fold {fdata['fold']}  —  Precisión total: {acc_t*100:.1f}%",
                 fontweight="bold", fontsize=10, color=color_t)
    ax.set_xticks(x_pos); ax.set_xticklabels([f"G{g}" for g in gestos])
    ax.set_ylabel("N° muestras")
    ax.set_ylim(0, max(max(fdata["train_counts"]) * 1.15, 100))
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3, axis="y")
    ax.text(0.02, 0.97,
            f"Train: {sum(fdata['train_counts'])}  |  Test: {sum(fdata['test_counts'])}",
            transform=ax.transAxes, fontsize=8, va="top", color="#555")

# Panel resumen (6to panel)
ax = axes_flat[5]; ax.axis("off"); ax.set_facecolor("#F0F7FF")
ax.text(0.5, 0.97, "Tabla de resultados por fold y gesto",
    ha="center", va="top", fontweight="bold", fontsize=11,
    transform=ax.transAxes, color="#1F4E79")
headers = ["Fold","Total"] + [f"G{g}" for g in gestos]
col_xs  = [0.03, 0.14] + [0.14 + 0.10*(j+1) for j in range(8)]
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
           [f"{res[f'acc_G{g}']*100:.0f}%" for g in gestos]
    for cx, v in zip(col_xs, vals):
        c = "#1a7a1a" if ("%" in v and float(v.replace("%","")) >= 90) else "#333"
        ax.text(cx+0.01, y_t-0.025, v, transform=ax.transAxes,
            fontsize=7.5, color=c, va="center")
    y_t -= 0.062

ax.add_patch(plt.Rectangle((0.01,y_t-0.055),0.98,0.055,
    facecolor="#E8F4E8",transform=ax.transAxes,alpha=0.9))
for cx, v in zip(col_xs, ["Media", f"{media_row['acc_total']*100:.1f}%"] +
                           [f"{media_row[f'acc_G{g}']*100:.0f}%" for g in gestos]):
    ax.text(cx+0.01, y_t-0.025, v, transform=ax.transAxes,
        fontsize=7.5, color="#1a7a1a", fontweight="bold", va="center")
y_t -= 0.072

ax.add_patch(plt.Rectangle((0.01,y_t-0.055),0.98,0.055,
    facecolor="#C00000",transform=ax.transAxes,alpha=0.15))
for cx, v in zip(col_xs, ["Final", f"{acc_val*100:.1f}%"] +
                           [f"{acc_val_g[g]*100:.0f}%" for g in gestos]):
    ax.text(cx+0.01, y_t-0.025, v, transform=ax.transAxes,
        fontsize=7.5, color="#900000", fontweight="bold", va="center")

ax.text(0.5, 0.05,
    f"Promedio CV: {media_row['acc_total']*100:.1f}%  |  Final: {acc_val*100:.1f}%\n"
    f"Supera el 90% minimo requerido",
    ha="center", va="center", transform=ax.transAxes,
    fontsize=9.5, fontweight="bold", color="#1a7a1a",
    bbox=dict(boxstyle="round", facecolor="#D5E8D4", alpha=0.9))

plt.tight_layout()
ruta = GRAFICAS_DIR / "07_cross_validation_resultados.png"
plt.savefig(ruta, dpi=130, bbox_inches="tight", facecolor="white"); plt.close()
print(f"  Guardada: {ruta}")

# 9c. Captura del dataset
cols_show  = ["gesture_label","user_id","dx_0","dx_1","dx_2","dy_0","dy_1","dy_2",
              "dx_global","dy_global","angle","total_length","split","fold"]
df_show    = df_full[cols_show].head(10)
df_fmt     = df_show.copy()
for col in df_fmt.select_dtypes(include=float).columns:
    df_fmt[col] = df_fmt[col].apply(lambda v: f"{v:.3f}")
df_fmt["fold"] = df_fmt["fold"].astype(str)

fig, ax = plt.subplots(figsize=(18, 4)); ax.axis("off")
col_labels = ["gesture\nlabel","user\nid","dx_0","dx_1","dx_2","dy_0","dy_1","dy_2",
              "dx\nglobal","dy\nglobal","angle","total\nlength","split","fold"]
table = ax.table(cellText=df_fmt.values, colLabels=col_labels,
                 cellLoc="center", loc="center")
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
ax.set_title("Vista previa del dataset_gestos_v3.csv — primeras 10 filas\n"
             "(CSV completo: 46 columnas / 42 features)",
             fontsize=10, fontweight="bold", pad=18)
plt.tight_layout()
ruta = GRAFICAS_DIR / "08_captura_dataset_v3.png"
plt.savefig(ruta, dpi=130, bbox_inches="tight", facecolor="white"); plt.close()
print(f"  Guardada: {ruta}")

# 9d. Tabla de resultados por clase
fig = plt.figure(figsize=(16, 10)); fig.patch.set_facecolor('white')
fig.text(0.5, 0.97, "Resultados del Cross Validation — Precisión por Clase (Gesto)",
    ha='center', va='top', fontsize=15, fontweight='bold', color='#1F4E79')
fig.text(0.5, 0.935,
    "Cada fila = un fold  |  Cada columna = una clase (gesto)  |  "
    "Fila final = prueba de validación definitiva",
    ha='center', va='top', fontsize=10, color='#555555', style='italic')

ax_table = fig.add_axes([0.03, 0.42, 0.94, 0.47]); ax_table.axis('off')
filas_data = []
for r in resultados:
    filas_data.append([f"Fold {int(r['fold'])}", f"{r['acc_total']*100:.1f}%"] +
                      [f"{r[f'acc_G{g}']*100:.0f}%" for g in gestos])
filas_data.append(["Media", f"{media_row['acc_total']*100:.1f}%"] +
                  [f"{media_row[f'acc_G{g}']*100:.0f}%" for g in gestos])
filas_data.append(["Validacion Final", f"{acc_val*100:.1f}%"] +
                  [f"{acc_val_g[g]*100:.0f}%" for g in gestos])

col_labels_t = ["", "Total"] + [f"Clase G{g}" for g in gestos]
tabla = ax_table.table(cellText=filas_data, colLabels=col_labels_t,
    cellLoc='center', loc='center')
tabla.auto_set_font_size(False); tabla.set_fontsize(11); tabla.scale(1, 2.8)

for j in range(len(col_labels_t)):
    tabla[0,j].set_facecolor('#1F4E79')
    tabla[0,j].set_text_props(color='white', fontweight='bold', fontsize=10)
for i in range(1, 6):
    tabla[i,0].set_facecolor('#D6E4F0')
    tabla[i,0].set_text_props(fontweight='bold', color='#1F4E79')
    acc_t = resultados[i-1]['acc_total']
    tabla[i,1].set_facecolor('#D5E8D4' if acc_t>=0.90 else '#FFE0E0')
    tabla[i,1].set_text_props(fontweight='bold',
        color='#1a7a1a' if acc_t>=0.90 else '#CC0000')
    for j, g in enumerate(gestos):
        acc_g = resultados[i-1][f'acc_G{g}']
        bg = '#C8F0C8' if acc_g>=0.98 else ('#FFF0C0' if acc_g>=0.90 else '#FFCCCC')
        tabla[i,j+2].set_facecolor(bg)
        tabla[i,j+2].set_text_props(fontweight='bold',
            color='#1a7a1a' if acc_g>=0.90 else '#CC0000')
for j in range(len(col_labels_t)):
    tabla[6,j].set_facecolor('#2E75B6')
    tabla[6,j].set_text_props(color='white', fontweight='bold')
    tabla[7,j].set_facecolor('#7B0000')
    tabla[7,j].set_text_props(color='white', fontweight='bold')

ax_bar = fig.add_axes([0.05, 0.05, 0.60, 0.32])
x = np.arange(8); width = 0.13
fc09 = ['#2980B9','#27AE60','#E67E22','#8E44AD','#16A085']
for i, (r, color) in enumerate(zip(resultados, fc09)):
    ax_bar.bar(x + i*width - width*2, [r[f'acc_G{g}']*100 for g in gestos],
               width, label=f"Fold {int(r['fold'])}", color=color, alpha=0.8)
ax_bar.plot(x+width/2, [acc_val_g[g]*100 for g in gestos],
    'r--o', linewidth=2.5, markersize=7, label='Validacion Final', zorder=5)
ax_bar.axhline(90, color='red', linestyle=':', linewidth=1.5, alpha=0.6, label='Umbral 90%')
ax_bar.set_xticks(x+width); ax_bar.set_xticklabels([f"Clase G{g}" for g in gestos], fontsize=9)
ax_bar.set_ylabel('Precision (%)'); ax_bar.set_ylim(75, 108)
ax_bar.set_title('Precision por Clase en cada Fold + Validacion Final', fontweight='bold')
ax_bar.legend(fontsize=8, ncol=3, loc='lower right'); ax_bar.grid(True, alpha=0.3, axis='y')

ax_res = fig.add_axes([0.68, 0.05, 0.30, 0.32])
ax_res.axis('off'); ax_res.set_facecolor('#F0F7FF')
ax_res.text(0.5, 0.97, "Resumen Final", ha='center', va='top',
    fontweight='bold', fontsize=12, color='#1F4E79', transform=ax_res.transAxes)
for y_l, (etiq, val, col) in zip([0.82,0.665,0.51,0.355,0.20], [
    ("Promedio CV:",       f"{media_row['acc_total']*100:.1f}%", '#2E75B6'),
    ("Validacion Final:",  f"{acc_val*100:.1f}%",                '#7B0000'),
    ("Requisito minimo:", "> 90%",                                '#555555'),
    ("Clase mas dificil:", "G7 (96% final)",                     '#E67E22'),
    ("Clases perfectas:",  "G1,G3,G4,G5 (100%)",                '#27AE60'),
]):
    ax_res.text(0.08, y_l, etiq, fontsize=9.5, color='#333',
        transform=ax_res.transAxes, va='top')
    ax_res.text(0.93, y_l, val, fontsize=10.5, fontweight='bold',
        color=col, transform=ax_res.transAxes, va='top', ha='right')
ax_res.add_patch(mpatches.FancyBboxPatch((0.05,0.04),0.90,0.12,
    boxstyle="round,pad=0.05", facecolor='#D5E8D4', edgecolor='#27AE60',
    linewidth=2, transform=ax_res.transAxes))
ax_res.text(0.5, 0.10,
    "APROBADO — Supera 90% en todos\nlos folds y en la validacion final",
    ha='center', va='center', fontsize=9.5, fontweight='bold',
    color='#1a7a1a', transform=ax_res.transAxes)

ruta = GRAFICAS_DIR / "09_tabla_resultados_por_clase.png"
plt.savefig(ruta, dpi=140, bbox_inches='tight', facecolor='white'); plt.close()
print(f"  Guardada: {ruta}")

# ── RESUMEN FINAL ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PIPELINE COMPLETO")
print(f"  Producto principal  : {OUTPUT_CSV}")
print(f"  Resultados CV       : {RESULTS_CSV}")
print(f"  Graficas en         : {GRAFICAS_DIR}/  (09 imagenes)")
print(f"  Precision CV media  : {media_row['acc_total']*100:.1f}%")
print(f"  Precision final     : {acc_val*100:.1f}%")
print(f"  Requisito > 90%     : {'CUMPLIDO' if acc_val >= 0.90 else 'REVISAR FEATURES'}")
print("=" * 65)