"""
DATA PREPARATION — Reconocimiento de Gestos en Tableta 2D
Inteligencia Artificial

PIPELINE:
  1. Carga archivos CSV de users_01_to_10/
  2. Detecta duplicados
  3. Limpieza DTW en 2 rondas (elimina basura por joyeria u otros objetos)
     R1: angulo global fuera del rango esperado del gesto
     R2: distancia DTW vs mediana de referencia (> media + K_STD*std)
  4. Extrae 42 features por muestra:
       19 dx + 19 dy (deltas spline, invariantes a traslacion)
       + dx_global + dy_global + angle + total_length
  5. Division: 20% validacion intocable, 80% trabajo
                del 80%: 80% train / 20% test
  6. Cross Validation 5-fold sobre TRAIN (Random Forest de validacion)
  7. Guarda dataset_gestos_final.csv (PRODUCTO PRINCIPAL)

NOTA sobre coordenadas de tableta:
  Y positivo apunta hacia ABAJO (convencion de pantalla).
  Por eso gesto arriba tiene dy negativo, y abajo tiene dy positivo.

  Gestos y sus angulos arctan2(dy, dx):
    G1 ↖  Diagonal arriba-izquierda  ~ -135 grados
    G2 ↗  Diagonal arriba-derecha    ~  -45 grados
    G3 ↙  Diagonal abajo-izquierda   ~  135 grados
    G4 ↘  Diagonal abajo-derecha     ~   45 grados
    G5 ↑  Vertical arriba            ~  -90 grados
    G6 ←  Horizontal izquierda       ~  180 grados
    G7 →  Horizontal derecha         ~    0 grados
    G8 ↓  Vertical abajo             ~   90 grados

USO: python data_final.py
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
    print("[AVISO] dtaidistance no encontrado. pip install dtaidistance")

# ─── CONFIGURACION ────────────────────────────────────────────────
DATA_DIR     = Path("users_01_to_10")
OUTPUT_CSV   = "dataset_gestos_final.csv"
BASURA_CSV   = "basura_detectada.csv"
RESULTS_CSV  = "resultados_cv_final.csv"
GRAFICAS_DIR = Path("graficas_final")
N_RESAMPLE   = 20
N_DTW        = 50
N_FOLDS      = 5
VAL_SIZE     = 0.20
RANDOM_STATE = 42
N_TREES      = 100
ANGULO_TOLERANCIA_DEG = 70
K_STD        = 2.0

# Angulos esperados segun coordenadas de tableta (Y positivo = abajo)
ANGULO_IDEAL = {
    1: -135,  # ↖ arriba-izquierda
    2:  -45,  # ↗ arriba-derecha
    3:  135,  # ↙ abajo-izquierda
    4:   45,  # ↘ abajo-derecha
    5:  -90,  # ↑ vertical arriba
    6:  180,  # ← horizontal izquierda
    7:    0,  # → horizontal derecha
    8:   90,  # ↓ vertical abajo
}

# Nombres con flechas consistentes con el sistema de coordenadas
NOMBRE = {
    1: "Diagonal ↖",   2: "Diagonal ↗",
    3: "Diagonal ↙",   4: "Diagonal ↘",
    5: "Vertical ↑",   6: "Horizontal ←",
    7: "Horizontal →", 8: "Vertical ↓"
}

GRAFICAS_DIR.mkdir(exist_ok=True)


# ── utilidades DTW ─────────────────────────────────────────────────
def resample_traj(x, y, n=N_DTW):
    t = np.linspace(0, 1, len(x))
    tn = np.linspace(0, 1, n)
    return np.column_stack([interp1d(t, x)(tn), interp1d(t, y)(tn)])

def normalize_traj(tr):
    tr = tr - tr[0]
    L = np.sum(np.linalg.norm(np.diff(tr, axis=0), axis=1))
    return tr / L if L > 1e-9 else tr

def dtw_dist(a, b):
    if DTW_AVAILABLE:
        return float(np.sqrt(
            dtw_lib.distance(a[:,0].astype(np.float64), b[:,0].astype(np.float64))**2 +
            dtw_lib.distance(a[:,1].astype(np.float64), b[:,1].astype(np.float64))**2))
    return float(np.sqrt(np.sum((a-b)**2)))

def ang_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


# ════════════════════════════════════════════════════════════════
#  PASO 1 — CARGA
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("PASO 1: Cargando archivos...")
print("=" * 60)

records = []
for udir in sorted(DATA_DIR.iterdir()):
    if not udir.is_dir(): continue
    uid = int(udir.name.split("_")[1])
    for f in sorted(udir.glob("*.csv")):
        parts = f.stem.split("_")
        df = pd.read_csv(f)
        records.append({"user_id": uid, "gesture_id": int(parts[1]),
                        "sample_id": int(parts[3]), "file": str(f),
                        "n_points": len(df), "df": df})

print(f"  {len(records)} archivos cargados")

df_inv = pd.DataFrame([{k:v for k,v in r.items() if k!="df"} for r in records])
pivot  = df_inv.pivot_table(index="user_id", columns="gesture_id",
                             values="sample_id", aggfunc="count", fill_value=0)
pivot.columns = [f"G{c}" for c in pivot.columns]
print("\n  Muestras por usuario/gesto:")
print(pivot.to_string())


# ════════════════════════════════════════════════════════════════
#  PASO 2 — LONGITUDES
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PASO 2: Analisis de longitudes...")
print("=" * 60)
prom = df_inv["n_points"].mean()
print(df_inv.groupby("gesture_id")["n_points"].agg(["min","max","mean"]).round(1).to_string())
print(f"\n  Promedio: {prom:.1f} pts | N_RESAMPLE={N_RESAMPLE} | N_DTW={N_DTW}")


# ════════════════════════════════════════════════════════════════
#  PASO 3 — DUPLICADOS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PASO 3: Duplicados...")
print("=" * 60)
seen, dup = {}, set()
for i, r in enumerate(records):
    k = hash(tuple(r["df"]["x"].round(4)) + tuple(r["df"]["y"].round(4)))
    if k in seen: dup.add(i)
    else: seen[k] = i
print(f"  Duplicados: {len(dup)}")


# ════════════════════════════════════════════════════════════════
#  PASO 3.5 — LIMPIEZA DTW
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PASO 3.5: Limpieza DTW (2 rondas)...")
print("=" * 60)

basura, basura_rows = {}, []
por_gesto = defaultdict(list)
for i, r in enumerate(records):
    if i not in dup and r["n_points"] >= 3:
        por_gesto[r["gesture_id"]].append(i)

for gid in range(1, 9):
    idxs  = por_gesto[gid]
    ideal = ANGULO_IDEAL[gid]
    print(f"\n  G{gid} {NOMBRE[gid]} — {len(idxs)} candidatas")

    # Ronda 1: angulo global
    r2 = []
    for i in idxs:
        r  = records[i]
        x, y = r["df"]["x"].values.astype(float), r["df"]["y"].values.astype(float)
        ang  = float(np.degrees(np.arctan2(y[-1]-y[0], x[-1]-x[0])))
        d    = ang_diff(ang, ideal)
        if gid == 6: d = min(d, ang_diff(ang, -180))
        if d > ANGULO_TOLERANCIA_DEG:
            basura[i] = f"R1: {ang:.1f}° vs ideal {ideal}° (diff={d:.1f}°)"
        else:
            r2.append(i)
    print(f"    R1: -{len(idxs)-len(r2)} eliminadas, {len(r2)} pasan")

    if len(r2) < 3:
        print(f"    R2: omitida (pocas muestras)")
        continue

    # Ronda 2: DTW
    trajs = [normalize_traj(resample_traj(
        records[i]["df"]["x"].values.astype(float),
        records[i]["df"]["y"].values.astype(float))) for i in r2]
    ref   = np.median(np.stack(trajs), axis=0)
    dists = np.array([dtw_dist(t, ref) for t in trajs])
    umbral = dists.mean() + K_STD * dists.std()
    print(f"    R2: media={dists.mean():.4f} std={dists.std():.4f} umbral={umbral:.4f}")
    r2_elim = 0
    for i, d in zip(r2, dists):
        if d > umbral:
            basura[i] = f"R2: DTW={d:.4f} > {umbral:.4f}"
            r2_elim += 1
    print(f"    R2: -{r2_elim} eliminadas")

for idx, mot in basura.items():
    r = records[idx]
    basura_rows.append({"user_id":r["user_id"], "gesture_id":r["gesture_id"],
                        "sample_id":r["sample_id"], "motivo":mot})
pd.DataFrame(basura_rows).to_csv(BASURA_CSV, index=False)

excluir = dup | set(basura.keys())
r1t = sum(1 for m in basura_rows if "R1" in m["motivo"])
r2t = sum(1 for m in basura_rows if "R2" in m["motivo"])
print(f"\n  Duplicados: {len(dup)} | Basura R1: {r1t} | Basura R2: {r2t}")
print(f"  Total eliminadas: {len(excluir)} | Validas: {len(records)-len(excluir)}")


# ════════════════════════════════════════════════════════════════
#  PASO 4 — VISUALIZACIONES
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PASO 4: Visualizaciones...")
print("=" * 60)

limpios = {g: [i for i,r in enumerate(records) if r["gesture_id"]==g and i not in excluir]
           for g in range(1,9)}
sucios  = {g: [i for i,r in enumerate(records) if r["gesture_id"]==g and i in excluir]
           for g in range(1,9)}
cu = cm.tab10(np.linspace(0,1,10))
cb = cm.Blues(np.linspace(0.35,0.9,5))

# 01 - user_01
fig, axes = plt.subplots(2, 4, figsize=(16,8))
fig.suptitle("Gestos — user_01  (● inicio, ▲ fin)", fontsize=12, fontweight="bold")
for gi, g in enumerate(range(1,9)):
    ax = axes[gi//4][gi%4]
    for k, f in enumerate(sorted((DATA_DIR/"user_01").glob(f"gesture_0{g}_sample_*.csv"))[:5]):
        df = pd.read_csv(f)
        ax.plot(df["x"], df["y"], color=cb[k], alpha=0.8, lw=1.8)
        ax.plot(df["x"].iloc[0], df["y"].iloc[0], "go", ms=5)
        ax.plot(df["x"].iloc[-1], df["y"].iloc[-1], "r^", ms=5)
    ax.set_title(f"G{g} {NOMBRE[g]}", fontweight="bold", fontsize=9)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    ax.invert_yaxis()  # <--- CORRECCIÓN AÑADIDA AQUÍ
plt.tight_layout()
plt.savefig(GRAFICAS_DIR/"01_trayectorias_user01.png", dpi=120, bbox_inches="tight")
plt.close(); print("  01_trayectorias_user01.png")

# 02 - Antes (basura rojo)
fig, axes = plt.subplots(2, 4, figsize=(18,9))
fig.suptitle("ANTES de limpieza  (rojo = eliminada)", fontsize=12, fontweight="bold")
for gi, g in enumerate(range(1,9)):
    ax = axes[gi//4][gi%4]
    for i, r in enumerate(records):
        if r["gesture_id"] != g: continue
        es_b = i in excluir
        ax.plot(r["df"]["x"], r["df"]["y"],
                color="red" if es_b else cu[r["user_id"]-1],
                alpha=0.7 if es_b else 0.20, lw=1.8 if es_b else 0.9,
                zorder=5 if es_b else 1)
    ax.set_title(f"G{g} {NOMBRE[g]}", fontweight="bold", fontsize=8)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    ax.invert_yaxis()  # <--- CORRECCIÓN AÑADIDA AQUÍ
axes[0][0].legend(handles=[
    Line2D([0],[0],color="gray",alpha=0.5,label="Valida"),
    Line2D([0],[0],color="red",alpha=0.8,label="Eliminada")], fontsize=7)
plt.tight_layout()
plt.savefig(GRAFICAS_DIR/"02_antes_limpieza.png", dpi=120, bbox_inches="tight")
plt.close(); print("  02_antes_limpieza.png")

# 03 - Despues (solo validas)
fig, axes = plt.subplots(2, 4, figsize=(18,9))
fig.suptitle("DESPUES de limpieza DTW", fontsize=12, fontweight="bold")
for gi, g in enumerate(range(1,9)):
    ax = axes[gi//4][gi%4]
    for i in limpios[g]:
        r = records[i]
        ax.plot(r["df"]["x"], r["df"]["y"], color=cu[r["user_id"]-1], alpha=0.3, lw=0.9)
    ax.set_title(f"G{g} {NOMBRE[g]}\n({len(limpios[g])} validas, {len(sucios[g])} elim.)",
                 fontweight="bold", fontsize=8)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    ax.invert_yaxis()  # <--- CORRECCIÓN AÑADIDA AQUÍ
plt.tight_layout()
plt.savefig(GRAFICAS_DIR/"03_despues_limpieza_DTW.png", dpi=120, bbox_inches="tight")
plt.close(); print("  03_despues_limpieza_DTW.png")


# ════════════════════════════════════════════════════════════════
#  PASO 5 — EXTRACCION DE FEATURES
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"PASO 5: Extraccion de features (N={N_RESAMPLE})...")
print("=" * 60)

def extract_features(df_s):
    x = df_s["x"].values.astype(float)
    y = df_s["y"].values.astype(float)
    try:
        tck,_ = splprep([x,y], s=0, k=min(3,len(x)-1))
        xr,yr = splev(np.linspace(0,1,N_RESAMPLE), tck)
    except Exception:
        t = np.linspace(0,1,len(x)); tn = np.linspace(0,1,N_RESAMPLE)
        xr = np.interp(tn,t,x); yr = np.interp(tn,t,y)
    dxg = float(x[-1]-x[0]); dyg = float(y[-1]-y[0])
    return (list(np.diff(xr)) + list(np.diff(yr)) +
            [dxg, dyg, float(np.arctan2(dyg,dxg)),
             float(np.sum(np.sqrt(np.diff(x)**2+np.diff(y)**2)))])

rows = []
for i, r in enumerate(records):
    if i in excluir or r["n_points"] < 3: continue
    rows.append([r["user_id"], r["gesture_id"]] + extract_features(r["df"]))

N_D = N_RESAMPLE - 1
feat_cols = ([f"dx_{i}" for i in range(N_D)] + [f"dy_{i}" for i in range(N_D)] +
             ["dx_global","dy_global","angle","total_length"])
df_full = pd.DataFrame(rows, columns=["user_id","gesture_label"]+feat_cols)
print(f"  Dataset: {df_full.shape[0]} muestras x {df_full.shape[1]} cols")
print(f"  Por gesto:"); print(df_full["gesture_label"].value_counts().sort_index().to_string())


# ════════════════════════════════════════════════════════════════
#  PASO 6 — DIVISION
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PASO 6: Division de datos...")
print("=" * 60)

X = df_full[feat_cols].values
y = df_full["gesture_label"].values

# Capa 1: 20% validacion intocable
ia = np.arange(len(df_full))
iw, iv = train_test_split(ia, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y)
Xw, yw = X[iw], y[iw]
Xv, yv = X[iv], y[iv]

# Capa 2: del 80%, 80% train / 20% test
is_ = np.arange(len(iw))
itr_s, ite_s = train_test_split(is_, test_size=0.20, random_state=RANDOM_STATE, stratify=yw)
itr, ite = iw[itr_s], iw[ite_s]
Xtr, ytr = X[itr], y[itr]
Xte, yte = X[ite], y[ite]

df_full["split"] = "validation"
df_full.loc[itr, "split"] = "train"
df_full.loc[ite, "split"] = "test"
df_full["fold"] = 0

N = len(df_full)
print(f"  Total: {N} | Train: {len(itr)} ({len(itr)/N*100:.0f}%) | "
      f"Test: {len(ite)} ({len(ite)/N*100:.0f}%) | Val: {len(iv)} ({len(iv)/N*100:.0f}%)")
print(f"\n  Por gesto (train/test/val):")
for g in range(1,9):
    m = df_full["gesture_label"]==g
    tr = (df_full.loc[m,"split"]=="train").sum()
    te = (df_full.loc[m,"split"]=="test").sum()
    va = (df_full.loc[m,"split"]=="validation").sum()
    print(f"    G{g} {NOMBRE[g]}: train={tr} test={te} val={va}")


# ════════════════════════════════════════════════════════════════
#  PASO 7 — CROSS VALIDATION (Random Forest — valida features)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"PASO 7: CV 5-fold sobre TRAIN (Random Forest)...")
print("=" * 60)

gestos = list(range(1,9))
skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
res    = []
fdata  = []

print(f"\n  {'Fold':<6} {'Total':>7}   " + "   ".join([f"G{g}" for g in gestos]))
print("  " + "-"*70)

for fold, (tri, tei) in enumerate(skf.split(Xtr, ytr), 1):
    clf = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(Xtr[tri], ytr[tri])
    yp  = clf.predict(Xtr[tei])
    yt  = ytr[tei]
    at  = accuracy_score(yt, yp)
    ag  = {g: accuracy_score(yt[yt==g], yp[yt==g]) if (yt==g).sum()>0 else 0.0 for g in gestos}
    res.append({"fold":fold,"acc_total":at,"n_train":len(tri),"n_test":len(tei),
                **{f"acc_G{g}":ag[g] for g in gestos}})
    fdata.append({"fold":fold,
                  "train_counts":[int((ytr[tri]==g).sum()) for g in gestos],
                  "test_counts" :[int((yt==g).sum()) for g in gestos],
                  "acc_total":at,"acc_gestos":[ag[g] for g in gestos]})
    print(f"  Fold {fold}  {at*100:>6.1f}%   "+"   ".join([f"{ag[g]*100:>4.0f}%" for g in gestos]))

df_res = pd.DataFrame(res)
med    = df_res[[f"acc_G{g}" for g in gestos]+["acc_total"]].mean()
print("  "+"-"*70)
print(f"  Media  {med['acc_total']*100:>6.1f}%   "+"   ".join([f"{med[f'acc_G{g}']*100:>4.0f}%" for g in gestos]))

# Modelo final RF: train+test → validacion
clf_f = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
clf_f.fit(Xw, yw)
ypv   = clf_f.predict(Xv)
av    = accuracy_score(yv, ypv)
avg   = {g: accuracy_score(yv[yv==g], ypv[yv==g]) if (yv==g).sum()>0 else 0.0 for g in gestos}
clf_t = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
clf_t.fit(Xtr, ytr)
ypt   = clf_t.predict(Xte)
at    = accuracy_score(yte, ypt)

print(f"\n  Test       : {at*100:.1f}%")
print(f"  Validacion : {av*100:.1f}%  {'CUMPLE >90%' if av>=0.90 else 'NO CUMPLE'}")
print(f"  Por gesto  : "+"  ".join([f"G{g}:{avg[g]*100:.0f}%" for g in gestos]))
df_res.to_csv(RESULTS_CSV, index=False)


# ════════════════════════════════════════════════════════════════
#  PASO 8 — GUARDAR CSV
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PASO 8: Guardando dataset...")
print("=" * 60)
df_full.to_csv(OUTPUT_CSV, index=False)
print(f"  {OUTPUT_CSV} — {df_full.shape[0]} filas x {df_full.shape[1]} cols")
print(f"  splits: {df_full['split'].value_counts().to_dict()}")


# ════════════════════════════════════════════════════════════════
#  PASO 9 — GRAFICAS DE RESULTADOS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PASO 9: Graficas de resultados...")
print("=" * 60)

# 05 - Distribucion final
fig, axes = plt.subplots(1, 2, figsize=(13,5))
fig.suptitle("Dataset limpio — distribucion final", fontsize=13, fontweight="bold")
cg = pd.Series({g:len(limpios[g]) for g in range(1,9)})
cu2= pd.Series({u:sum(1 for i,r in enumerate(records) if r["user_id"]==u and i not in excluir) for u in range(1,11)})
axes[0].bar([f"G{i}" for i in cg.index], cg.values, color="steelblue", edgecolor="white")
axes[0].set_title("Por Gesto"); axes[0].set_ylabel("N muestras")
for i,v in enumerate(cg.values): axes[0].text(i,v+0.5,str(v),ha="center",fontsize=9)
axes[1].bar([f"U{i}" for i in cu2.index], cu2.values, color="coral", edgecolor="white")
axes[1].set_title("Por Usuario"); axes[1].set_ylabel("N muestras")
for i,v in enumerate(cu2.values): axes[1].text(i,v+0.5,str(v),ha="center",fontsize=9)
plt.tight_layout()
plt.savefig(GRAFICAS_DIR/"05_distribucion_limpia.png", dpi=120, bbox_inches="tight")
plt.close(); print("  05_distribucion_limpia.png")

# 06 - Estructura de division
fig, ax = plt.subplots(figsize=(13,7)); ax.set_xlim(0,13); ax.set_ylim(0,9); ax.axis("off")
ax.set_title("Division de datos (indicacion de la maestra)", fontsize=12, fontweight="bold")
ax.add_patch(mpatches.FancyBboxPatch((0.3,7.5),12.2,1.0,boxstyle="round,pad=0.1",facecolor="#DDD",edgecolor="#555",lw=2))
ax.text(6.4,8.05,f"DATASET LIMPIO — {len(df_full)} muestras (100%)",ha="center",fontweight="bold",fontsize=10)
for xp,lbl in [(3.5,"80%"),(10,"20%")]:
    ax.annotate("",xy=(xp,7.2),xytext=(xp,7.5),arrowprops=dict(arrowstyle="->",color="black",lw=1.5))
ax.add_patch(mpatches.FancyBboxPatch((0.3,6.0),7.0,1.0,boxstyle="round,pad=0.1",facecolor="#2E75B6",edgecolor="#1F4E79",lw=2,alpha=0.85))
ax.text(3.8,6.55,f"80% Trabajo — {len(iw)} muestras",ha="center",color="white",fontweight="bold",fontsize=10)
ax.add_patch(mpatches.FancyBboxPatch((7.6,6.0),4.9,1.0,boxstyle="round,pad=0.1",facecolor="#C00000",edgecolor="#900000",lw=2,alpha=0.9))
ax.text(10.05,6.55,f"20% VALIDACION — {len(iv)} muestras\n(INTOCABLE)",ha="center",color="white",fontweight="bold",fontsize=9)
for xp in [2.2,5.6]:
    ax.annotate("",xy=(xp,5.7),xytext=(xp,6.0),arrowprops=dict(arrowstyle="->",color="#1F4E79",lw=1.5))
ax.add_patch(mpatches.FancyBboxPatch((0.3,4.5),3.7,1.0,boxstyle="round,pad=0.1",facecolor="#1F4E79",edgecolor="#1F4E79",lw=2,alpha=0.9))
ax.text(2.2,5.05,f"TRAIN — {len(itr)} ({len(itr)/N*100:.0f}%)",ha="center",color="white",fontweight="bold",fontsize=10)
ax.add_patch(mpatches.FancyBboxPatch((4.3,4.5),3.0,1.0,boxstyle="round,pad=0.1",facecolor="#27AE60",edgecolor="#1E8449",lw=2,alpha=0.9))
ax.text(5.8,5.05,f"TEST — {len(ite)} ({len(ite)/N*100:.0f}%)",ha="center",color="white",fontweight="bold",fontsize=10)
ax.annotate("",xy=(2.2,4.2),xytext=(2.2,4.5),arrowprops=dict(arrowstyle="->",color="#1F4E79",lw=2,linestyle="dashed"))
ax.text(2.2,3.9,f"CV {N_FOLDS}-Fold\n(verifica overfitting)",ha="center",fontsize=8.5,color="#1F4E79",fontweight="bold",
    bbox=dict(boxstyle="round",facecolor="#D6EAF8",alpha=0.85))
ax.text(6.4,1.5,f"CV media: {med['acc_total']*100:.1f}%  |  Test: {at*100:.1f}%  |  Val. final: {av*100:.1f}%",
    ha="center",fontsize=10,fontweight="bold",
    color="#1a7a1a" if av>=0.90 else "#CC0000",
    bbox=dict(boxstyle="round",facecolor="#D5E8D4" if av>=0.90 else "#FFE0E0",alpha=0.9))
plt.tight_layout()
plt.savefig(GRAFICAS_DIR/"06_estructura_division.png", dpi=120, bbox_inches="tight")
plt.close(); print("  06_estructura_division.png")

# 07 - Tabla de resultados CV
fig = plt.figure(figsize=(16,9)); fig.patch.set_facecolor('white')
fig.text(0.5,0.97,"Resultados CV — Random Forest (validacion de features)",
    ha='center',va='top',fontsize=13,fontweight='bold',color='#1F4E79')
ax_t = fig.add_axes([0.02,0.48,0.96,0.45]); ax_t.axis('off')
clt = ["","Total"]+[f"G{g}" for g in gestos]
fd2 = []
for r in res:
    fd2.append([f"Fold {int(r['fold'])}",f"{r['acc_total']*100:.1f}%"]+[f"{r[f'acc_G{g}']*100:.0f}%" for g in gestos])
fd2.append(["Media",f"{med['acc_total']*100:.1f}%"]+[f"{med[f'acc_G{g}']*100:.0f}%" for g in gestos])
fd2.append(["Val.Final",f"{av*100:.1f}%"]+[f"{avg[g]*100:.0f}%" for g in gestos])
tbl = ax_t.table(cellText=fd2,colLabels=clt,cellLoc='center',loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1,2.5)
for j in range(len(clt)):
    tbl[0,j].set_facecolor('#1F4E79'); tbl[0,j].set_text_props(color='white',fontweight='bold')
for i in range(1,N_FOLDS+1):
    tbl[i,0].set_facecolor('#D6E4F0'); tbl[i,0].set_text_props(fontweight='bold',color='#1F4E79')
    for j in range(1,len(clt)):
        v=float(fd2[i-1][j].replace('%',''))
        bg='#C8F0C8' if v>=99 else ('#D5E8D4' if v>=90 else '#FFE0E0')
        co='#1a7a1a' if v>=90 else '#CC0000'
        tbl[i,j].set_facecolor(bg); tbl[i,j].set_text_props(color=co,fontweight='bold')
for j in range(len(clt)):
    tbl[N_FOLDS+1,j].set_facecolor('#2E75B6'); tbl[N_FOLDS+1,j].set_text_props(color='white',fontweight='bold')
    tbl[N_FOLDS+2,j].set_facecolor('#7B0000'); tbl[N_FOLDS+2,j].set_text_props(color='white',fontweight='bold')
ax_b = fig.add_axes([0.03,0.04,0.60,0.37])
xp = np.arange(8); w = 0.13
for i,(r,c) in enumerate(zip(res,['#2980B9','#27AE60','#E67E22','#8E44AD','#16A085'])):
    ax_b.bar(xp+i*w-w*2,[r[f'acc_G{g}']*100 for g in gestos],w,label=f"Fold {int(r['fold'])}",color=c,alpha=0.8)
ax_b.plot(xp+w/2,[avg[g]*100 for g in gestos],'r--o',lw=2.5,ms=7,label='Val.Final',zorder=5)
ax_b.axhline(90,color='red',ls=':',lw=1.5,alpha=0.6,label='90%')
ax_b.set_xticks(xp+w); ax_b.set_xticklabels([f"G{g}\n{NOMBRE[g]}" for g in gestos],fontsize=7.5)
ax_b.set_ylim(60,108); ax_b.set_ylabel('Accuracy (%)'); ax_b.set_title('Accuracy por clase',fontweight='bold')
ax_b.legend(fontsize=8,ncol=3,loc='lower right'); ax_b.grid(True,alpha=0.3,axis='y')
ax_r = fig.add_axes([0.66,0.04,0.32,0.37]); ax_r.axis('off'); ax_r.set_facecolor('#F0F7FF')
ax_r.text(0.5,0.96,"Resumen",ha='center',va='top',fontweight='bold',fontsize=12,color='#1F4E79',transform=ax_r.transAxes)
for yi,(lbl,val,col) in zip(np.linspace(0.82,0.22,5),[
    ("CV media:",f"{med['acc_total']*100:.1f}%",'#2E75B6'),
    ("Test:",f"{at*100:.1f}%",'#27AE60'),
    ("Val.Final:",f"{av*100:.1f}%",'#7B0000'),
    ("Elim.DTW:",f"{len(basura)} muestras",'#E67E22'),
    ("Dataset:",f"{len(df_full)} muestras",'#27AE60'),
]):
    ax_r.text(0.08,yi,lbl,fontsize=10,color='#333',transform=ax_r.transAxes,va='center')
    ax_r.text(0.92,yi,val,fontsize=11,fontweight='bold',color=col,transform=ax_r.transAxes,va='center',ha='right')
fc='#D5E8D4' if av>=0.90 else '#FFE0E0'
ec='#27AE60' if av>=0.90 else '#CC0000'
ax_r.add_patch(mpatches.FancyBboxPatch((0.05,0.04),0.90,0.10,boxstyle="round,pad=0.05",
    facecolor=fc,edgecolor=ec,lw=2,transform=ax_r.transAxes))
ax_r.text(0.5,0.09,f"{'CUMPLE' if av>=0.90 else 'REVISAR'} >90%",ha='center',va='center',
    fontsize=11,fontweight='bold',color='#1a7a1a' if av>=0.90 else '#CC0000',transform=ax_r.transAxes)
plt.savefig(GRAFICAS_DIR/"07_resultados_cv.png",dpi=130,bbox_inches='tight',facecolor='white')
plt.close(); print("  07_resultados_cv.png")


print("\n" + "=" * 60)
print("  PIPELINE COMPLETO")
print(f"  CSV principal : {OUTPUT_CSV}")
print(f"  Basura        : {BASURA_CSV}")
print(f"  Resultados CV : {RESULTS_CSV}")
print(f"  Graficas      : {GRAFICAS_DIR}/ (7 imagenes)")
print(f"  Muestras      : {len(records)} originales → {len(df_full)} limpias")
print(f"  Eliminadas    : dup={len(dup)} R1={r1t} R2={r2t}")
print(f"  CV media RF   : {med['acc_total']*100:.1f}%")
print(f"  Validacion RF : {av*100:.1f}%")
print("=" * 60)