"""
=============================================================
  SVM - RECONOCIMIENTO DE GESTOS EN TABLETA
  Inteligencia Artificial  —  svm.py
=============================================================
  Lee el dataset limpio generado por data_final.py y entrena
  un clasificador SVM para reconocimiento de los 8 gestos.

  FLUJO:
  1. Cargar dataset_gestos_final.csv
  2. Separar splits: train / test / validation
  3. Normalizar features con StandardScaler
     (SVM es sensible a la escala — paso OBLIGATORIO)
  4. Cross Validation 5-Fold sobre TRAIN
     → verifica que no haya overfitting
  5. Evaluar en TEST → resultado intermedio
  6. Si TEST >= 90% → evaluar en VALIDACION (1 sola vez)
  7. Reportar accuracy total y por clase
  8. Guardar resultados y graficas

  POR QUE SVM:
  • Efectivo en espacios de alta dimension (42 features)
  • Margen maximo entre clases → buena generalizacion
  • Kernel RBF captura relaciones no lineales entre features
  • Requiere normalizacion previa (StandardScaler)

  DIFERENCIA CON RANDOM FOREST:
  • RF: conjunto de arboles, robusto sin normalizar
  • SVM: hiperplano de margen maximo, REQUIERE normalizacion
  • SVM suele ser mas preciso con features bien preprocesados

  USO:
    python svm.py

  ENTRADA:
    dataset_gestos_final.csv  (generado por data_final.py)

  SALIDA:
    resultados_svm.csv        <- accuracy por fold y clase
    graficas_svm/             <- graficas de resultados
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ─── CONFIGURACION ────────────────────────────────────────────
INPUT_CSV    = "dataset_gestos_final.csv"
RESULTS_CSV  = "resultados_svm.csv"
GRAFICAS_DIR = Path("graficas_svm")
N_FOLDS      = 5
RANDOM_STATE = 42

# Hiperparametros SVM
# kernel='rbf': Radial Basis Function, captura relaciones no lineales
# C=10: penalizacion por error (mas alto = menos margen, mas ajuste)
# gamma='scale': 1/(n_features * X.var()) — buena opcion por defecto
SVM_KERNEL = 'rbf'
SVM_C      = 10
SVM_GAMMA  = 'scale'

GRAFICAS_DIR.mkdir(exist_ok=True)

DIRECCIONES = {
    1: "Diagonal NW", 2: "Diagonal NE",
    3: "Diagonal SW", 4: "Diagonal SE",
    5: "Vertical U",  6: "Horizontal L",
    7: "Horizontal R", 8: "Vertical D"
}

# ════════════════════════════════════════════════════════════════
#  PASO 1 — CARGAR DATASET
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("PASO 1: Cargando dataset...")
print("=" * 65)

if not Path(INPUT_CSV).exists():
    raise FileNotFoundError(
        f"No se encontro '{INPUT_CSV}'.\n"
        "Ejecuta primero: python data_final.py"
    )

df = pd.read_csv(INPUT_CSV)
print(f"  Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"  Splits presentes: {df['split'].value_counts().to_dict()}")
print(f"  Clases: {sorted(df['gesture_label'].unique())}")

# Columnas de features (las 42)
feat_cols = [c for c in df.columns
             if c not in ("user_id", "gesture_label", "split", "fold")]
print(f"  Features: {len(feat_cols)}")


# ════════════════════════════════════════════════════════════════
#  PASO 2 — SEPARAR SPLITS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 2: Separando splits...")
print("=" * 65)

df_train = df[df["split"] == "train"].copy()
df_test  = df[df["split"] == "test"].copy()
df_val   = df[df["split"] == "validation"].copy()

X_train = df_train[feat_cols].values
y_train = df_train["gesture_label"].values
X_test  = df_test[feat_cols].values
y_test  = df_test["gesture_label"].values
X_val   = df_val[feat_cols].values
y_val   = df_val["gesture_label"].values

print(f"  Train      : {len(df_train):>5} muestras")
print(f"  Test       : {len(df_test):>5} muestras")
print(f"  Validacion : {len(df_val):>5} muestras  <- INTOCABLE hasta el final")

print(f"\n  Muestras por clase en TRAIN:")
for g in range(1, 9):
    n = (y_train == g).sum()
    print(f"    G{g} ({DIRECCIONES[g]}): {n}")


# ════════════════════════════════════════════════════════════════
#  PASO 3 — NORMALIZACION (OBLIGATORIO PARA SVM)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 3: Normalizando features (StandardScaler)...")
print("=" * 65)

# CRITICO: el scaler se ajusta SOLO con TRAIN
# Luego se aplica el mismo scaler a test y validacion
# Nunca ajustar con test o validacion -> data leakage
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # ajusta Y transforma
X_test_sc  = scaler.transform(X_test)        # solo transforma
X_val_sc   = scaler.transform(X_val)         # solo transforma

print(f"  Scaler ajustado sobre TRAIN ({len(X_train)} muestras)")
print(f"  Media de features (primeros 5): {scaler.mean_[:5].round(3)}")
print(f"  Std  de features (primeros 5): {scaler.scale_[:5].round(3)}")
print(f"  Aplicado a test y validacion con los mismos parametros")


# ════════════════════════════════════════════════════════════════
#  PASO 4 — CROSS VALIDATION SOBRE TRAIN
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"PASO 4: Cross Validation SVM ({N_FOLDS} folds sobre TRAIN)...")
print("=" * 65)
print(f"  Kernel={SVM_KERNEL}  C={SVM_C}  gamma={SVM_GAMMA}")

gestos   = list(range(1, 9))
skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_res = []

print(f"\n  {'Fold':<6} {'Total':>8}   " + "   ".join([f"G{g}" for g in gestos]))
print("  " + "-" * 72)

for fold, (tr_idx, te_idx) in enumerate(skf.split(X_train_sc, y_train), 1):
    X_tr = X_train_sc[tr_idx]; y_tr = y_train[tr_idx]
    X_te = X_train_sc[te_idx]; y_te = y_train[te_idx]

    svm = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA,
              random_state=RANDOM_STATE, decision_function_shape='ovr')
    svm.fit(X_tr, y_tr)
    y_pred    = svm.predict(X_te)
    acc_total = accuracy_score(y_te, y_pred)
    acc_g     = {g: accuracy_score(y_te[y_te==g], y_pred[y_te==g])
                 if (y_te==g).sum() > 0 else 0.0 for g in gestos}

    fold_res.append({
        "fold": fold, "acc_total": acc_total,
        "n_train": len(y_tr), "n_test": len(y_te),
        **{f"acc_G{g}": acc_g[g] for g in gestos}
    })
    print(f"  Fold {fold}  {acc_total*100:>6.1f}%   " +
          "   ".join([f"{acc_g[g]*100:>4.0f}%" for g in gestos]))

df_cv   = pd.DataFrame(fold_res)
media   = df_cv[[f"acc_G{g}" for g in gestos] + ["acc_total"]].mean()

print("  " + "-" * 72)
print(f"  Media  {media['acc_total']*100:>6.1f}%   " +
      "   ".join([f"{media[f'acc_G{g}']*100:>4.0f}%" for g in gestos]))

overfitting_ok = media['acc_total'] >= 0.85
print(f"\n  CV promedio: {media['acc_total']*100:.1f}%  ->  "
      f"{'Sin overfitting aparente, continuar' if overfitting_ok else 'Posible problema, revisar features o C'}")


# ════════════════════════════════════════════════════════════════
#  PASO 5 — ENTRENAR CON TRAIN COMPLETO, EVALUAR EN TEST
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 5: Evaluacion en TEST...")
print("=" * 65)

svm_train = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA,
                random_state=RANDOM_STATE, decision_function_shape='ovr')
svm_train.fit(X_train_sc, y_train)
y_pred_test = svm_train.predict(X_test_sc)
acc_test    = accuracy_score(y_test, y_pred_test)
acc_test_g  = {g: accuracy_score(y_test[y_test==g], y_pred_test[y_test==g])
               if (y_test==g).sum() > 0 else 0.0 for g in gestos}

print(f"  Modelo entrenado con TRAIN ({len(X_train)} muestras)")
print(f"  Evaluado  en TEST  ({len(X_test)} muestras)")
print(f"\n  Precision total en TEST: {acc_test*100:.1f}%")
print(f"  Por gesto: " + "   ".join([f"G{g}:{acc_test_g[g]*100:.0f}%" for g in gestos]))

test_ok = acc_test >= 0.90
print(f"\n  TEST {'>= 90% OK' if test_ok else '< 90% — considerar ajustes'}")


# ════════════════════════════════════════════════════════════════
#  PASO 6 — MODELO FINAL: TRAIN+TEST → VALIDACION
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 6: Evaluacion FINAL en VALIDACION (1 sola vez)...")
print("=" * 65)

# Combinar train + test para el modelo final
X_trabajo    = np.vstack([X_train_sc, X_test_sc])
y_trabajo    = np.concatenate([y_train, y_test])

svm_final = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA,
                random_state=RANDOM_STATE, decision_function_shape='ovr')
svm_final.fit(X_trabajo, y_trabajo)
y_pred_val = svm_final.predict(X_val_sc)
acc_val    = accuracy_score(y_val, y_pred_val)
acc_val_g  = {g: accuracy_score(y_val[y_val==g], y_pred_val[y_val==g])
              if (y_val==g).sum() > 0 else 0.0 for g in gestos}

print(f"  Modelo final entrenado con TRAIN+TEST ({len(X_trabajo)} muestras)")
print(f"  Evaluado en VALIDACION ({len(X_val)} muestras) — una sola vez")
print(f"\n  {'='*65}")
print(f"  RESULTADO FINAL")
print(f"  CV media     : {media['acc_total']*100:.1f}%")
print(f"  Test         : {acc_test*100:.1f}%")
print(f"  Validacion   : {acc_val*100:.1f}%")
print(f"  Por gesto    : " + "   ".join([f"G{g}:{acc_val_g[g]*100:.0f}%" for g in gestos]))
print(f"  {'='*65}")
print(f"  Requisito >90%: {'CUMPLIDO' if acc_val >= 0.90 else 'NO CUMPLIDO — ajustar C o kernel'}")


# ════════════════════════════════════════════════════════════════
#  PASO 7 — GUARDAR RESULTADOS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 7: Guardando resultados...")
print("=" * 65)

# Agregar filas de test y validacion al CSV
fila_test = {"fold": "test", "acc_total": acc_test,
             "n_train": len(X_train), "n_test": len(X_test),
             **{f"acc_G{g}": acc_test_g[g] for g in gestos}}
fila_val  = {"fold": "validacion", "acc_total": acc_val,
             "n_train": len(X_trabajo), "n_test": len(X_val),
             **{f"acc_G{g}": acc_val_g[g] for g in gestos}}

df_res = pd.concat([df_cv,
                    pd.DataFrame([fila_test]),
                    pd.DataFrame([fila_val])],
                   ignore_index=True)
df_res.to_csv(RESULTS_CSV, index=False)
print(f"  Guardado: {RESULTS_CSV}")


# ════════════════════════════════════════════════════════════════
#  PASO 8 — GRAFICAS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 8: Generando graficas...")
print("=" * 65)

# ── Grafica 1: Tabla de resultados CV + test + validacion ───────
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('white')
fig.text(0.5, 0.97,
    f"SVM — Resultados por Clase  (kernel={SVM_KERNEL}, C={SVM_C}, gamma={SVM_GAMMA})",
    ha='center', va='top', fontsize=13, fontweight='bold', color='#1F4E79')

ax_t = fig.add_axes([0.03, 0.48, 0.94, 0.45])
ax_t.axis('off')

col_labels = ["", "Total"] + [f"G{g}" for g in gestos]
filas = []
for r in fold_res:
    filas.append([f"Fold {int(r['fold'])}", f"{r['acc_total']*100:.1f}%"] +
                 [f"{r[f'acc_G{g}']*100:.0f}%" for g in gestos])
filas.append(["Media CV", f"{media['acc_total']*100:.1f}%"] +
             [f"{media[f'acc_G{g}']*100:.0f}%" for g in gestos])
filas.append(["Test", f"{acc_test*100:.1f}%"] +
             [f"{acc_test_g[g]*100:.0f}%" for g in gestos])
filas.append(["Validacion", f"{acc_val*100:.1f}%"] +
             [f"{acc_val_g[g]*100:.0f}%" for g in gestos])

tbl = ax_t.table(cellText=filas, colLabels=col_labels,
                 cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.6)

# Header
for j in range(len(col_labels)):
    tbl[0,j].set_facecolor('#1F4E79')
    tbl[0,j].set_text_props(color='white', fontweight='bold', fontsize=10)

# Folds
for i in range(1, N_FOLDS+1):
    tbl[i,0].set_facecolor('#D6E4F0')
    tbl[i,0].set_text_props(fontweight='bold', color='#1F4E79')
    at = fold_res[i-1]['acc_total']
    tbl[i,1].set_facecolor('#D5E8D4' if at>=0.90 else '#FFE0E0')
    tbl[i,1].set_text_props(fontweight='bold',
                             color='#1a7a1a' if at>=0.90 else '#CC0000')
    for j, g in enumerate(gestos):
        ag = fold_res[i-1][f'acc_G{g}']
        bg = '#C8F0C8' if ag>=0.98 else ('#FFF0C0' if ag>=0.90 else '#FFCCCC')
        tbl[i,j+2].set_facecolor(bg)
        tbl[i,j+2].set_text_props(fontweight='bold',
                                   color='#1a7a1a' if ag>=0.90 else '#CC0000')

# Media CV
row_m = N_FOLDS + 1
for j in range(len(col_labels)):
    tbl[row_m,j].set_facecolor('#2E75B6')
    tbl[row_m,j].set_text_props(color='white', fontweight='bold')

# Test
row_t = N_FOLDS + 2
for j in range(len(col_labels)):
    tbl[row_t,j].set_facecolor('#27AE60')
    tbl[row_t,j].set_text_props(color='white', fontweight='bold')

# Validacion
row_v = N_FOLDS + 3
for j in range(len(col_labels)):
    tbl[row_v,j].set_facecolor('#7B0000')
    tbl[row_v,j].set_text_props(color='white', fontweight='bold')

# Barras por clase abajo a la izquierda
ax_b = fig.add_axes([0.04, 0.05, 0.58, 0.36])
x     = np.arange(8); width = 0.20
colores_fold = ['#2980B9','#27AE60','#E67E22','#8E44AD','#16A085']
for i, (fr, col) in enumerate(zip(fold_res, colores_fold)):
    ax_b.bar(x + i*width - width*2,
             [fr[f'acc_G{g}']*100 for g in gestos],
             width, label=f"Fold {int(fr['fold'])}", color=col, alpha=0.75)
ax_b.plot(x + width/2, [acc_val_g[g]*100 for g in gestos],
          'r--o', linewidth=2.5, markersize=7, label='Validacion', zorder=5)
ax_b.plot(x + width/2, [acc_test_g[g]*100 for g in gestos],
          'g--s', linewidth=2.0, markersize=6, label='Test', zorder=4)
ax_b.axhline(90, color='red', linestyle=':', linewidth=1.5, alpha=0.6, label='90%')
ax_b.set_xticks(x + width/2)
ax_b.set_xticklabels([f"G{g}" for g in gestos], fontsize=9)
ax_b.set_ylabel('Precision (%)')
ax_b.set_ylim(60, 108)
ax_b.set_title('Precision por clase', fontweight='bold')
ax_b.legend(fontsize=8, ncol=3, loc='lower right')
ax_b.grid(True, alpha=0.3, axis='y')

# Resumen a la derecha
ax_r = fig.add_axes([0.65, 0.05, 0.33, 0.36])
ax_r.axis('off'); ax_r.set_facecolor('#F0F7FF')
ax_r.text(0.5, 0.97, "Resumen SVM", ha='center', va='top',
    fontweight='bold', fontsize=12, color='#1F4E79', transform=ax_r.transAxes)
items = [
    ("Kernel:",       SVM_KERNEL,                          '#555555'),
    ("C:",            str(SVM_C),                          '#555555'),
    ("Gamma:",        str(SVM_GAMMA),                      '#555555'),
    ("CV media:",     f"{media['acc_total']*100:.1f}%",   '#2E75B6'),
    ("Test:",         f"{acc_test*100:.1f}%",              '#27AE60'),
    ("Validacion:",   f"{acc_val*100:.1f}%",               '#7B0000'),
    ("Req. > 90%:",   "CUMPLIDO" if acc_val>=0.90 else "REVISAR",
                      '#1a7a1a' if acc_val>=0.90 else '#CC0000'),
]
for yi, (lbl, val, col) in zip(np.linspace(0.83, 0.15, 7), items):
    ax_r.text(0.08, yi, lbl, fontsize=10, color='#333',
        transform=ax_r.transAxes, va='center')
    ax_r.text(0.92, yi, val, fontsize=11, fontweight='bold',
        color=col, transform=ax_r.transAxes, va='center', ha='right')

estado = "APROBADO" if acc_val >= 0.90 else "REVISAR"
ax_r.add_patch(mpatches.FancyBboxPatch((0.05, 0.03), 0.90, 0.10,
    boxstyle="round,pad=0.05", facecolor='#D5E8D4' if acc_val>=0.90 else '#FFE0E0',
    edgecolor='#27AE60' if acc_val>=0.90 else '#CC0000',
    linewidth=2, transform=ax_r.transAxes))
ax_r.text(0.5, 0.08, estado, ha='center', va='center',
    fontsize=11, fontweight='bold',
    color='#1a7a1a' if acc_val>=0.90 else '#CC0000',
    transform=ax_r.transAxes)

plt.savefig(GRAFICAS_DIR / "01_resultados_svm.png",
            dpi=140, bbox_inches='tight', facecolor='white')
plt.close()
print("  Guardada: 01_resultados_svm.png")

# ── Grafica 2: Matriz de confusion (validacion) ─────────────────
fig, ax = plt.subplots(figsize=(9, 7))
cm_disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_val, y_pred_val,
                                      labels=gestos),
    display_labels=[f"G{g}" for g in gestos])
cm_disp.plot(ax=ax, colorbar=True, cmap='Blues', values_format='d')
ax.set_title(
    f"Matriz de Confusion — Validacion\n"
    f"SVM (kernel={SVM_KERNEL}, C={SVM_C})  |  "
    f"Accuracy = {acc_val*100:.1f}%",
    fontsize=11, fontweight='bold')
ax.set_xlabel("Clase Predicha")
ax.set_ylabel("Clase Real")
plt.tight_layout()
plt.savefig(GRAFICAS_DIR / "02_confusion_validacion.png",
            dpi=130, bbox_inches='tight', facecolor='white')
plt.close()
print("  Guardada: 02_confusion_validacion.png")

# ── Grafica 3: Comparativa SVM vs Random Forest ─────────────────
# (carga resultados_cv_final.csv si existe para comparar)
rf_csv = Path("resultados_cv_final.csv")
if rf_csv.exists():
    df_rf   = pd.read_csv(rf_csv)
    media_rf = df_rf[[f"acc_G{g}" for g in gestos] + ["acc_total"]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Comparativa: Random Forest vs SVM — Precision por clase (CV media)",
                 fontsize=12, fontweight='bold')

    x_pos = np.arange(8); w = 0.35
    for ax, (modelo, acc_g_dict, color, titulo) in zip(axes, [
        ("Random Forest",
         {g: media_rf[f'acc_G{g}'] for g in gestos},
         "#2E75B6", "Random Forest (CV media)"),
        ("SVM",
         {g: media[f'acc_G{g}'] for g in gestos},
         "#C00000", "SVM (CV media)"),
    ]):
        vals = [acc_g_dict[g]*100 for g in gestos]
        bars = ax.bar(x_pos, vals, color=color, alpha=0.8, edgecolor='white')
        ax.axhline(90, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='90%')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"G{g}" for g in gestos])
        ax.set_ylim(60, 108)
        ax.set_ylabel('Precision (%)')
        ax.set_title(titulo, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                    f"{v:.0f}%", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(GRAFICAS_DIR / "03_comparativa_rf_svm.png",
                dpi=130, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Guardada: 03_comparativa_rf_svm.png")
else:
    print("  [INFO] resultados_cv_final.csv no encontrado, "
          "se omite grafica comparativa.")

# ── Grafica 4: Barras de accuracy por etapa ─────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
etapas  = ['CV\n(media)', 'Test', 'Validacion']
valores = [media['acc_total']*100, acc_test*100, acc_val*100]
colores = ['#2E75B6', '#27AE60', '#7B0000']
bars    = ax.bar(etapas, valores, color=colores, alpha=0.85,
                 edgecolor='white', width=0.5)
ax.axhline(90, color='red', linestyle='--', linewidth=2,
           alpha=0.7, label='Requisito 90%')
ax.set_ylim(75, 105)
ax.set_ylabel('Precision (%)')
ax.set_title(f'SVM — Precision por etapa\n'
             f'(kernel={SVM_KERNEL}, C={SVM_C}, gamma={SVM_GAMMA})',
             fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, valores):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
            f"{v:.1f}%", ha='center', fontsize=13, fontweight='bold',
            color='#111111')
plt.tight_layout()
plt.savefig(GRAFICAS_DIR / "04_accuracy_por_etapa.png",
            dpi=130, bbox_inches='tight', facecolor='white')
plt.close()
print("  Guardada: 04_accuracy_por_etapa.png")


# ── RESUMEN FINAL ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SVM COMPLETO")
print(f"  Kernel        : {SVM_KERNEL}")
print(f"  C             : {SVM_C}")
print(f"  Gamma         : {SVM_GAMMA}")
print(f"  CV media      : {media['acc_total']*100:.1f}%")
print(f"  Test          : {acc_test*100:.1f}%")
print(f"  Validacion    : {acc_val*100:.1f}%")
print(f"  Resultados    : {RESULTS_CSV}")
print(f"  Graficas      : {GRAFICAS_DIR}/  (4 imagenes)")
print(f"  Req. > 90%    : {'CUMPLIDO' if acc_val >= 0.90 else 'NO CUMPLIDO'}")
print("=" * 65)