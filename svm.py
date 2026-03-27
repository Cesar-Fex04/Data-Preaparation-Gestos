"""
SVM — Reconocimiento de Gestos en Tableta 2D
Inteligencia Artificial

Lee dataset_gestos_final.csv (generado por data_final.py).
Division: 64% train / 16% test / 20% validacion (columna split del CSV).
Normaliza con StandardScaler ajustado SOLO en train.
Cross Validation 5-Fold sobre train para verificar overfitting.
Evalua en test y validacion final.

USO: python svm.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

INPUT_CSV    = "dataset_gestos_final.csv"
RESULTS_CSV  = "resultados_svm.csv"
GRAFICAS_DIR = Path("graficas_svm")
N_FOLDS      = 5
RANDOM_STATE = 42
SVM_KERNEL   = 'rbf'
SVM_C        = 10
SVM_GAMMA    = 'scale'

GRAFICAS_DIR.mkdir(exist_ok=True)

NOMBRE = {
    1:"Diagonal ↖", 2:"Diagonal ↗", 3:"Diagonal ↙", 4:"Diagonal ↘",
    5:"Vertical ↑",  6:"Horizontal ←", 7:"Horizontal →", 8:"Vertical ↓"
}
#arctan2(dy,dx) con Y positivo hacia abajo (convencion tableta)
ANGULO_IDEAL = {1:-135, 2:-45, 3:135, 4:45, 5:-90, 6:180, 7:0, 8:90}


# ════════════════════════════════════════════════════════════════
#  PASO 1 — CARGAR DATASET
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("PASO 1: Cargando dataset...")
print("=" * 65)

if not Path(INPUT_CSV).exists():
    raise FileNotFoundError(f"No se encontro '{INPUT_CSV}'.\nEjecuta: python data_final.py")

df = pd.read_csv(INPUT_CSV)
print(f"  {df.shape[0]} muestras x {df.shape[1]} columnas")
print(f"  Splits: {df['split'].value_counts().to_dict()}")

feat_cols = [c for c in df.columns
             if c not in ("user_id", "gesture_label", "split", "fold")]
print(f"  Features: {len(feat_cols)}")


# ════════════════════════════════════════════════════════════════
#  PASO 2 — SEPARAR SPLITS (lee columna split del CSV)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 2: Separando splits desde columna 'split' del CSV...")
print("=" * 65)

df_train = df[df["split"] == "train"].copy()
df_test  = df[df["split"] == "test"].copy()
df_val   = df[df["split"] == "validation"].copy()

X_train = df_train[feat_cols].values;  y_train = df_train["gesture_label"].values
X_test  = df_test[feat_cols].values;   y_test  = df_test["gesture_label"].values
X_val   = df_val[feat_cols].values;    y_val   = df_val["gesture_label"].values

total = len(df)
print(f"  Train      : {len(X_train):>5}  ({len(X_train)/total*100:.0f}%)")
print(f"  Test       : {len(X_test):>5}  ({len(X_test)/total*100:.0f}%)")
print(f"  Validacion : {len(X_val):>5}  ({len(X_val)/total*100:.0f}%)  <- INTOCABLE")

print(f"\n  Muestras por clase en TRAIN:")
for g in range(1, 9):
    print(f"    G{g} ({NOMBRE[g]}): {(y_train==g).sum()}")


# ════════════════════════════════════════════════════════════════
#  PASO 3 — NORMALIZACION
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 3: Normalizando con StandardScaler...")
print("=" * 65)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # aprende Y transforma
X_test_sc  = scaler.transform(X_test)        # solo transforma
X_val_sc   = scaler.transform(X_val)         # solo transforma

print(f"  Scaler ajustado con TRAIN ({len(X_train)} muestras)")
print(f"  Media features 1-5: {scaler.mean_[:5].round(2)}")
print(f"  Std  features 1-5: {scaler.scale_[:5].round(2)}")


# ════════════════════════════════════════════════════════════════
#  PASO 4 — CROSS VALIDATION 5-FOLD SOBRE TRAIN
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"PASO 4: Cross Validation {N_FOLDS}-Fold sobre TRAIN...")
print("=" * 65)
print(f"  kernel={SVM_KERNEL}  C={SVM_C}  gamma={SVM_GAMMA}")

gestos   = list(range(1, 9))
skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_res = []

print(f"\n  {'Fold':<6} {'Total':>8}   " + "   ".join([f"G{g}" for g in gestos]))
print("  " + "-" * 72)

for fold, (tr_i, te_i) in enumerate(skf.split(X_train_sc, y_train), 1):
    svm = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA,
              random_state=RANDOM_STATE, decision_function_shape='ovr')
    svm.fit(X_train_sc[tr_i], y_train[tr_i])
    y_p = svm.predict(X_train_sc[te_i])
    y_e = y_train[te_i]
    at  = accuracy_score(y_e, y_p)
    ag  = {g: accuracy_score(y_e[y_e==g], y_p[y_e==g])
           if (y_e==g).sum() > 0 else 0.0 for g in gestos}
    fold_res.append({"fold":fold, "acc_total":at,
                     "n_train":len(tr_i), "n_test":len(te_i),
                     **{f"acc_G{g}":ag[g] for g in gestos}})
    print(f"  Fold {fold}  {at*100:>6.1f}%   " +
          "   ".join([f"{ag[g]*100:>4.0f}%" for g in gestos]))

df_cv = pd.DataFrame(fold_res)
media = df_cv[[f"acc_G{g}" for g in gestos]+["acc_total"]].mean()
print("  " + "-" * 72)
print(f"  Media  {media['acc_total']*100:>6.1f}%   " +
      "   ".join([f"{media[f'acc_G{g}']*100:>4.0f}%" for g in gestos]))
print(f"\n  Variacion entre folds: "
      f"{df_cv['acc_total'].max()*100:.1f}% - {df_cv['acc_total'].min()*100:.1f}% = "
      f"{(df_cv['acc_total'].max()-df_cv['acc_total'].min())*100:.1f} pp")
print(f"  -> {'Sin overfitting — continuar' if media['acc_total']>=0.90 else 'Revisar modelo'}")


# ════════════════════════════════════════════════════════════════
#  PASO 5 — EVALUAR EN TEST
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
               if (y_test==g).sum()>0 else 0.0 for g in gestos}

print(f"  Entrenado: {len(X_train)} muestras | Evaluado: {len(X_test)} muestras")
print(f"  Accuracy total TEST: {acc_test*100:.1f}%")
print(f"  Por gesto: " + "  ".join([f"G{g}:{acc_test_g[g]*100:.0f}%" for g in gestos]))
print(f"  -> {'OK — proceder a validacion' if acc_test>=0.90 else 'Revisar: <90%'}")


# ════════════════════════════════════════════════════════════════
#  PASO 6 — MODELO FINAL: TRAIN+TEST → VALIDACION (1 SOLA VEZ)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 6: Evaluacion FINAL en VALIDACION (una sola vez)...")
print("=" * 65)

X_trabajo_sc  = np.vstack([X_train_sc, X_test_sc])
y_trabajo_all = np.concatenate([y_train, y_test])

svm_final = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA,
                random_state=RANDOM_STATE, decision_function_shape='ovr')
svm_final.fit(X_trabajo_sc, y_trabajo_all)
y_pred_val = svm_final.predict(X_val_sc)
acc_val    = accuracy_score(y_val, y_pred_val)
acc_val_g  = {g: accuracy_score(y_val[y_val==g], y_pred_val[y_val==g])
              if (y_val==g).sum()>0 else 0.0 for g in gestos}

print(f"  Entrenado: {len(X_trabajo_sc)} muestras (train+test)")
print(f"  Evaluado en VALIDACION: {len(X_val)} muestras — UNA SOLA VEZ")
print(f"\n  {'='*65}")
print(f"  RESULTADO FINAL")
print(f"  CV media   : {media['acc_total']*100:.1f}%")
print(f"  Test       : {acc_test*100:.1f}%")
print(f"  Validacion : {acc_val*100:.1f}%")
print(f"  {'='*65}")
print(f"  Por clase (validacion):")
for g in gestos:
    n = (y_val==g).sum()
    ok = round(acc_val_g[g]*n)
    err = n - ok
    print(f"    G{g} ({NOMBRE[g]:<14}): {acc_val_g[g]*100:.0f}%  "
          f"({ok}/{n} correctas, {err} errores)")
print(f"  {'='*65}")
print(f"  Req. >90%: {'CUMPLIDO' if acc_val>=0.90 else 'NO CUMPLIDO'}")


# ════════════════════════════════════════════════════════════════
#  PASO 7 — GUARDAR RESULTADOS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 7: Guardando resultados...")
print("=" * 65)

df_res = pd.concat([
    df_cv,
    pd.DataFrame([{"fold":"test", "acc_total":acc_test,
                   "n_train":len(X_train), "n_test":len(X_test),
                   **{f"acc_G{g}":acc_test_g[g] for g in gestos}}]),
    pd.DataFrame([{"fold":"validacion", "acc_total":acc_val,
                   "n_train":len(X_trabajo_sc), "n_test":len(X_val),
                   **{f"acc_G{g}":acc_val_g[g] for g in gestos}}]),
], ignore_index=True)
df_res.to_csv(RESULTS_CSV, index=False)
print(f"  {RESULTS_CSV}")


# ════════════════════════════════════════════════════════════════
#  PASO 8 — ANALISIS DE ERRORES (validacion)
#  Muestra exactamente que muestras se equivocaron y por que:
#  imprime sus features clave (angulo, dx, dy) para entender
#  de donde viene la confusion.
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 8: Analisis de errores en VALIDACION...")
print("=" * 65)

df_val_analisis = df_val.copy()
df_val_analisis["y_pred"] = y_pred_val
errores = df_val_analisis[
    df_val_analisis["gesture_label"] != df_val_analisis["y_pred"]
].copy()

n_errores = len(errores)
print(f"\n  Total errores: {n_errores} de {len(df_val)} muestras")
print(f"  Accuracy     : {(1 - n_errores/len(df_val))*100:.1f}%\n")

if n_errores == 0:
    print("  Clasificacion perfecta en validacion.")
else:
    # Tabla de errores con sus features reales
    print(f"  {'G_real':<8} {'G_pred':<8} {'angle_deg':>10} "
          f"{'dx_global':>10} {'dy_global':>10}  Nota")
    print("  " + "-" * 70)
    for _, row in errores.iterrows():
        gr  = int(row["gesture_label"])
        gp  = int(row["y_pred"])
        ang = np.degrees(row["angle"])
        dx  = row["dx_global"]
        dy  = row["dy_global"]
        # El angulo por si solo no explica la decision del SVM
        # (el SVM usa 42 features), pero da contexto sobre
        # si la muestra era "atipica" dentro de su clase
        dist_real = abs(((ang - ANGULO_IDEAL[gr] + 180) % 360) - 180)
        nota = f"angulo desviado {dist_real:.0f}deg de ideal G{gr}"
        print(f"  G{gr:<7} G{gp:<7} {ang:>9.1f}  {dx:>10.1f} {dy:>10.1f}  {nota}")

    # Patron de confusiones REAL (contando errores verdaderos)
    print(f"\n  PATRON DE CONFUSIONES (errores reales del modelo):")
    print(f"  {'Confusion':<15} {'Veces':>6}  Posible causa")
    print("  " + "-" * 60)
    from collections import Counter
    patron = Counter([(int(r["gesture_label"]), int(r["y_pred"]))
                      for _, r in errores.iterrows()])
    for (gr, gp), cnt in patron.most_common():
        # Calcular angulo promedio de las muestras erroneas
        mask = (errores["gesture_label"]==gr) & (errores["y_pred"]==gp)
        ang_prom = np.degrees(errores.loc[mask, "angle"].mean())
        causa = (f"angulo prom. {ang_prom:.0f}deg, "
                 f"entre ideal G{gr}({ANGULO_IDEAL[gr]}deg) "
                 f"y G{gp}({ANGULO_IDEAL[gp]}deg)")
        print(f"  G{gr}->G{gp:<10} {cnt:>6}  {causa}")


# ════════════════════════════════════════════════════════════════
#  PASO 9 — GRAFICAS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 9: Generando graficas...")
print("=" * 65)

# G1: Tabla completa de resultados
fig = plt.figure(figsize=(16, 9)); fig.patch.set_facecolor('white')
fig.text(0.5, 0.97,
    f"SVM — Resultados (kernel={SVM_KERNEL}, C={SVM_C}, gamma={SVM_GAMMA})",
    ha='center', va='top', fontsize=13, fontweight='bold', color='#1F4E79')

ax_t = fig.add_axes([0.02, 0.48, 0.96, 0.45]); ax_t.axis('off')
col_lbl = ["", "Total"] + [f"G{g}" for g in gestos]
filas = []
for r in fold_res:
    filas.append([f"Fold {int(r['fold'])}", f"{r['acc_total']*100:.1f}%"] +
                 [f"{r[f'acc_G{g}']*100:.0f}%" for g in gestos])
filas.append(["Media CV", f"{media['acc_total']*100:.1f}%"] +
             [f"{media[f'acc_G{g}']*100:.0f}%" for g in gestos])
filas.append([f"Test ({len(X_test)})", f"{acc_test*100:.1f}%"] +
             [f"{acc_test_g[g]*100:.0f}%" for g in gestos])
filas.append([f"Val. ({len(X_val)})", f"{acc_val*100:.1f}%"] +
             [f"{acc_val_g[g]*100:.0f}%" for g in gestos])

tbl = ax_t.table(cellText=filas, colLabels=col_lbl,
                 cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1, 2.5)

for j in range(len(col_lbl)):
    tbl[0,j].set_facecolor('#1F4E79')
    tbl[0,j].set_text_props(color='white', fontweight='bold')
for i in range(1, N_FOLDS+1):
    tbl[i,0].set_facecolor('#D6E4F0')
    tbl[i,0].set_text_props(fontweight='bold', color='#1F4E79')
    for j in range(1, len(col_lbl)):
        v = float(filas[i-1][j].replace('%',''))
        bg = '#C8F0C8' if v>=99 else ('#D5E8D4' if v>=90 else '#FFE0E0')
        co = '#1a7a1a' if v>=90 else '#CC0000'
        tbl[i,j].set_facecolor(bg); tbl[i,j].set_text_props(color=co, fontweight='bold')
for j in range(len(col_lbl)):
    tbl[N_FOLDS+1,j].set_facecolor('#2E75B6')
    tbl[N_FOLDS+1,j].set_text_props(color='white', fontweight='bold')
    tbl[N_FOLDS+2,j].set_facecolor('#27AE60')
    tbl[N_FOLDS+2,j].set_text_props(color='white', fontweight='bold')
    tbl[N_FOLDS+3,j].set_facecolor('#7B0000')
    tbl[N_FOLDS+3,j].set_text_props(color='white', fontweight='bold')

ax_b = fig.add_axes([0.03, 0.04, 0.60, 0.37])
x = np.arange(8); w = 0.18
for i,(fr,col) in enumerate(zip(fold_res,['#2980B9','#27AE60','#E67E22','#8E44AD','#16A085'])):
    ax_b.bar(x+i*w-w*2, [fr[f'acc_G{g}']*100 for g in gestos],
             w, label=f"Fold {int(fr['fold'])}", color=col, alpha=0.7)
ax_b.plot(x+w/2,[acc_val_g[g]*100 for g in gestos],'r--o',lw=2.5,ms=7,label='Validacion',zorder=5)
ax_b.plot(x+w/2,[acc_test_g[g]*100 for g in gestos],'g--s',lw=2,ms=6,label='Test',zorder=4)
ax_b.axhline(90,color='red',ls=':',lw=1.5,alpha=0.6,label='90%')
ax_b.set_xticks(x+w/2); ax_b.set_xticklabels([f"G{g}" for g in gestos],fontsize=9)
ax_b.set_ylim(60,108); ax_b.set_ylabel('Accuracy (%)')
ax_b.set_title('Accuracy por clase y etapa', fontweight='bold')
ax_b.legend(fontsize=8, ncol=3, loc='lower right'); ax_b.grid(True,alpha=0.3,axis='y')

ax_r = fig.add_axes([0.66,0.04,0.32,0.37]); ax_r.axis('off'); ax_r.set_facecolor('#F0F7FF')
ax_r.text(0.5,0.96,"Resumen",ha='center',va='top',fontweight='bold',
          fontsize=12,color='#1F4E79',transform=ax_r.transAxes)
for yi,(lbl,val,col) in zip(np.linspace(0.82,0.18,7),[
    ("Division:", f"{len(X_train)}/{len(X_test)}/{len(X_val)}", '#555555'),
    ("Kernel:", SVM_KERNEL, '#555555'), ("C:", str(SVM_C), '#555555'),
    ("Gamma:", str(SVM_GAMMA), '#555555'),
    ("CV media:", f"{media['acc_total']*100:.1f}%", '#2E75B6'),
    ("Test:", f"{acc_test*100:.1f}%", '#27AE60'),
    ("Validacion:", f"{acc_val*100:.1f}%", '#7B0000'),
]):
    ax_r.text(0.08,yi,lbl,fontsize=10,color='#333',transform=ax_r.transAxes,va='center')
    ax_r.text(0.92,yi,val,fontsize=11,fontweight='bold',color=col,
              transform=ax_r.transAxes,va='center',ha='right')
fc = '#D5E8D4' if acc_val>=0.90 else '#FFE0E0'
ec = '#27AE60' if acc_val>=0.90 else '#CC0000'
co = '#1a7a1a' if acc_val>=0.90 else '#CC0000'
ax_r.add_patch(mpatches.FancyBboxPatch((0.05,0.04),0.90,0.10,
    boxstyle="round,pad=0.05",facecolor=fc,edgecolor=ec,linewidth=2,
    transform=ax_r.transAxes))
ax_r.text(0.5,0.09,f"Req.>90%: {'CUMPLIDO' if acc_val>=0.90 else 'REVISAR'}",
    ha='center',va='center',fontsize=11,fontweight='bold',color=co,
    transform=ax_r.transAxes)
plt.savefig(GRAFICAS_DIR/"01_resultados_svm.png",dpi=140,bbox_inches='tight',facecolor='white')
plt.close(); print("  Guardada: 01_resultados_svm.png")

# G2: Matriz de confusion — validacion
fig, ax = plt.subplots(figsize=(9,7))
ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_val, y_pred_val, labels=gestos),
    display_labels=[f"G{g}" for g in gestos]
).plot(ax=ax, colorbar=True, cmap='Blues', values_format='d')
ax.set_title(f"Matriz de Confusion — Validacion\n"
             f"SVM (kernel={SVM_KERNEL}, C={SVM_C})  |  Accuracy={acc_val*100:.1f}%",
             fontsize=11, fontweight='bold')
ax.set_xlabel("Clase Predicha"); ax.set_ylabel("Clase Real")
plt.tight_layout()
plt.savefig(GRAFICAS_DIR/"02_confusion_validacion.png",dpi=130,bbox_inches='tight',facecolor='white')
plt.close(); print("  Guardada: 02_confusion_validacion.png")

# G3: Accuracy por etapa
fig, ax = plt.subplots(figsize=(8,5))
etapas  = [f'CV\n(media)', f'Test\n({len(X_test)})', f'Validacion\n({len(X_val)})']
valores = [media['acc_total']*100, acc_test*100, acc_val*100]
bars = ax.bar(etapas, valores, color=['#2E75B6','#27AE60','#7B0000'],
              alpha=0.85, edgecolor='white', width=0.45)
ax.axhline(90,color='red',ls='--',lw=2,alpha=0.7,label='Requisito 90%')
ax.set_ylim(75,105); ax.set_ylabel('Accuracy (%)')
ax.set_title(f'SVM — Accuracy por etapa\n(kernel={SVM_KERNEL}, C={SVM_C})',
             fontweight='bold')
ax.legend(fontsize=10); ax.grid(True,alpha=0.3,axis='y')
for bar,v in zip(bars,valores):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.3,
            f"{v:.1f}%", ha='center', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(GRAFICAS_DIR/"03_accuracy_por_etapa.png",dpi=130,bbox_inches='tight',facecolor='white')
plt.close(); print("  Guardada: 03_accuracy_por_etapa.png")

# G4: Scatter de angulos — donde estan los errores visualmente
fig, ax = plt.subplots(figsize=(12,5))
df_val_analisis = df_val.copy()
df_val_analisis["y_pred"] = y_pred_val

for g in gestos:
    mask_ok  = (df_val_analisis["gesture_label"]==g) & \
               (df_val_analisis["gesture_label"]==df_val_analisis["y_pred"])
    mask_err = (df_val_analisis["gesture_label"]==g) & \
               (df_val_analisis["gesture_label"]!=df_val_analisis["y_pred"])
    angs_ok  = np.degrees(df_val_analisis.loc[mask_ok, "angle"].values)
    angs_err = df_val_analisis.loc[mask_err].copy()

    ax.scatter([g]*len(angs_ok), angs_ok,
               color='#2E75B6', alpha=0.3, s=25, zorder=2)
    ax.hlines(ANGULO_IDEAL[g], g-0.45, g+0.45,
              colors='#E67E22', linewidths=2.5, linestyles='--', zorder=3)
    for _, row in angs_err.iterrows():
        ang_err = np.degrees(row["angle"])
        gp      = int(row["y_pred"])
        ax.scatter(g, ang_err, color='red', s=150, zorder=5, marker='X')
        ax.annotate(f"->G{gp}", xy=(g, ang_err), xytext=(g+0.15, ang_err),
                    fontsize=8, color='darkred', fontweight='bold', va='center')

ax.set_xticks(gestos)
ax.set_xticklabels([f"G{g}\n{NOMBRE[g]}" for g in gestos], fontsize=8)
ax.set_ylabel("Angulo de la trayectoria (grados)")
ax.set_title("Distribucion de angulos por clase — Validacion\n"
             "Azul=correctas  |  X rojo=error  |  Linea naranja=angulo ideal",
             fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(handles=[
    Line2D([0],[0],marker='o',color='w',markerfacecolor='#2E75B6',ms=8,label='Correctas',alpha=0.6),
    Line2D([0],[0],marker='X',color='w',markerfacecolor='red',ms=10,label='Error'),
    Line2D([0],[0],color='#E67E22',lw=2,ls='--',label='Angulo ideal'),
], fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig(GRAFICAS_DIR/"04_errores_por_angulo.png",dpi=130,bbox_inches='tight',facecolor='white')
plt.close(); print("  Guardada: 04_errores_por_angulo.png")

# G5: Comparativa RF vs SVM
rf_csv = Path("resultados_cv_final.csv")
if rf_csv.exists():
    try:
        df_rf  = pd.read_csv(rf_csv)
        df_rf_f = df_rf[pd.to_numeric(df_rf['fold'], errors='coerce').notna()]
        mrf    = df_rf_f[[f"acc_G{g}" for g in gestos]+["acc_total"]].mean()
        fig, axes = plt.subplots(1,2,figsize=(14,5))
        fig.suptitle("Random Forest vs SVM — Accuracy por clase (CV media)",
                     fontsize=12, fontweight='bold')
        for ax_c,(nom,acc_d,col) in zip(axes,[
            ("Random Forest",{g:mrf[f'acc_G{g}'] for g in gestos},"#2E75B6"),
            ("SVM (rbf C=10)",{g:media[f'acc_G{g}'] for g in gestos},"#C00000"),
        ]):
            vals = [acc_d[g]*100 for g in gestos]
            bars = ax_c.bar(range(8),vals,color=col,alpha=0.8,edgecolor='white')
            ax_c.axhline(90,color='red',ls='--',lw=1.5,alpha=0.7)
            ax_c.set_xticks(range(8)); ax_c.set_xticklabels([f"G{g}" for g in gestos])
            ax_c.set_ylim(60,108); ax_c.set_ylabel('Accuracy (%)')
            ax_c.set_title(f"{nom}\nMedia: {np.mean(vals):.1f}%", fontweight='bold')
            ax_c.grid(True,alpha=0.3,axis='y')
            for bar,v in zip(bars,vals):
                ax_c.text(bar.get_x()+bar.get_width()/2,v+0.5,
                          f"{v:.0f}%",ha='center',fontsize=9,fontweight='bold')
        plt.tight_layout()
        plt.savefig(GRAFICAS_DIR/"05_comparativa_rf_svm.png",
                    dpi=130,bbox_inches='tight',facecolor='white')
        plt.close(); print("  Guardada: 05_comparativa_rf_svm.png")
    except Exception as e:
        print(f"  [INFO] Comparativa RF vs SVM omitida: {e}")
else:
    print("  [INFO] resultados_cv_final.csv no encontrado — "
          "grafica comparativa omitida.")


# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SVM COMPLETO")
print(f"  Division    : {len(X_train)} train / {len(X_test)} test / {len(X_val)} val")
print(f"  Kernel      : {SVM_KERNEL}  C={SVM_C}  gamma={SVM_GAMMA}")
print(f"  CV media    : {media['acc_total']*100:.1f}%")
print(f"  Test        : {acc_test*100:.1f}%")
print(f"  Validacion  : {acc_val*100:.1f}%")
print(f"  Errores val : {n_errores} de {len(X_val)} muestras")
print(f"  Resultados  : {RESULTS_CSV}")
print(f"  Graficas    : {GRAFICAS_DIR}/")
print(f"  Req. >90%   : {'CUMPLIDO' if acc_val>=0.90 else 'NO CUMPLIDO'}")
print("=" * 65)