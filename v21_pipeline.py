# -*- coding: utf-8 -*-
"""
=============================================================================
V21 - CHAMPIONSHIP PIPELINE (COGNITIVE PERFORMANCE)
V19/V20 iskeleti uzerine sampiyonluk eklentileri:
  - Biyolojik Oznitelikler: Hafif Uyku %, Ideal Sicaklik Sapmasi, VKI Grup, Yas Grup
  - Model Cesitliligi: HistGradientBoosting, LGBM (Huber Loss) eklendi
  - Meta-Learner: BayesianRidge kullanildi
  - Pseudo Labeling: V20 pseudo etiketleri %30 agirlikla egitime dahil ediliyor
=============================================================================
"""
import pandas as pd
import numpy as np
import warnings
import os
import time
import gc

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    [tf.config.experimental.set_memory_growth(g, True) for g in gpus]

# ── AYARLAR ──
TRAIN_PATH  = r'd:\Datathonv2\train.csv'
TEST_PATH   = r'd:\Datathonv2\test_x.csv'
PSEUDO_PATH = r'd:\Datathonv2\submission_v20_pseudo_grandmaster.csv'
OUTPUT_DIR  = r'd:\Datathonv2'
TARGET_COL  = 'bilissel_performans_skoru'
SEEDS       = [42, 2024, 7, 123, 2026]
N_FOLDS     = 10
PSEUDO_W    = 0.30

try:
    _t = CatBoostRegressor(iterations=2, task_type='GPU', verbose=0)
    _t.fit([[1],[2],[3]], [1,2,3])
    CAT_TASK = 'GPU'
except:
    CAT_TASK = 'CPU'

print("="*70)
print("V21: CHAMPIONSHIP PIPELINE")
print(f"Seeds: {SEEDS} | Folds: {N_FOLDS} | CatBoost: {CAT_TASK} | Pseudo Weight: {PSEUDO_W}")
print("="*70)

# ── 1. VERİ YÜKLEME ──
train_raw = pd.read_csv(TRAIN_PATH)
test_raw  = pd.read_csv(TEST_PATH)
pseudo    = pd.read_csv(PSEUDO_PATH)
y_pseudo  = pseudo[TARGET_COL].values
print(f"Train: {train_raw.shape} | Test: {test_raw.shape} | Pseudo Labels Yuklendi.")

ulke_map = {'South Korea': 'Guney Kore', 'Spain': 'Ispanya', 'Sweden': 'Isvec', 'Mexico': 'Meksika', 'Netherlands': 'Hollanda'}
for df in [train_raw, test_raw]:
    df['ulke'] = df['ulke'].replace(ulke_map)

# ── 2. BİYOLOJİK FEATURE ENGINEERING ──
def create_features(df):
    df = df.copy()
    
    # -- Orijinal Features --
    df['toplam_uyku_kalitesi']  = df['rem_yuzdesi'] + df['derin_uyku_yuzdesi']
    df['uyku_kalitesi_orani']   = df['rem_yuzdesi'] / (df['derin_uyku_yuzdesi'] + 1e-5)
    df['uyku_bozulma_skoru']    = df['gecelik_uyanma_sayisi']*5 + df['uykuya_dalma_suresi_dk']*0.5
    df['stres_calisma']         = df['stres_skoru'] * df['gunluk_calisma_saati']
    df['kafein_ekran']          = df['uyku_oncesi_kafein_mg'] * df['uyku_oncesi_ekran_suresi_dk']
    df['aktivite_skoru']        = df['gunluk_adim_sayisi'] / 1000.0
    df['nabiz_stres']           = df['dinlenik_nabiz_bpm'] * df['stres_skoru']
    df['kalite_stres']          = df['toplam_uyku_kalitesi'] / (df['stres_skoru'] + 1e-5)
    df['uyanma_dalma']          = df['gecelik_uyanma_sayisi'] * df['uykuya_dalma_suresi_dk']
    df['log_adim']              = np.log1p(df['gunluk_adim_sayisi'])
    df['log_kafein']            = np.log1p(df['uyku_oncesi_kafein_mg'])
    df['log_ekran']             = np.log1p(df['uyku_oncesi_ekran_suresi_dk'])
    df['log_sekerleme']         = np.log1p(df['sekerleme_suresi_dk'])
    df['negatif_etki']          = (df['stres_skoru'] + df['gecelik_uyanma_sayisi']*1.5 + df['uykuya_dalma_suresi_dk']*0.2 + df['gunluk_calisma_saati']*0.5)
    df['pozitif_etki']          = (df['rem_yuzdesi']*0.5 + df['derin_uyku_yuzdesi']*0.3 + df['gunluk_adim_sayisi']/2000.0)
    df['net_etki']              = df['pozitif_etki'] - df['negatif_etki']
    df['hafta_sonu_abs']        = abs(df['hafta_sonu_uyku_farki_saat'])
    df['stres_kare']            = df['stres_skoru'] ** 2
    
    # -- Biyolojik Features (YENI) --
    df['hafif_uyku_yuzdesi']    = 100.0 - df['toplam_uyku_kalitesi']
    df['sicaklik_farki']        = abs(df['oda_sicakligi_celsius'] - 18.3)
    df['vki_grup']              = pd.cut(df['vucut_kitle_indeksi'], bins=[0, 18.5, 25, 30, 100], labels=['Zayif', 'Normal', 'Fazla_Kilolu', 'Obez']).astype(str)
    df['yas_grup']              = pd.cut(df['yas'], bins=[0, 20, 30, 40, 50, 60, 100], labels=['0-20', '21-30', '31-40', '41-50', '51-60', '60+']).astype(str)
    
    # -- Cross ve Diger Oznitelikler --
    df['cross_meslek_cinsiyet'] = df['meslek'].astype(str) + "_" + df['cinsiyet'].astype(str)
    df['cross_kronotip_ruh']    = df['kronotip'].astype(str) + "_" + df['ruh_sagligi_durumu'].astype(str)
    df['stres_missing']         = df['stres_skoru'].isnull().astype(int)
    df['ruh_missing']           = df['ruh_sagligi_durumu'].isnull().astype(int)
    df['meslek_missing']        = df['meslek'].isnull().astype(int)
    sev = {'Saglikli': 3, 'Anksiyete': 2, 'Depresyon': 1, 'Anksiyete ve depresyon': 0}
    df['ruh_sagligi_severity']  = df['ruh_sagligi_durumu'].map(sev).fillna(1.5)
    return df

train_raw = create_features(train_raw)
test_raw  = create_features(test_raw)

# YENI KATEGORIKLER EKLENDI
cat_cols = ['cinsiyet', 'meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi', 'cross_meslek_cinsiyet', 'cross_kronotip_ruh', 'vki_grup', 'yas_grup']

print("Target Encoding yapiliyor...")
for col in cat_cols:
    train_raw[col] = train_raw[col].replace('nan', 'MISSING').fillna('MISSING')
    test_raw[col]  = test_raw[col].replace('nan', 'MISSING').fillna('MISSING')

    kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
    train_raw[col + '_te'] = np.nan
    for tr_idx, val_idx in kf_te.split(train_raw):
        means = train_raw.iloc[tr_idx].groupby(col)[TARGET_COL].mean()
        train_raw.loc[val_idx, col + '_te'] = train_raw.iloc[val_idx][col].map(means)
    train_raw[col + '_te'].fillna(train_raw[TARGET_COL].mean(), inplace=True)
    test_raw[col + '_te']  = test_raw[col].map(train_raw.groupby(col)[TARGET_COL].mean())
    test_raw[col + '_te'].fillna(train_raw[TARGET_COL].mean(), inplace=True)

    le = LabelEncoder()
    le.fit(pd.concat([train_raw[col].astype(str), test_raw[col].astype(str)]))
    train_raw[col + '_le'] = le.transform(train_raw[col].astype(str))
    test_raw[col + '_le']  = le.transform(test_raw[col].astype(str))

drop_cols    = ['id', TARGET_COL] + cat_cols
feature_cols = [c for c in train_raw.columns if c not in drop_cols]

for c in feature_cols:
    med = train_raw[c].median()
    train_raw[c] = train_raw[c].fillna(med)
    test_raw[c]  = test_raw[c].fillna(med)

print("K-Means & PCA feature extraction...")
scaler_main = StandardScaler()
X_scaled_tr = scaler_main.fit_transform(train_raw[feature_cols])
X_scaled_te = scaler_main.transform(test_raw[feature_cols])

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_scaled_tr)
for i in range(5):
    train_raw[f'kmeans_dist_{i}'] = kmeans.transform(X_scaled_tr)[:,i]
    test_raw[f'kmeans_dist_{i}']  = kmeans.transform(X_scaled_te)[:,i]
    feature_cols.append(f'kmeans_dist_{i}')

pca = PCA(n_components=3, random_state=42)
pca_tr = pca.fit_transform(X_scaled_tr)
pca_te = pca.transform(X_scaled_te)
for i in range(3):
    train_raw[f'pca_{i}'] = pca_tr[:,i]
    test_raw[f'pca_{i}']  = pca_te[:,i]
    feature_cols.append(f'pca_{i}')

X        = train_raw[feature_cols].copy()
y        = train_raw[TARGET_COL].copy()
X_test   = test_raw[feature_cols].copy()
test_ids = test_raw['id'].copy()

print(f"Toplam feature sayisi: {len(feature_cols)}")

nn_scaler = StandardScaler()
X_nn      = nn_scaler.fit_transform(X)
X_test_nn = nn_scaler.transform(X_test)

def build_nn(input_dim, seed):
    tf.random.set_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='swish', input_dim=input_dim),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='swish'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='swish'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')
    return model

# ── 3. PIPELINE TRAINING (6 MODELLI) ──
all_test_preds = []
all_cv_scores  = []
start_total    = time.time()

for seed_idx, SEED in enumerate(SEEDS):
    print(f"\n{'='*70}")
    print(f"SEED {SEED}  ({seed_idx+1}/{len(SEEDS)})  —  {time.strftime('%H:%M:%S')}")
    print(f"{'='*70}")
    t0 = time.time()

    # Orijinal parametreler + yeni modeller
    lgb_p = dict(learning_rate=0.0289, n_estimators=5000, num_leaves=215, max_depth=6, min_child_samples=71, subsample=0.803, colsample_bytree=0.612, reg_alpha=3e-7, reg_lambda=7.939, random_state=SEED, n_jobs=-1, verbose=-1)
    # HUBER LGBM Parametreleri
    lgb_huber_p = lgb_p.copy()
    lgb_huber_p.update({'objective': 'huber', 'alpha': 1.2})
    
    xgb_p = dict(n_estimators=5000, max_depth=6, learning_rate=0.025, subsample=0.85, colsample_bytree=0.65, min_child_weight=6, reg_alpha=0.1, reg_lambda=2.0, random_state=SEED, n_jobs=-1, tree_method='hist', verbosity=0, early_stopping_rounds=200)
    cat_p = dict(iterations=5000, depth=7, learning_rate=0.03, l2_leaf_reg=4, random_seed=SEED, verbose=0, task_type=CAT_TASK, thread_count=-1, early_stopping_rounds=200)
    
    # HistGradientBoosting Parametreleri
    hgb_p = dict(max_iter=2000, learning_rate=0.03, max_depth=6, l2_regularization=2.0, random_state=SEED, early_stopping=True, validation_fraction=0.1, n_iter_no_change=100)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_nn    = np.zeros(len(X))
    oof_lgb   = np.zeros(len(X))
    oof_lgb_h = np.zeros(len(X))
    oof_xgb   = np.zeros(len(X))
    oof_cat   = np.zeros(len(X))
    oof_hgb   = np.zeros(len(X))

    tst_nn    = np.zeros(len(X_test))
    tst_lgb   = np.zeros(len(X_test))
    tst_lgb_h = np.zeros(len(X_test))
    tst_xgb   = np.zeros(len(X_test))
    tst_cat   = np.zeros(len(X_test))
    tst_hgb   = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        tf0 = time.time()
        
        X_tr,  X_val  = X.iloc[tr_idx],   X.iloc[val_idx]
        y_tr,  y_val  = y.iloc[tr_idx],   y.iloc[val_idx]
        Xn_tr, Xn_val = X_nn[tr_idx],     X_nn[val_idx]

        # 🎯 PSEUDO LABELING EKLEMESİ
        X_tr_p = pd.concat([X_tr, X_test], ignore_index=True)
        y_tr_p = np.concatenate([y_tr.values, y_pseudo])
        weights_p = np.concatenate([np.ones(len(tr_idx)), np.ones(len(y_pseudo)) * PSEUDO_W])
        Xn_tr_p = np.vstack([Xn_tr, X_test_nn])

        # 1. Keras NN
        nn = build_nn(Xn_tr_p.shape[1], SEED)
        nn.fit(Xn_tr_p, y_tr_p, sample_weight=weights_p, validation_data=(Xn_val, y_val),
               epochs=200, batch_size=256, verbose=0,
               callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=30, restore_best_weights=True),
                          tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=7)])
        oof_nn[val_idx] = nn.predict(Xn_val, verbose=0).flatten()
        tst_nn += nn.predict(X_test_nn, verbose=0).flatten() / N_FOLDS
        r_nn = root_mean_squared_error(y_val, oof_nn[val_idx])
        del nn; gc.collect()

        # 2. LightGBM (L2)
        m_lgb = lgb.LGBMRegressor(**lgb_p)
        m_lgb.fit(X_tr_p, y_tr_p, sample_weight=weights_p, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof_lgb[val_idx] = m_lgb.predict(X_val)
        tst_lgb += m_lgb.predict(X_test) / N_FOLDS
        r_lgb = root_mean_squared_error(y_val, oof_lgb[val_idx])

        # 3. LightGBM (Huber Loss)
        m_lgb_h = lgb.LGBMRegressor(**lgb_huber_p)
        m_lgb_h.fit(X_tr_p, y_tr_p, sample_weight=weights_p, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof_lgb_h[val_idx] = m_lgb_h.predict(X_val)
        tst_lgb_h += m_lgb_h.predict(X_test) / N_FOLDS
        r_lgb_h = root_mean_squared_error(y_val, oof_lgb_h[val_idx])

        # 4. XGBoost
        m_xgb = xgb.XGBRegressor(**xgb_p)
        m_xgb.fit(X_tr_p, y_tr_p, sample_weight=weights_p, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = m_xgb.predict(X_val)
        tst_xgb += m_xgb.predict(X_test) / N_FOLDS
        r_xgb = root_mean_squared_error(y_val, oof_xgb[val_idx])

        # 5. CatBoost
        m_cat = CatBoostRegressor(**cat_p)
        m_cat.fit(X_tr_p, y_tr_p, sample_weight=weights_p, eval_set=(X_val, y_val), verbose=False)
        oof_cat[val_idx] = m_cat.predict(X_val)
        tst_cat += m_cat.predict(X_test) / N_FOLDS
        r_cat = root_mean_squared_error(y_val, oof_cat[val_idx])
        
        # 6. HistGradientBoosting
        # HistGradientBoosting doesn't natively support sample_weight with missing values well sometimes, 
        # but sklearn's does support sample_weight.
        m_hgb = HistGradientBoostingRegressor(**hgb_p)
        m_hgb.fit(X_tr_p, y_tr_p, sample_weight=weights_p)
        oof_hgb[val_idx] = m_hgb.predict(X_val)
        tst_hgb += m_hgb.predict(X_test) / N_FOLDS
        r_hgb = root_mean_squared_error(y_val, oof_hgb[val_idx])

        elapsed = int(time.time() - tf0)
        print(f"  Fold {fold+1:2d}/{N_FOLDS} ({elapsed:3d}s) | "
              f"NN:{r_nn:.4f} LGB:{r_lgb:.4f} LGB-H:{r_lgb_h:.4f} HGB:{r_hgb:.4f} CAT:{r_cat:.4f} XGB:{r_xgb:.4f}")

    # 🎯 LEVEL-2: BayesianRidge Meta-Learner
    oof_meta = np.column_stack([oof_nn, oof_lgb, oof_lgb_h, oof_xgb, oof_cat, oof_hgb])
    tst_meta = np.column_stack([tst_nn, tst_lgb, tst_lgb_h, tst_xgb, tst_cat, tst_hgb])

    meta_model = BayesianRidge()
    meta_model.fit(oof_meta, y)
    seed_cv   = root_mean_squared_error(y, meta_model.predict(oof_meta))
    seed_pred = np.clip(meta_model.predict(tst_meta), 0, 10)

    print(f"\n  OOF -> NN:{root_mean_squared_error(y,oof_nn):.4f} LGB:{root_mean_squared_error(y,oof_lgb):.4f} "
          f"LGB-H:{root_mean_squared_error(y,oof_lgb_h):.4f} HGB:{root_mean_squared_error(y,oof_hgb):.4f} "
          f"CAT:{root_mean_squared_error(y,oof_cat):.4f} XGB:{root_mean_squared_error(y,oof_xgb):.4f}")
    print(f"  BayesianRidge katsayilari: {meta_model.coef_.round(4)}")
    print(f"  Pred std: {seed_pred.std():.4f}")
    print(f"  * SEED {SEED} Meta CV RMSE: {seed_cv:.5f}  ({int(time.time()-t0)//60}dk)")

    pd.DataFrame({'id': test_ids, 'bilissel_performans_skoru': seed_pred}).to_csv(os.path.join(OUTPUT_DIR, f'submission_v21_seed{SEED}.csv'), index=False)
    all_test_preds.append(seed_pred)
    all_cv_scores.append(seed_cv)
    gc.collect()

# ── 4. FINAL ──
print(f"\n{'='*70}")
print("V21 FINAL ENSEMBLE (CHAMPIONSHIP)")
print(f"{'='*70}")
for s, sc in zip(SEEDS, all_cv_scores):
    print(f"  Seed {s:4d}: {sc:.5f}")
print(f"  Ortalama Meta CV : {np.mean(all_cv_scores):.5f} +/- {np.std(all_cv_scores):.5f}")

final_preds = np.clip(np.mean(all_test_preds, axis=0), 0, 10)
pd.DataFrame({'id': test_ids, 'bilissel_performans_skoru': final_preds}).to_csv(os.path.join(OUTPUT_DIR, 'submission_v21_championship.csv'), index=False)

print(f"\nV21 TAMAMLANDI! ({int(time.time()-start_total)//60} dk)")
print(f"  -> submission_v21_championship.csv")
print(f"  mean={final_preds.mean():.4f}, std={final_preds.std():.4f}, min={final_preds.min():.4f}, max={final_preds.max():.4f}")
