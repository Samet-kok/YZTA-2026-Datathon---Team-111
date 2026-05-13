# -*- coding: utf-8 -*-
"""
=============================================================================
V27 - TOP 0.1% KAGGLE GRANDMASTER PIPELINE (CATBOOST SUPREMACY)
1. Hedef Değişken Katmanlandırması (Target-Binned Stratified K-Fold) ile 
   her fold'da eşit zorluk seviyesi ve maksimum Private LB stabilitesi.
2. Sızıntısız (Zero-Leakage) mimari. XGBoost ve LGBM elendi.
3. CATBOOST SUPREMACY: Zıt karakterli 3 farklı CatBoost mutasyonu:
   a) Muhafazakar (Sığ ağaç, yüksek L2 cezası)
   b) Agresif (Derin ağaç, düşük L2 cezası)
   c) Çeşitlilik (Yüksek rastgelelik gücü)
4. RankGauss (QuantileTransformer) uygulanmış Derin Keras Neural Network.
5. SLSQP Simplex Optimizasyonu ile OOF üzerinde ağırlıklandırma (Meta-Learner yok)
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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from scipy.optimize import minimize
from catboost import CatBoostRegressor
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    [tf.config.experimental.set_memory_growth(g, True) for g in gpus]

# ── AYARLAR ──
TRAIN_PATH  = r'd:\Datathonv2\train.csv'
TEST_PATH   = r'd:\Datathonv2\test_x.csv'
OUTPUT_DIR  = r'd:\Datathonv2'
TARGET_COL  = 'bilissel_performans_skoru'
SEEDS       = [42, 2024, 7, 123, 2026]
N_FOLDS     = 10

try:
    _t = CatBoostRegressor(iterations=2, task_type='GPU', verbose=0)
    _t.fit([['A'],['B'],['C']], [1,2,3], cat_features=[0])
    CAT_TASK = 'GPU'
except:
    CAT_TASK = 'CPU'

print("="*80)
print("V27: KAGGLE GRANDMASTER PIPELINE (CATBOOST SUPREMACY)")
print(f"Seeds: {SEEDS} | Folds: {N_FOLDS} | Target-Binned Stratified K-Fold")
print("="*80)

# ── 1. VERİ YÜKLEME ──
train_raw = pd.read_csv(TRAIN_PATH)
test_raw  = pd.read_csv(TEST_PATH)

ulke_map = {'South Korea': 'Guney Kore', 'Spain': 'Ispanya', 'Sweden': 'Isvec', 'Mexico': 'Meksika', 'Netherlands': 'Hollanda'}
for df in [train_raw, test_raw]:
    df['ulke'] = df['ulke'].replace(ulke_map)

# ── 2. FEATURE ENGINEERING (ELITE LEVEL) ──
def create_features(df):
    df = df.copy()
    # Biyolojik ve Etkileşim Oznitelikleri
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
    df['hafif_uyku_yuzdesi']    = 100.0 - df['toplam_uyku_kalitesi']
    df['sicaklik_farki']        = abs(df['oda_sicakligi_celsius'] - 18.3)
    df['vki_grup']              = pd.cut(df['vucut_kitle_indeksi'], bins=[0, 18.5, 25, 30, 100], labels=['Zayif', 'Normal', 'Fazla_Kilolu', 'Obez']).astype(str)
    df['yas_grup']              = pd.cut(df['yas'], bins=[0, 20, 30, 40, 50, 60, 100], labels=['0-20', '21-30', '31-40', '41-50', '51-60', '60+']).astype(str)
    
    df['cross_meslek_cinsiyet'] = df['meslek'].astype(str) + "_" + df['cinsiyet'].astype(str)
    df['cross_kronotip_ruh']    = df['kronotip'].astype(str) + "_" + df['ruh_sagligi_durumu'].astype(str)
    
    # Hedefe dönük sayısallaştırmalar
    sev = {'Saglikli': 3, 'Anksiyete': 2, 'Depresyon': 1, 'Anksiyete ve depresyon': 0}
    df['ruh_sagligi_severity']  = df['ruh_sagligi_durumu'].map(sev).fillna(1.5)
    return df

train_raw = create_features(train_raw)
test_raw  = create_features(test_raw)

cat_cols = ['cinsiyet', 'meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi', 'cross_meslek_cinsiyet', 'cross_kronotip_ruh', 'vki_grup', 'yas_grup']

# CatBoost icin tüm kategorikler String/Object kalmali
for col in cat_cols:
    train_raw[col] = train_raw[col].replace('nan', 'MISSING').fillna('MISSING').astype(str)
    test_raw[col]  = test_raw[col].replace('nan', 'MISSING').fillna('MISSING').astype(str)

drop_cols = ['id', TARGET_COL]
cont_cols = [c for c in train_raw.columns if c not in drop_cols and c not in cat_cols]

for c in cont_cols:
    med = train_raw[c].median()
    train_raw[c] = train_raw[c].fillna(med)
    test_raw[c]  = test_raw[c].fillna(med)

X_all = train_raw.drop(['id', TARGET_COL], axis=1)
y = train_raw[TARGET_COL]
X_test = test_raw.drop(['id'], axis=1)
test_ids = test_raw['id'].copy()

# Stratified K-Fold için Hedef Değişkeni 10 Gruba Ayırma
target_bins = pd.cut(y, bins=10, labels=False)

print(f"Toplam Oznitelik: {X_all.shape[1]} (Kategorik: {len(cat_cols)}, Sürekli: {len(cont_cols)})")

# ── 3. RANKGAUSS NEURAL NETWORK (TabNet Alternatifi) ──
def build_nn(input_dim, seed):
    tf.random.set_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='swish', input_dim=input_dim),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='swish'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='swish'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='mse')
    return model

# ── 4. SIZINTISIZ ÇAPRAZ DOĞRULAMA (LEAK-FREE CV) ──
all_test_preds = []
all_cv_scores  = []
start_total    = time.time()

for seed_idx, SEED in enumerate(SEEDS):
    print(f"\n{'='*70}")
    print(f"SEED {SEED}  ({seed_idx+1}/{len(SEEDS)})  —  {time.strftime('%H:%M:%S')}")
    print(f"{'='*70}")
    t0 = time.time()

    # CatBoost Mutasyonları
    cb_cons_p = {'learning_rate': 0.03, 'depth': 4, 'l2_leaf_reg': 10, 'bagging_temperature': 1.0, 'iterations': 4000, 'random_seed': SEED, 'task_type': CAT_TASK, 'verbose': 0}
    cb_aggr_p = {'learning_rate': 0.015, 'depth': 8, 'l2_leaf_reg': 1, 'bagging_temperature': 0.2, 'iterations': 4000, 'random_seed': SEED, 'task_type': CAT_TASK, 'verbose': 0}
    cb_div_p  = {'learning_rate': 0.04, 'depth': 6, 'l2_leaf_reg': 3, 'random_strength': 5.0, 'iterations': 4000, 'random_seed': SEED, 'task_type': CAT_TASK, 'verbose': 0}

    # Hedef Katmanlı Doğrulama (Target-Binned Stratified)
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_cons = np.zeros(len(X_all))
    oof_aggr = np.zeros(len(X_all))
    oof_div  = np.zeros(len(X_all))
    oof_nn   = np.zeros(len(X_all))

    tst_cons = np.zeros(len(X_test))
    tst_aggr = np.zeros(len(X_test))
    tst_div  = np.zeros(len(X_test))
    tst_nn   = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all, target_bins)):
        tf0 = time.time()
        
        X_tr, X_val = X_all.iloc[tr_idx].copy(), X_all.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[tr_idx].copy(), y.iloc[val_idx].copy()
        X_tst = X_test.copy()

        # NN İÇİN İÇERİDE TARGET ENCODING (Smooth=10)
        smooth = 10
        te_cols = []
        for col in cat_cols:
            global_mean = y_tr.mean()
            agg = y_tr.groupby(X_tr[col]).agg(['count', 'mean'])
            counts = agg['count']
            means = agg['mean']
            smooth_te = (counts * means + smooth * global_mean) / (counts + smooth)
            
            te_col = col + '_te'
            te_cols.append(te_col)
            X_tr[te_col]  = X_tr[col].map(smooth_te).fillna(global_mean)
            X_val[te_col] = X_val[col].map(smooth_te).fillna(global_mean)
            X_tst[te_col] = X_tst[col].map(smooth_te).fillna(global_mean)

        num_features = cont_cols + te_cols
        
        # NN İÇİN İÇERİDE RANKGAUSS (QuantileTransformer)
        # TabNet'in başarısının sırrı Gauss dağılımına dönüştürmesidir.
        qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=SEED)
        X_tr_nn  = qt.fit_transform(X_tr[num_features])
        X_val_nn = qt.transform(X_val[num_features])
        X_tst_nn = qt.transform(X_tst[num_features])

        # CatBoost için ham String kategorikler verilir. (Zero Leakage)
        X_tr_cat = X_tr[cont_cols + cat_cols]
        X_val_cat = X_val[cont_cols + cat_cols]
        X_tst_cat = X_tst[cont_cols + cat_cols]

        # 1. CatBoost Conservative (Sığ, Yüksek L2)
        m_cons = CatBoostRegressor(**cb_cons_p)
        m_cons.fit(X_tr_cat, y_tr, cat_features=cat_cols, eval_set=(X_val_cat, y_val), verbose=False, early_stopping_rounds=200)
        oof_cons[val_idx] = m_cons.predict(X_val_cat)
        tst_cons += m_cons.predict(X_tst_cat) / N_FOLDS
        r_cons = root_mean_squared_error(y_val, oof_cons[val_idx])

        # 2. CatBoost Aggressive (Derin, Düşük L2)
        m_aggr = CatBoostRegressor(**cb_aggr_p)
        m_aggr.fit(X_tr_cat, y_tr, cat_features=cat_cols, eval_set=(X_val_cat, y_val), verbose=False, early_stopping_rounds=200)
        oof_aggr[val_idx] = m_aggr.predict(X_val_cat)
        tst_aggr += m_aggr.predict(X_tst_cat) / N_FOLDS
        r_aggr = root_mean_squared_error(y_val, oof_aggr[val_idx])

        # 3. CatBoost Diverse (Rastgelelik Gücü Yüksek)
        m_div = CatBoostRegressor(**cb_div_p)
        m_div.fit(X_tr_cat, y_tr, cat_features=cat_cols, eval_set=(X_val_cat, y_val), verbose=False, early_stopping_rounds=200)
        oof_div[val_idx] = m_div.predict(X_val_cat)
        tst_div += m_div.predict(X_tst_cat) / N_FOLDS
        r_div = root_mean_squared_error(y_val, oof_div[val_idx])

        # 4. Keras NN (RankGauss)
        nn = build_nn(X_tr_nn.shape[1], SEED)
        nn.fit(X_tr_nn, y_tr, validation_data=(X_val_nn, y_val),
               epochs=200, batch_size=256, verbose=0,
               callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=30, restore_best_weights=True)])
        oof_nn[val_idx] = nn.predict(X_val_nn, verbose=0).flatten()
        tst_nn += nn.predict(X_tst_nn, verbose=0).flatten() / N_FOLDS
        r_nn = root_mean_squared_error(y_val, oof_nn[val_idx])
        del nn; gc.collect()

        elapsed = int(time.time() - tf0)
        print(f"  Fold {fold+1:2d}/{N_FOLDS} ({elapsed:3d}s) | "
              f"CB_Cons:{r_cons:.4f} CB_Aggr:{r_aggr:.4f} CB_Div:{r_div:.4f} NN:{r_nn:.4f}")

    # ── 5. SLSQP SIMPLEX ENSEMBLE OPTIMIZATION (RIDGE HATASI GİDERİLDİ) ──
    # OOF tahminlerini birleştir
    oof_meta = np.column_stack([oof_cons, oof_aggr, oof_div, oof_nn])
    tst_meta = np.column_stack([tst_cons, tst_aggr, tst_div, tst_nn])

    # Ağırlık optimizasyonu fonksiyonu
    def loss_func(weights):
        # Ağırlıklı ortalamanın RMSE'si
        pred = np.sum(weights * oof_meta, axis=1)
        return root_mean_squared_error(y, pred)

    # SLSQP Optimizasyonu (Ağırlıklar 0-1 arası ve toplamları 1 olmak zorunda)
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * 4
    init_weights = [0.40, 0.30, 0.20, 0.10] # CatBoost'a öncelik tanıyan başlangıç
    
    res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    best_w = res.x

    # Optimize edilmiş ağırlıklarla Seed tahmini
    seed_pred = np.sum(best_w * tst_meta, axis=1)
    seed_pred = np.clip(seed_pred, 0, 10)
    seed_cv = loss_func(best_w)

    print(f"\n  OOF -> CB_Cons:{root_mean_squared_error(y,oof_cons):.4f} CB_Aggr:{root_mean_squared_error(y,oof_aggr):.4f} "
          f"CB_Div:{root_mean_squared_error(y,oof_div):.4f} NN:{root_mean_squared_error(y,oof_nn):.4f}")
    
    print(f"  SLSQP Optimize Agirliklar: Cons(%.3f) Aggr(%.3f) Div(%.3f) NN(%.3f)" % tuple(best_w))
    print(f"  Pred std: {seed_pred.std():.4f}")
    print(f"  * SEED {SEED} Meta CV RMSE: {seed_cv:.5f}  ({int(time.time()-t0)//60}dk)")

    pd.DataFrame({'id': test_ids, 'bilissel_performans_skoru': seed_pred}).to_csv(os.path.join(OUTPUT_DIR, f'submission_v27_grandmaster_seed{SEED}.csv'), index=False)
    all_test_preds.append(seed_pred)
    all_cv_scores.append(seed_cv)
    gc.collect()

# ── 6. FINAL ──
print(f"\n{'='*70}")
print("V27 FINAL ENSEMBLE (TOP 0.1% KAGGLE GRANDMASTER - CATBOOST SUPREMACY)")
print(f"{'='*70}")
for s, sc in zip(SEEDS, all_cv_scores):
    print(f"  Seed {s:4d}: {sc:.5f}")
print(f"  Ortalama Meta CV : {np.mean(all_cv_scores):.5f} +/- {np.std(all_cv_scores):.5f}")

final_preds = np.clip(np.mean(all_test_preds, axis=0), 0, 10)
pd.DataFrame({'id': test_ids, 'bilissel_performans_skoru': final_preds}).to_csv(os.path.join(OUTPUT_DIR, 'submission_v27_kagglegm_final.csv'), index=False)

print(f"\nV27 TAMAMLANDI! ({int(time.time()-start_total)//60} dk)")
print(f"  -> submission_v27_kagglegm_final.csv")
print(f"  mean={final_preds.mean():.4f}, std={final_preds.std():.4f}, min={final_preds.min():.4f}, max={final_preds.max():.4f}")
