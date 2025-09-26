# enhanced_regression_forecast_pipeline.py
# Enhanced regression pipeline for future demand prediction
# Requirements: pandas, numpy, scikit-learn, lightgbm, catboost, optuna, matplotlib, seaborn, joblib

import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import optuna
from datetime import datetime, timedelta

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

SEED = 42
np.random.seed(SEED)

# -------------------------
# 0. Configuration & File paths
# -------------------------
ROOT = Path.cwd()
INV_FP = ROOT / "/content/Inventary.csv"
SALES_FP = ROOT / "/content/sales.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Forecast configuration
FORECAST_HORIZONS = [1, 3, 7, 14, 30]  # Multiple prediction horizons
PREDICTION_HORIZON = 7  # Main horizon for alerts
SAFETY_MULTIPLIER = 1.2  # Safety stock multiplier

print("🚀 Enhanced Regression Demand Forecasting Pipeline")
print(f"📁 Output directory: {OUT_DIR}")

# -------------------------
# 1. Load & Enhanced Data Cleaning
# -------------------------
print("\n📊 Loading and cleaning data...")

inv = pd.read_csv(INV_FP)
sales = pd.read_csv(SALES_FP)

# Normalize column names
sales.columns = [c.strip().lower().replace(' ', '_') for c in sales.columns]
inv.columns = [c.strip().lower().replace(' ', '_') for c in inv.columns]

# Parse and clean dates
sales['date'] = pd.to_datetime(sales['date'], errors='coerce')
sales = sales.dropna(subset=['date']).sort_values('date')

# Remove obvious outliers in units_sold (beyond 99.9th percentile)
q99 = sales['units_sold'].quantile(0.999)
sales = sales[sales['units_sold'] <= q99]

print(f"📈 Sales data: {sales.shape[0]:,} records")
print(f"🏪 Inventory data: {inv.shape[0]:,} records")
print(f"📅 Date range: {sales['date'].min()} → {sales['date'].max()}")

# -------------------------
# 2. Advanced Data Aggregation & Preprocessing
# -------------------------
print("\n🔄 Aggregating data to daily product-city level...")

# Comprehensive aggregation
sales_agg = sales.groupby([
    'date', 'product_id', 'product_name', 'category', 'sub_category', 'city_name'
]).agg({
    'units_sold': ['sum', 'count'],  # total units and number of transactions
    'selling_price': ['mean', 'std', 'min', 'max'],
    'mrp': ['mean', 'std'],
}).reset_index()

# Flatten column names
sales_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in sales_agg.columns]
sales_agg = sales_agg.rename(columns={
    'units_sold_sum': 'units_sold',
    'units_sold_count': 'transaction_count',
    'selling_price_mean': 'selling_price',
    'selling_price_std': 'price_volatility',
    'selling_price_min': 'min_price',
    'selling_price_max': 'max_price',
    'mrp_mean': 'mrp',
    'mrp_std': 'mrp_std'
})

# Fill price volatility NaN (single transaction days)
sales_agg['price_volatility'] = sales_agg['price_volatility'].fillna(0)

# Convert IDs to string for consistency
sales_agg['product_id'] = sales_agg['product_id'].astype(str)
sales_agg['city_name'] = sales_agg['city_name'].astype(str)
inv['product_id'] = inv['product_id'].astype(str)
inv['city_name'] = inv['city_name'].astype(str)

# Merge inventory (latest snapshot approach)
inv_latest = inv.groupby(['product_id', 'city_name']).agg({
    'stock_quantity': 'sum'
}).reset_index()

sales_agg = sales_agg.merge(inv_latest, on=['product_id', 'city_name'], how='left')
sales_agg['stock_quantity'] = sales_agg['stock_quantity'].fillna(0)

print(f"✅ Aggregated to {sales_agg.shape[0]:,} daily product-city records")

# -------------------------
# 3. Enhanced Feature Engineering
# -------------------------
print("\n🛠️ Creating advanced features...")

df = sales_agg.copy().sort_values(['product_id', 'city_name', 'date'])

# Fill missing prices with group medians
for col in ['selling_price', 'mrp', 'price_volatility']:
    df[col] = df[col].fillna(
        df.groupby('product_id')[col].transform('median')
    ).fillna(df[col].median())

# ===== CALENDAR FEATURES =====
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_month'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
df['is_month_start'] = (df['day_of_month'] <= 7).astype(int)
df['is_month_end'] = (df['day_of_month'] >= 24).astype(int)

# Season encoding
df['season'] = ((df['month'] % 12 + 3) // 3).map({
    1: 'winter', 2: 'spring', 3: 'summer', 4: 'autumn'
})

# ===== PRICE FEATURES =====
df['price_margin'] = (df['mrp'] - df['selling_price']).fillna(0)
df['price_ratio'] = (df['selling_price'] / (df['mrp'] + 1e-6)).clip(0, 10)
df['discount_percent'] = ((df['mrp'] - df['selling_price']) / (df['mrp'] + 1e-6) * 100).clip(0, 100)

# ===== LAG FEATURES (Extended) =====
LAG_PERIODS = [1, 2, 3, 7, 14, 21, 28, 35]  # Up to 5 weeks
ROLLING_WINDOWS = [3, 7, 14, 28]

# Create lag features
for lag in LAG_PERIODS:
    df[f'lag_units_{lag}'] = df.groupby(['product_id', 'city_name'])['units_sold'].shift(lag)
    df[f'lag_price_{lag}'] = df.groupby(['product_id', 'city_name'])['selling_price'].shift(lag)

# ===== ROLLING STATISTICS =====
for window in ROLLING_WINDOWS:
    # Units sold statistics
    df[f'roll_mean_{window}'] = df.groupby(['product_id', 'city_name'])['units_sold'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df[f'roll_std_{window}'] = df.groupby(['product_id', 'city_name'])['units_sold'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std()
    )
    df[f'roll_max_{window}'] = df.groupby(['product_id', 'city_name'])['units_sold'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).max()
    )
    df[f'roll_min_{window}'] = df.groupby(['product_id', 'city_name'])['units_sold'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).min()
    )
    
    # Price statistics
    df[f'roll_price_mean_{window}'] = df.groupby(['product_id', 'city_name'])['selling_price'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

# ===== EXPANDING FEATURES =====
df['expanding_mean'] = df.groupby(['product_id', 'city_name'])['units_sold'].transform(
    lambda x: x.shift(1).expanding(min_periods=1).mean()
)
df['expanding_std'] = df.groupby(['product_id', 'city_name'])['units_sold'].transform(
    lambda x: x.shift(1).expanding(min_periods=1).std()
)
df['expanding_count'] = df.groupby(['product_id', 'city_name'])['units_sold'].transform(
    lambda x: np.arange(1, len(x) + 1)
)

# ===== TREND FEATURES =====
# Short vs long term trend comparison
df['trend_ratio_7_28'] = (df['roll_mean_7'] / (df['roll_mean_28'] + 1e-6)).fillna(1)
df['trend_diff_7_28'] = (df['roll_mean_7'] - df['roll_mean_28']).fillna(0)

# ===== FREQUENCY FEATURES =====
df['sold_flag'] = (df['units_sold'] > 0).astype(int)
for window in [7, 14, 28]:
    df[f'sale_frequency_{window}'] = df.groupby(['product_id', 'city_name'])['sold_flag'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

# ===== VOLATILITY FEATURES =====
df['volatility_7'] = df.groupby(['product_id', 'city_name'])['units_sold'].transform(
    lambda x: x.shift(1).rolling(7, min_periods=2).apply(lambda y: np.std(y) / (np.mean(y) + 1e-6))
)

# Fill remaining NaNs
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

print(f"✅ Created {len([c for c in df.columns if c.startswith(('lag_', 'roll_', 'expanding_'))])} time-series features")

# -------------------------
# 4. Multiple Target Creation (Multi-horizon forecasting)
# -------------------------
print(f"\n🎯 Creating targets for horizons: {FORECAST_HORIZONS}")

# Create multiple targets
for horizon in FORECAST_HORIZONS:
    df[f'target_t{horizon}'] = df.groupby(['product_id', 'city_name'])['units_sold'].shift(-horizon)

# Use primary target for main model
df['target'] = df[f'target_t{PREDICTION_HORIZON}']

# -------------------------
# 5. Label Encoding for Categorical Features
# -------------------------
print("\n🏷️ Encoding categorical features...")

# Encode main categorical variables
encoders = {}
for col in ['product_id', 'city_name', 'category', 'sub_category', 'season']:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        joblib.dump(le, OUT_DIR / f"labelenc_{col}.joblib")

print(f"✅ Encoded {len(encoders)} categorical variables")

# -------------------------
# 6. Feature Selection & Data Preparation
# -------------------------
print("\n📋 Preparing feature matrix...")

# Define comprehensive feature list
base_features = [
    'product_id_enc', 'city_name_enc', 'category_enc', 'sub_category_enc', 'season_enc',
    'day_of_week', 'day_of_month', 'month', 'quarter', 'week_of_year',
    'is_weekend', 'is_month_start', 'is_month_end',
    'selling_price', 'mrp', 'price_margin', 'price_ratio', 'discount_percent',
    'price_volatility', 'transaction_count', 'stock_quantity',
    'expanding_mean', 'expanding_std', 'expanding_count',
    'trend_ratio_7_28', 'trend_diff_7_28', 'volatility_7'
]

# Add lag features
lag_features = [f'lag_units_{lag}' for lag in LAG_PERIODS] + [f'lag_price_{lag}' for lag in LAG_PERIODS]

# Add rolling features
roll_features = []
for window in ROLLING_WINDOWS:
    roll_features.extend([
        f'roll_mean_{window}', f'roll_std_{window}', f'roll_max_{window}', f'roll_min_{window}',
        f'roll_price_mean_{window}'
    ])

# Add frequency features
freq_features = [f'sale_frequency_{window}' for window in [7, 14, 28]]

# Combine all features
features = base_features + lag_features + roll_features + freq_features

# Keep only existing features
features = [f for f in features if f in df.columns]

print(f"✅ Selected {len(features)} features for modeling")

# Filter data: require minimum history and valid target
MIN_HISTORY_DAYS = max(LAG_PERIODS)
df_model = df.groupby(['product_id', 'city_name']).filter(
    lambda x: len(x) > MIN_HISTORY_DAYS
).reset_index(drop=True)

# Remove rows with missing targets or critical lag features
critical_features = [f'lag_units_{lag}' for lag in LAG_PERIODS[:4]] + ['target']
df_model = df_model.dropna(subset=critical_features)

X = df_model[features]
y = df_model['target']

print(f"📊 Final dataset: {len(df_model):,} samples with {len(features)} features")
print(f"📈 Target statistics: mean={y.mean():.2f}, std={y.std():.2f}, range=[{y.min():.2f}, {y.max():.2f}]")

# -------------------------
# 7. Advanced Model Training & Hyperparameter Optimization
# -------------------------
print(f"\n🤖 Training regression models with {PREDICTION_HORIZON}-day forecast horizon...")

# Time-based train/validation split
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# Enhanced evaluation metrics
def evaluate_regression(y_true, y_pred, model_name="Model"):
    """Comprehensive regression evaluation"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Handle MAPE calculation with zero values
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.inf
    
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    median_ae = np.median(np.abs(y_true - y_pred))
    q90_ae = np.percentile(np.abs(y_true - y_pred), 90)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE%': mape,
        'R²': r2,
        'Median_AE': median_ae,
        'Q90_AE': q90_ae
    }
    
    print(f"\n📊 {model_name} Performance:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    return metrics

# ===== LIGHTGBM OPTIMIZATION =====
def objective_lgb(trial):
    """Optuna objective for LightGBM optimization"""
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'seed': SEED,
        'num_leaves': trial.suggest_int('num_leaves', 31, 300),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
    }
    
    cv_scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=3000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        pred = model.predict(X_val, num_iteration=model.best_iteration)
        cv_scores.append(mean_absolute_error(y_val, pred))
    
    return np.mean(cv_scores)

print("🔍 Optimizing LightGBM hyperparameters...")
study_lgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=True, n_jobs=1)

best_lgb_params = study_lgb.best_params
best_lgb_params.update({'objective': 'regression', 'metric': 'mae', 'verbosity': -1, 'seed': SEED})

print(f"✅ Best LightGBM params found with CV MAE: {study_lgb.best_value:.4f}")

# ===== CATBOOST OPTIMIZATION =====
def objective_cat(trial):
    """Optuna objective for CatBoost optimization"""
    params = {
        'iterations': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 30.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'random_seed': SEED,
        'verbose': False,
        'early_stopping_rounds': 100
    }
    
    cv_scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        
        pred = model.predict(X_val)
        cv_scores.append(mean_absolute_error(y_val, pred))
    
    return np.mean(cv_scores)

print("🔍 Optimizing CatBoost hyperparameters...")
study_cat = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
study_cat.optimize(objective_cat, n_trials=30, show_progress_bar=True, n_jobs=1)

best_cat_params = study_cat.best_params
print(f"✅ Best CatBoost params found with CV MAE: {study_cat.best_value:.4f}")

# -------------------------
# 8. Final Model Training & Evaluation
# -------------------------
print("\n🎯 Training final models...")

# Create holdout test set (last 14 days)
holdout_days = 14
max_date = df_model['date'].max()
cutoff_date = max_date - pd.Timedelta(days=holdout_days)

train_mask = df_model['date'] <= cutoff_date
test_mask = df_model['date'] > cutoff_date

X_train_final = X[train_mask]
y_train_final = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"📊 Train set: {len(X_train_final):,} samples")
print(f"📊 Test set: {len(X_test):,} samples")

# ===== TRAIN FINAL LIGHTGBM =====
print("\n🌟 Training final LightGBM model...")
train_data = lgb.Dataset(X_train_final, label=y_train_final)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

lgb_final = lgb.train(
    best_lgb_params,
    train_data,
    num_boost_round=5000,
    valid_sets=[test_data],
    callbacks=[
        lgb.early_stopping(200, verbose=True),
        lgb.log_evaluation(100)
    ]
)

# Evaluate LightGBM
y_pred_lgb = lgb_final.predict(X_test, num_iteration=lgb_final.best_iteration)
metrics_lgb = evaluate_regression(y_test, y_pred_lgb, "LightGBM")

# Save LightGBM
joblib.dump(lgb_final, OUT_DIR / "model_lgb_final.joblib")

# ===== TRAIN FINAL CATBOOST =====
print("\n🌟 Training final CatBoost model...")
cat_final = CatBoostRegressor(**best_cat_params)
cat_final.fit(
    X_train_final, y_train_final,
    eval_set=(X_test, y_test),
    use_best_model=True,
    plot=False
)

# Evaluate CatBoost
y_pred_cat = cat_final.predict(X_test)
metrics_cat = evaluate_regression(y_test, y_pred_cat, "CatBoost")

# Save CatBoost
joblib.dump(cat_final, OUT_DIR / "model_cat_final.joblib")

# ===== ENSEMBLE MODEL =====
print("\n🔄 Creating ensemble model...")

# Simple weighted average (can be optimized)
weight_lgb = 1 / metrics_lgb['MAE'] if metrics_lgb['MAE'] > 0 else 1
weight_cat = 1 / metrics_cat['MAE'] if metrics_cat['MAE'] > 0 else 1
total_weight = weight_lgb + weight_cat

y_pred_ensemble = (weight_lgb * y_pred_lgb + weight_cat * y_pred_cat) / total_weight
metrics_ensemble = evaluate_regression(y_test, y_pred_ensemble, "Weighted Ensemble")

# Meta-learner (Stacking)
print("\n🧠 Training meta-learner...")
meta_features = np.column_stack([
    lgb_final.predict(X_train_final, num_iteration=lgb_final.best_iteration),
    cat_final.predict(X_train_final)
])
meta_test_features = np.column_stack([y_pred_lgb, y_pred_cat])

# Try different meta-learners
meta_models = {
    'Ridge': Ridge(alpha=1.0, random_state=SEED),
    'Lasso': Lasso(alpha=0.1, random_state=SEED),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=SEED)
}

best_meta = None
best_meta_score = float('inf')

for name, model in meta_models.items():
    model.fit(meta_features, y_train_final)
    meta_pred = model.predict(meta_test_features)
    score = mean_absolute_error(y_test, meta_pred)
    print(f"   Meta-{name} MAE: {score:.4f}")
    
    if score < best_meta_score:
        best_meta_score = score
        best_meta = model

# Final meta-learner evaluation
y_pred_meta = best_meta.predict(meta_test_features)
metrics_meta = evaluate_regression(y_test, y_pred_meta, "Meta-Learner (Stacking)")

# Save best meta-learner
joblib.dump(best_meta, OUT_DIR / "meta_learner.joblib")

# -------------------------
# 9. Feature Importance Analysis
# -------------------------
print("\n📊 Analyzing feature importance...")

# LightGBM feature importance
lgb_importance = pd.DataFrame({
    'feature': features,
    'importance': lgb_final.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

# Save feature importance
lgb_importance.to_csv(OUT_DIR / "feature_importance_lgb.csv", index=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = lgb_importance.head(25)
sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
plt.title(f'Top 25 Features - LightGBM Importance (Gain)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(OUT_DIR / "feature_importance_plot.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"\n🔝 Top 10 Most Important Features:")
for i, (_, row) in enumerate(lgb_importance.head(10).iterrows(), 1):
    print(f"   {i:2d}. {row['feature']: <25} ({row['importance']:,.0f})")

# -------------------------
# 10. Model Diagnostics & Visualization
# -------------------------
print("\n🔍 Creating diagnostic plots...")

# Create diagnostic plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Residuals histogram
residuals = y_test - y_pred_meta
axes[0,0].hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0,0].set_title('Residuals Distribution')
axes[0,0].set_xlabel('Residual')
axes[0,0].set_ylabel('Frequency')

# 2. True vs Predicted scatter
axes[0,1].scatter(y_test, y_pred_meta, alpha=0.6, s=10)
max_val = max(y_test.max(), y_pred_meta.max())
axes[0,1].plot([0, max_val], [0, max_val], 'r--', lw=2)
axes[0,1].set_xlabel('True Values')
axes[0,1].set_ylabel('Predicted Values')
axes[0,1].set_title('True vs Predicted')

# 3. Residuals vs Predicted
axes[0,2].scatter(y_pred_meta, residuals, alpha=0.6, s=10)
axes[0,2].axhline(y=0, color='r', linestyle='--')
axes[0,2].set_xlabel('Predicted Values')
axes[0,2].set_ylabel('Residuals')
axes[0,2].set_title('Residuals vs Predicted')

# 4. Time series of errors
test_dates = df_model[test_mask]['date'].values
daily_errors = pd.DataFrame({
    'date': test_dates,
    'error': np.abs(residuals)
}).groupby('date')['error'].mean()

axes[1,0].plot(daily_errors.index, daily_errors.values, marker='o', markersize=3)
axes[1,0].set_title('Daily Mean Absolute Error')
axes[1,0].set_xlabel('Date')
axes[1,0].set_ylabel('Mean Absolute Error')
axes[1,0].tick_params(axis='x', rotation=45)

# 5. Model comparison
model_names = ['LightGBM', 'CatBoost', 'Ensemble', 'Meta-Learner']
mae_scores = [metrics_lgb['MAE'], metrics_cat['MAE'], metrics_ensemble['MAE'], metrics_meta['MAE']]
axes[1,1].bar(model_names, mae_scores, color=['lightblue', 'lightgreen', 'orange', 'lightcoral'])
axes[1,1].set_title('Model Comparison (MAE)')
axes[1,1].set_ylabel('Mean Absolute Error')
# continuing from previous plotting section (finish axes formatting)
axes[1,1].tick_params(axis='x', rotation=15)

# 6. Top feature importance (reuse computed lgb_importance)
axes[1,2].barh(top_features['feature'][::-1], top_features['importance'][::-1])
axes[1,2].set_title('Top Features (LightGBM)')
axes[1,2].set_xlabel('Importance')
plt.tight_layout()

# Save full diagnostics figure
diag_fp = OUT_DIR / "diagnostic_plots.png"
fig.savefig(diag_fp, dpi=300, bbox_inches='tight')
print(f"✅ Saved diagnostics plot: {diag_fp}")

# Show in-line (optional)
plt.close(fig)

# -------------------------
# 11. Save artifacts & metadata
# -------------------------
print("\n💾 Saving artifacts and metadata...")

# Save trained models if not already saved
# (LightGBM saved earlier as model_lgb_final.joblib — joblib.dump used above)
# (CatBoost saved earlier as model_cat_final.joblib)

# Save ensemble predictions on test set
ensemble_df = X_test.copy()
ensemble_df['true'] = y_test
ensemble_df['pred_lgb'] = y_pred_lgb
ensemble_df['pred_cat'] = y_pred_cat
ensemble_df['pred_ensemble'] = y_pred_ensemble
ensemble_df['pred_meta'] = y_pred_meta
ensemble_fp = OUT_DIR / "test_predictions_summary.csv"
ensemble_df.to_csv(ensemble_fp, index=False)
print(f"✅ Saved test predictions summary: {ensemble_fp}")

# Save encoders mapping (already saved individually, but also save a manifest)
encoders_manifest = {k: str(OUT_DIR / f"labelenc_{k}.joblib") for k in encoders.keys()}
joblib.dump(encoders_manifest, OUT_DIR / "encoders_manifest.joblib")

# Save final selected feature list
import json
features_fp = OUT_DIR / "feature_list.json"
with open(features_fp, "w") as fh:
    json.dump(features, fh, indent=2)
print(f"✅ Saved feature list: {features_fp}")

# Save model metrics summary
metrics_summary = {
    "LightGBM": metrics_lgb,
    "CatBoost": metrics_cat,
    "Ensemble": metrics_ensemble,
    "MetaLearner": metrics_meta
}
metrics_fp = OUT_DIR / "metrics_summary.json"
with open(metrics_fp, "w") as fh:
    json.dump(metrics_summary, fh, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
print(f"✅ Saved metrics summary: {metrics_fp}")

# Save top feature importance CSV already saved above as feature_importance_lgb.csv
print(f"✅ Feature importance saved: {OUT_DIR / 'feature_importance_lgb.csv'}")
print(f"✅ Feature importance plot saved: {OUT_DIR / 'feature_importance_plot.png'}")

# -------------------------
# 12. Quick interactive examples (optional prints)
# -------------------------
print("\n🔎 Example forecasts (first 10 rows of test predictions):")
print(ensemble_df[['true','pred_lgb','pred_cat','pred_ensemble','pred_meta']].head(10).to_string(index=False))

# -------------------------
# 13. Done
# -------------------------
print("\n🎉 Pipeline complete. Artifacts written to:", OUT_DIR)
print("You can now load `outputs/model_lgb_final.joblib`, `outputs/model_cat_final.joblib`, and `outputs/meta_learner.joblib` in your Streamlit app.")
