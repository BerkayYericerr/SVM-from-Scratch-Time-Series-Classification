import numpy as np
from svm_model import SVM
from sklearn.preprocessing import StandardScaler
import time
import sys

def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return acc, prec, rec, f1


def time_series_splits(n_samples, n_splits=5):
    
    if n_samples < (n_splits + 1):
        raise ValueError(f"Not enough samples ({n_samples}) for {n_splits}-split time series CV.")

    test_size = n_samples // (n_splits + 1)
    if test_size < 1:
        raise ValueError("Computed fold size is < 1; reduce n_splits or provide more samples.")

    indices = np.arange(n_samples)
    splits = []
    for i in range(n_splits):
        train_end = test_size * (i + 1)
        val_start = train_end
        val_end = test_size * (i + 2) if i < (n_splits - 1) else n_samples

        train_idx = indices[:train_end]
        val_idx = indices[val_start:val_end]

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        splits.append((train_idx, val_idx))

    if len(splits) != n_splits:
        raise ValueError(f"Could only create {len(splits)} splits; need {n_splits}.")

    return splits


def time_series_cv_evaluate(X, y, svm_kwargs, n_splits=5):

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    splits = time_series_splits(len(y), n_splits=n_splits)

    fold_metrics = []
    cv_start = time.time()
    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        fold_start = time.time()
        print(f"\n[CV] Fold {fold}/5 START  | train={len(tr_idx)} val={len(va_idx)}", flush=True)
        
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)

        svm = SVM(**svm_kwargs)
        svm.fit(X_tr, y_tr)
        preds = svm.predict(X_va)

        fold_metrics.append(calc_metrics(y_va, preds))

        fold_time = time.time() - fold_start
        elapsed = time.time() - cv_start
        avg_per_fold = elapsed / fold
        eta = avg_per_fold * (5 - fold)

        print(f"[CV] Fold {fold}/5 DONE   | fold_time={fold_time:.1f}s | elapsed={elapsed:.1f}s | ETA={eta:.1f}s", flush=True)
        
    # average across folds
    fold_metrics = np.array(fold_metrics, dtype=float)  
    mean_metrics = fold_metrics.mean(axis=0)  
    std_metrics = fold_metrics.std(axis=0)

    return mean_metrics, std_metrics


def train_and_test(X, y, name, n_splits=2):
    print(f"\n--- {name} ---")

    # 80/20 splıt
    n = len(y)
    split = int(n * 0.8)
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    print(f"Train size: {len(y_train)} | Test size: {len(y_test)}")

    results = {}

    configs = {
        'linear': dict(C=1.0, kernel='linear'),
        'rbf': dict(C=1.0, kernel='rbf', sigma=1.0),
        'poly': dict(C=1.0, kernel='poly', degree=2, coef0=1.0),
        'custom': dict(C=1.0, kernel='custom', sigma=1.0, sigma_delta=1.0, k_alpha=1/3, k_beta=1/3, k_gamma=1/3),
    }

    # --- Time-series CV on training portion ---
    print(f"\nTime-series {n_splits}-fold CV on TRAIN (80%)")
    for kname, kwargs in configs.items():
        try:
            mean_m, std_m = time_series_cv_evaluate(X_train, y_train, kwargs, n_splits=n_splits)
            acc, prec, rec, f1 = mean_m
            acc_s, prec_s, rec_s, f1_s = std_m
            print(f"  {kname:6s} | Acc={acc:.4f}±{acc_s:.4f}  Prec={prec:.4f}±{prec_s:.4f}  Rec={rec:.4f}±{rec_s:.4f}  F1={f1:.4f}±{f1_s:.4f}")
            results[kname] = {'cv_acc': acc, 'cv_prec': prec, 'cv_rec': rec, 'cv_f1': f1}
        except TypeError as e:
            print(f"  {kname:6s} | skipped (SVM init args mismatch): {e}")
        except Exception as e:
            print(f"  {kname:6s} | CV failed: {e}")

    # --- Final fit on full training set, evaluate on test set ---
    print("\nFinal evaluation on TEST (trained on full 80%)")
    scaler_full = StandardScaler()
    X_tr_full = scaler_full.fit_transform(np.asarray(X_train, dtype=float))
    X_te = scaler_full.transform(np.asarray(X_test, dtype=float))

    for kname, kwargs in configs.items():
        try:
            print(f"\n{kname.upper()}:")
            svm = SVM(**kwargs)
            svm.fit(X_tr_full, y_train)
            preds = svm.predict(X_te)
            acc, prec, rec, f1 = calc_metrics(y_test, preds)
            print(f"  Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
            # keep backward-compatible keys used by main.py
            results.setdefault(kname, {})
            results[kname].update({'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1})
        except TypeError as e:
            print(f"  {kname} | skipped (SVM init args mismatch): {e}")
        except Exception as e:
            print(f"  {kname} | Test eval failed: {e}")

    return results
