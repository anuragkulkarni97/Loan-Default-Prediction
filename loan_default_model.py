"""
Loan Default Probability & Expected Loss Model
JPMorgan Chase – Quantitative Research Virtual Experience
-------------------------------------------------------
Models trained:  Logistic Regression | Random Forest | Gradient Boosting
Final output:    expected_loss(loan_properties) → $ expected loss
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
#  1.  LOAD & EXPLORE DATA
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv('/mnt/user-data/uploads/Task_3_and_4_Loan_Data.csv')

print("═"*60)
print("  DATASET OVERVIEW")
print("═"*60)
print(f"  Rows: {len(df):,}   |   Columns: {df.shape[1]}")
print(f"  Default rate: {df['default'].mean()*100:.1f}%")
print(f"\n{df.describe().round(2).to_string()}")
print(f"\n  Missing values:\n{df.isnull().sum().to_string()}")

FEATURES = ['credit_lines_outstanding', 'loan_amt_outstanding',
            'total_debt_outstanding', 'income',
            'years_employed', 'fico_score']
TARGET   = 'default'

X = df[FEATURES].values
y = df[TARGET].values

# ══════════════════════════════════════════════════════════════════════════════
#  2.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
df['debt_to_income']    = df['total_debt_outstanding'] / (df['income'] + 1)
df['loan_to_income']    = df['loan_amt_outstanding']   / (df['income'] + 1)
df['debt_service_ratio']= (df['loan_amt_outstanding']  / (df['income'] / 12 + 1))

FEATURES_ENG = FEATURES + ['debt_to_income', 'loan_to_income', 'debt_service_ratio']
X_eng = df[FEATURES_ENG].values

X_train, X_test, y_train, y_test = train_test_split(
    X_eng, y, test_size=0.2, random_state=42, stratify=y)

# ══════════════════════════════════════════════════════════════════════════════
#  3.  MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    RandomForestClassifier(n_estimators=200, max_depth=6,
                                          min_samples_leaf=10, random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                              learning_rate=0.05, random_state=42))
    ]),
}

results = {}
print("\n" + "═"*60)
print("  MODEL TRAINING & CROSS-VALIDATION (5-fold AUC-ROC)")
print("═"*60)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_eng, y, cv=StratifiedKFold(5),
                                scoring='roc_auc', n_jobs=-1)
    model.fit(X_train, y_train)
    y_prob  = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_prob)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    # Optimal threshold (Youden's J)
    j_scores = tpr - fpr
    best_thr = thresholds[np.argmax(j_scores)]
    y_pred   = (y_prob >= best_thr).astype(int)

    results[name] = {
        'model':     model,
        'y_prob':    y_prob,
        'auc':       auc,
        'cv_auc':    cv_scores.mean(),
        'cv_std':    cv_scores.std(),
        'fpr':       fpr,
        'tpr':       tpr,
        'threshold': best_thr,
        'y_pred':    y_pred,
    }
    print(f"\n  {name}")
    print(f"    CV AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Test AUC: {auc:.4f}   |   Threshold: {best_thr:.3f}")
    report = classification_report(y_test, y_pred, target_names=['No Default','Default'])
    print('\n'.join('    ' + line for line in report.splitlines()))

# Best model = highest test AUC
best_name  = max(results, key=lambda k: results[k]['auc'])
best_model = results[best_name]['model']
print(f"\n  ✅  Best model selected: {best_name}  (AUC={results[best_name]['auc']:.4f})")

# ══════════════════════════════════════════════════════════════════════════════
#  4.  EXPECTED LOSS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
RECOVERY_RATE = 0.10   # 10% recovery assumed

def predict_pd(loan: dict, model=best_model, feature_names=FEATURES_ENG) -> float:
    """
    Predict the Probability of Default (PD) for a single loan.

    Parameters
    ----------
    loan : dict with keys matching FEATURES_ENG columns
    Returns PD in [0, 1].
    """
    # Build raw feature row
    row = {k: loan.get(k, 0.0) for k in FEATURES}
    # Engineered features
    row['debt_to_income']    = row['total_debt_outstanding'] / (row['income'] + 1)
    row['loan_to_income']    = row['loan_amt_outstanding']   / (row['income'] + 1)
    row['debt_service_ratio']= row['loan_amt_outstanding']   / (row['income'] / 12 + 1)
    X_row = np.array([[row[f] for f in feature_names]])
    return float(model.predict_proba(X_row)[0, 1])


def expected_loss(loan: dict,
                  recovery_rate: float = RECOVERY_RATE,
                  verbose: bool = True) -> dict:
    """
    Compute Expected Loss for a loan.

    Expected Loss (EL) = PD × LGD × EAD
      PD  = Probability of Default       (from ML model)
      LGD = Loss Given Default           = 1 − recovery_rate
      EAD = Exposure at Default          = loan_amt_outstanding

    Parameters
    ----------
    loan          : dict of borrower/loan characteristics
    recovery_rate : fraction recovered if borrower defaults (default 10%)

    Returns dict with PD, LGD, EAD, and EL.
    """
    pd_val  = predict_pd(loan)
    ead     = loan.get('loan_amt_outstanding', 0.0)
    lgd     = 1.0 - recovery_rate
    el      = pd_val * lgd * ead

    result = {
        'PD':              pd_val,
        'EAD':             ead,
        'LGD':             lgd,
        'recovery_rate':   recovery_rate,
        'expected_loss':   el,
    }

    if verbose:
        print(f"\n  {'─'*45}")
        print(f"  EXPECTED LOSS REPORT")
        print(f"  {'─'*45}")
        print(f"  Loan Amount (EAD)    : ${ead:>12,.2f}")
        print(f"  Probability Default  : {pd_val:>12.4f}  ({pd_val*100:.2f}%)")
        print(f"  Loss Given Default   : {lgd:>12.2f}  (recovery={recovery_rate*100:.0f}%)")
        print(f"  {'─'*45}")
        print(f"  Expected Loss (EL)   : ${el:>12,.2f}")
        print(f"  {'─'*45}\n")

    return result

# ══════════════════════════════════════════════════════════════════════════════
#  5.  TEST CASES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("  EXPECTED LOSS — SAMPLE LOAN TESTS")
print("█"*60)

test_loans = [
    {   # Low-risk: high income, high FICO, low debt
        'label':                  'Low-risk borrower',
        'credit_lines_outstanding': 1,
        'loan_amt_outstanding':   5000,
        'total_debt_outstanding': 3000,
        'income':                 90000,
        'years_employed':         8,
        'fico_score':             750,
    },
    {   # Medium-risk
        'label':                  'Medium-risk borrower',
        'credit_lines_outstanding': 3,
        'loan_amt_outstanding':   15000,
        'total_debt_outstanding': 25000,
        'income':                 45000,
        'years_employed':         3,
        'fico_score':             620,
    },
    {   # High-risk: low income, low FICO, high debt
        'label':                  'High-risk borrower',
        'credit_lines_outstanding': 6,
        'loan_amt_outstanding':   25000,
        'total_debt_outstanding': 60000,
        'income':                 28000,
        'years_employed':         1,
        'fico_score':             540,
    },
]

test_results = []
for loan in test_loans:
    print(f"\n  ── {loan['label']} ──")
    r = expected_loss({k: v for k, v in loan.items() if k != 'label'})
    r['label'] = loan['label']
    test_results.append(r)

# Portfolio expected loss
print("\n" + "═"*60)
print("  PORTFOLIO EXPECTED LOSS (full test set)")
print("═"*60)
portfolio_el = []
for _, row in df.iterrows():
    pd_v = predict_pd(row.to_dict())
    el   = pd_v * (1 - RECOVERY_RATE) * row['loan_amt_outstanding']
    portfolio_el.append(el)
df['expected_loss'] = portfolio_el
print(f"  Total loans:           {len(df):,}")
print(f"  Total EAD:             ${df['loan_amt_outstanding'].sum():,.0f}")
print(f"  Total Expected Loss:   ${df['expected_loss'].sum():,.0f}")
print(f"  EL as % of EAD:        {df['expected_loss'].sum()/df['loan_amt_outstanding'].sum()*100:.2f}%")
print(f"  Avg PD per borrower:   {df['expected_loss'].sum()/(df['loan_amt_outstanding'].sum()*(1-RECOVERY_RATE))*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
#  6.  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 20))
fig.patch.set_facecolor('#F7F9FC')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                        left=0.07, right=0.97, top=0.94, bottom=0.04)
BLUE, RED, GREEN, ORANGE, PURPLE = '#1F77B4','#D62728','#2CA02C','#FF7F0E','#9467BD'

fig.text(0.5, 0.97, 'Loan Default Prediction — Model Analysis & Expected Loss',
         ha='center', fontsize=14, fontweight='bold')

# ── ROC curves ───────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#F7F9FC')
cols = [BLUE, GREEN, ORANGE]
for (name, res), col in zip(results.items(), cols):
    ax1.plot(res['fpr'], res['tpr'], lw=2, color=col,
             label=f"{name} (AUC={res['auc']:.3f})")
ax1.plot([0,1],[0,1], 'k--', lw=1)
ax1.set_title('ROC Curves — All Models', fontsize=10, fontweight='bold')
ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
ax1.legend(fontsize=7.5); ax1.grid(True, alpha=0.25)

# ── AUC bar comparison ────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#F7F9FC')
names = list(results.keys())
aucs  = [results[n]['auc'] for n in names]
cv_aucs = [results[n]['cv_auc'] for n in names]
cv_stds = [results[n]['cv_std'] for n in names]
x = np.arange(len(names))
ax2.bar(x - 0.2, aucs,    0.35, label='Test AUC',  color=cols, alpha=0.85)
ax2.bar(x + 0.2, cv_aucs, 0.35, label='CV AUC',    color=cols, alpha=0.45,
        yerr=cv_stds, capsize=4)
ax2.set_xticks(x); ax2.set_xticklabels([n.replace(' ','\n') for n in names], fontsize=8)
ax2.set_ylim(0.85, 1.0); ax2.set_title('AUC Comparison', fontsize=10, fontweight='bold')
ax2.set_ylabel('AUC-ROC'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25, axis='y')

# ── Confusion matrix (best model) ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
cm  = confusion_matrix(y_test, results[best_name]['y_pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['No Default', 'Default'])
disp.plot(ax=ax3, colorbar=False, cmap='Blues')
ax3.set_title(f'Confusion Matrix\n{best_name}', fontsize=10, fontweight='bold')

# ── Feature importance ────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor('#F7F9FC')
clf = best_model.named_steps['clf']
if hasattr(clf, 'feature_importances_'):
    imps = clf.feature_importances_
else:
    # Logistic regression: use absolute coefficients
    imps = np.abs(clf.coef_[0])
imps_norm = imps / imps.sum()
sorted_idx = np.argsort(imps_norm)
ax4.barh([FEATURES_ENG[i] for i in sorted_idx], imps_norm[sorted_idx],
         color=BLUE, alpha=0.8)
ax4.set_title(f'Feature Importance\n({best_name})', fontsize=10, fontweight='bold')
ax4.set_xlabel('Relative Importance'); ax4.grid(True, alpha=0.25, axis='x')

# ── PD distribution by default status ────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor('#F7F9FC')
pd_all = best_model.predict_proba(
    np.vstack([best_model.named_steps['scaler'].transform(X_eng)
               if False else X_eng])
)[:, 1]
pd_all = best_model.predict_proba(X_eng)[:, 1]
ax5.hist(pd_all[y == 0], bins=40, alpha=0.6, color=GREEN,  density=True, label='No Default')
ax5.hist(pd_all[y == 1], bins=40, alpha=0.6, color=RED,    density=True, label='Default')
ax5.set_title('Predicted PD Distribution\nby Actual Outcome', fontsize=10, fontweight='bold')
ax5.set_xlabel('Predicted PD'); ax5.set_ylabel('Density')
ax5.legend(fontsize=9); ax5.grid(True, alpha=0.25)

# ── Expected loss distribution ────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor('#F7F9FC')
ax6.hist(df['expected_loss'], bins=50, color=PURPLE, alpha=0.8, edgecolor='white')
ax6.axvline(df['expected_loss'].mean(), color=RED, lw=2, ls='--',
            label=f"Mean EL = ${df['expected_loss'].mean():,.0f}")
ax6.set_title('Expected Loss Distribution\n(Full Portfolio)', fontsize=10, fontweight='bold')
ax6.set_xlabel('Expected Loss ($)'); ax6.set_ylabel('Count')
ax6.legend(fontsize=9); ax6.grid(True, alpha=0.25)

# ── FICO score vs PD ─────────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
ax7.set_facecolor('#F7F9FC')
sc = ax7.scatter(df['fico_score'], pd_all, c=df['income'], cmap='viridis',
                 alpha=0.3, s=8)
plt.colorbar(sc, ax=ax7, label='Income ($)')
ax7.set_title('FICO Score vs Predicted PD\n(coloured by income)', fontsize=10, fontweight='bold')
ax7.set_xlabel('FICO Score'); ax7.set_ylabel('Predicted PD')
ax7.grid(True, alpha=0.25)

# ── Debt-to-income vs PD ──────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
ax8.set_facecolor('#F7F9FC')
sc2 = ax8.scatter(df['debt_to_income'], pd_all, c=df['fico_score'], cmap='RdYlGn',
                  alpha=0.3, s=8, vmin=500, vmax=800)
plt.colorbar(sc2, ax=ax8, label='FICO Score')
ax8.set_xlim(0, 3)
ax8.set_title('Debt-to-Income vs Predicted PD\n(coloured by FICO)', fontsize=10, fontweight='bold')
ax8.set_xlabel('Debt-to-Income Ratio'); ax8.set_ylabel('Predicted PD')
ax8.grid(True, alpha=0.25)

# ── Sample loan EL comparison ─────────────────────────────────────────────────
ax9 = fig.add_subplot(gs[2, 2])
ax9.set_facecolor('#F7F9FC')
labels = [r['label'] for r in test_results]
pds    = [r['PD'] for r in test_results]
els    = [r['expected_loss'] for r in test_results]
risk_colors = [GREEN, ORANGE, RED]
bars = ax9.bar(labels, els, color=risk_colors, edgecolor='white', width=0.5)
for bar, pd_v, el in zip(bars, pds, els):
    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             f'PD={pd_v*100:.1f}%\n${el:,.0f}', ha='center', va='bottom', fontsize=8)
ax9.set_title('Expected Loss by Risk Profile\n(Sample Test Loans)', fontsize=10, fontweight='bold')
ax9.set_ylabel('Expected Loss ($)')
ax9.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=8)
ax9.grid(True, alpha=0.25, axis='y')

plt.savefig('/mnt/user-data/outputs/loan_default_model.png', dpi=150,
            bbox_inches='tight', facecolor='#F7F9FC')
plt.close()
print("\n  Chart saved ✓")
