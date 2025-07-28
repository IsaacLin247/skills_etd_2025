
import warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.signal import welch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline   import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble  import RandomForestClassifier
from sklearn.metrics   import accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

df = pd.read_csv('data/ArmMovementDetection_Dataset.csv')

df['diff_TP']  = df['TP9'] - df['TP10']
df['diff_AF']  = df['AF7'] - df['AF8']
df['mean_all'] = df[['TP9','AF7','AF8','TP10']].mean(axis=1)
df['std_all']  = df[['TP9','AF7','AF8','TP10']].std(axis=1)

fs   = 256          
win  = 32 
half = win // 2

for ch in ['TP9','AF7','AF8','TP10']:
    mu, beta = [], []
    sig = df[ch].values
    for i in range(len(sig)):
        left  = max(0, i-half)
        right = min(len(sig), i+half)
        seg   = sig[left:right]
        f, P  = welch(seg, fs=fs, nperseg=len(seg))
        mu.append(np.trapz(P[(f>=8)&(f<13)],  f[(f>=8)&(f<13)]))
        beta.append(np.trapz(P[(f>=13)&(f<30)], f[(f>=13)&(f<30)]))
    df[f'{ch}_mu']   = mu
    df[f'{ch}_beta'] = beta

feat_cols = [c for c in df.columns if c != 'label']
X, y = df[feat_cols].values, df['label'].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('rf',   RandomForestClassifier(class_weight='balanced', random_state=42))
])
grid = GridSearchCV(
    pipe,
    param_grid={
        'rf__n_estimators':[200,400,800],
        'rf__max_depth':[None,10,20],
        'rf__min_samples_leaf':[1,3]
    },
    cv=3, scoring='accuracy', n_jobs=-1
)
grid.fit(X_tr, y_tr)

best = grid.best_estimator_
print("Best params:", grid.best_params_)

y_pred = best.predict(X_te)
acc = accuracy_score(y_te, y_pred)
print(f"Test accuracy: {acc*100:.1f}%  (n={len(y_te)})")

cm = confusion_matrix(y_te, y_pred)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=[0,1], yticklabels=[0,1],
            cmap='Blues', cbar=False)
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title('RF Confusion Matrix')
plt.tight_layout()
plt.show()