## 1) Données
- Patients annotés : 19
- Crises : 82
- Fenêtres : 33 206 × 740 features (743 colonnes), 3,54 % de crises

## 2) GroupKFold 5
- LightGBM: F1=0.659±0.121, PR-AUC=0.690±0.118
- XGBoost: F1=0.620±0.130, PR-AUC=0.682±0.104
- RandomForest: F1=0.402±0.325, PR-AUC=0.676±0.127

## 3) Améliorations
- MI_top250: F1=0.740±0.073, PR-AUC=0.753, n=250
- EEG_commun_8: F1=0.689±0.122, PR-AUC=0.704, n=260
- SMOTE: F1=0.654±0.129, PR-AUC=0.671, n=740

## 4) Métriques événementielles (LOPO RF, seuil 0.5)
- Moyenne tous patients (seuil 0.5) : taux détection=52.5%, FA/h=1.17, latence moy=3.1s
- Matrice de confusion événement (seuil 0.5) : TP=51, FP=10, FN=31 → taux=62.2%, FA/h=1.24 (base 19 patients annotés, 8.06 h)

## 4 ter) Matrices de confusion (fenêtres)
- Seuil 0.5 : TP=423, FP=110, FN=751, TN=27710 (F1=0.496)
- Seuil 0.18 (opt. F1) : TP=825, FP=313, FN=349, TN=27507 → F1=0.714
- Notebook de soumission : `eeg_project_soumission.ipynb` (section 7)

## 5) Sensibilité fenêtre/recouvrement
- 4.0s/50% : F1=0.663±0.188, PR-AUC=0.678
- 2.0s/50% : F1=0.635±0.149, PR-AUC=0.675

## 4b) Courbe seuils (post-traitement, cf. étape 5b)
- seuil 0.1: détection=93.9%, FA/h=10.67, latence=0.1s
- seuil 0.2: détection=86.6%, FA/h=3.35, latence=0.9s
- seuil 0.3: détection=74.4%, FA/h=1.99, latence=2.3s
- seuil 0.5: détection=62.2%, FA/h=1.24, latence=3.1s
- seuil 0.7: détection=40.2%, FA/h=0.74, latence=3.0s
