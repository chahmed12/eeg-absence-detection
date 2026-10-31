# Résultats P0 — protocole causal et strictement imbriqué

> [!IMPORTANT]
> Résultats finaux après régénération complète des caches `v3_causal`, filtrage amont causal (`sosfilt`), post-traitement causal et validation interne stricte. Le pli de test n'est jamais utilisé pour le seuil, l'early stopping ou la sélection de modèle.

## P0-1) LightGBM — EXP-00 (260 features, 8 canaux)
- Seuil par pli : 0,30 / 0,50 / 0,70 / 0,90 / 0,75 — moyenne **0,63 ± 0,23**
- F1 fenêtre : **0,705 poolé**, **0,700 ± 0,121** (moyenne des plis)
- Précision / rappel poolés : **0,706 / 0,703** ; PR-AUC poolé : **0,652**
- Événements : **77/82 crises détectées (93,9 %)**, 40 FP, 5 FN, 9,23 h
- FA/h : **4,33 hors-ligne**, **4,55 causal**
- Latence causale pondérée par événement : **4,46 s** (77 événements)

## P0-2) Deep Learning — validation interne 25 % des patients
| Expérience | Modèle | F1 fenêtre | Détection | FA/h | Latence causale |
|---|---|---|:---:|:---:|:---:|
| EXP-01 | CNN 1D temporel | 0,488 ± 0,119 | 96,3 % | 19,50 | 3,33 ± 3,81 s |
| EXP-02 | CNN spatio-temporel | 0,527 ± 0,140 | 93,9 % | 19,93 | 2,96 ± 3,99 s |
| EXP-03 | EEGNet, seuil 0,5 | 0,522 ± 0,100 | 93,9 % | 11,48 | 1,19 ± 8,85 s |
| EXP-04 | EEGNet, seuil calibré | **0,643 ± 0,077** | **93,9 % (77/82)** | **8,88** | **3,86 s** |

- EXP-04 : τ* = **0,77 ± 0,15**, sélectionné sur la validation interne de chaque pli. Moyenne des plis : 92,4 % ± 11,1 % et 8,29 ± 6,05 FA/h. Le balayage OOF global (F1 0,673 à τ=0,94) est optimiste et n'est pas le résultat principal.
- EXP-03 : latence élevée en écart-type à cause d'un outlier à 26 s (`230515B-G`).
- Les FA/h EXP-01 à EXP-04 utilisent le post-traitement causal et 9,23 h (22 enregistrements).

**Décision finale** : LightGBM EXP-00 est le modèle de référence retenu pour la détection ; EEGNet EXP-04 est l'extension Deep Learning retenue pour le monitoring embarqué. EEGNet est plus compact et plus rapide de bout en bout, mais LightGBM reste plus performant et émet moins de fausses alarmes.

---

# Archive historique (avant P0)

> [!WARNING]
> Document historique (12/10/2026). Les métriques FA/h de la section 4 utilisent 19 patients annotés / 8,06 h (RandomForest LOPO) et ne sont pas comparables aux résultats GroupKFold sur 22 enregistrements / 9,23 h. Les valeurs 86,6 % / 3,35 FA/h ne sont pas reproduites par les artefacts actuels. Voir `report/EXPERIMENTS_LOG.md` pour les résultats P0.

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
