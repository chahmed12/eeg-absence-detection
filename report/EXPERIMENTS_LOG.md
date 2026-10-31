# 📓 Cahier de Laboratoire Expérimental — Deep Learning EEG

Ce document retrace la conception **progressive et empirique** de l'architecture Deep Learning pour la détection des crises d'absence sur signaux EEG multi-canaux.

Chaque étape suit rigoureusement la démarche scientifique :
$$\text{Observation} \longrightarrow \text{Hypothèse testable} \longrightarrow \text{Expérience ciblée} \longrightarrow \text{Mesure objective} \longrightarrow \text{Décision d'architecture}$$

> [!IMPORTANT]
> **Statut P0 — résultats principaux** :
> - Toutes les expériences (**EXP-00 à EXP-04**) utilisent un **GroupKFold à 5 plis par patient** (3 à 5 enregistrements par pli de test), sans recouvrement de patient entre train et test.
> - **La sélection est strictement imbriquée** : pour chaque pli externe, un split interne par patient sert au choix du seuil, de l'early stopping et du checkpoint ; le modèle final est réentraîné sur tout le train externe, puis évalué une seule fois sur le test externe.
> - Le filtrage amont utilise `sosfilt` (causal, état initialisé) et le post-traitement causal ne fusionne plus rétroactivement les trous.
> - Les résultats ci-dessous remplacent les chiffres exploratoires antérieurs. La ligne **Random Forest (LOPO)** reste archivée séparément : protocole *Leave-One-Patient-Out* à 19 plis et 8,06 h annotées, non comparable au GroupKFold 22 enregistrements / 9,23 h.

---

## 📊 Étape 0 — La Référence (Baseline Machine Learning Classique)

Avant toute modélisation neuronale, les performances du pipeline existant (features manuelles + LightGBM / Random Forest) sont documentées pour servir de point de comparaison formel.

### 1. Données du problème et Partitionnement Patient → Pli (Identique pour tous les modèles)
- **Cohorte** : 19 patients annotés (82 crises) sur 22 enregistrements, dont 3 enregistrements sans crise. Les 22 identifiants uniques sont utilisés dans les plis ; les 3 enregistrements sans crise servent au calcul des FA/h mais pas au taux de détection.
- **Nombre de crises réelles** : 82 crises cliniques d'absence documentées.
- **Fenêtrage** : 2,0 secondes à 256 Hz (512 pas de temps), pas de 1,0 s (50 % overlap) = 33 206 fenêtres.
- **Déséquilibre extrême** : **3,54 %** de fenêtres de crise (96,46 % de signal hors-crise).

| Pli de test | Nombre d'enregistrements | Identifiants des patients testés (GroupKFold 5 plis) |
|:---:|:---:|---|
| **Fold 1** | 3 | `230105B-H`, `230406B-A`, `230515B-G` |
| **Fold 2** | 4 | `210504B-C`, `210914B-A`, `211027B-C`, `220106A-A` |
| **Fold 3** | 5 | `191113A-D`, `210208B-G`, `210427B-C`, `210928B-B`, `230519A-E` |
| **Fold 4** | 5 | `200625A-F`, `210204B-C`, `211104B-D`, `230210B-D`, `230224B-B` |
| **Fold 5** | 5 | `190304A-E`, `210406A-F`, `220121B-D`, `230102B-F`, `230718A-A` |

### 2. Performances établies (Baseline Machine Learning)

| Modèle / Configuration | Protocole d'évaluation | Features | F1-Score (fenêtre) | PR-AUC | Taux détection (événement) | Fausses Alarmes / h | Latence causale estimée |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM EXP-00 (8 canaux, P0)** | **GroupKFold 5 plis, sélection interne stricte** | 260 | **0,705 poolé ; 0,700 ± 0,121** | **0,652 poolé** | **93,9 %** (77/82 crises) | **4,33 hors-ligne / 4,55 causal** | **4,46 s** |
| LightGBM historique (archivé) | GroupKFold 5 plis, seuil OOF non strict | 260 | 0,677 ± 0,091 | 0,673 ± 0,167 | 89,0 % ± 10,6 % | 4,44 ± 3,79 FA/h | 6,83 ± 2,88 s |
| LightGBM balayage indicatif (archivé) | GroupKFold 5 plis | 260 | 0,689 ± 0,122 | 0,704 | ~86,6 % (seuil 0,20) | 3,35 FA/h | 0,9 s (non causal) |
| Random Forest (archivé) | LOPO (19 plis, 8,06 h) | 740 | 0,496 | 0,676 | 62,2 % | 1,24 FA/h | 3,1 s |

> [!NOTE]
> **Traçabilité des chiffres LightGBM** :
> - **Ligne P0** : 77 crises sur 82 détectées, 40 FP et 5 FN sur 9,23 h ; FA/h 4,33 hors-ligne et 4,55 avec post-traitement causal. F1 et PR-AUC sont poolés sur les fenêtres ; F1 ± écart-type est la moyenne des plis.
> - **Chiffres historiques (86,6 % / 3,35 FA/h)** : seuil 0,20 repéré rétrospectivement sur la courbe OOF. Ils ne sont pas reproduits par les artefacts actuels et ne constituent plus l'estimation principale. La ligne 62,2 % / 1,24 FA/h est un RandomForest LOPO.
> - **Latence** : la valeur P0 est **4,46 s**, pondérée par événement. Les anciennes valeurs 6,83 s et 5,26 s sont des moyennes de moyennes par patient/pli et restent archivées.

### 3. Faiblesses et anomalies observées sur la baseline
1. **Faux négatifs résiduels** :
   - Après sélection stricte du seuil, 5 crises sur 82 restent manquées (77/82 détectées). Les faux négatifs se concentrent sur les crises très brèves et le pli 3 (71,4 % de détection).
2. **Latence causale** :
   - La latence événementielle P0 est **4,46 s** avec le filtrage amont causal et le post-traitement causal. Les anciennes latences 0,9 s / 3,1 s / 6,83 s provenaient de modes non causaux ou d'agrégations différentes.
3. **Fausses alarmes** :
   - La baseline P0 émet **4,55 FA/h** en mode causal (4,33 hors-ligne) sur 22 enregistrements / 9,23 h. C'est mieux que les modèles DL, mais encore insuffisant pour un usage clinique non supervisé.

---

## 🗺️ Registre de Suivi des Expériences

| ID | Hypothèse testée | Modification d'architecture | Statut | Résultat principal |
|---|---|---|:---:|---|
| **EXP-00** | Référence Machine Learning | Baseline LightGBM (260 feats, GroupKFold) | ✅ **Validé P0** | F1=0,705 poolé (0,700 ± 0,121), Détection=93,9 %, FA/h=4,55 causal, Latence=4,46 s |
| **EXP-01** | L'apprentissage sur signal brut capture les pointes-ondes | CNN 1D Spatio-Temporel Dense (non factorisé) | ✅ **P0** | F1=0,488 ± 0,119, Rappel=0,850, Détection=96,3 %, FA/h=19,50, Latence=3,33 s |
| **EXP-02** | La factorisation spatiale réduit les fausses alarmes | Séparation temporelle 1D + convolution spatiale Depthwise | ✅ **P0** | F1=0,527 ± 0,140, Précision=0,413, Détection=93,9 %, FA/h=19,93, Latence=2,96 s |
| **EXP-03** | La séparation spatio-temporelle complète (EEGNet) optimise la compacité | Architecture EEGNet complète (1 489 paramètres) | ✅ **P0** | F1=0,522 ± 0,100, Précision=0,391, Détection=93,9 %, FA/h=11,48, Latence=1,19 s |
| **EXP-04** | La calibration du seuil sur une grille étendue optimise le point clinique | Calibration sur grille étendue, validation interne par patient | ✅ **P0** | τ*=0,77 ± 0,15, F1=0,643 ± 0,077, Détection=93,9 % (77/82), FA/h=8,88 causal (moyenne plis 8,29 ± 6,05), Latence=3,86 s |

---

## 📝 Fiches détaillées des expériences

### 🧪 EXP-01 : CNN 1D Spatio-Temporel Dense Non-Factorisé
* **Contexte** : Validation de l'Hypothèse 1.
* **Question / Hypothèse scientifique** :
  La forme d'onde pointe-onde caractéristique (décharge à 3 Hz, soit 250–330 ms par cycle) permet-elle à un réseau convolutif 1D d'apprendre directement depuis le signal brut sans features manuelles, réduisant ainsi les faux négatifs observés avec LightGBM ?
* **Architecture & Hyperparamètres modifiés** :
  - **Entrée** : `(B, 8, 512)` (8 canaux communs, 2 secondes à 256 Hz).
  - **Étage 1** : `Conv1d(in_channels=8, out_channels=16, kernel_size=63, padding="same")`.
    > [!NOTE]
    > **Précision architecturale** : Cette couche combine linéairement les 8 canaux EEG simultanément avec l'axe temporel (tenseur de poids $16 \times 8 \times 63$). Elle n'est donc pas « univariée temporelle pure », mais constitue une convolution spatio-temporelle dense et non factorisée.
  - **Étage 2** : `Conv1d(16, 32, kernel_size=31, padding="same")` $\to$ BatchNorm $\to$ ELU $\to$ AdaptiveAvgPool1D($1$).
  - **Classifieur** : Dropout(0.3) $\to$ Linear(32, 1) $\to$ **logit brut** (pas de Sigmoid en entraînement).
  - **Loss** : `BCEWithLogitsLoss(pos_weight=~27.2)` — attend des logits bruts, `pos_weight` calculé strictement sur le train de chaque pli. Sigmoid utilisé uniquement en inférence via `torch.sigmoid(logits)`.
  - **Optimiseur** : Adam (`lr=1e-3`, `weight_decay=1e-4`, scheduler `ReduceLROnPlateau`).
  - **Entraînement** : 10 époques par fold sur CPU.
* **Protocole d'évaluation P0** :
  - **GroupKFold 5 plis par patient** (3 à 5 patients par pli de test).
  - Pour chaque pli externe, **25 % des patients annotés du train externe** forment la validation interne (`train_fold_nested`). L'early stopping choisit le meilleur epoch sur cette validation ; le modèle est ensuite réentraîné sur tout le train externe.
  - Normalisation `FoldScaler` calculée strictement sur le train de chaque fold.
  - Filtrage amont causal (`sosfilt`) et post-traitement causal ; test externe évalué une seule fois.
* **Résultats obtenus** :
  - **F1-score fenêtre** : **$0,488 \pm 0,119$**.
  - **Rappel fenêtre** : **$0,850$**.
  - **PR-AUC** : $0,646$.
  - **Taux de détection clinique (événements)** : **$96,3\ \%$** (79/82).
  - **Latence causale pondérée par événement** : **$3,33\text{ s}$** (écart-type $3,81\text{ s}$).
  - **Fausses alarmes par heure** : **$19,50\ \text{FA/h}$** (180 FP sur 9,23 h).
* **Analyse des erreurs** :
  - **Validation de l'apprentissage sur signal brut** : Le rappel élevé (0,850) confirme que le réseau apprend directement la dynamique des pointes-ondes sans calcul de features manuelles.
  - **Cause des faux positifs** : En combinant tous les canaux de manière dense dès la première couche sans séparation structurelle, le réseau s'active sur des transitoires locaux et produit beaucoup de fausses alarmes.
* **Décision argumentée** :
  ➡️ **Tester la factorisation spatio-temporelle (EXP-02)** en séparant le filtrage temporel par canal et le filtrage spatial inter-électrodes.

---

### 🧪 EXP-02 : CNN Spatio-Temporel Factorisé (Convolution Spatiale Depthwise)
* **Contexte** : Validation de l'Hypothèse 2.
* **Question / Hypothèse scientifique** :
  La factorisation explicite de l'opération en deux étages distincts — un filtrage temporel 1D par canal, suivi d'une convolution spatiale Depthwise combinant les 8 électrodes — permet-elle d'améliorer la précision par rapport à la convolution dense non factorisée d'EXP-01 ?
* **Architecture & Hyperparamètres modifiés** :
  - **Entrée** : `(B, 1, 8, 512)`.
  - **Étage 1 (Temporel pur)** : Conv2D `(1, 64)`, 8 filtres $\to$ BatchNorm2D (isole les motifs fréquentiels le long du temps, sans mélanger les électrodes).
  - **Étage 2 (Spatial Depthwise)** : Conv2D `(8, 1)`, groups=8 ($D=2 \to 16$ filtres) $\to$ BatchNorm2D $\to$ ELU $\to$ AvgPool2D(1, 4) $\to$ Dropout(0.3) (combine les 8 électrodes scalp).
  - **Étage 3 (Combinaison)** : Conv2D `(1, 32)` $\to$ BatchNorm2D $\to$ ELU $\to$ AdaptiveAvgPool2D(1, 1) $\to$ Dropout(0.3) $\to$ Linear(32, 1) logit brut.
  - **Loss & Optimizer** : Identiques à EXP-01 (`pos_weight=~27.2` calculé sur train uniquement, Adam `1e-3`, 10 époques).
* **Protocole d'évaluation P0** :
  - Strictement identique à EXP-01 : GroupKFold 5 plis, validation interne par patient, FoldScaler train-only, filtrage et post-traitement causaux.
* **Résultats obtenus** :
  - **F1-score fenêtre** : **$0,527 \pm 0,140$** ($\mathbf{+0,039}$ par rapport à EXP-01).
  - **Précision fenêtre** : **$0,413$**.
  - **Rappel fenêtre** : $0,833$.
  - **PR-AUC** : $0,679$.
  - **Taux de détection clinique** : **$93,9\ \%$** (77/82).
  - **Latence causale pondérée par événement** : **$2,96\text{ s}$** (écart-type $3,99\text{ s}$).
  - **Fausses alarmes par heure** : **$19,93\ \text{FA/h}$** (184 FP sur 9,23 h).
* **Analyse des erreurs** :
  - **Gain de F1** : +3,9 points par rapport à EXP-01. La factorisation apporte une amélioration modérée mais reste loin de la baseline LightGBM.
  - **Nuance d'interprétation** : Ce gain confirme l'utilité de la séparation des filtres fréquentiels et topographiques, mais ne prouve pas formellement que le réseau rejette spécifiquement les artefacts oculaires sans étude d'ablation.
  - **Persistance des FA à seuil 0,5** : Le taux de $19,93\text{ FA/h}$ reste élevé car le modèle entraîné sous `pos_weight=27.2` génère des scores décalés vers le haut.
* **Décision argumentée** :
  ➡️ **Valider l'architecture spatio-temporelle factorisée**. Passer à **EXP-03** avec **EEGNet** (Lawhern et al., 2018) pour ajouter des convolutions séparables en profondeur et un budget de paramètres minimal.

---

### 🧪 EXP-03 : Architecture EEGNet Complète (Convolutions Séparables & Régularisation)
* **Contexte** : Validation de l'Hypothèse 3.
* **Question / Hypothèse scientifique** :
  EEGNet — combinant filtrage temporel ($F_1=8$ filtres), filtrage spatial depthwise ($D=2 \to 16$ filtres), et convolutions séparables en profondeur avec pooling régulier — permet-elle de réduire la sur-paramétrisation tout en consolidant les motifs spatio-temporels caractéristiques ?
* **Architecture & Hyperparamètres vérifiés dans le code** :
  - **Entrée** : `(B, 1, 8, 512)` (8 canaux, 512 pas de temps à 256 Hz).
  - **Bloc 1 (Temporel)** : Conv2D `(1, 64)`, 8 filtres $\to$ BatchNorm2D (512 + 16 params).
  - **Bloc 2 (Spatial Depthwise)** : Conv2D `(8, 1)`, groups=8 ($D=2 \to 16$ canaux) $\to$ BatchNorm2D $\to$ ELU $\to$ AvgPool2D(1, 4) $\to$ Dropout(0.25) (128 + 32 params).
  - **Bloc 3 (Séparable)** : Conv2D Depthwise `(1, 16)`, groups=16 (256 params) $\to$ Conv2D Pointwise `(1, 1)` (16 filtres, 256 params) $\to$ BatchNorm2D (32 params) $\to$ ELU $\to$ AvgPool2D(1, 8) $\to$ Dropout(0.25).
  - **Tête Dense** : Linear(256, 1) (257 params).
  - **Nombre total de paramètres vérifié** : **1 489 paramètres exacts** (tous entraînables).
  - **Entraînement** : 10 époques par fold sur CPU, Adam `lr=1e-3`, `weight_decay=1e-4`, `pos_weight=~27.2` calculé sur le train de chaque pli.
* **Protocole d'évaluation P0** :
  - Strictement identique : **GroupKFold 5 plis par patient**, validation interne par patient, FoldScaler train-only, filtrage et post-traitement causaux.
  - Seuil fixe $\tau = 0,5$ pour cette expérience ; la calibration est réservée à EXP-04.
* **Résultats obtenus** :
  - **F1-score fenêtre** : **$0,522 \pm 0,100$** ($-0,005$ vs EXP-02, $+0,034$ vs EXP-01).
  - **Précision fenêtre** : **$0,391$**.
  - **Rappel fenêtre** : $0,861$.
  - **PR-AUC** : $0,692$.
  - **Taux de détection clinique** : **$93,9\ \%$** (77/82).
  - **Latence causale pondérée par événement** : **$1,19\text{ s}$** (écart-type $8,85\text{ s}$).
  - **Fausses alarmes par heure** : **$11,48\ \text{FA/h}$** (106 FP sur 9,23 h).
* **Analyse des erreurs** :
  - **Compacité** : Avec 1 489 paramètres, EEGNet reste le modèle le plus compact, mais son F1 reste proche d'EXP-02.
  - **Sensibilité** : Le rappel de 0,861 est le plus élevé des trois CNN ; les faux positifs restent nombreux.
  - **Source résiduelle de FA** : Le seuil fixe 0,5 est inadapté à un modèle entraîné avec `pos_weight=27.2`. La calibration du seuil par validation interne (EXP-04) est l'étape suivante.
* **Décision argumentée** :
  ➡️ **Retenir EEGNet comme architecture candidate**. Passer à **EXP-04** pour une calibration du point de fonctionnement sur validation interne stricte.

---

### 🧪 EXP-04 : Calibration du Seuil sur Validation Interne Stricte
* **Contexte** : Évaluation P0 de l'Hypothèse 4. Les scores DL proviennent de checkpoints sélectionnés sur validation interne, sans utiliser le pli de test.
* **Question / Hypothèse scientifique** :
  L'utilisation d'une perte pondérée (`pos_weight = 27.2`) décale la fonction logistique vers le haut. Sur une grille étendue $\tau \in [0,50 ; 0,98]$, quel seuil $\tau^*$ minimise les fausses alarmes sous contrainte clinique, et quelle est la latence en mode causal ?
* **Protocole P0** :
  - Pour chaque pli externe $k$, un sous-ensemble de **25 % des patients annotés du train externe** forme la validation interne.
  - `train_fold_nested` sélectionne le meilleur epoch sur cette validation, puis réentraîne le modèle sur tout le train externe.
  - $\tau^*_k$ est sélectionné sur les prédictions de validation interne sous contrainte de détection, puis appliqué au pli de test $k$ ; le test n'intervient jamais dans la sélection.
  - Évaluation conjointe hors-ligne (métriques rétrospectives) et causale (filtrage amont `sosfilt`, post-traitement sans fusion rétroactive des trous).

#### 1. Résultats détaillés par pli de test (P0)

| Pli de test | Patients testés | Seuil $\tau^*$ | F1 fenêtre | Précision | Rappel | Détection hors-ligne | FA/h hors-ligne | Latence causale |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | 3 | **0,80** | 0,788 | 0,750 | 0,830 | **100,0 %** | **0,60 FA/h** | 3,85 s |
| **Fold 2** | 4 | **0,50** | 0,638 | 0,557 | 0,747 | 90,6 % | 15,35 FA/h | 3,66 s |
| **Fold 3** | 5 | **0,75** | 0,571 | 0,452 | 0,774 | **71,4 %** | 4,23 FA/h | 4,00 s |
| **Fold 4** | 5 | **0,96** | 0,583 | 0,513 | 0,675 | **100,0 %** | 5,80 FA/h | 4,67 s |
| **Fold 5** | 5 | **0,85** | 0,634 | 0,493 | 0,885 | **100,0 %** | 15,46 FA/h | 3,76 s |
| **Moyenne ± Écart-type** | — | **0,77 ± 0,15** | **0,643 ± 0,077** | **0,553 ± 0,104** | **0,782 ± 0,072** | **92,4 % ± 11,1 %** | **8,29 ± 6,05 FA/h** | **3,86 s** |

> [!NOTE]
> **Métriques événementielles poolées (principales)** : 77 crises détectées sur 82 (**93,9 %**), 5 FN, 82 FP, **8,88 FA/h causal** (8,45 hors-ligne), latence causale **3,86 s** sur 9,23 h. Le pli 3 concentre les 5 FN sur des crises très brèves ; les plis 1, 4 et 5 atteignent 100 %.
> Le seuil $\tau^*_k$ est sélectionné sur la validation interne du pli, jamais sur le test. Les FA/h du tableau sont hors-ligne ; les FA/h causales par pli sont dans `EXP-04_metrics.json`.

---

#### 2. Balayage indicatif étendu sur l'ensemble OOF réuni (optimiste)

> [!WARNING]
> Ce tableau est purement exploratoire et **ne remplace pas** la calibration par pli. Les métriques sont calculées sur le pool global des 33 206 fenêtres, ce qui surestime les performances. Le meilleur point OOF (τ=0,94, F1 0,673) n'est pas le résultat principal.

| Seuil $\tau$ | F1 fenêtre | Précision | Rappel | Détection | FA/h | Latence causale |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0,50 | 0,491 | 0,347 | 0,836 | 92,7 % | 10,40 | 1,19 s |
| 0,60 | 0,526 | 0,388 | 0,817 | 95,1 % | 11,16 | 2,32 s |
| 0,70 | 0,568 | 0,439 | 0,803 | 95,1 % | 11,48 | 3,65 s |
| 0,75 | 0,589 | 0,470 | 0,789 | 93,9 % | 11,38 | 3,78 s |
| 0,80 | 0,616 | 0,512 | 0,773 | 93,9 % | 9,32 | 3,94 s |
| 0,85 | 0,641 | 0,561 | 0,749 | 93,9 % | 7,04 | 4,19 s |
| 0,90 | 0,660 | 0,614 | 0,715 | 92,7 % | 5,31 | 4,53 s |
| 0,92 | 0,669 | 0,641 | 0,698 | 91,5 % | 4,01 | 4,68 s |
| **0,94** | **0,673** | **0,669** | **0,676** | **91,5 %** | **3,68** | **4,81 s** |
| 0,96 | 0,665 | 0,692 | 0,640 | 89,0 % | 3,36 | 4,74 s |
| 0,98 | 0,641 | 0,731 | 0,571 | 86,6 % | 2,93 | 5,08 s |

---

## 🏆 Synthèse Finale Comparative : Machine Learning vs Deep Learning

### Tableau A — Comparaison sous GroupKFold 5 plis (P0, sélection strictement imbriquée)

Les partitions de test sont disjointes par patient et le test externe n'intervient ni dans le seuil, ni dans l'early stopping, ni dans le choix du checkpoint. Les métriques fenêtre sont des moyennes de plis ; les métriques événementielles principales sont poolées sur 22 enregistrements / 9,23 h.

| Métrique d'évaluation | Baseline LightGBM (EXP-00) | CNN Dense (EXP-01) | CNN Factorisé (EXP-02) | EEGNet $\tau=0,5$ (EXP-03) | **EEGNet calibré (EXP-04)** |
|---|:---:|:---:|:---:|:---:|:---:|
| **Type de données d'entrée** | 260 features manuelles (FFT/Ondelettes) | Signal brut (8 canaux) | Signal brut (8 canaux) | Signal brut (8 canaux) | **Signal brut (8 canaux)** |
| **Nombre de paramètres** | Arbres gradient boosting (~100 arbres) | ~35 000 | ~18 000 | **1 489** | **1 489** |
| **Poids du modèle seuls** | ~1,2 Mo | ~140 Ko | ~72 Ko | **5,96 Ko** | **5,96 Ko** |
| **F1-Score (fenêtre)** | **0,700 ± 0,121** (0,705 poolé) | 0,488 ± 0,119 | 0,527 ± 0,140 | 0,522 ± 0,100 | **0,643 ± 0,077** |
| **Précision / rappel (fenêtre)** | **0,706 / 0,703 poolés** | — / 0,850 | 0,413 / 0,833 | 0,391 / 0,861 | **0,553 / 0,782** |
| **PR-AUC** | **0,652 poolé** | 0,646 | 0,679 | 0,692 | **0,692 ± 0,163** |
| **Taux Détection Crises** | **93,9 %** (77/82) | **96,3 %** | 93,9 % | 93,9 % | **93,9 %** (77/82 ; moyenne plis 92,4 ± 11,1 %) |
| **Fausses Alarmes / h** | **4,55 causal** (4,33 hors-ligne) | 19,50 | 19,93 | 11,48 | **8,88 causal** (8,45 hors-ligne ; moyenne plis 8,29 ± 6,05) |
| **Latence causale** | **4,46 s** | 3,33 s | 2,96 s | 1,19 s | **3,86 s** |
| **Temps calcul / fenêtre (médiane)** | **16,81 ± 5,33 ms** | — | — | — | **10,35 ± 4,82 ms** (pipeline total) ‡ |
| **Extraction manuelle de features** | OUI (lourde, 260 features) | NON (end-to-end) | NON (end-to-end) | NON (end-to-end) | **NON (end-to-end)** |

> [!NOTE]
> Les latences sont pondérées par événement, avec filtrage amont causal et post-traitement causal.
> ‡ Le ratio ~1,6× compare le **pipeline total** (16,81 vs 10,35 ms) sur du bruit synthétique avec un EEGNet non entraîné. En inférence seule, LightGBM est plus rapide (0,95 ms vs 1,46 ms en moyenne).

---

### 💡 Conclusions objectives et Matrice de Décision pour la soutenance

### 1. Matrice d'Arbitrage : 7 Dimensions d'Ingénierie (LightGBM vs EEGNet)

Le choix final de l'architecture ne se résume pas à sélectionner le F1-score fenêtre le plus haut. Il découle d'un arbitrage rigoureux sur **7 dimensions d'ingénierie** :

| Dimension d'ingénierie | Question résolue | Baseline LightGBM (EXP-00) | Candidat EEGNet Calibré (EXP-04) | Analyse de l'arbitrage |
|---|---|:---:|:---:|---|
| **1. Détection clinique (événements)** | Quel modèle détecte le mieux les crises sur des patients non vus ? | **93,9 %** (77/82, seuil interne strict) | **92,4 % ± 11,1** (seuil interne strict, τ* = 0,77 ± 0,15) | Détection proche ; LightGBM garde un F1 fenêtre supérieur (0,705 poolé vs 0,643 calibré). |
| **2. Taux de fausses alarmes** | Combien d'alertes erronées sont émises par heure ? | **4,33 FA/h hors-ligne / 4,55 causal** | **8,29 ± 6,05 FA/h** | LightGBM émet moins de fausses alarmes ; dispersion inter-plis élevée pour EEGNet. |
| **3. Robustesse inter-patients** | Les résultats restent-ils stables d'un patient à l'autre ? | F1 $\pm 0,121$ ; détection 71,4–100 % | F1 $\pm 0,077$ ; détection 71,4–100 % | EEGNet calibré a un F1 plus stable entre plis ; les deux restent sensibles aux crises brèves du pli 3. |
| **4. Latence causale en flux continu** | Le pipeline respecte-t-il la causalité sans fuite du futur ? | **4,46 s** | **3,86 s** | Latences pondérées par événement, filtrage amont causal `sosfilt` et post-traitement causal sans fusion rétroactive des trous. |
| **5. Coût de calcul de bout en bout** | Quel est le temps total de traitement par fenêtre de 2 s ? | 16,81 ± 5,33 ms (médiane) | **10,35 ± 4,82 ms** (médiane) | **Pipeline total ~1,6× plus rapide pour EEGNet** sur bruit synthétique et modèle non entraîné. En inférence seule, LightGBM est plus rapide (0,95 vs 1,46 ms). |
| **6. Empreinte mémoire des poids** | Quelle est la taille des poids et l'embarquabilité ? | ~1,2 Mo (100 arbres LightGBM) | **5,96 Ko** (1 489 poids Float32) | **Poids seuls plus compacts (5,96 Ko vs ~1,2 Mo)**, facilitant un futur portage microcontrôleur (hors runtime Python ~150 Mo). |
| **7. Intégration et maintenance** | Le code est-il testé, reproductible et modulaire ? | 260 features manuelles (scipy, pywt, numba) | Pipeline PyTorch compact, **130 fonctions de test** | **Candidat pour une future implémentation embarquée, à confirmer sur cible matérielle**. Exécution du 31/10/2026 : **130 passed, 0 xfailed**. |

---

### 2. Benchmark Équitable de Bout en Bout (50 répétitions, warm-up=5)

Chronométré sur la même machine avec des fenêtres de 2,0 s (512 échantillons à 256 Hz) générées à partir de **bruit synthétique**, et un **EEGNet non entraîné** (poids initiaux). Ce benchmark mesure le coût du pipeline, pas la performance clinique :

```text
=====================================================================================
Étape                               | LightGBM (médiane ± IQR) | EEGNet (médiane ± IQR)
-------------------------------------------------------------------------------------
1. Filtrage (0.5-40Hz + 50Hz notch) |   7,66 ± 3,05 ms      |   8,54 ± 3,91 ms
2. Préparation (Features vs Tenseur) |   7,72 ± 2,35 ms      |   0,08 ± 0,03 ms
3. Inférence (100 arbres vs ConvNet) |   0,80 ± 0,34 ms      |   1,32 ± 0,66 ms
4. Post-traitement causal glissant  |   0,17 ± 0,08 ms      |   0,19 ± 0,10 ms
-------------------------------------------------------------------------------------
TOTAL DE BOUT EN BOUT               |  16,81 ± 5,33 ms      |  10,35 ± 4,82 ms
=====================================================================================

Ratio pipeline total (médianes) : ~1,62x en faveur d'EEGNet
Inférence seule (moyennes) : LightGBM 0,95 ms vs EEGNet 1,46 ms
Budget temps réel disponible (pas de 1,0 s) : 1 000,00 ms
Charge CPU relative du pipeline : LightGBM = 1,68 % | EEGNet = 1,04 %
Latence causale événementielle P0 : 3,86 s (EEGNet calibré) vs 4,46 s (LightGBM)
```

> [!NOTE]
> Le ratio ~1,62× ne signifie pas qu'EEGNet est plus rapide en inférence : l'écart vient surtout de l'extraction des 260 features LightGBM (7,72 ms vs 0,08 ms). En inférence seule, LightGBM est plus rapide. Les délais P0 (3,86 s et 4,46 s) incluent la fenêtre de 2 s, la confirmation médiane causale et le filtrage amont causal `sosfilt`.

---

### 3. Précisions méthodologiques essentielles pour le jury

1. **Sélection du seuil : protocole P0 strictement imbriqué** :
   - LightGBM : pour chaque pli externe, 25 % des patients annotés du train externe forment la validation interne. Un modèle interne sélectionne $\tau^*_k$ ; un modèle final est réentraîné sur tout le train externe, puis évalué sur le test externe.
   - EEGNet : `train_fold_nested` sélectionne le meilleur epoch sur la validation interne, réentraîne sur tout le train externe, puis EXP-04 sélectionne $\tau^*_k$ sur les prédictions de validation interne de chaque pli.
   - Seuils finaux : LightGBM $\tau^* = 0,63 \pm 0,23$ ; EEGNet calibré $\tau^* = 0,77 \pm 0,15$. Le test externe n'intervient jamais dans ces choix.
   - Limite restante : les seuils sont choisis sur des métriques événementielles rétrospectives de validation interne, pas sur une estimation prospective.

2. **Formulation de la latence** :
   - Latences P0 pondérées par événement : **4,46 s** pour LightGBM et **3,86 s** pour EEGNet calibré. Elles incluent le filtrage amont causal (`sosfilt`), la fenêtre de 2 s et le post-traitement causal.
   - Le post-traitement causal ne fusionne plus rétroactivement les trous : `merge_gaps` reste disponible uniquement pour l'analyse rétrospective. Les tests couvrent l'invariance du début et de la fin des alarmes passées face à des scores futurs.

3. **Distinction entre poids du modèle et mémoire vive d'exécution** :
   - Les **1 489 paramètres float32** d'EEGNet occupent exactement **5,96 Ko** de stockage de poids.
   - L'empreinte mémoire d'exécution en environnement de développement Python/PyTorch dépend de l'interpréteur (~150 Mo). La compacité des poids démontre le potentiel d'un portage sur microcontrôleur embarqué (TFLite Micro / ONNX C++), mais toute qualification de « basse consommation » devra faire l'objet de mesures physiques de puissance sur cible matérielle.

4. **Couverture de tests et robustesse logicielle** :
   - La suite compte **130 fonctions de test** couvrant l'ingestion, les filtres causaux glissants, l'extraction de tenseurs, les architectures, la sélection imbriquée et les métriques. Exécution complète du 31/10/2026 : **130 passed, 0 xfailed**. Un taux de réussite élevé ne prouve pas la validité scientifique.

5. **Positionnement final pour le PFE** :
   - **EEGNet est retenu comme candidat pour la suite du projet** (monitoring ambulatoire continu et systèmes embarqués) en raison de sa compacité extrême (5,96 Ko) et de son fonctionnement de bout en bout sans extraction de 260 features. Son avantage de vitesse concerne le pipeline total (~1,62×) ; en inférence seule, LightGBM reste plus rapide. Après calibration stricte : F1 fenêtre 0,643, 93,9 % de détection poolée et 8,88 FA/h causal (moyenne des plis 92,4 ± 11,1 % et 8,29 ± 6,05 FA/h).
   - **LightGBM reste la référence de performance** pour l'analyse rétrospective en milieu hospitalier : F1 fenêtre 0,705 poolé, 93,9 % de détection et 4,55 FA/h en post-traitement causal.
   - En tant que prototype académique sur 19 patients pédiatriques, ce système ne constitue pas un dispositif médical certifié et nécessite une validation externe préalable à tout déploiement clinique.

---

## 🧾 Statut de reproductibilité (P0 terminé)

Les artefacts P0 ont été régénérés et alignés avec le code :

- `cache/features.parquet` et `cache/raw_windows.npz` reconstruits avec `force=True` après passage au filtrage causal `v3_causal`.
- `cache/experiments/EXP-00_nested_results.json`, `EXP-01_metrics.json`, `EXP-02_metrics.json`, `EXP-03_metrics.json`, `EXP-04_metrics.json` et les prédictions `EXP-03_oof_predictions.pkl` / `EXP-03_inner_val_predictions.pkl` régénérés.
- EXP-00 et EXP-04 utilisent une validation interne par patient ; EXP-01 à EXP-03 utilisent `train_fold_nested` ; EXP-04 ne sélectionne le seuil que sur la validation interne.
- Filtrage amont causal (`sosfilt`), post-traitement causal et agrégation des latences pondérées par événement.
- Suite complète du 31/10/2026 : **130 passed, 0 xfailed** (`python -m pytest tests -q`).
- Les anciens caches `v2_filtered` et le mode `merge_gaps` rétrospectif restent sur disque pour comparaison, mais ne sont pas utilisés pour les résultats P0.

**Règle P0** : les chiffres principaux sont ceux des expériences régénérées avec test externe aveugle. Les valeurs historiques 86,6 % / 3,35 FA/h restent archivées et ne doivent pas être présentées comme l'estimation actuelle. Une validation externe indépendante reste nécessaire.
