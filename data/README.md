# Données brutes (à ne pas committer)

Ce projet nécessite les fichiers EEG au format MCOS et le fichier d'annotations.

## Fichiers attendus
- *_0000d.mat (un fichier par patient/enregistrement)
- Annotations MAJ.xlsx

## Emplacement
Par défaut, le dossier est configuré dans src/config.py via DATA_DIR = C:\Users\MLAB\CRISES ABSENCES.

Tu peux le surcharger sans modifier le code en définissant la variable d'environnement :

```powershell
$env:EEG_DATA_DIR = "D:\chemin\vers\ton\dossier"
```

## Important
- **Ne pas** ajouter ces fichiers au dépôt Git.
- **Ne pas** les uploader sur GitHub (ce sont de vraies données cliniques/expérimentales).
- Les caches générés (cache/) sont ignorés par .gitignore.
