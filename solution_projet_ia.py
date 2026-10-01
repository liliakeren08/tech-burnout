import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
# Modèles Scikit-Learn
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ===============================================================================================================================
# 1. ALGORITHMES DE CLASSIFICATION DÉVELOPPÉS "FROM SCRATCH"
# ===============================================================================================================================

class PerceptronScratch:
    """Implémentation du Perceptron de Rosenblatt de A à Z."""
    def __init__(self, eta=0.01, n_iter=50, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        # Initialisation des poids avec de petites valeurs aléatoires
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        
        for _ in range(self.n_iter):
            for xi, target in zip(X, y):
                # Règle de mise à jour de Rosenblatt
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update  # Le biais w0
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


class KNNScratch:
    """Implémentation de l'algorithme des K plus proches voisins (KNN) de A à Z."""
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        predictions = [self._predict_single(xi) for xi in np.array(X)]
        return np.array(predictions)

    def _predict_single(self, x):
        # 1. Calculer les distances euclidiennes entre le point x et tous les points d'entraînement
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        
        # 2. Trouver les indices des K plus proches voisins
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Récupérer les classes de ces K voisins
        k_nearest_labels = self.y_train[k_indices]
        
        # 4. Vote majoritaire (classe la plus fréquente)
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common


# ===============================================================================================================================
# 2. FONCTION PRINCIPALE DE CHARGEMENT ET D'ENTRAÎNEMENT
# ===============================================================================================================================

def run_experiment(df_base, balance_data=False):
    """Exécute l'analyse complète (prétraitement et entraînement) pour un scénario donné."""
    # 1. Échantillonnage ou Équilibrage
    if balance_data:
        # Prendre 100% des cas de classe 1 (14 450 lignes) et un échantillon équivalent de classe 0
        df_pos = df_base[df_base['seeks_professional_help'] == 1]
        df_neg = df_base[df_base['seeks_professional_help'] == 0].sample(n=len(df_pos), random_state=42)
        df_balanced = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        # Sélectionner 10 000 lignes équilibrées
        df = df_balanced.sample(n=10000, random_state=42).reset_index(drop=True)
    else:
        # Sélectionner 10 000 lignes brutes
        df = df_base.sample(n=10000, random_state=42).reset_index(drop=True)

    # 2. Prétraitement des colonnes
    df = df.drop(columns=['burnout_level'], errors='ignore')
    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])
    df = pd.get_dummies(df, columns=['job_role', 'company_size', 'work_mode'])

    # 3. Séparation X et y
    nom_cible = 'seeks_professional_help'
    X = df.drop(columns=[nom_cible])
    y = df[nom_cible]

    # 4. Split Train/Test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 5. Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Dictionnaire de modèles
    modeles = {
        "SVM (RBF)": SVC(kernel='rbf', C=1.0, random_state=42),
        "Neural Network MLP": MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Perceptron Scratch": PerceptronScratch(eta=0.01, n_iter=50, random_state=42),
        "KNN Scratch (K=5)": KNNScratch(k=5)
    }

    resultats = {}
    for nom, model in modeles.items():
        if "Scratch" in nom:
            model.fit(X_train_scaled, y_train.values)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        # Métriques
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        resultats[nom] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ConfusionMatrix": cm
        }
    return resultats


def main():
    print("="*90)
    print("      PIPELINE D'ENTRAINEMENT ET BENCHMARK COMPARATIF (IMBALANCED VS BALANCED)")
    print("="*90)

    # Resolution dynamique et portable du jeu de donnees
    base_dir = Path(__file__).resolve().parent
    search_paths = [
        base_dir / "tech_mental_health_burnout.csv",
        base_dir / "data" / "tech_mental_health_burnout.csv",
        Path.cwd() / "tech_mental_health_burnout.csv",
        Path.cwd() / "data" / "tech_mental_health_burnout.csv"
    ]

    df_base = None
    for path in search_paths:
        if path.exists():
            print(f"Chargement des donnees depuis : {path.name}")
            df_base = pd.read_csv(path)
            break

    if df_base is None:
        print("Erreur : Fichier 'tech_mental_health_burnout.csv' introuvable dans le repertoire.")
        return

    # --- ETUDE A : DONNEES DESEQUILIBREES ---
    print("\n[Etude A] Entrainement sur donnees d'origine desequilibrees...")
    res_imbalanced = run_experiment(df_base, balance_data=False)

    # --- ETUDE B : DONNEES REEQUILIBREES ---
    print("\n[Etude B] Entrainement sur donnees reequilibrees (Undersampling)...")
    res_balanced = run_experiment(df_base, balance_data=True)

    # --- RENDU DE L'ETUDE COMPARATIVE ---
    comparatif_txt = ""
    comparatif_txt += "="*90 + "\n"
    comparatif_txt += "             BENCHMARK PREDICTIF & EVALUATION DES MODELES DE CLASSIFICATION\n"
    comparatif_txt += "="*90 + "\n\n"

    comparatif_txt += "SCENARIO A : DONNEES BRUTES (DESEQUILIBREES - 90% Classe 0, 10% Classe 1)\n"
    comparatif_txt += "-"*90 + "\n"
    comparatif_txt += f"{'Algorithme':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}\n"
    comparatif_txt += "-"*90 + "\n"
    for nom, met in res_imbalanced.items():
        comparatif_txt += f"{nom:<25} | {met['Accuracy']*100:8.2f}% | {met['Precision']*100:8.2f}% | {met['Recall']*100:8.2f}% | {met['F1-Score']*100:8.2f}%\n"
    comparatif_txt += "-"*90 + "\n\n"

    comparatif_txt += "SCENARIO B : DONNEES REEQUILIBREES (EQUILIBREES - 50% Classe 0, 50% Classe 1)\n"
    comparatif_txt += "-"*90 + "\n"
    comparatif_txt += f"{'Algorithme':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}\n"
    comparatif_txt += "-"*90 + "\n"
    for nom, met in res_balanced.items():
        comparatif_txt += f"{nom:<25} | {met['Accuracy']*100:8.2f}% | {met['Precision']*100:8.2f}% | {met['Recall']*100:8.2f}% | {met['F1-Score']*100:8.2f}%\n"
    comparatif_txt += "-"*90 + "\n\n"

    comparatif_txt += "MATRICES DE CONFUSION COMPARATIVES\n"
    comparatif_txt += "="*90 + "\n"
    for nom in res_imbalanced.keys():
        comparatif_txt += f"*** {nom} ***\n"
        comparatif_txt += f"  -> Scenario A (Desequilibre) :\n{res_imbalanced[nom]['ConfusionMatrix']}\n"
        comparatif_txt += f"  -> Scenario B (Equilibre)    :\n{res_balanced[nom]['ConfusionMatrix']}\n\n"

    # Affichage sur la console
    print("\n" + comparatif_txt)

    # Sauvegarde dans le fichier texte final
    out_file = base_dir / "resultats_modeles_projet.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(comparatif_txt)
    print(f"-> Rendu final et matrices de confusion enregistres dans '{out_file.name}' !")

    # ===============================================================================================================================
    # 3. GENERATION DES GRAPHIQUES REQUIS PAR LE PROJET
    # ===============================================================================================================================
    print("\nGeneration des graphiques en cours... (Fermez chaque fenetre pour continuer)")
    
    # 3.1 Distribution des classes (Biais de depart)
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='seeks_professional_help', data=df_base, hue='seeks_professional_help', palette='Set2', legend=False)
    plt.title("Distribution de la variable cible (Donnees Brutes)", fontsize=14)
    plt.xlabel("Cherche de l'aide professionnelle (0 = Non, 1 = Oui)")
    plt.ylabel("Nombre d'employes")
    for p in ax.patches:
        ax.annotate(f'{100 * p.get_height() / len(df_base):.1f}%', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

    # 3.2 Matrice de correlation globale
    df_corr = df_base.copy()
    df_corr = df_corr.drop(columns=['burnout_level'], errors='ignore')
    le_tmp = LabelEncoder()
    # Handle object types mapping
    for col in df_corr.select_dtypes(exclude=['number']).columns:
        df_corr[col] = le_tmp.fit_transform(df_corr[col].astype(str))
            
    plt.figure(figsize=(12, 10))
    corr = df_corr.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .7}, annot=False)
    plt.title("Matrice de Correlation Globale", fontsize=16)
    plt.tight_layout()
    plt.show()

    # 3.3 Comparaison des F1-Scores
    models = list(res_imbalanced.keys())
    f1_imbalanced = [res_imbalanced[m]['F1-Score'] * 100 for m in models]
    f1_balanced = [res_balanced[m]['F1-Score'] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(12, 6))
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, f1_imbalanced, width, label='Scenario A (Desequilibre)', color='lightcoral')
    rects2 = ax.bar(x + width/2, f1_balanced, width, label='Scenario B (Equilibre)', color='mediumseagreen')

    ax.set_ylabel('F1-Score (%)')
    ax.set_title("Comparaison des F1-Scores : Avant vs Apres Equilibrage", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
