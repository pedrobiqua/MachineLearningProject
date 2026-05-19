import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, cohen_kappa_score

from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns

### Modelos que serão avaliados
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

# # -------------------------
# # Path do dataset para ser avaliado nos modelos
# # -------------------------

### TROCAR APENAS AQUI
EXTRACTION_TYPE = "HOG" # RESNET | HOG
WITH_PCA = False
n_components = 100

if EXTRACTION_TYPE == "HOG":
    PATH = "/home/pedro/Projetos/MachineLearningProject/features_csv"
    TRAIN = PATH + "/train_features_hog.csv"
    TEST = PATH + "/test_features_hog.csv"
elif EXTRACTION_TYPE == "RESNET":
    PATH = "/home/pedro/Datasets/FER"
    TRAIN = PATH + "/train_features_model.csv"
    TEST = PATH + "/test_features_model.csv"
else:
    os._exit(0)

def train_and_evaluate(X_train, y_train, X_test, y_test, model, param_grid):

    if WITH_PCA:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components)),
            ('model', model)
        ])
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

    print(f"Modelo usado no experimento: {pipeline}")

    param_grid = {
        f"model__{key}": value for key, value in param_grid.items()
    }
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    ## Grid search usando o pipeline, utiliza o scaler de maneira correta
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring='f1_macro',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nMelhores parâmetros encontrados:")
    print(grid.best_params_)

    print("Score médio (CV):")
    print(grid.best_score_)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, normalize='true')

    # Métrica Kappa
    kappa = cohen_kappa_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    result_text = ""
    result_text += "Parâmetros:\n"
    result_text += str(grid.best_params_) + "\n\n"

    result_text += "Score médio (CV):\n"
    result_text += str(grid.best_score_) + "\n\n"

    result_text += "Kappa:\n"
    result_text += str(kappa) + "\n\n"

    result_text += "F1 Macro:\n"
    result_text += str(f1_macro) + "\n\n"

    result_text += "F1 Weighted:\n"
    result_text += str(f1_weighted) + "\n\n"

    result_text += "Reports:\n"
    result_text += report + "\n"

    metrics = {
        "model": model.__class__.__name__,
        "best_params": str(grid.best_params_),
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "kappa": kappa
    }

    return metrics, result_text, cm

def save_confusion_matrix(cm, path_output_file):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')

    plt.xlabel("Predito")
    plt.ylabel("Real")

    plt.tight_layout()
    plt.savefig(f"{path_output_file}_confusion_matrix.png", dpi=300)
    plt.close()


models_and_parameters = [
    {
        "name": "LogisticRegression",
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.01, 0.1, 1, 10]
        }
    },
    {
        "name": "KNN",
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9]
        }
    },
    {
        "name": "SVM",
        "model": SVC(),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf"]
        }
    },
    {
        "name": "MLP",
        "model": MLPClassifier(max_iter=1000),
        "params": {
            "hidden_layer_sizes": [(50,), (50, 50)],
            "alpha": [0.0001, 0.001]
        }
    },
    {
        "name": "RandomForest",
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10]
        }
    },
    {
        "name": "DecisionTree",
        "model": DecisionTreeClassifier(),
        "params": {
            "max_depth": [None, 5, 10]
        }
    },
    {
        "name": "NaiveBayes",
        "model": GaussianNB(),
        "params": {
            # vazio mesmo
        }
    },
    {
        "name": "LDA",
        "model": LinearDiscriminantAnalysis(),
        "params": {
            # vazio mesmo
        }
    }
]

DATASET = TRAIN.split("/")[-1].replace(".csv", "")

# Carregar dados
X_train = pd.read_csv(TRAIN)
y_train = X_train['label']
X_train = X_train.drop('label', axis=1)

X_test = pd.read_csv(TEST)
y_test = X_test['label']
X_test = X_test.drop('label', axis=1)

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("TESTE DE CLASSIFICADORES FER")
print(f"CONFIGS:\n Tipo de extração: {EXTRACTION_TYPE} | PCA: {WITH_PCA}")

experiment_results = []

for item in models_and_parameters:
    print("\nMODELO:", item["name"])

    metrics_dict, result_text, cm = train_and_evaluate(
        X_train, y_train,
        X_test, y_test,
        item["model"],
        item["params"]
    )

    if WITH_PCA:
        name = f"{item['name']}_{EXTRACTION_TYPE}_PCA"
    else:
        name = f"{item['name']}_{EXTRACTION_TYPE}"

    file_path = os.path.join(OUTPUT_DIR, f"{name}.txt")
    with open(file_path, "w") as f:
        f.write(result_text)

    save_confusion_matrix(cm, path_output_file=f"{OUTPUT_DIR}/{name}")
    experiment_results.append(metrics_dict)

results_df = pd.DataFrame(experiment_results)
results_df = results_df.sort_values(
    by="test_f1_macro",
    ascending=False
)

results_df = results_df.round(4)

summary_path = os.path.join(
    OUTPUT_DIR,
    f"summary_{EXTRACTION_TYPE}_{f'PCA_{n_components}' if WITH_PCA else 'NO_PCA'}.csv"
)

results_df.to_csv(summary_path, index=False)

print("\nResumo dos experimentos:")
print(results_df)