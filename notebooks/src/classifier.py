"""
src/classifier.py

Classificador supervisionado -- adaptado de
https://github.com/Laboratorio-de-Analise-de-Dados/disciplina_python_ml,
dev/script/aulas_teoricas/src/classifier.py. Mesmo pipeline (KNN/SVM/RF/GBM/NB/NN, cada um com
ColumnTransformer + VarianceThreshold + SelectKBest + GridSearchCV(StratifiedKFold)),
mesma forma de reportar (métricas, matriz de confusão, variáveis selecionadas).
"""

# Importando modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Visualização dos dados
import matplotlib.pyplot as plt

# Edição das databases
import pandas as pd
import numpy as np

# Seleção de hiperparâmetros e validação cruzada
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict

# Estruturação dos dados e pré-processamento
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Organização do fluxo de trabalho (Pipeline)
from sklearn.pipeline import Pipeline

# Seleção de variáveis
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Métricas da classificação
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class Classifier:
    """
    Classificador genérico que encapsula diferentes modelos de aprendizado de máquina.

        Parâmetros
        ----------
        df : pd.DataFrame
            Treino: features + coluna alvo.
        target : str
            Nome da coluna alvo (default `"Risco"`).
        df_test : pd.DataFrame | None
            Teste, com as mesmas colunas de `df`. 
        descartar : list[str]
            Colunas que não são feature nem alvo (ex.: `["Species"]`). As que
            não existirem no df são ignoradas.
        scoring : str
            Métrica do GridSearchCV (default `"balanced_accuracy"`).

        Atributos (depois de `classify`)
        --------------------------------
        modelos_ : dict[str, Pipeline]
            Melhor pipeline de cada modelo já treinado (ex.: `clf.modelos_["rf"]`),
            pronto para `.predict()` em dados novos.

        Métodos
        -------
        classify(modelo, n_splits=5)
            Treina, avalia e devolve (métricas, predições).

        Exemplo
        -------
        >>> clf = Classifier(df_train, target="cluster", df_test=df_test, descartar=["Species"])
        >>> metricas, y_pred = clf.classify("rf")
        >>> clf.modelos_["rf"].predict(df_novo[clf.X.columns])
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str = "Risco",
        df_test: pd.DataFrame | None = None,
        descartar: list[str] | tuple[str, ...] = (),
        scoring: str = "balanced_accuracy",
    ) -> None:
        self.target = target
        self.scoring = scoring
        self.descartar = [c for c in descartar if c in df.columns]
        self.X = df.drop(columns=[target, *self.descartar])
        self.y = df[target]
        self.X_test = None if df_test is None else df_test[self.X.columns]
        self.y_test = None if df_test is None else df_test[target]
        self.modelos_ = {}  # só existe depois do .fit
        self.melhores_parametros_ = {} 

        return None

    def __avaliar(self, grid_search: GridSearchCV, cv: StratifiedKFold) -> tuple[pd.Series, np.ndarray, str]:
        """
        Predições em dados que o modelo não viu: o teste, se existir; senão,
        out-of-fold no treino (cada genoma é previsto por um modelo treinado
        sem ele, com os hiperparâmetros do melhor modelo).
            Retorna
            -------
            (y_verdadeiro, y_pred, nome_do_conjunto)
        """
        if self.X_test is not None:
            return self.y_test, grid_search.best_estimator_.predict(self.X_test), "teste"
        y_pred = cross_val_predict(grid_search.best_estimator_, self.X, self.y, cv=cv)
        return self.y, y_pred, "out-of-fold (treino)"

    def __metricas_pontuais(self, grid_search: GridSearchCV) -> None:
        """
        Exibe os melhores hiperparâmetros e o score de CV do melhor modelo
        encontrado pelo GridSearchCV (já ajustado).
        """
        print("\n--- Melhores Resultados do Grid Search ---")
        print(f"Melhores hiperparâmetros: {grid_search.best_params_}")
        print(f"Melhor {self.scoring} (CV): {grid_search.best_score_:.4f}")
        return None

    def __matrix_confusao(self, y_true: pd.Series, y_pred: np.ndarray, conjunto: str) -> None:
        """
        Exibe a matriz de confusão e o classification report.
        `labels` fixa a ordem das classes, e a mesma lista vai para
        `display_labels`, então os nomes dos eixos batem com linhas/colunas.
        """
        labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Greens)
        plt.title(
            f"{conjunto} | accuracy {accuracy_score(y_true, y_pred):.2f} | "
            f"balanced {balanced_accuracy_score(y_true, y_pred):.2f}"
        )
        plt.show()
        print(f"\nClassification Report ({conjunto}):")
        print(classification_report(y_true, y_pred, zero_division=0))
        return None

    def __retornar_metricas(self, y_true: pd.Series, y_pred: np.ndarray, conjunto: str) -> pd.DataFrame:
        """
        Métricas em uma linha. "weighted" pondera pelo suporte de cada classe;
        "macro" (F1 macro, balanced accuracy) dá o mesmo peso a todas, então
        uma classe pequena mal prevista aparece.
        """
        return pd.DataFrame(
            {
                "Conjunto": [conjunto],
                "Acurácia": [accuracy_score(y_true, y_pred)],
                "Acurácia balanceada": [balanced_accuracy_score(y_true, y_pred)],
                "Precision (weighted)": [precision_score(y_true, y_pred, average="weighted", zero_division=0)],
                "Recall (weighted)": [recall_score(y_true, y_pred, average="weighted", zero_division=0)],
                "F1 (weighted)": [f1_score(y_true, y_pred, average="weighted", zero_division=0)],
                "F1 (macro)": [f1_score(y_true, y_pred, average="macro", zero_division=0)],
            }
        ).round(4)

    def __variaveis_selecionadas(self, grid_search: GridSearchCV) -> pd.DataFrame:
        """
        Exibe as variáveis selecionadas pelo melhor modelo encontrado pelo
        GridSearchCV.
            Parâmetros
            ----------
            grid_search : GridSearchCV
                Objeto GridSearchCV já ajustado com os dados.

            Retorna
            -------
            df_summary : pd.DataFrame
                DataFrame contendo o status de cada atributo no pipeline.
        """
        # 1. Extrai o melhor pipeline ajustado pelo GridSearchCV
        best_pipeline = grid_search.best_estimator_

        # 2. Recupera os nomes de TODAS as colunas geradas após a
        # codificação/escala (ColumnTransformer)
        feat_names_orig = best_pipeline["preprocessor"].get_feature_names_out()

        # 3. Etapa 1: Aplica a máscara do VarianceThreshold
        mask_variance = best_pipeline["var_threshold"].get_support()
        features_after_variance = feat_names_orig[mask_variance]

        # 4. Etapa 2: Aplica a máscara do SelectKBest sobre as colunas
        # sobressalentes
        mask_kbest = best_pipeline["feature_selection"].get_support()
        selected_features = features_after_variance[mask_kbest]

        # 5. Imprime o resultado final de forma amigável
        m = "Total de variáveis "
        print(f"{m}originais pré-processadas: {len(feat_names_orig)}")
        print(f"{m}após VarianceThreshold:    {len(features_after_variance)}")
        print(f"{m}selecionadas no modelo:    {len(selected_features)}")

        print("\n--- Lista das Variáveis Selecionadas ---")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i}. {feature}")

        # Cria uma máscara final combinando as duas seleções
        final_mask = mask_variance.copy()
        final_mask[mask_variance] = mask_kbest

        # Constrói o relatório em DataFrame
        df_summary = pd.DataFrame(
            {
                "Atributo_Preprocessado": feat_names_orig,
                "Passou_Variancia": mask_variance,
                "Selecionado_Final": final_mask,
            }
        )

        print("\n--- Status de Cada Atributo no Pipeline ---")
        print(df_summary.to_string(index=False))

        return df_summary

    def __preprocessador(self) -> ColumnTransformer:
        """
        Cria um pré-processador que padroniza variáveis numéricas e
        aplica One-Hot Encoding em variáveis categóricas (colunas de `self.X`,
        que já não tem o alvo nem as colunas de `descartar`).
            Retorna
            -------
            preprocessor : ColumnTransformer
                Objeto ColumnTransformer para pré-processamento.
        """
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    self.X.select_dtypes(include=["number", "bool"]).columns,
                ),
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    self.X.select_dtypes(include=["object", "category"]).columns,
                ),
            ]
        )
        return preprocessor

    def __knn_classify(self) -> tuple[Pipeline, dict]:
        """
        Cria um pipeline para o classificador KNN e define a grade de
        hiperparâmetros para busca.
            Retorna
            -------
            pipeline : Pipeline
                Objeto Pipeline configurado com pré-processamento e KNN.
            param_grid : dict
                Dicionário contendo a grade de hiperparâmetros para busca.
        """
        # KNeighborsClassifier não tem `class_weight` para colunas desbalanceadas
        pipeline = Pipeline(  # lista de tuplas
            [
                ("preprocessor", self.__preprocessador()),
                ("var_threshold", VarianceThreshold(threshold=1e-4)),
                (
                    "feature_selection",
                    SelectKBest(score_func=f_classif, k=min(10, self.X.shape[1])),
                ),
                ("knn", KNeighborsClassifier()),
            ]
        )
        param_grid = {
            "var_threshold__threshold": [1e-4, 0.01, 0.05],
            "feature_selection__k": [1, 2, "all"],
            "knn__n_neighbors": [1, 3, 5],
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan"],
        }

        return pipeline, param_grid

    def __svm_classify(self) -> tuple[Pipeline, dict]:
        """
        Cria um pipeline para o classificador SVM e define a grade de
        hiperparâmetros para busca.
            Retorna
            -------
            pipeline : Pipeline
                Objeto Pipeline configurado com pré-processamento e SVM.
            param_grid : dict
                Dicionário contendo a grade de hiperparâmetros para busca.
        """

        pipeline = Pipeline(
            [
                ("preprocessor", self.__preprocessador()),
                ("var_threshold", VarianceThreshold(threshold=1e-4)),
                (
                    "feature_selection",
                    SelectKBest(score_func=f_classif, k=min(10, self.X.shape[1])),
                ),
                ("svm", SVC(random_state=42, 
                            class_weight="balanced")), # pondera a função de perda pelo inverso da frequência de cada classe
            ]
        )
        param_grid = {
            "var_threshold__threshold": [1e-4, 0.01, 0.05],
            "feature_selection__k": [1, 2, "all"],
            # Parâmetro de regularização
            "svm__C": [0.1, 1, 10, 100],
            # Tipo de kernel (Linear ou Radial Basis Function)
            "svm__kernel": ["linear", "rbf"],
            # Coeficiente do kernel RBF
            "svm__gamma": ["scale", "auto", 0.01, 0.1],
        }

        return pipeline, param_grid

    def __rf_classify(self) -> tuple[Pipeline, dict]:
        """
        Cria um pipeline para o classificador Random Forest e define a
        grade de hiperparâmetros para busca.
            Retorna
            -------
            pipeline : Pipeline
                Objeto Pipeline configurado com pré-processamento e
                Random Forest.
            param_grid : dict
                Dicionário contendo a grade de hiperparâmetros para busca.
        """
        # class_weight="balanced" -- mesmo motivo do SVM acima
        pipeline = Pipeline(
            [
                ("preprocessor", self.__preprocessador()),
                ("var_threshold", VarianceThreshold(threshold=1e-4)),
                (
                    "feature_selection",
                    SelectKBest(score_func=f_classif, k=min(10, self.X.shape[1])),
                ),
                ("rf", RandomForestClassifier(random_state=42, class_weight="balanced")),
            ]
        )

        param_grid = {
            "var_threshold__threshold": [1e-4, 0.01, 0.05],
            "feature_selection__k": [1, 2, "all"],
            # Número de árvores na floresta
            "rf__n_estimators": [50, 100, 200],
            # Profundidade máxima de cada árvore
            "rf__max_depth": [None, 5, 10],
            # Mínimo de amostras para dividir um nó
            "rf__min_samples_split": [2, 5],
            # Critério de medição de qualidade da divisão
            "rf__criterion": ["gini", "entropy"],
        }

        return pipeline, param_grid

    def __gbm_classify(self) -> tuple[Pipeline, dict]:
        """
        Cria um pipeline para o classificador Gradient Boosting e define
        a grade de hiperparâmetros para busca.
            Retorna
            -------
            pipeline : Pipeline
                Objeto Pipeline configurado com pré-processamento e
                Gradient Boosting.
            param_grid : dict
                Dicionário contendo a grade de hiperparâmetros para busca.
        """
        # GradientBoostingClassifier não tem `class_weight`
        pipeline = Pipeline(
            [
                ("preprocessor", self.__preprocessador()),
                ("var_threshold", VarianceThreshold(threshold=1e-4)),
                (
                    "feature_selection",
                    SelectKBest(score_func=f_classif, k=min(10, self.X.shape[1])),
                ),
                ("gb", GradientBoostingClassifier(random_state=42)),
            ]
        )

        param_grid = {
            "var_threshold__threshold": [1e-4, 0.01, 0.05],
            "feature_selection__k": [1, 2, "all"],
            # Número de estágios de boosting (árvores)
            "gb__n_estimators": [50, 100, 150],
            # Taxa de aprendizado (encolhimento do impacto de cada árvore)
            "gb__learning_rate": [0.01, 0.1, 0.2],
            # Profundidade máxima dos estimadores individuais
            "gb__max_depth": [3, 5],
            # Fração de amostras usadas para ajustar os estimadores base
            "gb__subsample": [0.8, 1.0],
        }

        return pipeline, param_grid

    def __nb_classify(self) -> tuple[Pipeline, dict]:
        """
        Cria um pipeline para o classificador Gaussian Naive Bayes e
        define a grade de hiperparâmetros para busca.
            Retorna
            -------
            pipeline : Pipeline
                Objeto Pipeline configurado com pré-processamento e
                Gaussian Naive Bayes.
            param_grid : dict
                Dicionário contendo a grade de hiperparâmetros para busca.
        """
        pipeline = Pipeline(
            [
                ("preprocessor", self.__preprocessador()),
                ("var_threshold", VarianceThreshold(threshold=1e-4)),
                (
                    "feature_selection",
                    SelectKBest(score_func=f_classif, k=min(10, self.X.shape[1])),
                ),
                ("nb", GaussianNB()),
            ]
        )

        param_grid = {
            "var_threshold__threshold": [1e-4, 0.01, 0.05],
            "feature_selection__k": [1, 2, "all"],
            # Suavização de variância para estabilidade numérica
            "nb__var_smoothing": np.logspace(0, -9, num=10),
        }

        return pipeline, param_grid

    def __nn_classify(self) -> tuple[Pipeline, dict]:
        """
        Cria um pipeline para o classificador Rede Neural (MLP) e define
        a grade de hiperparâmetros para busca.
            Retorna
            -------
            pipeline : Pipeline
                Objeto Pipeline configurado com pré-processamento e
                Rede Neural (MLP).
            param_grid : dict
                Dicionário contendo a grade de hiperparâmetros para busca.
        """
        # max_iter expandido para garantir convergência durante a otimização
        pipeline = Pipeline(
            [
                ("preprocessor", self.__preprocessador()),
                ("var_threshold", VarianceThreshold(threshold=1e-4)),
                (
                    "feature_selection",
                    SelectKBest(score_func=f_classif, k=min(10, self.X.shape[1])),
                ),
                ("mlp", MLPClassifier(max_iter=1000, random_state=42)),
            ]
        )

        param_grid = {
            "var_threshold__threshold": [1e-4, 0.01, 0.05],
            "feature_selection__k": [1, 2, "all"],
            # Arquiteturas: 1 camada com 10/50 neurônios ou 2 camadas (20, 10)
            "mlp__hidden_layer_sizes": [(10,), (20, 10), (50,)],
            # Funções de ativação
            "mlp__activation": ["relu", "tanh"],
            # Otimizadores
            "mlp__solver": ["adam", "lbfgs"],
            # Termo de regularização L2 (penalty)
            "mlp__alpha": [0.0001, 0.01],
        }

        return pipeline, param_grid

    def classify(
        self, modelo: str, n_splits: int = 5
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Treina (GridSearchCV no treino) e avalia o modelo especificado.
            Parâmetros
            ----------
            modelo : str
                Nome do modelo a ser treinado. Opções:
                                'knn', 'svm', 'rf', 'gbm', 'nb', 'nn'.
            n_splits : int, opcional
                Número de divisões para a validação cruzada (default é 5).
            Retorna
            -------
            metricas : pd.DataFrame
                1 linha com as métricas no teste (ou out-of-fold).
            y_pred : np.ndarray
                Predições nesse mesmo conjunto.
        """
        construtores = {
            "knn": self.__knn_classify,
            "svm": self.__svm_classify,
            "rf": self.__rf_classify,
            "gbm": self.__gbm_classify,
            "nb": self.__nb_classify,
            "nn": self.__nn_classify,
        }
        if modelo not in construtores:
            raise ValueError(f"modelo {modelo!r} desconhecido -- use um de {list(construtores)}")

        estimator, param_grid = construtores[modelo]()
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=self.scoring,
            n_jobs=-1,
        )

        print(f"--- Treinando {modelo} ({len(self.X)} amostras, {self.X.shape[1]} features) ---")
        grid_search.fit(self.X, self.y)
        self.modelos_[modelo] = grid_search.best_estimator_
        self.melhores_parametros_[modelo] = grid_search.best_params_

        self.__metricas_pontuais(grid_search=grid_search) # printa .best_param e .best_score
        y_true, y_pred, conjunto = self.__avaliar(grid_search=grid_search, cv=cv)
        self.__matrix_confusao(y_true, y_pred, conjunto)
        self.__variaveis_selecionadas(grid_search=grid_search)

        return self.__retornar_metricas(y_true, y_pred, conjunto), y_pred
