"""
src/classifier.py

Classificador supervisionado -- adaptado de
https://github.com/Laboratorio-de-Analise-de-Dados/disciplina_python_ml,
dev/script/aulas_teoricas/src/classifier.py. Mesmo pipeline (KNN/SVM/RF/GBM/NB/NN, cada um com
ColumnTransformer + VarianceThreshold + SelectKBest + GridSearchCV(StratifiedKFold, scoring=accuracy)),
mesma forma de reportar (accuracy/precision/recall/F1, matriz de confusão, variáveis selecionadas).

Adaptações em relação à original:
- Alvo configurável (`target`, default `"Risco"`) em vez de `"classe"` fixo -- a coluna alvo aqui é a
  categoria de risco derivada dos clusters de `06_clusterizacao_risco.ipynb` (ver notebook de
  classificação, ainda não escrito), não um dataset didático já rotulado.
- `class_weight="balanced"` em SVM e Random Forest -- os 2 únicos dos 6 modelos com esse parâmetro
  nativo no scikit-learn. Motivo: a discussão que levou a este arquivo já identificou que "Baixo"
  tende a virar a classe MAJORITÁRIA disparada (a maioria dos genomas High/Medium/Bin não tem carga
  beta-lactâmica -- ver `06_clusterizacao_risco.ipynb`, seção 10), então treinar sem correção de peso
  tende a colapsar pra sempre prever a classe majoritária. `KNeighborsClassifier`/
  `GradientBoostingClassifier`/`GaussianNB`/`MLPClassifier` NÃO têm `class_weight` nativo -- se o
  desbalanceamento continuar um problema depois de testar estes 6, a próxima ferramenta é
  reamostragem (SMOTE) antes do `.fit()`, não um parâmetro do estimador -- fora de escopo aqui de
  propósito, ver conversa/notebook de clusterização pra contexto.
- `__preprocessador`: a original monta a lista de colunas NUMÉRICAS a partir de `self.df` inteiro, sem
  excluir `"classe"` (só a lista de colunas CATEGÓRICAS excluía). Funcionava na prática porque o alvo
  didático era string (`select_dtypes(["number"])` já excluía sozinho); aqui excluo `target`
  explicitamente dos dois lados -- se `Risco` algum dia virar código numérico (0/1/2) em vez de string,
  a original quebraria (`ColumnTransformer` tentaria selecionar uma coluna que não existe mais em
  `self.X` na hora do fit).
- `classify()`: os 6 blocos `if modelo == "..."` (idênticos exceto qual `__<modelo>_classify()` chamar)
  viraram um dict de despacho -- mesmo comportamento, menos repetição.
- Resto do pipeline (pré-processamento/seleção de variável/grid search/relatório) sem mudança de
  lógica, só apontado pra `target` em vez de `"classe"`.
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
            DataFrame contendo as features e a coluna alvo.
        target : str
            Nome da coluna alvo (default `"Risco"` -- a categoria de risco derivada dos clusters,
            ver `06_clusterizacao_risco.ipynb`).

        Métodos
        -------
        classify(modelo)
            Treina e avalia o modelo especificado.
    """

    def __init__(self, df: pd.DataFrame, target: str = "Risco") -> None:
        self.df = df
        self.target = target
        self.X = df.drop(target, axis=1)
        self.y = df[target]

        return None

    def __matrix_confusao(self, grid_search: GridSearchCV) -> None:
        """
        Exibe a matriz de confusão do melhor modelo encontrado pelo GridSearchCV.
            Parâmetros
            ----------
            grid_search : GridSearchCV
                Objeto GridSearchCV já ajustado com os dados.
        """
        grid_search.fit(self.X, self.y)

        # 10. Avaliação no conjunto de Teste
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X)

        accuracy = accuracy_score(self.y, y_pred)
        cm = confusion_matrix(self.y, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.y.value_counts().index,
        )
        disp.plot(cmap=plt.cm.Greens)
        plt.title(f"Accuracy {accuracy:.2f}")
        plt.show()
        print("\nClassification Report:")
        print(classification_report(self.y, y_pred))
        return None

    def __retornar_metricas(
        self, grid_search: GridSearchCV
    ) -> tuple[pd.DataFrame, np.ndarray]:
        grid_search.fit(self.X, self.y)

        # 10. Avaliação no conjunto de Teste
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X)

        #  Acurácia
        acc = accuracy_score(self.y, y_pred)

        # Precision -- "weighted" pondera pelo suporte de cada classe, mais
        # honesto que "macro" quando o alvo é desbalanceado (esperado aqui,
        # ver nota do módulo sobre "Baixo" como classe majoritária)
        precision = precision_score(self.y, y_pred, average="weighted")

        # Recall
        recall = recall_score(self.y, y_pred, average="weighted")

        # F1-score
        f1 = f1_score(self.y, y_pred, average="weighted")

        retorno = pd.DataFrame(
            {
                "Acurácia": [acc],
                "Precision": [precision],
                "Recall": [recall],
                "F1-Score": [f1],
            }
        ).round(4)

        return retorno, y_pred

    def __metricas_pontuais(self, grid_search: GridSearchCV) -> None:
        """
        Exibe os melhores hiperparâmetros e a acurácia do melhor modelo
        encontrado pelo GridSearchCV.
            Parâmetros
            ----------
            grid_search : GridSearchCV
                Objeto GridSearchCV já ajustado com os dados.
        """
        # 10. Treinamento Final no Dataset Completo
        # Após validar a capacidade de generalização via Nested CV, ajustamos
        # o GridSearch nos dados totais

        print("--- Treinando modelo com Pipeline de pré-processamento ---")
        grid_search.fit(self.X, self.y)

        # 9. Exibição dos Resultados
        print("\n--- Melhores Resultados do Grid Search ---")
        print(f"Melhores hiperparâmetros: {grid_search.best_params_}")
        print(f"Melhor acurácia (CV): {grid_search.best_score_:.4f}")

        # 10. Avaliação no conjunto de Teste
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X)

        print("\n--- Desempenho no Conjunto de Teste ---")
        print(f"Acurácia final: {accuracy_score(self.y, y_pred):.4f}\n")

        return None

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
        aplica One-Hot Encoding em variáveis categóricas.
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
                    (self.df.select_dtypes(include=["number"]).columns.drop(
                        self.target, errors="ignore"
                    )),
                ),
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    (
                        self.df.drop(self.target, axis=1)
                        .select_dtypes(include=["object"])
                        .columns
                    ),
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
        # KNeighborsClassifier não tem `class_weight` nativo -- se o
        # desbalanceamento pesar aqui, a saída é reamostragem (SMOTE) antes
        # do fit, não um parâmetro deste estimador
        pipeline = Pipeline(
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
        # class_weight="balanced" -- pondera a função de perda pelo inverso
        # da frequência de cada classe, pra "Baixo" (esperado majoritário,
        # ver docstring do módulo) não dominar sozinho a fronteira de decisão
        pipeline = Pipeline(
            [
                ("preprocessor", self.__preprocessador()),
                ("var_threshold", VarianceThreshold(threshold=1e-4)),
                (
                    "feature_selection",
                    SelectKBest(score_func=f_classif, k=min(10, self.X.shape[1])),
                ),
                ("svm", SVC(random_state=42, class_weight="balanced")),
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
        # GradientBoostingClassifier não tem `class_weight` nativo (só
        # `sample_weight` em .fit(), que o Pipeline/GridSearchCV não passa
        # por padrão) -- mesma ressalva de KNN acima
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
        self, modelo: str, n_splits: int = 3
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Treina e avalia o modelo especificado.
            Parâmetros
            ----------
            modelo : str
                Nome do modelo a ser treinado. Opções:
                                'knn', 'svm', 'rf', 'gbm', 'nb', 'nn'.
            n_splits : int, opcional
                Número de divisões para a validação cruzada (default é 3).
            Retorna
                return retorno, grid_search.predict(self.X)
                pd.DataFrame
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
        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="accuracy",
            n_jobs=-1,
        )

        self.__metricas_pontuais(grid_search=grid_search)
        self.__matrix_confusao(grid_search=grid_search)
        self.__variaveis_selecionadas(grid_search=grid_search)

        retorno = self.__retornar_metricas(grid_search=grid_search)

        return retorno
