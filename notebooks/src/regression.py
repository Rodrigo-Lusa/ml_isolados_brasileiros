# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Importando modelos
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV

# Visualização dos dados
import matplotlib.pyplot as plt
import seaborn as sns

# Edição das databases
import pandas as pd
import numpy as np
from util import utils as ut

# Seleção de hiperparâmetros e validação cruzada
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# Estruturação dos dados e pré-processamento
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

# Organização do fluxo de trabalho (Pipeline)
from sklearn.pipeline import Pipeline

# Seleção de variáveis
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# Métricas da regressão
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import spearmanr


class Regression:
    """
    Regressor genérico que encapsula diferentes modelos de
    aprendizado de máquina.
        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame contendo as features e a coluna alvo.

        Métodos
        -------
        regress(modelo)
            Treina e avalia o modelo especificado.
    """

    def __init__(self, df: pd.DataFrame, target: str) -> None:
        self.df = df
        self.target = target  # <- ALTERADO (linha nova)
        self.X = df.drop(target, axis=1)
        self.y = df[target]
        self.modelos_ = {}  # só existe depois do .regress

    def __obter_metricas(
        self, grid_search: GridSearchCV | RandomizedSearchCV, 
        modelo: str
    ) -> tuple[pd.DataFrame, np.ndarray]:
        y_true = self.y
        y_pred = grid_search.predict(self.X)

        df_test = pd.DataFrame()
        df_test["y_true"] = y_true
        df_test["y_pred"] = y_pred

        r_cal, r_cal_p = spearmanr(df_test["y_true"], df_test["y_pred"])
        r2_cal = r2_score(df_test["y_true"], df_test["y_pred"])
        mae_cal = mean_absolute_error(df_test["y_true"], df_test["y_pred"])
        rmse_cal = np.sqrt(mean_squared_error(df_test["y_true"], df_test["y_pred"]))
        mape_cal = mean_absolute_percentage_error(df_test["y_true"], df_test["y_pred"])

        mensagem = f"""
    R Spearman: {r_cal:.4f}
    R p-value: {r_cal_p:.4f}
    R2: {r2_cal:.4f}
    MAE: {mae_cal:.4f}
    RMSE: {rmse_cal:.4f}
    MAPE: {mape_cal:.4f}
        """

        p = sns.jointplot(
            x="y_true",
            y="y_pred",
            kind="reg",
            marginal_ticks=True,
            data=df_test,
        )
        p.fig.suptitle(f"Correlação predito / observado - {modelo}", y=1.02)
        plt.text(
            0.05,
            0.99,
            mensagem,
            transform=p.ax_joint.transAxes,
            fontsize=12,
            ha="left",
            va="top",
        )
        plt.xlabel("Observado")
        plt.ylabel("Predito")
        plt.show()

        metricas = pd.DataFrame(
            {
                "R Spearman": [r_cal],
                "R p-value": [r_cal_p],
                "R2": [r2_cal],
                "MAE": [mae_cal],
                "RMSE": [rmse_cal],
                "MAPE": [mape_cal],
            }
        ).round(4)

        return metricas, y_pred

    def __variaveis_selecionadas(
        self, grid_search: GridSearchCV | RandomizedSearchCV
    ) -> pd.DataFrame:
        best_pipeline = grid_search.best_estimator_
        feat_names_orig = best_pipeline["preprocessor"].get_feature_names_out()

        mask_variance = best_pipeline["var_threshold"].get_support()
        features_after_variance = feat_names_orig[mask_variance]

        mask_kbest = best_pipeline["feature_selection"].get_support()
        selected_features = features_after_variance[mask_kbest]

        m = "Total de variáveis "
        print(f"{m}originais pré-processadas: {len(feat_names_orig)}")
        print(f"{m}após VarianceThreshold:    {len(features_after_variance)}")
        print(f"{m}selecionadas no modelo:    {len(selected_features)}")

        print("\n--- Lista das Variáveis Selecionadas ---")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i}. {feature}")

        final_mask = mask_variance.copy()
        final_mask[mask_variance] = mask_kbest

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
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    (
                        self.df.drop(self.target, axis=1)  # <- ALTERADO (era "IFN-γ")
                        .select_dtypes(include=["number"])
                        .columns
                    ),
                ),
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),  # <- ALTERADO
                    (self.df.select_dtypes(include=["object"]).columns),
                    ),
            ]
        )
        return preprocessor

    def __regressor_classify(self) -> tuple[Pipeline, dict]:
        pipeline = Pipeline(
            [
                ("preprocessor", self.__preprocessador()),
                ("var_threshold", VarianceThreshold(threshold=1e-4)),
                (
                    "feature_selection",
                    SelectKBest(score_func=f_regression, k=min(10, self.X.shape[1])),
                ),
                ("regressor", LinearRegression()),
            ]
        )

        param_grid = {
            "var_threshold__threshold": [1e-4, 0.01, 0.05],
            "feature_selection__k": [1, 2, "all"],
            "regressor__fit_intercept": [True, False],
            "regressor__positive": [True, False],
        }

        return pipeline, param_grid

    def __svm_classify(self) -> tuple[Pipeline, dict]:
        pipeline = Pipeline(
            [
                ("preprocessor", self.__preprocessador()),
                ("var_threshold", VarianceThreshold(threshold=1e-4)),
                (
                    "feature_selection",
                    SelectKBest(score_func=f_regression, k=min(10, self.X.shape[1])),
                ),
                ("svm", SVR()),
            ]
        )

        param_grid = {
            "var_threshold__threshold": [1e-4, 0.01, 0.05],
            "feature_selection__k": [1, 2, "all"],
            "svm__kernel": ["linear", "rbf"],
            "svm__C": [0.1, 1, 10],
            "svm__epsilon": [0.01, 0.1, 0.2],
        }

        return pipeline, param_grid

    def __bayesian_ridge_classify(self) -> tuple[Pipeline, dict]:
        pipeline = Pipeline(
            [
                ("preprocessor", self.__preprocessador()),
                ("var_threshold", VarianceThreshold(threshold=1e-4)),
                (
                    "feature_selection",
                    SelectKBest(score_func=f_regression, k=min(10, self.X.shape[1])),
                ),
                ("nb", BayesianRidge()),
            ]
        )

        param_grid = {
            "var_threshold__threshold": [1e-4, 0.01, 0.05],
            "feature_selection__k": [1, 2, "all"],
            "nb__max_iter": [100, 300],
            "nb__alpha_1": [1e-6, 1e-4],
            "nb__lambda_1": [1e-6, 1e-4],
        }

        return pipeline, param_grid

    # Transformado em método estático para permitir a serialização (pickle)
    @staticmethod
    def __neural_network_model(
        optimizer="SGD", activation1="relu", activation2="relu"
    ) -> Sequential:
        model = Sequential(
            [
                Dense(64, activation=activation1),
                Dense(32, activation=activation2),
                Dense(1, activation="linear"),
            ]
        )

        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def __neural_network_classify(self) -> tuple[Pipeline, dict]:
        pipeline = Pipeline(
            [
                ("preprocessor", self.__preprocessador()),
                ("var_threshold", VarianceThreshold(threshold=1e-4)),
                (
                    "feature_selection",
                    SelectKBest(score_func=f_regression, k=min(10, self.X.shape[1])),
                ),
                (
                    "to_tensor",
                    FunctionTransformer(
                        func=lambda x: tf.convert_to_tensor(x, dtype=tf.float32)
                    ),
                ),
                ("nn", KerasRegressor(build_fn=self.__neural_network_model, verbose=0)),
            ]
        )

        param_grid = {
            "var_threshold__threshold": [1e-4, 0.01, 0.05],
            "feature_selection__k": [1, 2, "all"],
            "nn__optimizer": ["SGD", "RMSprop", "Adam"],
            "nn__epochs": [10, 100],
            "nn__batch_size": [16, 32],
            "nn__activation1": ["relu", "tanh", "sigmoid", "softmax"],
            "nn__activation2": ["relu", "tanh", "sigmoid", "softmax"],
        }

        return pipeline, param_grid

    def regress(
        self, modelo: str, n_splits: int = 5, n_iter_search: int = 20, n_jobs: int = -1
    ) -> tuple[pd.DataFrame, np.ndarray]:
        if modelo == "regressor":
            estimator, param_grid = self.__regressor_classify()
        elif modelo == "svm":
            estimator, param_grid = self.__svm_classify()
        elif modelo == "bayesian_ridge":
            estimator, param_grid = self.__bayesian_ridge_classify()
        elif modelo == "neural_network":
            estimator, param_grid = self.__neural_network_classify()
        else:
            mensagem = f"Modelo inválido: {modelo}. "
            mensagem += "Escolha 'regressor', 'svm', 'bayesian_ridge'"
            mensagem += " ou 'neural_network'."
            raise ValueError(mensagem)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        if modelo == "neural_network":
            grid_search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=n_iter_search,
                cv=kf,
                scoring="neg_mean_squared_error",
                n_jobs=n_jobs,
                random_state=42,
            )
        else:
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=kf,
                scoring="neg_mean_squared_error",
                n_jobs=n_jobs,
            )

        print("Iniciando o Grid Search com Cross Validation...")
        if modelo == "neural_network":
            early_stopping = EarlyStopping(
                monitor="loss",
                mode="min",
                patience=10,
                min_delta=0.001,
                restore_best_weights=True,
            )
            y_fit = self.y.astype(np.float32)
            grid_search.fit(self.X, y_fit, nn__callbacks=[early_stopping])
        else:
            grid_search.fit(self.X, self.y)

        self.modelos_[modelo] = grid_search  # .predict() em dados novos
        print("Grid Search concluído.")
        print("=" * 50)
        print("Melhores parâmetros encontrados:")
        ut.printDic(grid_search.best_params_)
        print("=" * 50)
        self.__variaveis_selecionadas(grid_search=grid_search)
        metricas, y_pred = self.__obter_metricas(grid_search=grid_search, modelo=modelo)

        return metricas, y_pred