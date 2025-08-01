from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple
import pandas as pd
import numpy as np
from typing import Dict, Any

class FeatureEngineering:
    def __init__(self):
        # Definición de columnas por tipo
        self.numeric_t1 = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11_X12','X13','X14']
        self.numeric_t2 = ['X15']
        self.nominal = ['X16']
        self.numeric_y = ['TC_I_UTAM']
        self.counts = ['TC_COUNTS']

        # Inicialización de transformadores
        self.transformer_x_t1 = ColumnTransformer(
            transformers=[
                ("numeric_t1", StandardScaler(), self.numeric_t1),
            ]
        )

        self.transformer_x_t2 = ColumnTransformer(
            transformers=[
                ("numeric_t2", StandardScaler(), self.numeric_t2),
                ("nominal", OneHotEncoder(), self.nominal)
            ]
        )

        self.transformer_tc_i_utam = ColumnTransformer(
            transformers=[
                ("numeric_y", StandardScaler(), self.numeric_y)
            ],
            remainder='drop'
        )

    def get_data_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aplica las transformaciones a las variables exógenas y endógenas.

        Args:
            df (pd.DataFrame): DataFrame procesado con columnas originales.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - x_t1: Variables Type I transformadas.
                - x_t2: Variables Type II + nominal transformadas.
                - tc_i_utam_tr: Variable endógena transformada.
        """
        # Aplicar transformaciones
        x_t1 = self.transformer_x_t1.fit_transform(df)
        x_t2 = self.transformer_x_t2.fit_transform(df)
        tc_i_utam_tr = self.transformer_tc_i_utam.fit_transform(df)

        print("Transformaciones completadas:")
        print(f" - x_t1 shape: {x_t1.shape}")
        print(f" - x_t2 shape: {x_t2.shape}")
        print(f" - tc_i_utam_tr shape: {tc_i_utam_tr.shape}")
        return x_t1, x_t2, tc_i_utam_tr
    
    def split_data(self, x_t1: np.ndarray, x_t2: np.ndarray, tc_i_utam_tr: np.ndarray) -> Dict[str, Any]:
        """
        Divide los datos en Train (70%), Val (10%) y Test (20%) para X (Type I y II) e y (Type III).

        Args:
            x_t1 (np.ndarray): Variables exógenas Type I transformadas.
            x_t2 (np.ndarray): Variables exógenas Type II transformadas.
            tc_i_utam_tr (np.ndarray): Variable endógena transformada.

        Returns:
            Dict[str, Any]: Diccionario con los splits para X e y.
        """
        utam_index = len(x_t1)
        idx_train = int(0.7 * utam_index)
        idx_val = int(0.8 * utam_index)

        # Exógenas
        X_train_t1, X_train_t2 = x_t1[:idx_train], x_t2[:idx_train]
        X_val_t1, X_val_t2 = x_t1[idx_train:idx_val], x_t2[idx_train:idx_val]
        X_test_t1, X_test_t2 = x_t1[idx_val:], x_t2[idx_val:]

        # Endógena
        y_train = tc_i_utam_tr[:idx_train]
        y_val = tc_i_utam_tr[idx_train:idx_val]
        y_test = tc_i_utam_tr[idx_val:]

        print(f"División completada:")
        print(f" - Train: {X_train_t1.shape[0]} filas")
        print(f" - Val: {X_val_t1.shape[0]} filas")
        print(f" - Test: {X_test_t1.shape[0]} filas")

        return {
        "X_train_t1": X_train_t1,
        "X_val_t1": X_val_t1,
        "X_test_t1": X_test_t1,
        "X_train_t2": X_train_t2,
        "X_val_t2": X_val_t2,
        "X_test_t2": X_test_t2,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
        }   
    
    def feature_engineer_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Ejecuta:
        - Transformación de variables
        - División Train/Val/Test
        Retorna dict con splits.
        """
        x_t1, x_t2, y = self.get_data_features(df)
        splits = self.split_data(x_t1, x_t2, y)
        print("Ejecución exitosa de la etapa: Ingenieria de caracteristicas")
        return splits