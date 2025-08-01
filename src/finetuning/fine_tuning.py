import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

class FineTuning:
    def __init__(self, export_dir="pry-spatioconvgrunet-2025/models"):
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

    # ---------------------- TRANSFORMACIÓN ZAT ----------------------
    def transform_data(self, df: pd.DataFrame):
        numeric_t1 = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11_X12','X13','X14']
        numeric_t2 = ['X15']
        nominal = ['X16']
        numeric_y = ['TC_I_ZAT']

        transformer_x_t1 = ColumnTransformer([("numeric_t1", StandardScaler(), numeric_t1)])
        transformer_x_t2 = ColumnTransformer([("numeric_t2", StandardScaler(), numeric_t2),
                                              ("nominal", OneHotEncoder(), nominal)])
        transformer_tc_i_zat = ColumnTransformer([("numeric_y", StandardScaler(), numeric_y)], remainder='drop')

        x_t1 = transformer_x_t1.fit_transform(df)
        x_t2 = transformer_x_t2.fit_transform(df)
        y = transformer_tc_i_zat.fit_transform(df)

        return x_t1, x_t2, y

    # ---------------------- SPLIT Y RESHAPE ----------------------
    def split_and_reshape(self, x_t1, x_t2, y):
        idx = len(x_t1)
        idx_train = int(0.7 * idx)
        idx_val = int(0.8 * idx)

        # Splits
        X_train_t1, X_val_t1, X_test_t1 = x_t1[:idx_train], x_t1[idx_train:idx_val], x_t1[idx_val:]
        X_train_t2, X_val_t2, X_test_t2 = x_t2[:idx_train], x_t2[idx_train:idx_val], x_t2[idx_val:]
        y_train, y_val, y_test = y[:idx_train], y[idx_train:idx_val], y[idx_val:]

        # Reshape
        x_t1_train = X_train_t1.reshape(X_train_t1.shape[0], X_train_t1.shape[1], 1)
        x_t1_val = X_val_t1.reshape(X_val_t1.shape[0], X_val_t1.shape[1], 1)
        x_t1_test = X_test_t1.reshape(X_test_t1.shape[0], X_test_t1.shape[1], 1)

        x_t2_train = X_train_t2.reshape(X_train_t2.shape[0], X_train_t2.shape[1], 1)
        x_t2_val = X_val_t2.reshape(X_val_t2.shape[0], X_val_t2.shape[1], 1)
        x_t2_test = X_test_t2.reshape(X_test_t2.shape[0], X_test_t2.shape[1], 1)

        y_train_r = y_train.reshape(y_train.shape[0], 1, 1, 1, y_train.shape[-1])
        y_val_r = y_val.reshape(y_val.shape[0], 1, 1, 1, y_val.shape[-1])
        y_test_r = y_test.reshape(y_test.shape[0], 1, 1, 1, y_test.shape[-1])

        return (x_t1_train, x_t1_val, x_t1_test,
                x_t2_train, x_t2_val, x_t2_test,
                y_train_r, y_val_r, y_test_r,
                y_train, y_val, y_test)

    # ---------------------- MODELO FINE TUNNING ----------------------
    def create_finetuned_model(self, pre_trained_model):
        # Congelar capas del modelo base
        for layer in pre_trained_model.layers:
            layer.trainable = False

        # Agregar nuevas capas
        new_layers = Dense(2048, activation='relu')(pre_trained_model.output)
        new_layers = Dropout(0.5)(new_layers)
        new_layers = Dense(1024, activation='relu')(new_layers)
        new_layers = Dropout(0.5)(new_layers)
        new_layers = Dense(512, activation='relu')(new_layers)
        new_layers = Dropout(0.5)(new_layers)
        new_output = Dense(1, activation='exponential')(new_layers)

        fine_tuned_model = Model(inputs=pre_trained_model.inputs, outputs=new_output)
        fine_tuned_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape'])
        return fine_tuned_model

    # ---------------------- REGRESSION METRICS ----------------------
    def regression_report(self, y_real, y_pred):
        return pd.DataFrame({
            'MSE': [mean_squared_error(y_real, y_pred)],
            'MAE': [mean_absolute_error(y_real, y_pred)],
            'MAPE': [mean_absolute_percentage_error(y_real, y_pred) * 100],
            'R²': [r2_score(y_real, y_pred)]
        })

    def export_finetuned_model(self, fine_tuned_model, model_name="fine_tuned_spatioconvgru-net_model"):
        """
        Exporta la arquitectura del modelo (JSON) y sus pesos (.h5) en la carpeta 'models' del proyecto.
        """
        if fine_tuned_model is None:
            raise ValueError("No hay modelo fine-tuned para exportar.")

        # Definir carpeta de exportación fija
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        export_dir = os.path.join(project_root, "models")

        # Crear carpeta si no existe
        os.makedirs(export_dir, exist_ok=True)

        # Guardar arquitectura
        model_json = fine_tuned_model.to_json()
        json_path = os.path.join(export_dir, f"{model_name}.json")
        with open(json_path, "w") as json_file:
            json_file.write(model_json)

        # Guardar pesos
        weights_path = os.path.join(export_dir, f"{model_name}.weights.h5")
        fine_tuned_model.save_weights(weights_path)

        print(f"Modelo fine-tuned exportado en:\n- {json_path}\n- {weights_path}")


    # ---------------------- PIPELINE COMPLETO ----------------------
    def run_finetuning_pipeline(self, zat_df, pre_trained_model, epochs=100, batch_size=64, load_weights_path=None):
        print("Iniciando Fine-Tuning para ZAT...")
        
        # 1. Transformación y Split
        x_t1, x_t2, y = self.transform_data(zat_df)
        splits = self.split_and_reshape(x_t1, x_t2, y)

        (x_t1_train, x_t1_val, x_t1_test,
        x_t2_train, x_t2_val, x_t2_test,
        y_train_r, y_val_r, y_test_r,
        y_train, y_val, y_test) = splits

        # 2. Crear modelo fine-tuned
        fine_tuned_model = self.create_finetuned_model(pre_trained_model)

        # ✅ Si hay pesos previos, cargarlos
        if load_weights_path and os.path.exists(load_weights_path):
            print(f"📂 Cargando pesos previos desde: {load_weights_path}")
            fine_tuned_model.load_weights(load_weights_path)

        # 3. Callbacks
        best_model_path = os.path.join(self.export_dir, "fine_tuned_spatioconvgru-net_model.weights.h5")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor="val_loss",
                                            save_best_only=True, save_weights_only=True, mode="min"),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, min_delta=0.0001)
        ]

        # 4. Entrenamiento
        history = fine_tuned_model.fit(
            [x_t1_train, x_t2_train, y_train_r], y_train,
            validation_data=([x_t1_val, x_t2_val, y_val_r], y_val),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks
        )

        # 5. Predicciones y métricas
        preds_train = fine_tuned_model.predict([x_t1_train, x_t2_train, y_train_r])
        preds_val = fine_tuned_model.predict([x_t1_val, x_t2_val, y_val_r])
        preds_test = fine_tuned_model.predict([x_t1_test, x_t2_test, y_test_r])

        metrics = {
            "train": self.regression_report(y_train.ravel(), preds_train.ravel()),
            "val": self.regression_report(y_val.ravel(), preds_val.ravel()),
            "test": self.regression_report(y_test.ravel(), preds_test.ravel())
        }

        # ✅ 6. Exportar el modelo fine-tuned
        self.export_finetuned_model(fine_tuned_model)
        print("Fine-Tuning completado.")
        return fine_tuned_model, metrics
    

    def predict_and_evaluate(self, zat_df, fine_tuned_model):
        print("Ejecutando predicción sobre TODOS los registros ZAT...")

        # 1. Transformar y fragmentar datos
        x_t1, x_t2, y = self.transform_data(zat_df)
        splits = self.split_and_reshape(x_t1, x_t2, y)

        (x_t1_train, x_t1_val, x_t1_test,
        x_t2_train, x_t2_val, x_t2_test,
        y_train_r, y_val_r, y_test_r,
        y_train, y_val, y_test) = splits

        # 2. Predicciones
        preds_train = fine_tuned_model.predict([x_t1_train, x_t2_train, y_train_r])
        preds_val = fine_tuned_model.predict([x_t1_val, x_t2_val, y_val_r])
        preds_test = fine_tuned_model.predict([x_t1_test, x_t2_test, y_test_r])

        # 3. Métricas
        metrics = {
            "train": self.regression_report(y_train.ravel(), preds_train.ravel()),
            "val": self.regression_report(y_val.ravel(), preds_val.ravel()),
            "test": self.regression_report(y_test.ravel(), preds_test.ravel())
        }
        print("Métricas sobre TODOS los registros:\n", metrics)
        print("Predicción y evaluación completadas.")
        return metrics


