import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, MaxPooling1D, Flatten, Dense, GRU, Concatenate, ConvLSTM2D
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf

class Models:
    def __init__(self):
        self.model1 = None
        self.model2 = None
        self.model3 = None
        self.combined_model = None

    # ---------------------- MODELOS BASE ----------------------
    def build_model1(self, input_shape: Tuple[int]) -> Model:
        inp = Input(shape=input_shape)
        x = Conv1D(512, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(inp)
        x = BatchNormalization()(x)
        x = Conv1D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.5)(x)
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = Dropout(0.5)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        out = Dense(1)(x)
        self.model1 = Model(inputs=inp, outputs=out)
        return self.model1

    def build_model2(self, input_shape: Tuple[int]) -> Model:
        inp = Input(shape=input_shape)
        x = GRU(512, return_sequences=True, kernel_regularizer=l2(0.01))(inp)
        x = BatchNormalization()(x)
        x = GRU(256)(x)
        x = Dropout(0.5)(x)
        out = Dense(1)(x)
        self.model2 = Model(inputs=inp, outputs=out)
        return self.model2

    def build_model3(self, input_shape: Tuple[int]) -> Model:
        inp = Input(shape=input_shape)
        x = ConvLSTM2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same',
                       return_sequences=True, kernel_regularizer=l2(0.01))(inp)
        x = Dropout(0.5)(x)
        x = ConvLSTM2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Flatten()(x)
        out = Dense(1)(x)
        self.model3 = Model(inputs=inp, outputs=out)
        return self.model3

    def build_combined_model(self) -> Model:
        combined_input = Concatenate()([self.model1.output, self.model2.output, self.model3.output])
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(combined_input)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        prediction = Dense(1, activation='linear')(x)
        self.combined_model = Model(inputs=[self.model1.input, self.model2.input, self.model3.input],
                                    outputs=prediction)
        return self.combined_model

    # ---------------------- REFORMATEAR DATOS ----------------------
    def reshape_inputs(self, X_train_t1, X_test_t1, X_val_t1,
                       X_train_t2, X_test_t2, X_val_t2,
                       y_train, y_test, y_val):
        x_t1_train = X_train_t1.reshape(X_train_t1.shape[0], X_train_t1.shape[1], 1)
        x_t1_test = X_test_t1.reshape(X_test_t1.shape[0], X_test_t1.shape[1], 1)
        x_t1_val = X_val_t1.reshape(X_val_t1.shape[0], X_val_t1.shape[1], 1)

        x_t2_train = X_train_t2.reshape(X_train_t2.shape[0], X_train_t2.shape[1], 1)
        x_t2_test = X_test_t2.reshape(X_test_t2.shape[0], X_test_t2.shape[1], 1)
        x_t2_val = X_val_t2.reshape(X_val_t2.shape[0], X_val_t2.shape[1], 1)

        y_train_r = y_train.reshape(y_train.shape[0], 1, 1, 1, y_train.shape[-1])
        y_test_r = y_test.reshape(y_test.shape[0], 1, 1, 1, y_test.shape[-1])
        y_val_r = y_val.reshape(y_val.shape[0], 1, 1, 1, y_val.shape[-1])

        return (x_t1_train, x_t1_test, x_t1_val,
                x_t2_train, x_t2_test, x_t2_val,
                y_train_r, y_test_r, y_val_r)    

    # ---------------------- FORECASTING ----------------------
    def forecast(self, reshaped_inputs: Tuple, dataset_type: str = "test") -> np.ndarray:
        x_t1, x_t2, y_data = reshaped_inputs
        preds = self.combined_model.predict([x_t1, x_t2, y_data])
        print(f"✅ Predicciones completadas para {dataset_type} -> Shape: {preds.shape}")
        return preds
    # ---------------------- MÉTRICAS ----------------------
    def regression_report(self, y_real: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        mse = mean_squared_error(y_real, y_pred)
        mae = mean_absolute_error(y_real, y_pred)
        mape = mean_absolute_percentage_error(y_real, y_pred) * 100
        r2 = r2_score(y_real, y_pred)
        report = pd.DataFrame({
            'MSE': [mse],
            'MAE': [mae],
            'MAPE': [mape],
            'R²': [r2]
        })
        return report

    # ---------------------- PIPELINE COMPLETO ----------------------
    def pipeline_training_models(self, splits: Dict, epochs=100, batch_size=64):
        """
        Ejecuta el pipeline completo:
        - Reformatea los datos
        - Construye modelos
        - Compila y entrena el modelo combinado
        """
        # 1. Reformatear entradas
        (x_t1_train, x_t1_test, x_t1_val,
         x_t2_train, x_t2_test, x_t2_val,
         y_train_r, y_test_r, y_val_r) = self.reshape_inputs(
            splits["X_train_t1"], splits["X_test_t1"], splits["X_val_t1"],
            splits["X_train_t2"], splits["X_test_t2"], splits["X_val_t2"],
            splits["y_train"], splits["y_test"], splits["y_val"]
        )

        # 2. Construir modelos
        self.build_model1((x_t1_train.shape[1], x_t1_train.shape[2]))
        self.build_model2((x_t2_train.shape[1], x_t2_train.shape[2]))
        self.build_model3((y_train_r.shape[1], y_train_r.shape[2], y_train_r.shape[3], y_train_r.shape[4]))
        self.build_combined_model()

        # 3. Compilar modelo combinado
        self.combined_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape'])

        # 4. Definir callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, min_delta=0.0001)
        ]

        # 5. Entrenar modelo
        history = self.combined_model.fit(
            [x_t1_train, x_t2_train, y_train_r],
            splits["y_train"],
            validation_data=([x_t1_val, x_t2_val, y_val_r], splits["y_val"]),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks
        )
        # 6. Forecast & métricas
        preds_train = self.forecast((x_t1_train, x_t2_train, y_train_r), "train")
        preds_val = self.forecast((x_t1_val, x_t2_val, y_val_r), "val")
        preds_test = self.forecast((x_t1_test, x_t2_test, y_test_r), "test")

        self.metrics_report = {
            "train": self.regression_report(splits["y_train"].ravel(), preds_train.ravel()),
            "val": self.regression_report(splits["y_val"].ravel(), preds_val.ravel()),
            "test": self.regression_report(splits["y_test"].ravel(), preds_test.ravel())
        }

        print("\n Métricas obtenidas:")
        print("Train:\n", self.metrics_report["train"])
        print("Validation:\n", self.metrics_report["val"])
        print("Test:\n", self.metrics_report["test"])

        return self.combined_model

    # ---------------------- EXPORTAR MODELO ----------------------
    def export_models(self):
        """
        Exporta arquitectura (JSON) y pesos (.h5) en la carpeta 'models' del proyecto.
        """
        if self.combined_model is None:
            raise ValueError("No hay modelo entrenado. Ejecute pipeline_training_models primero.")

        # Definir carpeta de exportación fija: pry-spatioconvgrunet-2025/models
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        export_dir = os.path.join(project_root, "models")

        # Crear carpeta si no existe
        os.makedirs(export_dir, exist_ok=True)

        # Guardar arquitectura
        model_json = self.combined_model.to_json()
        json_path = os.path.join(export_dir, "spatioconvgru-net_model_tmau.json")
        with open(json_path, "w") as json_file:
            json_file.write(model_json)

        # Guardar pesos
        weights_path = os.path.join(export_dir, "spatioconvgru-net_model_tmau.weights.h5")
        self.combined_model.save_weights(weights_path)

        print(f"Modelo exportado en:\n- {json_path}\n- {weights_path}")

