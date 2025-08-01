# SpatioConvGru-Net: Predicting Urban Traffic Accidents in Bogotá  
**Repositorio experimental que respalda la investigación “SpatioConvGRU-Net for Short-Term Traffic Crash Frequency Prediction in Bogotá: A Macroscopic Spatiotemporal Deep Learning Approach with Urban Factors”**  

Este repositorio contiene el soporte experimental del artículo científico *“SpatioConvGRU-Net for Short-Term Traffic Crash Frequency Prediction in Bogotá: A Macroscopic Spatiotemporal Deep Learning Approach with Urban Factors”*, ver en https://www.mdpi.com/2673-3951/6/3/71, orientado a abordar uno de los retos más complejos en la **seguridad vial urbana**: la predicción de siniestros de tránsito en entornos metropolitanos con alta densidad poblacional y dinámicas intensas de movilidad, como la ciudad de Bogotá.  

La propuesta introduce **SpatioConvGru-Net**, una arquitectura híbrida de *deep learning* diseñada para modelar **dependencias espaciotemporales** en series de tiempo con granularidad horaria. El modelo integra:  

- **CNN** (*Convolutional Neural Networks*) para la extracción de patrones espaciales desde estructuras geográficas.  
- **GRU** (*Gated Recurrent Units*) para la captura de secuencias temporales.  
- **ConvLSTM** (*Convolutional Long Short-Term Memory*) como conector espaciotemporal, combinando convoluciones con dinámicas recurrentes.  

El desarrollo experimental se implementó en **Python**, utilizando **Keras** sobre **TensorFlow**, y se estructura bajo una arquitectura modular que soporta todo el ciclo:  
- Preprocesamiento y limpieza de datos.  
- Ingeniería de características (espaciales y temporales).  
- Entrenamiento y ajuste fino (*fine-tuning*) del modelo.  
- Exportación y documentación de resultados.  

---

## Table of Contents

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura proyecto](#estructura-proyecto)
- [Reproducibilidad](#reproducibilidad)
- [Autores](#autores)

---

## Características

- Implementación de la arquitectura **SpatioConvGru-Net** para predicción de siniestros en granularidad horaria.  
- Integración de múltiples flujos: **CNN + GRU + ConvLSTM**.  
- Pipeline modular orientado a experimentación reproducible:  
  - **Bronze**: datos brutos.  
  - **Silver**: datos transformados y limpios.  
  - **Gold**: datasets definitivos y modelos exportados.  
- Entrenamiento base + **Fine-Tuning** sobre datos heterogéneos.  
- Documentación clara del proceso experimental para soporte científico. 
- **Nota**: Debido a que los archivos de datos exceden el límite permitido por GitHub, estos no se encuentran en el repositorio. Para acceder a ellos, solicítelos a través del correo electrónico **alsandoval@unal.edu.co** 

---

## Requisitos

- Python >= 3.9  
- Bibliotecas principales:  
  - `tensorflow`  
  - `keras`  
  - `pandas`, `numpy`  
  - `scikit-learn`  
  - `geopandas`  
- Administrador de entornos recomendado: **conda** o **mamba**  

---

## Instalación

### Creación de entorno virtual desde YAML

```bash
# Clonar repositorio
git clone https://github.com/alsandoval/spatioconvgru-net.git
cd pry-spatioconvgrunet-2025

# Crear entorno virtual con conda
conda env create -f environment.yml

# Activar entorno
conda activate spatioconvgrunet
```

---

## Uso

Ejecutar el pipeline completo (preprocesamiento → entrenamiento → exportación):

```bash
python main.py
```

El script `main.py` permite la ejecución modular del flujo experimental:  

```python
# Ejecutar pipeline completo
processor = DataPreprocessing()
utam_col, zat_col = processor.run_preprocessing(...)
processor.export_to_golden(utam_col, zat_col)

# Ingeniería de características y entrenamiento
data_splits = FeatureEngineering().feature_engineer_pipeline(utam_col)
models = Models()
combined_model = models.pipeline_training_models(data_splits, epochs=10, batch_size=64)
models.export_models()

# Fine-tuning con datos ZAT
fine_tuning = FineTuning()
fine_tuned_model, metrics = fine_tuning.run_finetuning_pipeline(zat_col, pre_trained_model=combined_model)
```

---

## Estructura proyecto

```
spatioconvgrunet/
├── config/
│   ├── experiment.yml
├── data/
│   ├── bronze/
│   │   ├── precipitation/
│   │   |   ├── bgta_precipitation.csv
│   │   ├── taz/
│   │   |   ├── TAZ.shp
│   │   ├── tmau/
│   │   |   ├── TMAU.shp
│   │   ├── traffic_crashes/
│   │   |   ├── bgta_2019_traffic_crashes.shp
│   │   ├── weather_stations_bgta/
│   │   |   ├── EM_BGTA_2019.shp
│   ├── silver/
│   ├── golden/
│   │   ├── taz.parquet
│   │   ├── tmau.parquet
├── models/
│   ├── spatioconvgru-net_model_tmau.weights.h5
│   ├── spatioconvgru-net_model_tmau.json
│   ├── fine_tuned_spatioconvgru-net_model.weights.h5
│   ├── fine_tuned_spatioconvgru-net_model.json
├── notebooks/
│   ├── experiments_spatioconvgru.ipynb
├── src/
│   ├── data/
│   │   ├── data_preprocessing.py
│   ├── preprocessing/
│   │   ├── feature_engineering.py
│   ├── modesl/
│   │   ├── models.py
│   ├── finetuning/
│   │   ├── fine_tuning.py
│   ├── utils/
│   │   ├── functions.py
│   └── main.py
└── README.md
```

---

## Reproducibilidad

Para garantizar la replicación de los resultados:  
- **Dataset**: Repositorio incluye enlaces a datos brutos y scripts para su procesamiento.  
- **Configuración experimental**: definida en `config/experiment.yml`.  
- **Hardware recomendado**:  
  - GPU: NVIDIA Tesla T4 o superior.  
  - RAM: ≥ 16GB.  
  - CUDA 11.2 y cuDNN configurado.  

---

## Autores

**Built by [Alejandro Sandoval Pineda] y colaborador: [Cesar Augusto Pedraza Bonilla]**  
