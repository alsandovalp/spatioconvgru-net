import warnings
from data.data_preprocessing import DataPreprocessing
from features.feature_engineering import FeatureEngineering
from models.models import Models
from finetuning.fine_tuning import FineTuning
warnings.filterwarnings("ignore")

def main():   
    
    """Función principal para ejecutar el entrenamiento del modelo"""
    
    #----- LECTURA Y PREPROCESAMIENTO DATOS-----#
    processor = DataPreprocessing()
    # Ejecutar el pipeline de preprocesamiento completo
    utam_col, zat_col = processor.run_preprocessing(r"weather_stations_bgta\EM_BGTA_2019.shp",
                                                    r"tmau\TMAU.shp",
                                                    r"taz\TAZ.shp",
                                                    r"precipitation\bgta_precipitation.csv",
                                                    r"traffic_crashes\bgta_2019_traffic_crashes.shp")
    # Exportar en golden datasets definitivos
    processor.export_to_golden(utam_col, zat_col)
    #----- INGENIERIA DE CARACTERISTICAS-----# 
    data_splits = FeatureEngineering().feature_engineer_pipeline(utam_col)
    #----- ENTRENAMIENTO DEL MODELO-----# 
    models = Models()
    combined_model = models.pipeline_training_models(data_splits, epochs=1, batch_size=128)
    # Exportar en golden mejor modelo 
    models.export_models()
    #----- FINE TUNNING MODELO PARA DATASET ZAT --------#
    zat_col_0 = zat_col[zat_col['TC_I_ZAT']!=0] # Sobre registros con valor diferente de 0 en TC_I_ZAT
    fine_tuning = FineTuning()
    fine_tuned_model, metrics_0 = fine_tuning.run_finetuning_pipeline(zat_col_0, pre_trained_model=combined_model, epochs=1, batch_size=128)
    fine_tuning.predict_and_evaluate(zat_col, fine_tuned_model) # Sobre totalidad de registros   

if __name__ == "__main__":
    main()
