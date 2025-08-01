import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm
from typing import Tuple
from prince import MCA
import numpy as np
import os
from utils.functions import get_output_path

class DataPreprocessing:
    def __init__(self):
        pass

    def get_from_bronze(self,name_1: str, name_2: str, name_3: str,name_4: str, name_5: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Carga cinco DataFrames desde archivos Parquet guardados en la carpeta 'bronze'.

        Args:
            name_1 (str): Nombre del primer archivo sin extensión.
            name_2 (str): Nombre del segundo archivo sin extensión.
            name_3 (str): Nombre del tercer archivo sin extensión.
            name_4 (str): Nombre del cuarto archivo sin extensión.
            name_5 (str): Nombre del cuarto archivo sin extensión.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Los cinco DataFrames cargados.
        """
        try:
            # Carga archivo 1
            if name_1.lower().endswith('.shp'):
                df1 = gpd.read_file(get_output_path("bronze", name_1))
            elif name_1.lower().endswith('.csv'):
                df1 = pd.read_csv(get_output_path("bronze", name_1), sep=",", encoding='latin1', parse_dates=["Fecha"])
            else:
                raise ValueError(f"Formato no soportado para {name_1}")
            # Carga archivo 2
            if name_2.lower().endswith('.shp'):
                df2 = gpd.read_file(get_output_path("bronze", name_2))
            elif name_2.lower().endswith('.csv'):
                df2 = pd.read_csv(get_output_path("bronze", name_2), sep=",", encoding='latin1', parse_dates=["Fecha"])
            else:
                raise ValueError(f"Formato no soportado para {name_2}")
            # Carga archivo 3
            if name_3.lower().endswith('.shp'):
                df3 = gpd.read_file(get_output_path("bronze", name_3))
            elif name_3.lower().endswith('.csv'):
                df3 = pd.read_csv(get_output_path("bronze", name_3), sep=",", encoding='latin1', parse_dates=["Fecha"])
            else:
                raise ValueError(f"Formato no soportado para {name_3}")
            # Carga archivo 4
            if name_4.lower().endswith('.shp'):
                df4 = gpd.read_file(get_output_path("bronze", name_4))
            elif name_4.lower().endswith('.csv'):
                df4 = pd.read_csv(get_output_path("bronze", name_4), sep=",", encoding='latin1', parse_dates=["Fecha"])
            else:
                raise ValueError(f"Formato no soportado para {name_4}")
            # Carga archivo 5
            if name_5.lower().endswith('.shp'):
                df5 = gpd.read_file(get_output_path("bronze", name_5))
            elif name_5.lower().endswith('.csv'):
                df5 = pd.read_csv(get_output_path("bronze", name_5), sep=",", encoding='latin1', parse_dates=["Fecha"])
            else:
                raise ValueError(f"Formato no soportado para {name_5}")            
            print(f"Archivos cargados correctamente desde Bronze:\n -{name_1}\n -{name_2}\n -{name_3}\n -{name_4}\n -{name_5}")

            return df1, df2, df3, df4, df5
        except Exception as e:
            raise RuntimeError(f"Error cargando archivos desde Bronze: {e}")
        

    def calculate_new_variables(self, utam: gpd.GeoDataFrame, zat: gpd.GeoDataFrame,
                                em_bgta_2019: gpd.GeoDataFrame, prec_bgta: pd.DataFrame,
                                traffic_acc: gpd.GeoDataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calcula nuevas variables derivadas a partir de UTAM, ZAT, estaciones meteorológicas,
        precipitaciones y siniestros viales. Retorna datasets finales listos para análisis.
        Args:
            utam (gpd.GeoDataFrame): Polígonos UTAM.
            zat (gpd.GeoDataFrame): Polígonos ZAT.
            em_bgta_2019 (gpd.GeoDataFrame): Puntos de estaciones meteorológicas.
            prec_bgta (pd.DataFrame): Precipitaciones por hora.
            traffic_acc (gpd.GeoDataFrame): Puntos de siniestros viales.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames transformados utam_col y zat_col.
        """
        # ---------------- MCA para X11 y X12 ----------------
        mca_utam = MCA(n_components=8, copy=True, check_input=True, engine='sklearn', random_state=42)
        mca_zat = MCA(n_components=8, copy=True, check_input=True, engine='sklearn', random_state=42)

        mca_utam.fit(utam[['X11', 'X12']])
        mca_zat.fit(zat[['X11', 'X12']])

        eigenvalues_utam = mca_utam.eigenvalues_
        eigenvalues_zat = mca_zat.eigenvalues_

        explained_var_utam = eigenvalues_utam / np.sum(eigenvalues_utam)
        explained_var_zat = eigenvalues_zat / np.sum(eigenvalues_zat)

        coords_utam = mca_utam.row_coordinates(utam[['X11', 'X12']])
        coords_zat = mca_zat.row_coordinates(zat[['X11', 'X12']])

        utam['X11_X12'] = np.sqrt(np.sum((coords_utam.iloc[:, :6] ** 2) * explained_var_utam[:6], axis=1))
        zat['X11_X12'] = np.sqrt(np.sum((coords_zat.iloc[:, :6] ** 2) * explained_var_zat[:6], axis=1))

        # ---------------- Centroides y asignación de estación más cercana ----------------
        utam_cen = gpd.GeoDataFrame(utam.drop('geometry', axis=1), geometry=utam['geometry'].centroid)
        zat_cen = gpd.GeoDataFrame(zat.drop('geometry', axis=1), geometry=zat['geometry'].centroid)

        for idx, point in utam_cen.iterrows():
            nearest_idx = em_bgta_2019.geometry.distance(point.geometry).idxmin()
            for col in em_bgta_2019.columns[1:3]:
                if col != 'geometry':
                    utam.at[idx, col] = em_bgta_2019.at[nearest_idx, col]
                    utam_cen.at[idx, col] = em_bgta_2019.at[nearest_idx, col]

        for idx, point in zat_cen.iterrows():
            nearest_idx = em_bgta_2019.geometry.distance(point.geometry).idxmin()
            for col in em_bgta_2019.columns[1:3]:
                if col != 'geometry':
                    zat.at[idx, col] = em_bgta_2019.at[nearest_idx, col]
                    zat_cen.at[idx, col] = em_bgta_2019.at[nearest_idx, col]

        # ---------------- Expansión temporal (8760 horas = 1 año) ----------------
        fecha_inicio = pd.to_datetime("2019-01-01 00:00:00")
        horas = [fecha_inicio + pd.Timedelta(hours=i) for i in range(8760)]

        def expandir(df, row):
            data = {col: [row[col]] * 8760 for col in df.columns}
            data['FECHA_HORA'] = horas
            return pd.DataFrame(data)

        expanded_utam = utam.apply(lambda r: expandir(utam, r), axis=1)
        expanded_zat = zat.apply(lambda r: expandir(zat, r), axis=1)

        utam_fecha = pd.concat(expanded_utam.to_list(), ignore_index=True)
        zat_fecha = pd.concat(expanded_zat.to_list(), ignore_index=True)

        # ---------------- Integrar precipitaciones ----------------
        prec_bgta_2019 = prec_bgta[pd.to_datetime(prec_bgta['Fecha']).dt.year == 2019]
        utam_fecha['Concatenada'] = utam_fecha['CodigoEsta'].astype(str) + '_' + utam_fecha['FECHA_HORA'].astype(str)
        zat_fecha['Concatenada'] = zat_fecha['CodigoEsta'].astype(str) + '_' + zat_fecha['FECHA_HORA'].astype(str)
        prec_bgta_2019['Concatenada'] = prec_bgta_2019['CodigoEstacion'].astype(str) + '_' + prec_bgta_2019['Fecha'].astype(str)

        utam_fecha['X15'] = utam_fecha['Concatenada'].map(prec_bgta_2019.set_index('Concatenada')['Valor'])
        zat_fecha['X15'] = zat_fecha['Concatenada'].map(prec_bgta_2019.set_index('Concatenada')['Valor'])

        utam_fecha.drop('Concatenada', axis=1, inplace=True)
        zat_fecha.drop('Concatenada', axis=1, inplace=True)

        # ---------------- Iluminación (X16) ----------------
        def clasificar_horas(fecha):
            return "Optima" if 6 <= fecha.hour < 18 else "Limitada"

        utam_fecha['X16'] = utam_fecha['FECHA_HORA'].apply(clasificar_horas)
        zat_fecha['X16'] = zat_fecha['FECHA_HORA'].apply(clasificar_horas)

        # ---------------- Conteo de accidentes ----------------
        utam_tracc = gpd.sjoin(traffic_acc, utam, how='inner', predicate='intersects')
        utam_tracc['FECHA_HORAS'] = pd.to_datetime(utam_tracc['FECHA_H'], format='%d/%m/%Y %H:%M:%S').dt.floor('H')
        utam_tracc_hours = utam_tracc.groupby(['UTAM', 'FECHA_HORAS']).size().reset_index(name='Conteo')

        zat_tracc = gpd.sjoin(traffic_acc, zat, how='inner', predicate='intersects')
        zat_tracc['FECHA_HORAS'] = pd.to_datetime(zat_tracc['FECHA_H'], format='%d/%m/%Y %H:%M:%S').dt.floor('H')
        zat_tracc_hours = zat_tracc.groupby(['ZAT_', 'FECHA_HORAS']).size().reset_index(name='Conteo')

        utam_fecha['Concatenada'] = utam_fecha['UTAM'] + '_' + utam_fecha['FECHA_HORA'].astype(str)
        zat_fecha['Concatenada'] = zat_fecha['ZAT_'] + '_' + zat_fecha['FECHA_HORA'].astype(str)
        utam_tracc_hours['Concatenada'] = utam_tracc_hours['UTAM'] + '_' + utam_tracc_hours['FECHA_HORAS'].astype(str)
        zat_tracc_hours['Concatenada'] = zat_tracc_hours['ZAT_'] + '_' + zat_tracc_hours['FECHA_HORAS'].astype(str)

        utam_fecha['TC_COUNTS'] = utam_fecha['Concatenada'].map(utam_tracc_hours.set_index('Concatenada')['Conteo']).fillna(0)
        zat_fecha['TC_COUNTS'] = zat_fecha['Concatenada'].map(zat_tracc_hours.set_index('Concatenada')['Conteo']).fillna(0)

        utam_fecha.drop('Concatenada', axis=1, inplace=True)
        zat_fecha.drop('Concatenada', axis=1, inplace=True)

        # ---------------- Índices TCIRP ----------------
        utam_fecha['TC_I_UTAM'] = utam_fecha['TC_COUNTS'] / utam_fecha['LEN_KM']
        zat_fecha['TC_I_ZAT'] = zat_fecha['TC_COUNTS'] / zat_fecha['LEN_KM']

        # ---------------- Orden y columnas finales ----------------
        utam_fecha_or = utam_fecha.sort_values(by='FECHA_HORA').reset_index(drop=True)
        zat_fecha_or = zat_fecha.sort_values(by='FECHA_HORA').reset_index(drop=True)

        cols_utam = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11_X12','X13','X14','X15','X16','TC_I_UTAM','TC_COUNTS','LEN_KM','FECHA_HORA','UTAM']
        cols_zat = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11_X12','X13','X14','X15','X16','TC_I_ZAT','TC_COUNTS','LEN_KM','FECHA_HORA','UTAM','ZAT_']

        utam_col = utam_fecha_or[cols_utam]
        zat_col = zat_fecha_or[cols_zat]

        return utam_col, zat_col


    def run_preprocessing(self, name_1: str, name_2: str, name_3: str, name_4: str, name_5: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta el pipeline completo de preprocesamiento:
        - Lee los datos desde Bronze.
        - Calcula nuevas variables.
        - Devuelve los datasets finales para UTAM y ZAT.

        Args:
            name_1 (str): Archivo estaciones meteorológicas (.shp).
            name_2 (str): Archivo polígonos UTAM (.shp).
            name_3 (str): Archivo polígonos ZAT (.shp).
            name_4 (str): Archivo precipitaciones (.csv).
            name_5 (str): Archivo siniestros viales (.shp).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: utam_col y zat_col listos para análisis.
        """
        try:
            # 1. Leer archivos desde Bronze
            em_bgta_2019, utam, zat, prec_bgta, traffic_acc = self.get_from_bronze(
                name_1, name_2, name_3, name_4, name_5
            )

            # 2. Calcular nuevas variables
            utam_final, zat_final = self.calculate_new_variables(
                utam, zat, em_bgta_2019, prec_bgta, traffic_acc
            )

            print("Ejecución exitosa de la etapa: Preprocesaiento de datos")
            return utam_final, zat_final

        except Exception as e:
            raise RuntimeError(f"Error en el pipeline de preprocesamiento: {e}")
        

    def export_to_golden(self, utam_df: pd.DataFrame, zat_df: pd.DataFrame) -> None:
        """
        Exporta los DataFrames procesados a la carpeta Golden fuera de src.

        Args:
            utam_df (pd.DataFrame): DataFrame procesado para UTAM.
            zat_df (pd.DataFrame): DataFrame procesado para ZAT.
        """
        try:
            # Ruta base del proyecto (un nivel arriba de src)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            golden_dir = os.path.join(project_root, "data", "golden")
            # Definir rutas
            utam_path = os.path.join(golden_dir, "tmau.parquet")
            zat_path = os.path.join(golden_dir, "taz.parquet")
            # Exportar
            utam_df.to_parquet(utam_path, index=False, engine="pyarrow")
            zat_df.to_parquet(zat_path, index=False, engine="pyarrow")

            print(f"Exportación exitosa a Golden:\n - {utam_path}\n - {zat_path}")

        except Exception as e:
            raise RuntimeError(f"❌ Error exportando a Golden: {e}")
        
