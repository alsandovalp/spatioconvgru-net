import os
import pandas as pd

# Funciones de utilidad


def get_project_path(*path_parts):
    """
    Retorna la ruta absoluta a un archivo dentro del proyecto.

    Parámetros:
    - path_parts: Partes de la ruta relativas a la raíz del proyecto (ej. "input", "archivo.xlsx")

    Retorna:
    - Ruta absoluta al archivo.
    """
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(base_path, *path_parts)

def get_output_path(subfolder: str, filename: str) -> str:
    """
    Construye y retorna la ruta completa para guardar un archivo en 'output/<subfolder>/filename'.

    Parámetros:
    - subfolder: 'bronze', 'silver' o 'gold'
    - filename: nombre del archivo, ej. 'data.parquet'

    Retorna:
    - Ruta completa donde guardar el archivo
    """
    if subfolder not in {"bronze", "silver", "gold"}:
        raise ValueError("Subcarpeta inválida. Use: 'bronze', 'silver' o 'gold'.")

    base_path = os.path.dirname(os.getcwd())
    output_path = os.path.join(base_path, "data", subfolder)

    os.makedirs(output_path, exist_ok=True)

    return os.path.join(output_path, filename)

def get_input_path(filename: str) -> str:
    """
    Construye y retorna la ruta completa para leer un archivo ubicado directamente en la carpeta 'input/'.

    Parámetros:
    - filename (str): Nombre del archivo, por ejemplo: 'regis01.sas7bdat'

    Retorna:
    - Ruta completa donde se encuentra el archivo.
    """
    base_path = os.path.dirname(os.getcwd())
    input_path = os.path.join(base_path, "input")
    return os.path.join(input_path, filename)

def get_input_path_fix(filename: str) -> str:
    """
    Construye y retorna la ruta completa para leer un archivo ubicado directamente en la carpeta 'input/'.

    Parámetros:
    - filename (str): Nombre del archivo, por ejemplo: 'regis01.sas7bdat'

    Retorna:
    - Ruta completa donde se encuentra el archivo.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_path, "..", ".."))
    input_path = os.path.join(project_root, "input")
    return os.path.join(input_path, filename)

def get_output_paths(subfolder: str, filename: str) -> str:
    """
    Construye y retorna la ruta completa para guardar un archivo en 'output/<subfolder>/filename'.

    Parámetros:
    - subfolder: 'bronze', 'silver' o 'gold'
    - filename: nombre del archivo, ej. 'data.parquet'

    Retorna:
    - Ruta completa donde guardar el archivo
    """
    if subfolder not in {"bronze", "silver", "gold"}:
        raise ValueError("Subcarpeta inválida. Use: 'bronze', 'silver' o 'gold'.")

    base_path = os.path.dirname(os.getcwd())
    output_path = os.path.join(base_path, "output", subfolder)

    os.makedirs(output_path, exist_ok=True)

    filename = str(filename)

    return os.path.join(output_path, filename)
