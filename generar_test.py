# -*- coding: utf-8 -*-
import pandas as pd
import argparse

def generate_test_data(input_file, output_file, test_fraction=0.2, random_seed=42):
    # Cargar el archivo CSV de la plantilla
    df = pd.read_csv(input_file)

    # Obtener una fracción aleatoria de las filas para el conjunto de datos de prueba
    test_data = df.sample(frac=test_fraction, random_state=random_seed)

    # Guardar los datos de prueba en un nuevo archivo CSV
    test_data.to_csv(output_file, index=False)
    print("Se han creado los datos de prueba en '{}'.".format(output_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera un conjunto de datos de prueba a partir de un archivo CSV de plantilla.")
    parser.add_argument("input_file", help="Ruta al archivo CSV de plantilla")
    parser.add_argument("output_file", help="Ruta para guardar el archivo CSV de prueba")
    parser.add_argument("--test_fraction", type=float, default=0.2, help="Fracción de datos a utilizar como prueba (por defecto: 0.2)")
    parser.add_argument("--random_seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad (por defecto: 42)")
    args = parser.parse_args()

    generate_test_data(args.input_file, args.output_file, args.test_fraction, args.random_seed)
