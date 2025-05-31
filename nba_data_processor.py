import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Ignorar FutureWarnings para evitar saturar la salida
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_and_preprocess_data(filepath, stats_columns):
    """
    Carga el dataset de jugadores, selecciona columnas de estadísticas,
    convierte a numérico y maneja valores nulos.

    Args:
        filepath (str): Ruta al archivo Excel del dataset.
        stats_columns (list): Lista de nombres de columnas a usar para el clustering.

    Returns:
        tuple: (DataFrame de estadísticas limpias, DataFrame original con nombres, Series de nombres de jugadores)
    """
    if not os.path.exists(filepath):
        print(f"Error: El archivo no se encontró en '{filepath}'")
        return None, None, None

    player_data = pd.read_excel(filepath)

    # Filtrar solo las columnas que realmente existen en el DataFrame
    actual_stats_columns = [col for col in stats_columns if col in player_data.columns]
    
    if not actual_stats_columns:
        print("Error: Ninguna de las columnas de estadísticas especificadas se encontró en el dataset.")
        return None, None, None

    stats_for_clustering = player_data[actual_stats_columns].copy()

    # Convertir a numérico, forzando errores a NaN, y luego eliminando NaN
    for col in stats_for_clustering.columns:
        stats_for_clustering[col] = pd.to_numeric(stats_for_clustering[col], errors='coerce')
    stats_for_clustering = stats_for_clustering.dropna()
    
    # Es crucial que los índices del DataFrame original se mantengan para mapear de vuelta los clústeres
    # Filtrar el player_data original para que coincida con los índices de stats_for_clustering
    player_data_cleaned = player_data.loc[stats_for_clustering.index].copy()
    
    # Guarda los nombres de los jugadores correspondientes a las estadísticas limpias
    player_names_for_clustering = player_data_cleaned['PLAYER_NAME']
    
    print(f"Dataset cargado y preprocesado. Jugadores para clustering: {stats_for_clustering.shape[0]}")
    return stats_for_clustering, player_data_cleaned, player_names_for_clustering

def scale_data(stats_df):
    """
    Escala las características usando StandardScaler.

    Args:
        stats_df (pd.DataFrame): DataFrame de estadísticas numéricas.

    Returns:
        tuple: (DataFrame de estadísticas escaladas, StandardScaler entrenado)
    """
    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(stats_df)
    scaled_stats_df = pd.DataFrame(scaled_stats, columns=stats_df.columns, index=stats_df.index)
    print("Datos escalados exitosamente.")
    return scaled_stats_df, scaler

def find_optimal_k(scaled_stats_df, max_k=10):
    """
    Usa el método del codo para sugerir un K óptimo.

    Args:
        scaled_stats_df (pd.DataFrame): DataFrame de estadísticas escaladas.
        max_k (int): Número máximo de clústeres a probar.

    Returns:
        None: Muestra un gráfico del método del codo.
    """
    wcss = []
    print(f"Calculando WCSS para K de 1 a {max_k} para el método del codo...")
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(scaled_stats_df)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
    plt.title('Método del Codo para K-Means')
    plt.xlabel('Número de Clústeres (K)')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.show()
    print("\nVisualiza el gráfico del 'Método del Codo' para elegir un valor de K.")
    print("Busca un 'codo' donde la disminución de WCSS se ralentice significativamente.")

def perform_kmeans_clustering(scaled_stats_df, n_clusters):
    """
    Realiza el clustering K-Means y devuelve las etiquetas de clúster.

    Args:
        scaled_stats_df (pd.DataFrame): DataFrame de estadísticas escaladas.
        n_clusters (int): Número de clústeres deseado.

    Returns:
        numpy.ndarray: Etiquetas de clúster para cada jugador.
        sklearn.cluster.KMeans: Modelo KMeans entrenado.
    """
    print(f"Realizando clustering K-Means con {n_clusters} clústeres...")
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(scaled_stats_df)
    print("Clustering completado.")
    return clusters, kmeans

def analyze_clusters(player_data_with_clusters, stats_columns):
    """
    Calcula y muestra las estadísticas promedio por clúster.

    Args:
        player_data_with_clusters (pd.DataFrame): DataFrame de jugadores con etiquetas de clúster.
        stats_columns (list): Lista de columnas de estadísticas usadas para el clustering.

    Returns:
        pd.DataFrame: Estadísticas promedio por clúster.
    """
    print("\nCalculando estadísticas promedio por clúster para entender sus perfiles...")
    cluster_means = player_data_with_clusters.groupby('CLUSTER')[stats_columns].mean()
    print("\nEstadísticas promedio por clúster:")
    print(cluster_means)
    # Guarda las estadísticas promedio por clúster en un archivo Excel
    output_filename = 'cluster_means_report.xlsx'
    try:
        cluster_means.to_excel(output_filename)
        print(f"\nEstadísticas promedio por clúster guardadas en: {output_filename}")
    except Exception as e:
        print(f"Error al guardar las estadísticas promedio en Excel: {e}")
    
    overall_means = player_data_with_clusters[stats_columns].mean()
    print("\nEstadísticas promedio generales de la liga:")
    print(overall_means)
    
    # Opcional: Visualización para comparar perfiles de clúster
    # Puedes crear gráficos de barras o de radar para visualizar esto mejor.
    # Por ejemplo, para las top 5 métricas o algo representativo.
    
    return cluster_means

if __name__ == '__main__':
    # Aquí puedes ejecutar un test rápido para ver si el archivo funciona
    filepath = 'nba_active_player_stats_2023-24_Regular_Season_100min.xlsx' # Asegúrate de que este sea el nombre de tu archivo
    
    # Columnas de estadísticas que usarás para el clustering
    # Estas son las que se van a escalar y sobre las que se aplicará el clustering
    clustering_stats_columns = [
        'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
        'BLK', 'TOV', 'PF', 'PTS', 'GP', 'GS'
    ]

    stats_for_clustering, player_data_cleaned, player_names = load_and_preprocess_data(filepath, clustering_stats_columns)

    if stats_for_clustering is not None:
        scaled_stats_df, scaler = scale_data(stats_for_clustering)
        
        # Para elegir el K, ejecuta esto una vez y mira el gráfico
        # find_optimal_k(scaled_stats_df) 
        # Después de ejecutarlo, elige un K y ponlo abajo.
        # Por ejemplo, si el codo está en 4 o 5:
        
        optimal_k = 5 # <--- Sustituye esto con el K que elegiste del método del codo
        
        clusters, kmeans_model = perform_kmeans_clustering(scaled_stats_df, optimal_k)
        
        player_data_cleaned['CLUSTER'] = clusters
        
        cluster_means = analyze_clusters(player_data_cleaned, clustering_stats_columns)
        
        print("\nPrueba completa de procesamiento de datos.")