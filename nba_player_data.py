import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
import time
import math

def get_nba_active_player_stats(season='2023-24', season_type='Regular Season', min_minutes_played=100):
    """
    Obtiene estadísticas de carrera por temporada para jugadores activos de la NBA
    y con un mínimo de minutos jugados en la temporada especificada.

    Args:
        season (str): La temporada de la que obtener las estadísticas (ej. '2023-24').
        season_type (str): Tipo de temporada ('Regular Season', 'Playoffs').
        min_minutes_played (int): Mínimo de minutos jugados para incluir al jugador.

    Returns:
        pandas.DataFrame: Un DataFrame con las estadísticas de los jugadores por temporada.
    """
    # Obtener solo jugadores activos
    active_nba_players = players.get_active_players()
    all_player_stats = []
    
    print(f"Obteniendo IDs de {len(active_nba_players)} jugadores activos...")
    
    # Define la lista de columnas que esperas y quieres convertir a numérico
    # Esto ayuda a manejar inconsistencias si alguna columna no es numérica al inicio
    numeric_stats_columns = [
        'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
        'BLK', 'TOV', 'PF', 'PTS'
    ]

    for i, player in enumerate(active_nba_players):
        player_id = player['id']
        player_name = player['full_name']
        
        # Pequeño retraso para evitar time-outs
        # Puedes ajustar este valor si aún experimentas muchos errores
        time.sleep(0.5) 
        
        print(f"Procesando jugador {i+1}/{len(active_nba_players)}: {player_name} (ID: {player_id})")

        try:
            career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)
            player_df = pd.DataFrame() # Inicializar como DataFrame vacío

            if season_type == 'Regular Season':
                player_df = career_stats.get_data_frames()[0]
            elif season_type == 'Playoffs':
                # Asegúrate de que el índice 1 realmente contenga los datos de playoffs
                # Algunas veces un jugador puede no tener stats de playoffs
                if len(career_stats.get_data_frames()) > 1:
                    player_df = career_stats.get_data_frames()[1]
                else:
                    # print(f"No hay datos de playoffs para {player_name}")
                    continue # Saltar si no hay datos de playoffs disponibles
            else:
                print(f"Tipo de temporada '{season_type}' no manejado. Saltando a {player_name}.")
                continue

            # Filtrar por la temporada deseada
            season_stats = player_df[player_df['SEASON_ID'] == season].copy()
            
            if not season_stats.empty:
                # Asegurarse de que las columnas numéricas sean realmente numéricas para el filtro de MIN
                for col in numeric_stats_columns:
                    if col in season_stats.columns:
                        season_stats[col] = pd.to_numeric(season_stats[col], errors='coerce')
                
                # Filtrar por minutos jugados para asegurar participación significativa
                # Comprobar que 'MIN' no sea NaN antes de la comparación
                if 'MIN' in season_stats.columns and not math.isnan(season_stats['MIN'].iloc[0]) and season_stats['MIN'].iloc[0] >= min_minutes_played:
                    season_stats['PLAYER_NAME'] = player_name
                    all_player_stats.append(season_stats)
                    # print(f"Estadísticas obtenidas para {player_name} ({season}) - MIN: {season_stats['MIN'].iloc[0]}")
                # else:
                    # print(f"Saltando a {player_name}: No jugó suficientes minutos ({min_minutes_played}) o MIN es NaN.")
            # else:
                # print(f"No hay estadísticas para {player_name} en la temporada {season_type} {season}")

        except Exception as e:
            print(f"Error al obtener estadísticas para {player_name} (ID: {player_id}): {e}")
            # Puedes añadir un retraso más largo aquí si el error es persistente
            # time.sleep(2) 
            
    if all_player_stats:
        combined_df = pd.concat(all_player_stats, ignore_index=True)
        return combined_df
    else:
        print(f"No se encontraron estadísticas para la temporada {season} ({season_type}) con {min_minutes_played} minutos mínimos.")
        return pd.DataFrame()

if __name__ == '__main__':
    target_season = '2023-24'
    target_season_type = 'Regular Season' 
    min_minutes = 100 # Puedes ajustar este umbral

    print(f"Iniciando la obtención de datos para la temporada {target_season} ({target_season_type}) para jugadores con al menos {min_minutes} minutos...")
    player_data = get_nba_active_player_stats(
        season=target_season, 
        season_type=target_season_type,
        min_minutes_played=min_minutes
    )

    if not player_data.empty:
        # Reordenar columnas para que el nombre del jugador esté al principio
        cols = ['PLAYER_NAME'] + [col for col in player_data.columns if col != 'PLAYER_NAME']
        player_data = player_data[cols]

        output_filename = f'nba_active_player_stats_{target_season}_{target_season_type.replace(" ", "_")}_{min_minutes}min.xlsx'
        player_data.to_excel(output_filename, index=False)
        print(f"\n¡Dataset guardado exitosamente en '{output_filename}'!")
        print(f"Dimensiones del dataset: {player_data.shape}")
        print("\nPrimeras 5 filas del dataset:")
        print(player_data.head())
        
        # Prepara para el clustering: Selecciona solo columnas numéricas de estadísticas
        # Asegúrate de que los IDs y nombres no estén aquí
        stats_columns = [
            'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
            'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
            'BLK', 'TOV', 'PF', 'PTS', 'GP', 'GS' # 'GP' y 'GS' también son relevantes
        ]
        
        # Filtrar solo las columnas que realmente existen en el DataFrame
        # y que son parte de las estadísticas a considerar para el clustering
        actual_stats_columns = [col for col in stats_columns if col in player_data.columns]
        
        stats_for_clustering = player_data[actual_stats_columns].copy()
        
        # Convertir a numérico, forzando errores a NaN, y luego eliminando NaN
        for col in stats_for_clustering.columns:
            stats_for_clustering[col] = pd.to_numeric(stats_for_clustering[col], errors='coerce')
        stats_for_clustering = stats_for_clustering.dropna() 
        
        # Guarda los nombres de los jugadores correspondientes a las estadísticas limpias
        # Esto es importante para poder mapear los clústeres de vuelta a los nombres
        player_names_for_clustering = player_data.loc[stats_for_clustering.index, 'PLAYER_NAME']

        print(f"\nDatos preparados para clustering (primeras 5 filas):")
        print(stats_for_clustering.head())
        print(f"\nNúmero de jugadores en el dataset final para clustering: {stats_for_clustering.shape[0]}")

    else:
        print("No se pudo obtener el dataset. Revisa la temporada, el tipo de temporada o el umbral de minutos.")