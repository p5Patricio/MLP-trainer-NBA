import pandas as pd
from nba_api.stats.static import teams
from nba_data_processor import load_and_preprocess_data, scale_data, perform_kmeans_clustering, analyze_clusters
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from xhtml2pdf import pisa
import datetime 

# Define categorías de estadísticas para gráficos de radar más legibles
radar_stats_categories = {
    "Ofensivas Clave": ['PTS', 'AST', 'FGM', 'FGA', 'TOV'],
    "Rebotes": ['REB', 'OREB', 'DREB'],
    "Tiro": ['FG_PCT', 'FG3_PCT', 'FT_PCT'], # Los porcentajes son clave y se grafican bien juntos
    "Defensivas": ['STL', 'BLK', 'PF'] # PF es "malas", considerarla bien
}

# Mapeo de abreviaturas de estadísticas a nombres completos para mejor visualización
STAT_NAMES_MAP = {
    'MIN': 'Minutos',
    'FGM': 'Tiros de Campo Anotados',
    'FGA': 'Tiros de Campo Intentados',
    'FG_PCT': 'Tiros de Campo %',
    'FG3M': 'Triples Anotados',
    'FG3A': 'Triples Intentados',
    'FG3_PCT': 'Triples %',
    'FTM': 'Tiros Libres Anotados',
    'FTA': 'Tiros Libres Intentados',
    'FT_PCT': 'Tiros Libres %',
    'OREB': 'Rebotes Ofensivos',
    'DREB': 'Rebotes Defensivos',
    'REB': 'Rebotes Totales',
    'AST': 'Asistencias',
    'STL': 'Robos',
    'BLK': 'Bloqueos',
    'TOV': 'Pérdidas de Balón',
    'PF': 'Faltas Personales',
    'PTS': 'Puntos',
    'GP': 'Partidos Jugados',
    'GS': 'Partidos Iniciados',
    # Puedes añadir más si tienes otras estadísticas en tu dataset
}

# Diccionario de ejercicios detallados (puedes expandirlo)
NBA_DRILLS = {
    "puntería/eficiencia": [
        "1. Drill de Tiro de Campo: 50 tiros de cada spot (cerca del aro, media distancia, tres puntos). Enfocarse en la forma y el seguimiento.",
        "2. Tiro después de drible: 20 tiros de pull-up desde la media distancia, 20 tiros de step-back.",
        "3. Tiros Libres: 10 series de 5 tiros libres, buscando al menos 80% de efectividad."
    ],
    "Reducir TOV": [
        "1. Drill de Manejo de Balón bajo Presión: Driblear con dos balones, driblear mientras un compañero intenta quitar el balón sin contacto.",
        "2. Pase y Movimiento: Ejercicios de pase en movimiento, con énfasis en la precisión y la toma de decisiones rápidas (ej. 3 contra 2, 4 contra 3).",
        "3. Lectura de Defensa: Simulaciones de situaciones de juego donde el jugador debe leer la defensa antes de pasar o driblear."
    ],
    "Reducir PF": [
        "1. Defensa Posicional: Drills de movimiento lateral y posicionamiento defensivo sin contacto excesivo.",
        "2. Contención sin Falta: Practicar la defensa de balón, forzando al atacante a ir hacia su mano débil sin cometer falta.",
        "3. Carga Ofensiva vs. Falta Defensiva: Simular situaciones de carga ofensiva para entender cuándo es falta ofensiva y cuándo defensiva."
    ],
    "REB": [
        "1. Drill de Box-Out: Practicar bloquear al oponente para asegurar el rebote defensivo y ofensivo.",
        "2. Saltos de Rebote: 30 saltos continuos para rebote, enfocándose en el timing y la posición.",
        "3. Rebote en Tráfico: Ejercicios de rebote con múltiples jugadores simulando situaciones de juego."
    ],
    "AST": [
        "1. Drills de Visión de Juego: Ver y reaccionar a los compañeros que cortan, identificar al hombre abierto.",
        "2. Pases de Entrada al Poste: Practicar pases precisos al poste bajo contra defensa."
    ],
    "STL": [
        "1. Anticipación de Pase: Drills para leer la ofensiva y anticipar las líneas de pase.",
        "2. Defensa de Balón Activa: Ejercicios para robar el balón sin cometer falta."
    ],
    "BLK": [
        "1. Timing de Bloqueo: Practicar el salto y el timing para bloquear tiros sin cometer falta.",
        "2. Defensa de P&R (Pick and Roll): Drills para defender el pick and roll y proteger el aro."
    ],
    "MIN": [ # Esto es más general, enfocado en rendimiento general y consistencia
        "1. Acondicionamiento Físico: Rutinas de entrenamiento de fuerza y resistencia específicas para baloncesto.",
        "2. Mejora de Habilidades Fundamentales: Trabajar en todas las áreas básicas: dribleo, pase, tiro, defensa individual."
    ],
    "GP": [ # Similar a MIN, enfocado en durabilidad y consistencia
        "1. Prevención de Lesiones: Rutinas de estiramiento, fortalecimiento de articulaciones y músculos clave.",
        "2. Consistencia en el Rendimiento: Mantener un alto nivel de energía y concentración durante todo el partido y la temporada."
    ],
    "GS": [ # Enfocado en liderazgo y habilidades de titular
        "1. Liderazgo en Cancha: Drills de comunicación y toma de decisiones como líder.",
        "2. Habilidades de Iniciador: Enfocarse en el impacto inmediato al inicio del partido y la capacidad de llevar el ritmo."
    ],
    "PTS": [
        "1. Drills de Anotación en Transición: Practicar tiros y finalizaciones en contraataque.",
        "2. Ataque al Aro: Drills de penetración y finalización cerca del canasto, incluyendo flotadoras y bandejas de contacto.",
        "3. Creación de Espacio para Tiro: Practicar movimientos para liberarse del defensor y conseguir un buen tiro."
    ]
}

def display_team_selection_menu(all_teams_df):
    """Muestra un menú de selección de equipos."""
    print("\n--- Selecciona un Equipo NBA ---")
    for i, row in all_teams_df.iterrows():
        print(f"{i+1}. {row['full_name']} ({row['abbreviation']})")
    
    while True:
        try:
            choice = int(input(f"Ingresa el número del equipo (1-{len(all_teams_df)}): "))
            if 1 <= choice <= len(all_teams_df):
                selected_team_abbrev = all_teams_df.iloc[choice - 1]['abbreviation']
                return selected_team_abbrev
            else:
                print("Número de equipo inválido. Intenta de nuevo.")
        except ValueError:
            print("Entrada inválida. Por favor, ingresa un número.")

def display_player_selection_menu(players_in_team_df):
    """Muestra un menú de selección de jugadores para un equipo dado."""
    if players_in_team_df.empty:
        print("No hay jugadores en el dataset para este equipo en la temporada actual.")
        return None

    print(f"\n--- Selecciona un Jugador ---")
    for i, row in players_in_team_df.iterrows():
        print(f"{i+1}. {row['PLAYER_NAME']}")
    
    while True:
        try:
            choice = int(input(f"Ingresa el número del jugador (1-{len(players_in_team_df)}): "))
            if 1 <= choice <= len(players_in_team_df):
                selected_player_name = players_in_team_df.iloc[choice - 1]['PLAYER_NAME']
                return selected_player_name
            else:
                print("Número de jugador inválido. Intenta de nuevo.")
        except ValueError:
            print("Entrada inválida. Por favor, ingresa un número.")

def create_radar_chart(player_name, stats_to_plot, cluster_avg_stats, projected_stats, stats_labels, player_cluster_role, title="Rendimiento del Jugador"):
    """
    Crea un gráfico de radar para comparar las estadísticas del jugador
    con el promedio del clúster y las estadísticas proyectadas. Este se muestra en pantalla.
    """
    num_vars = len(stats_labels)

    # Calcular el ángulo para cada eje del gráfico de radar
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Cierra el círculo

    # Convertir los datos a listas y cerrar el círculo
    player_values = stats_to_plot.values.flatten().tolist()
    player_values += player_values[:1]

    cluster_avg_values = cluster_avg_stats.values.flatten().tolist()
    cluster_avg_values += cluster_avg_values[:1]

    projected_values = projected_stats.values.flatten().tolist()
    projected_values += projected_values[:1]

    # Aumentar el tamaño de la figura para mayor claridad
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Graficar las estadísticas del jugador
    ax.plot(angles, player_values, color='green', linewidth=2, label=f'{player_name} (Actual)')
    ax.fill(angles, player_values, color='green', alpha=0.25)

    # Graficar el promedio del clúster
    ax.plot(angles, cluster_avg_values, color='red', linewidth=2, label=f'Promedio {player_cluster_role} ({cluster_avg_stats.name})')
    ax.fill(angles, cluster_avg_values, color='red', alpha=0.1)

    # Graficar las estadísticas proyectadas (después del entrenamiento)
    ax.plot(angles, projected_values, color='blue', linewidth=2, linestyle='--', label=f'{player_name} (Proyectado)')
    ax.fill(angles, projected_values, color='blue', alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0) # Posición de las etiquetas radiales
    
    # Ajustar los límites del eje radial dinámicamente
    all_values = player_values + cluster_avg_values + projected_values
    max_val_plot = np.nanmax([v for v in all_values if np.isfinite(v) and not math.isnan(v)]) # Ignorar NaN y no finitos
    max_val_plot = max_val_plot if max_val_plot > 0 else 1.0

    # MODIFICACIÓN 1: Aumentar el margen superior del eje radial
    # Originalmente era max_val_plot * 1.1, lo cambiamos a 1.2 o incluso 1.25 si necesitas más espacio
    ax.set_ylim(0, max_val_plot * 1.25) #

    ax.set_xticks(angles[:-1])
    display_stats_labels = [STAT_NAMES_MAP.get(s, s) for s in stats_labels] #
    ax.set_xticklabels(display_stats_labels, fontsize=10) #

    # MODIFICACIÓN 2: Aumentar el pad del título del gráfico
    # Originalmente el pad era 25, lo aumentamos (ej. a 30 o 35)
    ax.set_title(title, va='bottom', fontsize=14, pad=35) #
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.3), fontsize=10) #

    # MODIFICACIÓN 3: Ajustar el layout general de la figura
    # Esto ayuda a que los elementos no se superpongan y se use bien el espacio.
    # El valor de 'pad' aquí es el relleno alrededor de los elementos del subplot.
    try:
        fig.tight_layout(pad=3.0)
    except Exception:
        # A veces tight_layout puede fallar con subplots polares si hay advertencias
        # que se tratan como errores. Podemos intentar con plt.
        plt.tight_layout(pad=3.0)


    plt.show()

# Esta función es una copia de create_radar_chart pero para guardar en archivo, no mostrar.
# Es necesario pasarle `ax` (el Axes de matplotlib) en lugar de crearlo internamente.
def create_radar_chart_to_file(ax, player_name, stats_to_plot, cluster_avg_stats, projected_stats, stats_labels, player_cluster_role, title="Rendimiento del Jugador"):
    """
    Crea un gráfico de radar para comparar las estadísticas del jugador
    con el promedio del clúster y las estadísticas proyectadas, en un Axes dado.
    """
    num_vars = len(stats_labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] 

    player_values = stats_to_plot.values.flatten().tolist()
    player_values += player_values[:1]

    cluster_avg_values = cluster_avg_stats.values.flatten().tolist()
    cluster_avg_values += cluster_avg_values[:1]

    projected_values = projected_stats.values.flatten().tolist()
    projected_values += projected_values[:1]

    ax.plot(angles, player_values, color='green', linewidth=2, label=f'{player_name} (Actual)')
    ax.fill(angles, player_values, color='green', alpha=0.25)

    ax.plot(angles, cluster_avg_values, color='red', linewidth=2, label=f'Promedio {player_cluster_role} ({cluster_avg_stats.name})')
    ax.fill(angles, cluster_avg_values, color='red', alpha=0.1)

    ax.plot(angles, projected_values, color='blue', linewidth=2, linestyle='--', label=f'{player_name} (Proyectado)')
    ax.fill(angles, projected_values, color='blue', alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    
    all_values = player_values + cluster_avg_values + projected_values #
    max_val_plot = np.nanmax([v for v in all_values if np.isfinite(v) and not math.isnan(v)]) #
    max_val_plot = max_val_plot if max_val_plot > 0 else 1.0 #
    
    # MODIFICACIÓN 1: Aumentar el margen superior del eje radial
    ax.set_ylim(0, max_val_plot * 1.25) #
    
    ax.set_xticks(angles[:-1]) #
    display_stats_labels = [STAT_NAMES_MAP.get(s, s) for s in stats_labels] #
    ax.set_xticklabels(display_stats_labels, fontsize=10) #

    # MODIFICACIÓN 2: Aumentar el pad del título del gráfico
    ax.set_title(title, va='bottom', fontsize=12, pad=30) #
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.3), fontsize=9)


def analyze_player_weak_spots(player_name, player_data_with_clusters, cluster_means, cluster_roles, stats_columns, scaler, threshold_multiplier=0.75):
    """
    Analiza las áreas débiles de un jugador comparando sus estadísticas con
    las de su clúster promedio y genera gráficos de radar con proyección de mejora,
    además de preparar el contenido para el reporte PDF.
    """
    player_row = player_data_with_clusters[player_data_with_clusters['PLAYER_NAME'] == player_name]

    if player_row.empty:
        print(f"Error: Jugador '{player_name}' no encontrado en el dataset preprocesado. Asegúrate de que haya jugado suficientes minutos.")
        return

    player_cluster = player_row['CLUSTER'].iloc[0]
    player_stats_raw = player_row[stats_columns].iloc[0] # Estadísticas originales del jugador
    
    # Obtener el promedio del clúster (Cluster_means ya está en valores originales)
    cluster_avg_stats_raw = cluster_means.loc[player_cluster]
    cluster_avg_stats_raw.name = player_cluster # Asignar nombre para la leyenda del gráfico

    player_cluster_role = cluster_roles.get(player_cluster, "Rol Desconocido") # Obtener el rol, con fallback
    print(f"\n--- Análisis de Áreas Débiles para {player_name} (Clúster: {player_cluster} - Rol: {player_cluster_role}) ---")
    print("Estadísticas del Jugador vs. Promedio de su Clúster:")

    comparison_df = pd.DataFrame({
        'Player Stats': player_stats_raw,
        'Cluster Average': cluster_avg_stats_raw,
        'Difference': player_stats_raw - cluster_avg_stats_raw,
        'Percentage Difference': (player_stats_raw - cluster_avg_stats_raw) / cluster_avg_stats_raw * 100
    })
    
    # Formatear el porcentaje para una mejor lectura
    comparison_df['Percentage Difference'] = comparison_df['Percentage Difference'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")

    print(comparison_df)

    weak_areas = []
    print("\nÁreas Potencialmente Débiles (significativamente por debajo del promedio del clúster):")
    
    # Para la proyección, haremos una copia de las estadísticas del jugador
    projected_stats_raw = player_stats_raw.copy()

    # Identificar áreas débiles y simular mejora
    for stat in stats_columns:
        if stat in player_stats_raw.index and stat in cluster_avg_stats_raw.index:
            player_val = player_stats_raw[stat]
            cluster_avg_val = cluster_avg_stats_raw[stat]

            # Manejo de valores NaN y cero para evitar errores y lógica inconsistente
            if pd.isna(player_val) or pd.isna(cluster_avg_val):
                projected_stats_raw[stat] = player_val # No proyectar si es NaN
                continue
            if cluster_avg_val == 0:
                if player_val == 0:
                    continue # Ambos son 0, no es una debilidad significativa
                else:
                    # Si el promedio es 0 pero el jugador tiene valor, no es una debilidad (ej. BLK=5 pero promedio=0).
                    # A menos que sea un porcentaje y el jugador tenga 0, lo cual es una debilidad.
                    if '_PCT' in stat and player_val == 0: # Caso especial para porcentajes si son 0
                         weak_areas.append(f"{stat}: {player_val:.3f} (Promedio Clúster: {cluster_avg_val:.3f}) - [Necesita mejorar {stat}]")
                         projected_stats_raw[stat] = 0.05 # Proyectar a un mínimo de 0.05 (5%) si era 0
                    continue 
            
            # Lógica para identificar debilidades y proyectar mejoras
            if '_PCT' in stat: # Para porcentajes, la mejora es un incremento en puntos porcentuales
                # Considerar un mínimo de GP para que los porcentajes sean significativos
                if player_data_with_clusters.loc[player_row.index[0], 'GP'] > 10 and (cluster_avg_val - player_val) > 0.05: # Umbral de 5 puntos porcentuales
                    weak_areas.append(f"{stat}: {player_val:.3f} (Promedio Clúster: {cluster_avg_val:.3f}) - [Necesita mejorar puntería/eficiencia]")
                    # Proyectar mejora: acercarse al promedio del clúster, sin superarlo demasiado
                    projected_stats_raw[stat] = min(player_val + 0.03, cluster_avg_val + 0.01) # Mejora de 3 puntos porcentuales
            elif stat in ['TOV', 'PF']: # Estadísticas donde un valor alto es "malo", la mejora es una reducción
                if player_val > (cluster_avg_val * 1.20) and cluster_avg_val > 0: # Si está 20% por encima del promedio del clúster
                    weak_areas.append(f"{stat}: {player_val} (Promedio Clúster: {cluster_avg_val:.2f}) - [Reducir {stat}]")
                    # Proyectar mejora: reducir hacia el promedio del clúster
                    projected_stats_raw[stat] = max(player_val * 0.90, cluster_avg_val * 0.95) # Reducción del 10%
            elif player_val < (cluster_avg_val * threshold_multiplier): # Generalmente, si está 25% por debajo del promedio del clúster
                weak_areas.append(f"{stat}: {player_val} (Promedio Clúster: {cluster_avg_val:.2f}) - [Necesita mejorar {stat}]")
                # Proyectar mejora: aumentar hacia el promedio del clúster
                projected_stats_raw[stat] = min(player_val * 1.15, cluster_avg_val * 0.95) # Aumento del 15%

    detailed_drills_html = "" # Inicializar aquí para usarlo en el PDF incluso si no hay debilidades
    if weak_areas:
        for area in weak_areas:
            print(f"- {area}")
        
        print("\n--- Sugerencias de Entrenamiento Generalizadas para Áreas Débiles ---")
        # Aquí es donde enlazaremos con los drills detallados
        detailed_drills_html = "<h3>Sugerencias de Entrenamiento Detalladas:</h3><ul>"
        for area_desc in weak_areas:
            # Intentar mapear la descripción general a una clave en NBA_DRILLS
            found_key = None
            for drill_key in NBA_DRILLS.keys():
                # Busca si la clave del drill está en la descripción de la debilidad
                # Normaliza ambas cadenas a minúsculas para una búsqueda más robusta
                if drill_key.lower() in area_desc.lower(): 
                    found_key = drill_key
                    break
            
            if found_key:
                print(f"  - Área: {area_desc.split(':')[0]}") # Imprime el nombre de la estadística
                print(f"    Sugerencias:")
                detailed_drills_html += f"<li><strong>{area_desc.split(':')[0]}</strong><ul>" # Título del área en HTML
                for drill in NBA_DRILLS[found_key]:
                    print(f"      - {drill}")
                    detailed_drills_html += f"<li>{drill}</li>"
                detailed_drills_html += "</ul></li>"
            else:
                print(f"  - No hay sugerencias detalladas para: {area_desc.split(':')[0]}")
                detailed_drills_html += f"<li><strong>{area_desc.split(':')[0]}</strong>: No hay sugerencias detalladas disponibles.</li>"
        detailed_drills_html += "</ul>"
    else:
        print("¡Este jugador no muestra áreas de debilidad significativas en comparación con su clúster!")
        print("Podría ser un jugador muy completo o estar en un clúster con compañeros de equipo de rendimiento similar.")
        projected_stats_raw = player_stats_raw.copy() # Si no hay debilidades, la proyección es la misma que la actual
        detailed_drills_html = "<p>El jugador no presenta debilidades significativas en comparación con su clúster.</p>"

    # --- Generación de Gráficos de Radar (en pantalla) ---
    print("\n--- Generando Gráficos de Rendimiento en pantalla ---")
    
    # Para cada categoría de estadísticas, crear un gráfico de radar
    for category_name, stats_list in radar_stats_categories.items():
        # Filtrar las estadísticas disponibles en el dataset y en la categoría
        current_stats_to_plot = [s for s in stats_list if s in stats_columns]
        
        if not current_stats_to_plot:
            print(f"No hay estadísticas para la categoría '{category_name}' en el dataset. Saltando gráfico.")
            continue

        player_stats_cat = player_stats_raw[current_stats_to_plot].fillna(0)
        cluster_avg_stats_cat = cluster_avg_stats_raw[current_stats_to_plot].fillna(0)
        projected_stats_cat = projected_stats_raw[current_stats_to_plot].fillna(0)
        
        # Normalizar los valores para este gráfico de radar.
        max_for_category = max(player_stats_cat.max(), cluster_avg_stats_cat.max(), projected_stats_cat.max())
        max_for_category = max_for_category if max_for_category > 0 else 1 # Evitar división por cero
        
        normalized_player_stats = player_stats_cat / max_for_category
        normalized_cluster_avg_stats = cluster_avg_stats_cat / max_for_category
        normalized_projected_stats = projected_stats_cat / max_for_category
        
        create_radar_chart(
            player_name,
            normalized_player_stats,
            normalized_cluster_avg_stats,
            normalized_projected_stats,
            current_stats_to_plot,
            player_cluster_role, # <-- ¡Añadir aquí!
            title=f'Rendimiento de {player_name} - {category_name}'
        )
    
    print("\n--- Generando Gráfico de Rendimiento Global (Todas las Estadísticas Clave) en pantalla ---")
    
    # Volver a usar todas las stats relevantes para un radar "todo en uno"
    all_radar_stats_for_global = [s for s in stats_columns if s not in ['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'PLAYER_AGE']]
    
    player_stats_all = player_stats_raw[all_radar_stats_for_global].fillna(0)
    cluster_avg_stats_all = cluster_avg_stats_raw[all_radar_stats_for_global].fillna(0)
    projected_stats_all = projected_stats_raw[all_radar_stats_for_global].fillna(0)
    
    # Normalizar por el máximo de todas las estadísticas globales
    max_val_all = max(player_stats_all.max(), cluster_avg_stats_all.max(), projected_stats_all.max())
    max_val_all = max_val_all if max_val_all > 0 else 1
    
    normalized_player_stats_all = player_stats_all / max_val_all
    normalized_cluster_avg_stats_all = cluster_avg_stats_all / max_val_all
    normalized_projected_stats_all = projected_stats_all / max_val_all
    
    create_radar_chart(
        player_name,
        normalized_player_stats_all,
        normalized_cluster_avg_stats_all,
        normalized_projected_stats_all,
        all_radar_stats_for_global,
        player_cluster_role, # <-- ¡Añadir aquí!
        title=f'Rendimiento Global de {player_name} vs. Clúster y Proyección'
    )

    print("\nGenerando reporte detallado en PDF...")
    generate_player_report_pdf(
        player_name,
        player_row,
        player_cluster,
        cluster_roles, # ¡Añadir cluster_roles aquí!
        player_stats_raw,
        cluster_avg_stats_raw,
        projected_stats_raw,
        comparison_df,
        weak_areas,
        detailed_drills_html,
        stats_columns,
        radar_stats_categories
    )
    print("Reporte PDF generado exitosamente.")


def generate_player_report_pdf(player_name, player_row, player_cluster, cluster_roles, player_stats_raw, 
                               cluster_avg_stats_raw, projected_stats_raw, comparison_df, 
                               weak_areas_list, detailed_drills_html,
                               all_stats_columns, radar_categories):
    player_cluster_role = cluster_roles.get(player_cluster, "Rol Desconocido")
    """
    Genera un reporte PDF con el análisis del jugador, incluyendo gráficos y rutinas.
    """
    # 1. Crear las imágenes de los gráficos de radar
    # Guardar los gráficos temporalmente como PNG
    img_filenames = []

    # Gráfico global para el PDF
    all_radar_stats_for_pdf = [s for s in all_stats_columns if s not in ['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'PLAYER_AGE']]
    player_stats_all_pdf = player_stats_raw[all_radar_stats_for_pdf].fillna(0)
    cluster_avg_stats_all_pdf = cluster_avg_stats_raw[all_radar_stats_for_pdf].fillna(0)
    projected_stats_all_pdf = projected_stats_raw[all_radar_stats_for_pdf].fillna(0)
    
    max_val_all_pdf = max(player_stats_all_pdf.max(), cluster_avg_stats_all_pdf.max(), projected_stats_all_pdf.max())
    max_val_all_pdf = max_val_all_pdf if max_val_all_pdf > 0 else 1
    
    normalized_player_stats_all_pdf = player_stats_all_pdf / max_val_all_pdf
    normalized_cluster_avg_stats_all_pdf = cluster_avg_stats_all_pdf / max_val_all_pdf
    normalized_projected_stats_all_pdf = projected_stats_all_pdf / max_val_all_pdf
    
    fig_global, ax_global = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True)) #
    create_radar_chart_to_file(ax_global, player_name, normalized_player_stats_all_pdf, normalized_cluster_avg_stats_all_pdf, 
                               normalized_projected_stats_all_pdf, all_radar_stats_for_pdf, 
                               player_cluster_role,
                               title=f'Rendimiento Global de {player_name} vs. Clúster y Proyección') #
    
    # MODIFICACIÓN 3: Aplicar tight_layout a la figura antes de guardar
    fig_global.tight_layout(pad=3.0) # Puedes ajustar el valor de pad según sea necesario

    global_radar_img = f"radar_global_{player_name.replace(' ', '_')}.png" #
    fig_global.savefig(global_radar_img, bbox_inches='tight', dpi=150) #
    plt.close(fig_global) #
    img_filenames.append(global_radar_img)


    # Gráficos por categoría
    for category_name, stats_list in radar_categories.items():
        current_stats_to_plot_pdf = [s for s in stats_list if s in all_stats_columns]
        if not current_stats_to_plot_pdf: continue

        player_stats_cat_pdf = player_stats_raw[current_stats_to_plot_pdf].fillna(0)
        cluster_avg_stats_cat_pdf = cluster_avg_stats_raw[current_stats_to_plot_pdf].fillna(0)
        projected_stats_cat_pdf = projected_stats_raw[current_stats_to_plot_pdf].fillna(0)
        
        max_for_category_pdf = max(player_stats_cat_pdf.max(), cluster_avg_stats_cat_pdf.max(), projected_stats_cat_pdf.max())
        max_for_category_pdf = max_for_category_pdf if max_for_category_pdf > 0 else 1
        
        # Estas son las variables que causaban el error de "no definida"
        normalized_player_stats_pdf = player_stats_cat_pdf / max_for_category_pdf
        normalized_cluster_avg_stats_pdf = cluster_avg_stats_cat_pdf / max_for_category_pdf
        normalized_projected_stats_pdf = projected_stats_cat_pdf / max_for_category_pdf

        fig_cat, ax_cat = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True)) #
        create_radar_chart_to_file(ax_cat, player_name, normalized_player_stats_pdf, normalized_cluster_avg_stats_pdf, 
                                   normalized_projected_stats_pdf, current_stats_to_plot_pdf, 
                                   player_cluster_role,
                                   title=f'Rendimiento de {player_name} - {category_name}') #
        
        # MODIFICACIÓN 3: Aplicar tight_layout a la figura antes de guardar
        fig_cat.tight_layout(pad=3.0) # Puedes ajustar el valor de pad

        cat_radar_img = f"radar_{category_name.replace(' ', '_')}_{player_name.replace(' ', '_')}.png" #
        fig_cat.savefig(cat_radar_img, bbox_inches='tight', dpi=120) #
        plt.close(fig_cat) #
        img_filenames.append(cat_radar_img)


    # 2. Crear el contenido HTML del reporte
    report_html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Reporte de Análisis de Jugador NBA: {player_name}</title>
        <style>
            body {{ font-family: 'Arial', sans-serif; margin: 20px; color: #333; }}
            h1, h2, h3 {{ color: #2C3E50; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .player-info, .cluster-info {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .comparison-table td:nth-child(even) {{ background-color: #f9f9f9; }}
            .weak-areas ul {{ list-style-type: none; padding: 0; }}
            .weak-areas li {{ background-color: #ffe6e6; border-left: 5px solid #ff4d4d; margin-bottom: 8px; padding: 10px; border-radius: 4px; }}
            .drills-section ul {{ list-style-type: disc; margin-left: 20px; }}
            .drills-section li {{ margin-bottom: 5px; }}
            .chart-container {{ text-align: center; margin-bottom: 30px; }}
            .chart-container img {{ max-width: 90%; height: auto; border: 1px solid #ccc; border-radius: 5px; }}
            .footer {{ text-align: center; font-size: 0.8em; color: #777; margin-top: 30px; }}
            /* Clases para el rendimiento en la tabla */
            .good-performance {{ color: green; font-weight: bold; }}
            .weak-spot {{ color: red; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Reporte de Análisis de Jugador NBA</h1>
            <h2>{player_name}</h2>
            <p><strong>Temporada:</strong> {player_row['SEASON_ID'].iloc[0]} | 
                <strong>Equipo:</strong> {player_row['TEAM_ABBREVIATION'].iloc[0]} | 
                <strong>Edad:</strong> {player_row['PLAYER_AGE'].iloc[0]:.0f} años</p>
        </div>

        <div class="cluster-info">
            <h3>Información del Clúster</h3>
            <p>Este jugador ha sido agrupado en el <strong>Clúster {player_cluster} ({player_cluster_role})</strong>.</p>
            <p>Los jugadores de este clúster comparten perfiles estadísticos similares. El análisis compara el rendimiento de {player_name} con el promedio de jugadores de su mismo clúster.</p>
            <h4>Estadísticas Promedio del Clúster:</h4>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Estadística</th>
                        <th>Promedio Clúster</th>
                    </tr>
                </thead>
                <tbody>
    """
    # MODIFICACIÓN AQUI: Muestra las estadísticas promedio del clúster con nombres completos
    for stat in [s for s in all_stats_columns if s in cluster_avg_stats_raw.index]:
        display_stat_name = STAT_NAMES_MAP.get(stat, stat) # Obtener el nombre completo
        report_html += f"<tr><td>{display_stat_name}</td><td>{cluster_avg_stats_raw[stat]:.2f}</td></tr>"
    report_html += """
                </tbody>
            </table>
        </div>

        <div class="player-info">
            <h3>Comparación de Estadísticas</h3>
            <p>A continuación, se muestra la comparación de las estadísticas de {player_name} con el promedio de su clúster, incluyendo la diferencia y el porcentaje de diferencia.</p>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Estadística</th>
                        <th>Estadísticas del Jugador</th>
                        <th>Promedio Clúster</th>
                        <th>Diferencia</th>
                        <th>% Diferencia</th>
                    </tr>
                </thead>
                <tbody>
    """.format(player_name=player_name)
    # MODIFICACIÓN AQUI: Usar el comparison_df para la tabla con nombres completos y formato condicional
    for index, row in comparison_df.iterrows():
        player_val = row['Player Stats']
        cluster_val = row['Cluster Average']
        diff_val = row['Difference']
        
        # --- MODIFICACIÓN IMPORTANTE AQUÍ ---
        # Asegurarse de que percent_diff_val sea un número antes de formatearlo.
        # Si ya es un string (e.g., "15.23%"), quita el '%' y conviértelo a float.
        raw_percent_diff = row['Percentage Difference']
        if isinstance(raw_percent_diff, str):
            # Eliminar el '%' si está presente y luego convertir a float
            percent_diff_val = float(raw_percent_diff.replace('%', ''))
        else:
            # Ya es un número (float o int), úsalo directamente
            percent_diff_val = float(raw_percent_diff)
        # --- FIN MODIFICACIÓN IMPORTANTE ---

        # Formateo condicional para el porcentaje de diferencia
        percent_diff_str = f"{percent_diff_val:.2f}%" # Ahora percent_diff_val es definitivamente un float
        color_class = ""
        if percent_diff_val > 5:
            color_class = "good-performance"
        elif percent_diff_val < -5:
            color_class = "weak-spot"
        
        # Obtener el nombre completo de la estadística
        display_stat_name = STAT_NAMES_MAP.get(index, index) # 'index' es el nombre de la estadística en comparison_df
        
        report_html += f"""
                    <tr>
                        <td>{display_stat_name}</td>
                        <td>{player_val:.2f}</td>
                        <td>{cluster_val:.2f}</td>
                        <td>{diff_val:.2f}</td>
                        <td class="{color_class}">{percent_diff_str}</td>
                    </tr>
        """
    report_html += """
                </tbody>
            </table>
        </div>

        <div class="weak-areas">
            <h3>Áreas Potencialmente Débiles Identificadas</h3>
            <ul>
    """
    if weak_areas_list:
        for area in weak_areas_list:
            report_html += f"<li>{area}</li>"
    else:
        report_html += "<li>No se identificaron áreas de debilidad significativas en comparación con el promedio de su clúster.</li>"
    report_html += """
            </ul>
        </div>
        
        <div class="chart-container">
            <h3>Gráficos de Rendimiento (Actual vs. Clúster vs. Proyección)</h3>
            <img src="{global_radar_img}" alt="Rendimiento Global">
            <p>Este gráfico muestra el perfil de rendimiento global de {player_name} en comparación con el promedio de su clúster y una proyección de mejora después del entrenamiento.</p>
        </div>
    """.format(global_radar_img=global_radar_img, player_name=player_name) # Formato para la imagen global

    # Añadir los gráficos por categoría
    for category_name, stats_list in radar_categories.items():
        cat_radar_img = f"radar_{category_name.replace(' ', '_')}_{player_name.replace(' ', '_')}.png"
        if os.path.exists(cat_radar_img): # Solo si la imagen existe
            report_html += f"""
            <div class="chart-container">
                <h4>Gráfico de Rendimiento: {category_name}</h4>
                <img src="{cat_radar_img}" alt="Rendimiento {category_name}">
                <p>Análisis detallado de las estadísticas {category_name.lower()} de {player_name}.</p>
            </div>
            """

    report_html += f"""
        <div class="drills-section">
            {detailed_drills_html}
        </div>

        <div class="footer">
            <p>Reporte generado por NBA Player Analyzer - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """

    # 3. Guardar el HTML y convertirlo a PDF
    output_filename = f"Reporte_{player_name.replace(' ', '_')}.pdf"
    
    # xhtml2pdf
    try:
        with open(output_filename, "wb") as f:
            # base_url=os.getcwd() es importante para que pisa encuentre las imágenes temporales
            pisa_status = pisa.CreatePDF(
                    report_html,                # the HTML to convert
                    dest=f,                     # file handle to receive result
                    encoding='utf-8',           # specify encoding for special characters
                    link_callback=lambda uri, rel: os.path.join(os.getcwd(), uri) # Callback para resolver rutas de imágenes
            )
        if pisa_status.err:
            print(f"¡Error al generar PDF con xhtml2pdf para {player_name}!")
            print(pisa_status.err) # Imprimir detalles del error
        else:
            print(f"Reporte PDF generado y guardado como: {output_filename}")
    except Exception as e:
        print(f"Error inesperado al generar PDF: {e}")

    # 4. Limpiar las imágenes temporales
    for img_file in img_filenames:
        if os.path.exists(img_file):
            os.remove(img_file)
            print(f"Imagen temporal eliminada: {img_file}")

def assign_cluster_roles(cluster_means):
    """
    Asigna un rol descriptivo a cada clúster basándose en sus estadísticas promedio.
    Esta función es heurística y ha sido ajustada según el análisis de los datos.
    Se asume que las estadísticas de cluster_means son TOTALES de la temporada.
    """
    cluster_roles = {}
    
    # Asume que las estadísticas son TOTALES de la temporada y no por partido.
    # Ajusta los umbrales basándote en esta suposición.
    
    for cluster_id, stats in cluster_means.iterrows():
        role = "Indefinido" # Rol por defecto

        # Convertir a Series para fácil acceso a los valores y manejo de NaNs
        stats = stats.fillna(0) # Rellenar NaN con 0 para comparaciones numéricas

        # Orden de prioridad de los roles: de los más específicos/extremos a los más generales.

        # --- Roles de Baja Participación / Desarrollo ---
        # Clúster con muy pocos minutos y partidos, producción muy baja.
        # Esto captura jugadores con un impacto mínimo.
        if stats['MIN'] < 200 and stats['GP'] < 25 and stats['PTS'] < 100:
            role = "Jugador de Límite de Roster / Desarrollo"
        
        # --- Interiores Dominantes / Rebotadores Eficientes / Protectores de Aro ---
        # Muy alto FG_PCT, alto REB, BLK decente. (Centros puros o Pívots)
        # Separamos los protectores de aro más específicos
        elif stats['BLK'] > 70 and stats['REB'] > 300 and stats['MIN'] > 800:
            role = "Protector de Aro / Pívot Defensivo"
        elif stats['FG_PCT'] > 0.60 and stats['REB'] > 400 and stats['PTS'] > 300:
            role = "Centro Dominante / Interior Eficiente"
        # Hombres grandes con buen rebote y eficiencia, pero no tan dominantes.
        elif stats['REB'] > 350 and stats['MIN'] > 800 and stats['PTS'] < 600:
            role = "Hombre Grande Rebotador / De Rol"

        # --- Creadores de Juego / Bases ---
        # Alta AST, minutos significativos. Exigimos más AST para este rol.
        elif stats['AST'] > 350 and stats['MIN'] > 1200:
            # Si tiene también buen volumen de anotación y triple, es un creador completo
            if stats['PTS'] > 800 and stats['FG3M'] > 80:
                role = "Base Creador de Juego / Anotador"
            else:
                role = "Base Creador de Juego Puro"
        
        # --- Bases de Rol / Facilitadores (Ahora más específico para no sobrecargar) ---
        # Jugadores que facilitan el juego con asistencias decentes para sus minutos,
        # pero que no son los creadores primarios. Podrían tener un buen tiro de 3.
        # Aquí es donde hemos hecho el mayor ajuste.
        elif stats['AST'] > 150 and stats['MIN'] > 500 and stats['PTS'] < 500:
            if stats['FG3M'] > 70 and stats['FG3_PCT'] > 0.35:
                role = "Base de Rol / Facilitador (Tirador)"
            else:
                role = "Base de Rol / Manejador de Balón"

        # --- Especialistas Ofensivos: Tiradores de Élite / Escoltas Anotadores ---
        # Alto volumen y eficiencia en triples, buen volumen de tiro general.
        # Mayor exigencia para ser "Especialista en Triples"
        elif stats['FG3_PCT'] > 0.38 and stats['FG3M'] > 150 and stats['FGA'] > 600:
            role = "Especialista en Triples / Escolta Anotador"
        
        # --- Anotadores de Volumen / Aleros Ofensivos ---
        # Alto PTS y FGA, no encaja en un rol de tirador o centro específico.
        # Ahora el umbral de PTS es más alto.
        elif stats['PTS'] > 800 and stats['FGA'] > 650:
            role = "Anotador de Volumen / Alero Ofensivo"

        # --- Defensores de Rol ---
        # Buen número de robos y/o bloqueos, con minutos significativos.
        elif (stats['STL'] > 80 or stats['BLK'] > 40) and stats['MIN'] > 700:
            role = "Defensor de Rol"

        # --- Jugador Completo (All-Around) ---
        # Buen balance en múltiples categorías, minutos significativos.
        # Este rol es para jugadores que contribuyen en muchas áreas y no encajan en un especialista.
        # Hacemos los umbrales más estrictos para este rol.
        elif stats['PTS'] > 500 and stats['AST'] > 150 and stats['REB'] > 250 and \
             stats['MIN'] > 1000 and stats['FG_PCT'] > 0.45:
            role = "Jugador Completo (All-Around)"

        # --- Jugador de Rol General (último recurso) ---
        # Si aún no se asignó un rol específico, es un jugador de rol más general.
        if role == "Indefinido":
            role = "Jugador de Rol General"

        cluster_roles[cluster_id] = role

    return cluster_roles


if __name__ == '__main__':
    filepath = 'nba_active_player_stats_2023-24_Regular_Season_100min.xlsx' 
    
    clustering_stats_columns = [
        'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
        'BLK', 'TOV', 'PF', 'PTS', 'GP', 'GS'
    ]

    # --- Cargar y Preprocesar Datos ---
    stats_for_clustering, player_data_cleaned, player_names = load_and_preprocess_data(filepath, clustering_stats_columns)

    if stats_for_clustering is None:
        print("No se pudo cargar o preprocesar el dataset. Saliendo.")
        exit()

    # --- Escalar Datos ---
    scaled_stats_df, scaler = scale_data(stats_for_clustering)

    # --- Realizar Clustering ---
    # AJUSTA ESTO CON EL K ÓPTIMO QUE ENCONTRASTE CON EL MÉTODO DEL CODO
    optimal_k = 5

    clusters, kmeans_model = perform_kmeans_clustering(scaled_stats_df, optimal_k)

    # Añadir los clústeres al DataFrame de datos limpios
    player_data_cleaned['CLUSTER'] = clusters

    # --- Analizar Clústeres (para entender los roles) ---
    cluster_means = analyze_clusters(player_data_cleaned, clustering_stats_columns)
    # --- Asignar Roles a los Clústeres ---
    cluster_roles = assign_cluster_roles(cluster_means)
    print("\n--- Roles de los Clústeres Asignados ---")
    for c_id, role_name in cluster_roles.items():
        print(f"Clúster {c_id}: {role_name}")

    # --- Menú de Selección de Jugadores ---
    all_nba_teams = pd.DataFrame(teams.get_teams())
    
    # Filtra los equipos para asegurar que solo mostramos los que tienen jugadores en nuestro dataset
    teams_in_dataset = player_data_cleaned['TEAM_ABBREVIATION'].unique()
    filtered_teams_df = all_nba_teams[all_nba_teams['abbreviation'].isin(teams_in_dataset)].reset_index(drop=True)

    selected_team_abbrev = display_team_selection_menu(filtered_teams_df)

    if selected_team_abbrev:
        # Obtener jugadores de ese equipo en el dataset, excluyendo 'TOT' para este menú
        players_in_selected_team = player_data_cleaned[
            (player_data_cleaned['TEAM_ABBREVIATION'] == selected_team_abbrev)
        ].sort_values(by='PLAYER_NAME').reset_index(drop=True)

        # Si el equipo es 'TOT' para un jugador específico (por ejemplo, si el jugador fue traspasado
        # y solo aparece con TOT), podríamos considerar una lógica alternativa o simplemente no listarlo
        # bajo un equipo específico. Para un análisis por equipo, lo mejor es excluir TOT.

        selected_player_name = display_player_selection_menu(players_in_selected_team)

        if selected_player_name:
            analyze_player_weak_spots(
                selected_player_name, 
                player_data_cleaned, 
                cluster_means, 
                cluster_roles, # Asegúrate de que este esté aquí
                clustering_stats_columns, # Este también es necesario
                scaler # <-- ¡Asegúrate de que 'scaler' esté aquí!
            )
        else:
            print("No se seleccionó ningún jugador.")
    else:
        print("No se seleccionó ningún equipo.")