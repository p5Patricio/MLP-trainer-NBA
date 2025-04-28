import os
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import playercareerstats, commonplayerinfo
from nba_api.stats.endpoints import commonteamroster

os.makedirs('data', exist_ok=True)

team_name_input = input("Escribe el nombre del equipo de la NBA: ").strip().lower()

nba_teams = teams.get_teams()

team_mapping = {team['full_name'].lower(): team['id'] for team in nba_teams}

if team_name_input not in team_mapping:
    print(f"\nEquipo '{team_name_input}' no encontrado.\n")
else:
    team_id = team_mapping[team_name_input]
    print(f"\nBuscando jugadores del equipo: {team_name_input.title()}\n")

    roster = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
    player_ids = roster['PLAYER_ID']

    all_stats = []

    for player_id in player_ids:
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
            player_name = info['DISPLAY_FIRST_LAST'][0]
            print(f"Obteniendo estadísticas de {player_name}...")

            stats = playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0]
            stats['PLAYER_NAME'] = player_name

            all_stats.append(stats)
        except Exception as e:
            print(f"Error obteniendo datos del jugador con ID {player_id}: {e}")

    if all_stats:
        team_stats = pd.concat(all_stats, ignore_index=True)
        excel_filename = f"data/{team_name_input.replace(' ', '_')}_players_stats.xlsx"
        team_stats.to_excel(excel_filename, index=False)
        print(f"\n¡Archivo de estadísticas guardado en '{excel_filename}'!\n")
    else:
        print("\nNo se pudieron obtener estadísticas de los jugadores.\n")
