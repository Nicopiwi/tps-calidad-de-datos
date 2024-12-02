import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic

# VER METADATOS DE LOS PUNTOS DE EXTRACCION

df_pozos_general = pd.read_csv('./data/capitulo-iv-pozos.csv', low_memory=False)
df_pozos_convencional = pd.read_csv('./data/produccin-de-pozos-de-gas-y-petrleo-2024.csv', low_memory=False)
df_pozos_noconvencional = pd.read_csv('./data/produccin-de-pozos-de-gas-y-petrleo-no-convencional.csv', low_memory=False)
df_puntos_de_extraccion = pd.read_csv('./data/puntos_de_extraccion_AA050.csv', low_memory=False)

# Extract coordinates from 'geom' in df_puntos_de_extraccion
df_puntos_de_extraccion[['longitude', 'latitude']] = df_puntos_de_extraccion['geom'].str.extract(r'POINT\(([-\d\.]+) ([-\d\.]+)\)').astype(float)

# Extract coordinates from 'geojson' in df_pozos_general
df_pozos_general[['longitude', 'latitude']] = df_pozos_general['geojson'].str.extract(r'"coordinates":\[\s*([-\d\.]+),\s*([-\d\.]+)\]').astype(float)


# Function to calculate closest matches based on geographic proximity
def match_points(df_points, df_pozos):
    matches = []
    for _, point_row in df_points.iterrows():
        point_coords = (point_row['latitude'], point_row['longitude'])
        min_distance = float('inf')
        match_row = None

        for _, pozo_row in df_pozos.iterrows():
            pozo_coords = (pozo_row['latitude'], pozo_row['longitude'])
            distance = geodesic(point_coords, pozo_coords).meters
            if distance < min_distance:
                min_distance = distance
                match_row = pozo_row
        
        matches.append((point_row['fna'], match_row['sigla'], min_distance))
    
    return pd.DataFrame(matches, columns=['fna', 'matched_sigla', 'distance'])

df_matches = match_points(df_puntos_de_extraccion, df_pozos_general)

# Merge dataframes with `idpozo`
df_pozos_general = df_pozos_general.merge(df_pozos_convencional, on='idpozo', how='left', suffixes=('', '_convencional'))
df_pozos_general = df_pozos_general.merge(df_pozos_noconvencional, on='idpozo', how='left', suffixes=('', '_noconvencional'))

# Metrics
total_points = len(df_puntos_de_extraccion)
matched_points = df_matches[df_matches['distance'] < 50].shape[0]  # Points matched within 50 meters
unmatched_points = total_points - matched_points

print(f"Total Points: {total_points}")
print(f"Matched Points: {matched_points}")
print(f"Unmatched Points: {unmatched_points}")

# Scatter plot of matched and unmatched points
plt.figure(figsize=(10, 6))
plt.scatter(df_pozos_general['longitude'], df_pozos_general['latitude'], color='blue', label='Pozos')
plt.scatter(df_puntos_de_extraccion['longitude'], df_puntos_de_extraccion['latitude'], color='red', label='Points')
plt.legend()
plt.title('Geographic Locations of Pozos and Points')
plt.show()

# Bar chart of match statistics
plt.bar(['Matched', 'Unmatched'], [matched_points, unmatched_points], color=['green', 'orange'])
plt.title('Matching Statistics')
plt.show()

df_matches_matched = df_matches[df_matches['distance'] < 50]
df_matches_unmatched = df_matches[df_matches['distance'] >= 50]

df_matches_matched.to_csv('./data/matched_points.csv', index=False)
df_matches_unmatched.to_csv('./data/unmatched_points.csv', index=False)
