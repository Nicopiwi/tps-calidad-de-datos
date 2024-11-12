import pandas as pd
from mpl_toolkits.basemap import Basemap
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from xlrd import open_workbook

# Load workbook and default sheet
wb = open_workbook('initial_data.xls', formatting_info=True)
default_sheet = wb.sheet_by_index(0)

# Set parameters and matchings
HEADER = 'HEADER'
BLANK = 'BLANK'
AIR_SEA = 'AIR/SEA'
UNPROVOKED = 'UNPROVOKED'
PROVOKED = 'PROVOKED'
QUESTIONABLE = 'QUESTIONABLE'
BOAT = 'BOAT'
MISMATCH = 'MISMATCH'

incident_types_names = {
    HEADER: "Header row (not an attack)",
    BLANK: "Blank",
    AIR_SEA: "Air/Sea Distasters",
    UNPROVOKED: "Unprovoked Incidents",
    PROVOKED: "Provoked Incidents",
    QUESTIONABLE: "Questionable Incidents",
    BOAT: "Attacks on Boats"
}

color_to_string = {
    # Header
    (0, 0, 128): incident_types_names[HEADER],
    # No color
    None: incident_types_names[BLANK],
    # Yellow
    (255, 255, 204): incident_types_names[AIR_SEA],
    # Tan
    (237, 218, 181): incident_types_names[UNPROVOKED],
    # Orange
    (255, 204, 153): incident_types_names[PROVOKED],
    # Blue
    (153, 204, 255): incident_types_names[QUESTIONABLE],
    # Green
    (207, 238, 204): incident_types_names[BOAT],
    # Yellow (another "tone", but practically the same)
    (255, 255, 197): incident_types_names[AIR_SEA]
}

type_and_color_matching = {
    "Unprovoked": incident_types_names[UNPROVOKED],
    "Provoked": incident_types_names[PROVOKED],
    "Questionable": incident_types_names[QUESTIONABLE],
    "?": incident_types_names[QUESTIONABLE],
    "Unconfirmed": incident_types_names[QUESTIONABLE],
    "Unverified": incident_types_names[QUESTIONABLE],
    "Invalid": incident_types_names[QUESTIONABLE], # Revisar
    "Under Investigation": incident_types_names[QUESTIONABLE],
    "Boat": incident_types_names[BOAT],
    "Watercraft": incident_types_names[BOAT],
    "Sea Disaster": incident_types_names[AIR_SEA],
    "": incident_types_names[BLANK],
}

info_graph_color = {
    incident_types_names[HEADER]: (255, 255, 255),
    incident_types_names[BLANK]: (0, 0, 0),
    incident_types_names[AIR_SEA]: (255, 255, 204),
    incident_types_names[UNPROVOKED]: (237, 218, 181),
    incident_types_names[PROVOKED]: (255, 204, 153),
    incident_types_names[QUESTIONABLE]: (153, 204, 255),
    incident_types_names[BOAT]: (207, 238, 204),
    MISMATCH: (128, 128, 128)
}

stats = {}

# (STEP) Initial Data Loading
data = []
for row in range(default_sheet.nrows):
    row_data = [default_sheet.cell(row, col).value for col in range(default_sheet.ncols)]
    data.append(row_data)

columns = [default_sheet.cell(0, col).value for col in range(default_sheet.ncols)]  # Use first row as header
df = pd.DataFrame(data, columns=columns)

# (STEP) Capture initial stats
stats['Initial Rows'] = df.shape[0]
stats['Initial Columns'] = df.shape[1]

# (STEP) Extract Incident Types (Color Matching)
def extract_incident_types(sheet):
    incident_types = []
    for row in range(sheet.nrows):
        c = sheet.cell(row, 1)
        xf = wb.xf_list[c.xf_index]
        color = wb.colour_map.get(xf.background.pattern_colour_index)
        incident_type = color_to_string.get(color, "Unknown")  # Default to "Unknown"
        incident_types.append(incident_type)

    return incident_types

incident_types_col = extract_incident_types(default_sheet)
df['Incident Type (color)'] = incident_types_col

# (STEP) Remove First Two Rows (Headers or Redundant Rows)
df = df.drop(index=[0, 1]).reset_index(drop=True)

# (STEP) Remove Empty Columns
initial_cols = df.shape[1]
df = df.loc[:, (df != '').any(axis=0)]
df.dropna(how='all', axis=1, inplace=True)
stats['Remaining Columns'] = df.shape[1]

# (STEP) Remove Empty Rows
initial_rows = df.shape[0]
df = df.dropna(how='all')
stats['Remaining Rows'] = df.shape[0]

# (STEP) Remove rows where all three "important" values are empty (Country, Type, and Incident Type (color))
df = df[~((df['Country'].isna() | (df['Country'] == '')) & 
          (df['Type'].isna() | (df['Type'] == '')) & 
          (df['Incident Type (color)'] == incident_types_names[BLANK]))]
stats['Remaining Rows Empty'] = df.shape[0]

# Reset the index after filtering rows
df = df.reset_index(drop=True)

# (STEP) Normalize, Lowercase, and Strip Country Column
df['Country'] = (
    df['Country']
    .str.normalize('NFKD')
    .str.encode('ascii', errors='ignore')
    .str.decode('utf-8')
    .str.upper()
    .str.strip()
)
stats['Unique Countries'] = df['Country'].nunique()

# (STEP) Clean countries thoroughly, and match both countries and islands
df_islas = pd.read_csv('islas_data_copy.csv')
country_dict = pd.Series(df_islas['NATION(S)'].values, index=df_islas['ISLAND']).to_dict()

with open('countries.txt', 'r') as file:
    lines = file.readlines()
    
country_list = [line.strip().upper() for line in lines]

# Add countries as both key and value.
for country in country_list:
    country_dict[country] = country

# Special cases (ADD TO CSV LATER)
country_dict['USA'] = 'USA'
country_dict['United States'] = 'USA'
country_dict['United States of America'] = 'USA'
country_dict[''] = 'Blank'
country_dict['ENGLAND'] = 'UNITED KINGDOM'
country_dict['HONG KONG'] = 'HONG KONG' # China?
country_dict['PACIFIC OCEAN'] = 'PACIFIC OCEAN' 
country_dict['ATLANTIC OCEAN'] = 'ATLANTIC OCEAN'
country_dict['INDIAN OCEAN'] = 'INDIAN OCEAN'
country_dict['CARIBBEAN SEA'] = 'CARIBBEAN SEA'
country_dict['COLUMBIA'] = 'COLUMBIA'
country_dict['SCOTLAND'] = 'SCOTLAND'

df['Matched Country'] = df['Country'].map(country_dict).fillna('NO MATCH')
df.loc[df['Country'].str.contains('/'), 'Matched Country'] = "NON-IDENTIFIABLE"

stats['Countries'] = len(country_list)  # Number of Countries (unique countries from country_list)
stats['Islands'] = len(country_dict) - len(country_list)  # Number of Islands (keys in country_dict)
stats['Matched Countries/Islands'] = (df['Matched Country'] != 'NO MATCH').sum()
stats['Countries Left After Matching'] = df['Matched Country'].nunique()
stats['Countries Not Matched'] = (~(df['Matched Country'] != 'NO MATCH')).sum()
stats['Null Countries'] = (df['Country'] == '').sum()

# (STEP) Match Types and Incident Types (Color) - Type Matching Stats
def match_incident_types(df):
    def match_type(row):
        type_value = type_and_color_matching.get(row['Type'], MISMATCH)
        color_value = row['Incident Type (color)']
        
        return type_value if type_value == color_value else MISMATCH
    
    df['Matched Type'] = df.apply(match_type, axis=1)
    return df

df = match_incident_types(df)
stats['Type Matches'] = (df['Matched Type'] != MISMATCH).sum()
stats['Type Mismatches'] = (df['Matched Type'] == MISMATCH).sum()
stats['Final Matched Types'] = df['Matched Type'].value_counts().to_dict()

# (STEP) Save to CSV
df.to_csv('./script_data.csv', index=False)

# Display statistics for rows, columns, and mismatches
print(f"Numero de islas (segun excel): {stats['Islands']}")
print(f"Numero de paises (segun lista .txt): {stats['Countries']}")
print(f"Total de islas/paises matcheados: {stats['Matched Countries/Islands']}")
print(f"Total de casos sin matchear: {stats['Countries Not Matched']}")
print(f"Total de casos en blanco: {stats['Null Countries']}")
print(f"Valores unicos en columna 'Country' antes de limpiar: {stats['Unique Countries']}")
print(f"Valores unicos luego de matchear: {stats['Countries Left After Matching']}")

print()

top_countries = df[df['Matched Country'] != 'NO MATCH']['Matched Country'].value_counts().head(20)
no_match_rows = df[df['Matched Country'] == 'NO MATCH']

print("Top 20 paises con mayores casos")
print(top_countries)

print()

top_no_match = no_match_rows['Country'].value_counts()
print("Paises/islas sin matchear")
print(top_no_match)

print()

print("Filas iniciales:", stats['Initial Rows'])
print("Filas restantes (luego de eliminar vacias):", stats['Remaining Rows'])
print("Filas restantes (luego de eliminar 'Country', 'Type' y 'Incident Type (color)' nulos):", stats['Remaining Rows Empty'])
print("Columnas iniciales:", stats['Initial Columns'])
print("Columnas restantes (luego de eliminar vacias):", stats['Remaining Columns'])
print("Total mismatch incidentes:", stats['Type Mismatches'])


# Function to display percentages only for slices larger than 3%
def autopct_func(pct):
    return f'{pct:.1f}%' if pct > 3 else ''

# Normalize RGB colors to the range [0, 1]
def normalize_rgb(rgb):
    return tuple(c / 255 for c in rgb)

# Pie Chart for Matched Types Distribution with normalized colors
matched_type_labels, matched_type_counts = zip(*stats['Final Matched Types'].items())
explode = [0.05] * len(matched_type_labels)  # Slightly "explode" each slice for visibility

# Extract normalized colors from info_graph_color for the pie chart
pie_chart_colors = [normalize_rgb(info_graph_color[type_]) for type_ in matched_type_labels]

total_count = sum(matched_type_counts)
legend_labels = [f"{label} ({(count / total_count) * 100:.1f}%)" for label, count in zip(matched_type_labels, matched_type_counts)]

plt.figure(figsize=(8, 8))
plt.pie(
    matched_type_counts, labels=[None] * len(matched_type_labels), autopct=autopct_func, 
    startangle=140, explode=explode, colors=pie_chart_colors, textprops={'fontsize': 10}
)
plt.title('Tipos de incidentes (matcheados)', fontsize=14)
plt.legend(legend_labels, title="Incident Types")
plt.tight_layout()

plt.savefig('./pie_chart.png')
plt.close()

# Bar Chart for Incident Types Count with normalized colors
plt.figure(figsize=(8, 6))

# Extract normalized colors from info_graph_color for the bar chart
bar_chart_colors = [normalize_rgb(info_graph_color[type_]) for type_ in matched_type_labels]

plt.bar(matched_type_labels, matched_type_counts, color=bar_chart_colors)
plt.title('Número de Incidentes por Tipo', fontsize=14)
plt.xlabel('Tipo de Incidente', fontsize=12)
plt.ylabel('Número de Incidentes', fontsize=12)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility

# Optional: add count labels on top of the bars
for i, count in enumerate(matched_type_counts):
    plt.text(i, count + 1, str(count), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('./incident_type_counts.png')
plt.close()

# Top countries
plt.figure(figsize=(10, 6))
top_countries.plot(kind='bar', color='skyblue')

plt.title('20 Países con más casos', fontsize=16)
plt.xlabel('País', fontsize=12)
plt.ylabel('Número de casos', fontsize=12)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('./top_countries.png')
plt.close()

# Load the world shapefile
url = "./ne_110m_admin_0_countries.zip"
world = gpd.read_file(url)

country_counts = df['Matched Country'].value_counts().reset_index()
country_counts.columns = ['Matched Country', 'EventCount']
country_dict = dict(zip(country_counts['Matched Country'], country_counts['EventCount']))

fig, ax = plt.subplots(figsize=(15, 10))

m = Basemap(projection='robin', resolution='c', lat_0=0, lon_0=0)

# Draw coastlines and countries
m.drawcoastlines()
m.drawcountries()

# Draw the map boundaries and fill color
m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='lightgreen', lake_color='lightblue')

# Iterate through each country in the shapefile
for _, row in world.iterrows():
    country_name = row['SOVEREIGNT'].upper()
    print(country_name)
    if country_name in country_dict:
        # Get the centroid coordinates of the country
        centroid = row['geometry'].centroid
        lat, lon = centroid.y, centroid.x
        
        x, y = m(lon, lat)
        
        ax.text(x, y, str(country_dict[country_name]), fontsize=8, ha='center', color='black')

plt.title('World Map of Event Counts')
plt.savefig('./world_map.png')
plt.close()