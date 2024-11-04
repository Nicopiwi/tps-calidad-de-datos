import pandas as pd
import numpy as np
from xlrd import open_workbook

wb = open_workbook('initial_data.xls', formatting_info=True)
default_sheet = wb.sheet_by_index(0) 

color_to_string = {
    # Header
    (0, 0, 128): "",
    # No color
    None: "",
    # Yellow
    (255, 255, 204): "Air/Sea Disasters",
    # Tan
    (237, 218, 181): "Unprovoked Incidents",
    # Orange
    (255, 204, 153): "Provoked Incidents",
    # Blue
    (153, 204, 255): "Questionable Incidents",
    # Green
    (207, 238, 204): "Attacks on Boats",
    # Yellow (another "tone", but practically the same)
    (255, 255, 197): "Air/Sea Disasters"
}

def extract_incident_types(sheet):
    incident_types = []

    for row in range(sheet.nrows):
        c = sheet.cell(row, 1)
        xf = wb.xf_list[c.xf_index]
        color = wb.colour_map.get(xf.background.pattern_colour_index)

        incident_type = color_to_string.get(color, "Unknown")  # Default to "Unknown"
        incident_types.append(incident_type)

    return incident_types

incident_types = extract_incident_types(default_sheet)

data = []
for row in range(default_sheet.nrows):
    row_data = [default_sheet.cell(row, col).value for col in range(default_sheet.ncols)]
    data.append(row_data)

columns = [default_sheet.cell(0, col).value for col in range(default_sheet.ncols)]  # Use first row as header
df = pd.DataFrame(data, columns=columns)

df['Incident Type'] = incident_types

# Remove first two rows (redundant)
df = df.drop(index=0)
df = df.drop(index=1)
df = df.reset_index(drop=True)  # Reset the index

# Remove empty columns
df = df.loc[:, (df != '').any(axis=0)]
df.dropna(how='all', axis=1, inplace=True)

df.to_csv('../data_versions/1_initial_csv.csv', index=False)