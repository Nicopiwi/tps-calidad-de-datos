import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

def identify_outliers_grouped(df, group_col, value_col):
    df = df.reset_index(drop=True)
    grouped = df.groupby(group_col)
    df['zscore'] = grouped[value_col].transform(lambda x: zscore(x, nan_policy='omit'))
    df['outlier'] = df['zscore'].abs() > 3
    outliers = df[df['outlier']]
    return df, outliers


data = pd.read_csv('./data/capitulo-iv-pozos.csv')
general_2024 = pd.read_csv('./data/produccin-de-pozos-de-gas-y-petrleo-2024.csv', low_memory=False)
noconvencional = pd.read_csv('./data/produccin-de-pozos-de-gas-y-petrleo-no-convencional.csv', low_memory=False)

noconvencional_2024 = noconvencional[noconvencional['anio'] == 2024]
convencional_2024 = general_2024[general_2024['tipo_de_recurso'] == 'CONVENCIONAL']
noconvencional_2024_general = general_2024[general_2024['tipo_de_recurso'] == 'NO CONVENCIONAL']

print(general_2024.columns)
print(noconvencional.columns)

# MILES DE M3
df_combined_gas = pd.concat([
    convencional_2024[['tipoextraccion', 'prod_gas', 'idpozo', 'mes']].assign(tipo='convencional'),
    noconvencional_2024[['tipoextraccion', 'prod_gas', 'idpozo', 'mes']].assign(tipo='no_convencional'),
    noconvencional_2024_general[['tipoextraccion', 'prod_gas', 'idpozo', 'mes']].assign(tipo='no_convencional')
])

print(df_combined_gas.head())

# M3
df_combined_pet = pd.concat([
    convencional_2024[['tipoextraccion', 'prod_pet', 'idpozo', 'mes']].assign(tipo='convencional'),
    noconvencional_2024[['tipoextraccion', 'prod_pet', 'idpozo', 'mes']].assign(tipo='no_convencional'),
    noconvencional_2024_general[['tipoextraccion', 'prod_pet', 'idpozo', 'mes']].assign(tipo='no_convencional')
])

print(df_combined_pet.columns)
print(df_combined_gas.columns)

# Loop through datasets for gas and petroleum
for idx, (df_combined, value_col, title) in enumerate([
    (df_combined_gas, 'prod_gas', r'gas, miles de $m^3$'),
    (df_combined_pet, 'prod_pet', r'petroleo, $m^3$')]):

    df_combined, outliers = identify_outliers_grouped(df_combined, 'tipoextraccion', value_col)
    
    # Calculate mean production per extraction type
    mean_per_tipoextraccion = df_combined.groupby('tipoextraccion')[value_col].mean()
    outliers_mean_per_tipoextraccion = outliers.groupby('tipoextraccion')[value_col].mean()
    num_outliers_per_tipoextraccion = outliers.groupby('tipoextraccion')[value_col].size()
    total_per_tipoextraccion = df_combined.groupby('tipoextraccion').size()

    percentage_outliers = (num_outliers_per_tipoextraccion / total_per_tipoextraccion * 100).round(2)

    print(f"\nAverage {title} for each 'tipoextraccion':")
    print(mean_per_tipoextraccion)

    print(f"\nAverage {title} for outliers by 'tipoextraccion':")
    print(outliers_mean_per_tipoextraccion)

    print(f"\nNumber of outliers in {title} by 'tipoextraccion':")
    print(num_outliers_per_tipoextraccion)

    print(f"\nPercentage of outliers in {title} by 'tipoextraccion':")
    print(percentage_outliers)
    for tipo, percentage in percentage_outliers.items():
        if tipo in total_per_tipoextraccion.keys() and tipo in num_outliers_per_tipoextraccion.keys():
            print(f"{tipo:<20}: {percentage:>6.2f}% ({num_outliers_per_tipoextraccion[tipo]:>4}/{total_per_tipoextraccion[tipo]:>4})")

    # Plot mean production by extraction type
    plt.figure(figsize=(12, 6))
    mean_per_tipoextraccion.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Promedio mensual ({title}) por tipo de extracción', fontsize=16)
    plt.ylabel(f'Promedio mensual ({title})', fontsize=12)
    plt.xlabel('Tipo de extracción', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'./mean_{value_col}_by_extraction_type.png')

     # Boxplot to visualize distribution and outliers
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='tipoextraccion', y=value_col, data=df_combined, width=0.6, palette='Set2')

    # Optional: Increase the y-axis range if needed for extreme outliers
    plt.title(f'Boxplot de {title} por tipo de extracción', fontsize=16)
    plt.ylabel(f'{value_col} Producción', fontsize=12)
    plt.xlabel('Tipo de extracción', fontsize=12)
    plt.xticks(rotation=45)

    # Adjust y-axis range to ensure visibility of box, especially when outliers are extreme
    plt.ylim(df_combined[value_col].quantile(0.01) - 10, df_combined[value_col].quantile(0.99) + 10)

    plt.tight_layout()
    plt.savefig(f'./boxplot_{value_col}_by_extraction_type.png')
    plt.show()

# ----------

def identify_outliers(df, value_col):
    """Identify outliers based on z-score."""
    df['zscore'] = zscore(df[value_col], nan_policy='omit')
    df['outlier'] = df['zscore'].abs() > 3
    outliers = df[df['outlier']]
    return df, outliers

for idx, (df_combined, value_col, title) in enumerate([
    (df_combined_gas, 'prod_gas', r'gas, miles de $m^3$'),
    (df_combined_pet, 'prod_pet', r'petroleo, m^3')]):

    df_combined, outliers = identify_outliers(df_combined, value_col)
    mean_value = df_combined[value_col].mean()
    outliers_mean_value = outliers[value_col].mean()
    num_outliers = outliers.shape[0]
    total_entries = df_combined.shape[0]
    percentage_outliers = (num_outliers / total_entries) * 100

    print(f"\nAverage {title} for the entire dataset:")
    print(f"{mean_value:.2f}")

    print(f"\nAverage {title} for outliers:")
    print(f"{outliers_mean_value:.2f}")

    print(f"\nNumber of outliers in {title}:")
    print(f"{num_outliers}")

    print(f"\nPercentage of outliers in {title}:")
    print(f"{percentage_outliers:.2f}% ({num_outliers}/{total_entries})")

    # Plot mean production for entire dataset
    plt.figure(figsize=(12, 6))
    df_combined[value_col].plot(kind='hist', bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Distribución {title}', fontsize=16)
    plt.xlabel(f'Producción ({value_col})', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'./distribution_{value_col}_entire_dataset.png')

# -----

df_combined_gas = df_combined_gas.sort_values(by=['idpozo', 'mes'])
df_combined_pet = df_combined_pet.sort_values(by=['idpozo', 'mes'])

# Function to perform anomaly detection and visualize results
def detect_anomalies(df, value_col, title, threshold=3):
    """
    Detects anomalies in the time series data for each well using z-scores
    and highlights sudden changes in performance.
    """
    # Group by well ('idpozo') and iterate
    anomalies = []

    for well, group in df.groupby('idpozo'):
        print(well)

        group = group.set_index('mes').sort_index()

        # Calculate rolling mean and std for anomaly detection (e.g., 3-month window)
        group['rolling_mean'] = group[value_col].rolling(window=3).mean()
        group['rolling_std'] = group[value_col].rolling(window=3).std()

        # Calculate z-score (for anomaly detection)
        group['zscore'] = (group[value_col] - group['rolling_mean']) / group['rolling_std']
        group['anomaly'] = group['zscore'].abs() > threshold

        anomalies_group = group[group['anomaly']]

        if not anomalies_group.empty:
            print("added_anomaly")

            anomalies_group = anomalies_group[['idpozo', 'tipoextraccion', 'anomaly']]
            anomalies_group['value'] = anomalies_group[value_col]
            anomalies.append(anomalies_group)

    # Combine all anomalies into one DataFrame
    anomalies_df = pd.concat(anomalies)

    # Print number of anomalies and details
    print(f"\nNumber of anomalies detected: {len(anomalies_df)}")
    print("\nAnomalies details (idpozo, tipoextraccion, production value):")
    print(anomalies_df[['idpozo', 'tipoextraccion', 'value']])

    return anomalies_df

# Detect anomalies for gas production
# anomalies_gas = detect_anomalies(df_combined_gas, 'prod_gas', 'Gas Production')

# # Detect anomalies for petroleum production
# anomalies_pet = detect_anomalies(df_combined_pet, 'prod_pet', 'Petroleum Production')