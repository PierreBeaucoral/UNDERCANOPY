import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set working directory
import os
wd = os.getenv("PROJECT_DIR", ".")  # Use an environment variable or default to current dir

# Helper function for data import
def csv_import(filepath, delimiter="|"):
    """Helper function to safely import CSV files."""
    try:
        return pd.read_csv(filepath, encoding='utf8', delimiter=delimiter, dtype={'text': str, 'USD_Disbursement': float})
    except FileNotFoundError:
        print(f'Error: File {filepath} not found.')
        return None
    return pd.read_csv(filepath, encoding='utf8', delimiter=delimiter, dtype={
        'text': str, "USD_Disbursement": float
    })

# Import datasets
df_origin = pd.read_csv('os.path.join(wd, "data", "DataPB.csv")', encoding='utf8', delimiter='|')
df = csv_import(os.path.join(wd, "data", "climate_finance_total.csv"))

# Create disbursement timeline (Refactored)
def create_funding_timeline(df, output_path):
    df_time = df.rename(columns={'Year': 'effective_year', 'USD_Disbursement_Defl': 'effective_funding'})
    df_time['adaptation_funding'] = np.where(df_time['meta_category'] == 'Adaptation', df_time['effective_funding'], 0)
    df_time['mitigation_funding'] = np.where(df_time['meta_category'] == 'Mitigation', df_time['effective_funding'], 0)
    df_time['environment_funding'] = np.where(df_time['meta_category'] == 'Environment', df_time['effective_funding'], 0)
    
    df_time = df_time[['effective_year', 'adaptation_funding', 'mitigation_funding', 'environment_funding']].groupby('effective_year').sum().reset_index()
    df_time.to_csv(output_path, index=False)
    return df_time

def create_funding_timeline_by_donor(df, output_path):
    # Rename columns for clarity
    df_time = df.rename(columns={'Year': 'effective_year', 'USD_Disbursement_Defl': 'effective_funding'})
    
    # Create separate funding categories based on 'meta_category'
    df_time['adaptation_funding'] = np.where(df_time['meta_category'] == 'Adaptation', df_time['effective_funding'], 0)
    df_time['mitigation_funding'] = np.where(df_time['meta_category'] == 'Mitigation', df_time['effective_funding'], 0)
    df_time['environment_funding'] = np.where(df_time['meta_category'] == 'Environment', df_time['effective_funding'], 0)
    
    # Group by both 'DonorName' and 'effective_year', then sum the funding values
    df_time_grouped = df_time.groupby(['DonorName', 'effective_year']).agg({
        'adaptation_funding': 'sum',
        'mitigation_funding': 'sum',
        'environment_funding': 'sum'
    }).reset_index()
    
    # Save the grouped data to a CSV file
    df_time_grouped.to_csv(output_path, index=False)
    
    return df_time_grouped


# Create commitment timeline (Refactored for USD_commitment_Defl)
def create_commitment_timeline(df, output_path):
    df_time = df.rename(columns={'Year': 'effective_year', 'USD_Commitment_Defl': 'effective_commitment'})
    df_time['adaptation_commitment'] = np.where(df_time['meta_category'] == 'Adaptation', df_time['effective_commitment'], 0)
    df_time['mitigation_commitment'] = np.where(df_time['meta_category'] == 'Mitigation', df_time['effective_commitment'], 0)
    df_time['environment_commitment'] = np.where(df_time['meta_category'] == 'Environment', df_time['effective_commitment'], 0)
    
    df_time = df_time[['effective_year', 'adaptation_commitment', 'mitigation_commitment', 'environment_commitment']].groupby('effective_year').sum().reset_index()
    df_time.to_csv(output_path, index=False)
    return df_time

# Rio marker preparation (Adaptation & Mitigation)
def prepare_rio_data(df_origin):
    df_origin = df_origin[df_origin['Year'] >= 2000].reset_index(drop=True)

    # Add DonorType classification
    df_origin['DonorType'] = np.select(
        [
            df_origin.DonorCode < 807,
            df_origin.DonorCode > 1600,
            df_origin.DonorCode == 104,
            df_origin.DonorCode == 820
        ],
        ['Donor Country', 'Private Donor', 'Multilateral Donor Organization', 'Donor Country'],
        default='Multilateral Donor Organization'
    )

    # Filter to only include rows with Donor Country
    df_origin = df_origin[df_origin['DonorType'] == 'Donor Country'].reset_index(drop=True)

    # Group by climate adaptation/mitigation markers for disbursements and commitments
    def group_marker(marker_col, funding_col, df, significant_value, principal_value, start_year):
        rio_data_significant = df[df[marker_col] == significant_value][[funding_col, 'Year']].groupby('Year').sum().reset_index()
        rio_data_principal = df[df[marker_col] == principal_value][[funding_col, 'Year']].groupby('Year').sum().reset_index()
        
        rio_data_filtered_significant = rio_data_significant[rio_data_significant['Year'] >= start_year][funding_col] / 1000
        rio_data_filtered_principal = rio_data_principal[rio_data_principal['Year'] >= start_year][funding_col] / 1000
        
        zero_padding_significant = [0] * (start_year - 2000)
        zero_padding_principal = [0] * (start_year - 2000)
        
        return zero_padding_significant + rio_data_filtered_significant.tolist(), zero_padding_principal + rio_data_filtered_principal.tolist()

    # For adaptation (starting from 2008)
    rio_adapt1_disbursement, rio_adapt2_disbursement = group_marker('ClimateAdaptation', 'USD_Disbursement_Defl', df_origin, significant_value=1, principal_value=2, start_year=2008)
    rio_adapt1_commitment, rio_adapt2_commitment = group_marker('ClimateAdaptation', 'USD_Commitment_Defl', df_origin, significant_value=1, principal_value=2, start_year=2008)

    # For mitigation (starting from 2000)
    rio_miti1_disbursement, rio_miti2_disbursement = group_marker('ClimateMitigation', 'USD_Disbursement_Defl', df_origin, significant_value=1, principal_value=2, start_year=2000)
    rio_miti1_commitment, rio_miti2_commitment = group_marker('ClimateMitigation', 'USD_Commitment_Defl', df_origin, significant_value=1, principal_value=2, start_year=2000)

    return (rio_adapt1_disbursement, rio_adapt2_disbursement, rio_adapt1_commitment, rio_adapt2_commitment,
            rio_miti1_disbursement, rio_miti2_disbursement, rio_miti1_commitment, rio_miti2_commitment)

def prepare_rio_data_by_country(df_origin, start_year=2000):
    # Filter by Donor Country
    df_origin = df_origin[df_origin['DonorName'] == 'Donor Country'].reset_index(drop=True)
    
    # Create the necessary groupings by provider country and year for disbursements and commitments
    def group_by_provider_and_marker(marker_col, funding_col, df, significant_value, principal_value, start_year):
        # Group by DonorCountry and Year for both significant and principal markers
        rio_data_significant = df[df[marker_col] == significant_value].groupby(['DonorCountry', 'Year'])[funding_col].sum().reset_index()
        rio_data_principal = df[df[marker_col] == principal_value].groupby(['DonorCountry', 'Year'])[funding_col].sum().reset_index()

        # Filter for relevant years
        rio_data_filtered_significant = rio_data_significant[ rio_data_significant['Year'] >= start_year]
        rio_data_filtered_principal = rio_data_principal[ rio_data_principal['Year'] >= start_year]

        return rio_data_filtered_significant, rio_data_filtered_principal

    # For adaptation (starting from 2008)
    rio_adapt1_disbursement, rio_adapt2_disbursement = group_by_provider_and_marker('ClimateAdaptation', 'USD_Disbursement_Defl', df_origin, significant_value=1, principal_value=2, start_year=2008)
    rio_adapt1_commitment, rio_adapt2_commitment = group_by_provider_and_marker('ClimateAdaptation', 'USD_Commitment_Defl', df_origin, significant_value=1, principal_value=2, start_year=2008)

    # For mitigation (starting from 2000)
    rio_miti1_disbursement, rio_miti2_disbursement = group_by_provider_and_marker('ClimateMitigation', 'USD_Disbursement_Defl', df_origin, significant_value=1, principal_value=2, start_year=2000)
    rio_miti1_commitment, rio_miti2_commitment = group_by_provider_and_marker('ClimateMitigation', 'USD_Commitment_Defl', df_origin, significant_value=1, principal_value=2, start_year=2000)

    return (rio_adapt1_disbursement, rio_adapt2_disbursement, rio_adapt1_commitment, rio_adapt2_commitment,
            rio_miti1_disbursement, rio_miti2_disbursement, rio_miti1_commitment, rio_miti2_commitment)



# Stacked area plot
def stacked_area(df, output_folder, input_colors, max_number, data_type):
    y = df.drop('effective_year', axis=1)
    legend = list(y.columns)
    y = y.T
    x = list(df['effective_year'])
    
    fig, ax = plt.subplots()
    ax.stackplot(x, y / 1000, colors=input_colors)
    ax.tick_params(labelsize=11)
    
    # Set the Y-axis label based on the data type
    ylabel_text = f'Aggregated {"disbursements" if data_type == "disbursements" else "commitments"} (billion USD)'
    ax.set_ylabel(ylabel_text, labelpad=15, color='#333333', fontsize=11)
    
    xticks = np.arange(min(x), max(x) + 1, 1.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks.astype(int), rotation=45)

    plt.xlim(2000, 2023)
    
    legend_list = [Patch(facecolor=input_colors[i], label=legend[i]) for i in range(max_number)]
    
    ax.legend(handles=legend_list, loc="upper left", borderaxespad=0, prop={'size': 8}, frameon=True)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    plt.savefig(output_folder, bbox_inches="tight", dpi=1200)
    plt.show()

def rio_stacked_area(cluster, rio_principal, rio_significant, output_folder, rio_colors, cluster_color, climate_type, max_y, data_type):
    # Filter the data for years from 2000 to 2022
    years = list(range(2000, 2023))
    cluster_filtered = cluster[-len(years):]
    rio_principal_filtered = rio_principal[-len(years):]
    rio_significant_filtered = rio_significant[-len(years):]

    # Plot the figure
    fig, ax = plt.subplots(figsize=(24, 8))
    
    # Ensure the order is principal first, then significant
    ax.stackplot(years, [rio_principal_filtered, rio_significant_filtered], colors=[rio_colors[1], rio_colors[0]], alpha=0.5)
    
    # Plot the cluster data
    ax.plot(years, cluster_filtered, color=cluster_color, linewidth=6)
    
    # Add annotations for significant climate events
    events = [(2010, 'USD 100bn target'), (2015, 'Paris Agreement')]
    for year, event in events:
        ax.vlines(x=year, ymin=0, ymax=max_y - 3, linestyle='--', color='#636363')
        ax.text(x=year, y=max_y - 2, s=event, ha='center')
    
    # Set labels and tick marks
    ylabel_text = f'Aggregated aid {data_type} (billion USD)\nfor Climate Change {climate_type}'
    ax.set_ylabel(ylabel_text, labelpad=15, color='#333333', fontsize=12)
    
    plt.xticks(np.arange(min(years), max(years) + 1, 2), rotation=45)
    plt.xlim(2000, 2023)
    plt.ylim(0, max_y)
    
    # Add legend
    legend_list = [
        Patch(facecolor=cluster_color, label='ClimateFinanceBERT'),
        Patch(facecolor=rio_colors[1], label='Rio markers principal'),  # Change order for legend
        Patch(facecolor=rio_colors[0], label='Rio markers significant')
    ]
    ax.legend(handles=legend_list, loc="upper left", prop={'size': 12}, frameon=True)
    
    # Save the figure using plt.savefig
    plt.tight_layout()
    plt.savefig(output_folder, bbox_inches="tight", dpi=1200)
    plt.show()

    years = list(range(2000, 2023))
    
    # Create subplots: 1 row, 2 columns, one for each provider country
    fig, axs = plt.subplots(len(provider_countries), 2, figsize=(24, 6 * len(provider_countries)), sharey=True)
    
    events = [(2010, 'USD 100bn target'), (2015, 'Paris Agreement')]

    for i, country in enumerate(provider_countries):
        # For Adaptation
        axs[i, 0].stackplot(years, [rio_adapt2[country], rio_adapt1[country]], colors=['#fec44f', '#fff7bc'], alpha=0.5)
        axs[i, 0].plot(years, cluster_adap[country], color='#d95f0e', linewidth=6)
        axs[i, 0].set_title(f'Adaptation Funding ({country})', fontsize=14)
        axs[i, 0].set_ylabel('Aggregated aid disbursements (billion USD)', labelpad=15, color='#333333', fontsize=12)
        axs[i, 0].set_xlim(2000, 2023)
        axs[i, 0].set_ylim(0, 23)
        axs[i, 0].set_xticks(np.arange(2000, 2024, 2))
        
        # Annotate significant events for Adaptation
        for year, event in events:
            axs[i, 0].vlines(x=year, ymin=0, ymax=22, linestyle='--', color='#636363')
            axs[i, 0].text(x=year, y=21, s=event, ha='center')
        
        # For Mitigation
        axs[i, 1].stackplot(years, [rio_miti2[country], rio_miti1[country]], colors=['#addd8e', '#f7fcb9'], alpha=0.5)
        axs[i, 1].plot(years, cluster_miti[country], color='#31a354', linewidth=6)
        axs[i, 1].set_title(f'Mitigation Funding ({country})', fontsize=14)
        axs[i, 1].set_ylabel('Aggregated aid disbursements (billion USD)', labelpad=15, color='#333333', fontsize=12)
        axs[i, 1].set_xlim(2000, 2023)
        axs[i, 1].set_ylim(0, 23)
        axs[i, 1].set_xticks(np.arange(2000, 2024, 2))
        
        # Annotate significant events for Mitigation
        for year, event in events:
            axs[i, 1].vlines(x=year, ymin=0, ymax=22, linestyle='--', color='#636363')
            axs[i, 1].text(x=year, y=21, s=event, ha='center')
    
    # Add legend below the plot
    legend_list = [
        Patch(facecolor='#d95f0e', label='ClimateFinanceBERT (Adaptation)'),
        Patch(facecolor='#fec44f', label='Rio markers principal (Adaptation)'),
        Patch(facecolor='#fff7bc', label='Rio markers significant (Adaptation)'),
        Patch(facecolor='#31a354', label='ClimateFinanceBERT (Mitigation)'),
        Patch(facecolor='#addd8e', label='Rio markers principal (Mitigation)'),
        Patch(facecolor='#f7fcb9', label='Rio markers significant (Mitigation)')
    ]
    fig.legend(handles=legend_list, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize=14)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'combined_adaptation_mitigation_by_country_plot.png'), bbox_inches='tight')
    plt.show()
def combined_adaptation_mitigation_plot_by_country(rio_adapt1_disbursement, rio_adapt2_disbursement, rio_miti1_disbursement, rio_miti2_disbursement, cluster_adap, cluster_miti, output_folder):
    # Get unique provider countries (DonorCountry) for plotting
    provider_countries = rio_adapt1_disbursement['DonorCountry'].unique()
    
    # Loop through each provider country and create a separate plot
    for country in provider_countries:
        fig, axs = plt.subplots(2, 1, figsize=(24, 12), sharey=True)

        # Filter data for the current provider country
        country_adap1 = rio_adapt1_disbursement[ rio_adapt1_disbursement['DonorCountry'] == country ]
        country_adap2 = rio_adapt2_disbursement[ rio_adapt2_disbursement['DonorCountry'] == country ]
        country_miti1 = rio_miti1_disbursement[ rio_miti1_disbursement['DonorCountry'] == country ]
        country_miti2 = rio_miti2_disbursement[ rio_miti2_disbursement['DonorCountry'] == country ]
        
        # Adaptation plot
        axs[0].stackplot(country_adap1['Year'], [country_adap2['USD_Disbursement_Defl'], country_adap1['USD_Disbursement_Defl']], colors=['#fec44f', '#fff7bc'], alpha=0.5)
        axs[0].plot(country_adap1['Year'], cluster_adap[cluster_adap['DonorCountry'] == country], color='#d95f0e', linewidth=6)
        axs[0].set_title(f'Adaptation Funding - {country}', fontsize=14)
        axs[0].set_ylabel('Aggregated aid disbursements (billion USD)', labelpad=15, color='#333333', fontsize=12)
        axs[0].set_xlim(2000, 2023)
        axs[0].set_ylim(0, 23)
        axs[0].set_xticks(np.arange(2000, 2024, 2))

        # Mitigation plot
        axs[1].stackplot(country_miti1['Year'], [country_miti2['USD_Disbursement_Defl'], country_miti1['USD_Disbursement_Defl']], colors=['#addd8e', '#f7fcb9'], alpha=0.5)
        axs[1].plot(country_miti1['Year'], cluster_miti[cluster_miti['DonorCountry'] == country], color='#31a354', linewidth=6)
        axs[1].set_title(f'Mitigation Funding - {country}', fontsize=14)
        axs[1].set_ylabel('Aggregated aid disbursements (billion USD)', labelpad=15, color='#333333', fontsize=12)
        axs[1].set_xlim(2000, 2023)
        axs[1].set_ylim(0, 23)
        axs[1].set_xticks(np.arange(2000, 2024, 2))

        # Add legend
        legend_list = [
            Patch(facecolor='#d95f0e', label='ClimateFinanceBERT (Adaptation)'),
            Patch(facecolor='#fec44f', label='Rio markers principal (Adaptation)'),
            Patch(facecolor='#fff7bc', label='Rio markers significant (Adaptation)'),
            Patch(facecolor='#31a354', label='ClimateFinanceBERT (Mitigation)'),
            Patch(facecolor='#addd8e', label='Rio markers principal (Mitigation)'),
            Patch(facecolor='#f7fcb9', label='Rio markers significant (Mitigation)')
        ]
        fig.legend(handles=legend_list, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize=14)

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'combined_adaptation_mitigation_plot_{country}.png'), bbox_inches='tight')
        plt.show()


# Combined Adaptation and Mitigation Plot
def combined_adaptation_mitigation_plot(cluster_adap, rio_adapt1, rio_adapt2, cluster_miti, rio_miti1, rio_miti2, output_folder):
    years = list(range(2000, 2023))
    cluster_adap = cluster_adap[-len(years):]
    cluster_miti = cluster_miti[-len(years):]
    
    # Create subplots: 1 row, 2 columns
    fig, axs = plt.subplots(2, 1, figsize=(24, 12), sharey=True)
    
    # Plot for Adaptation
    axs[0].stackplot(years, [rio_adapt2, rio_adapt1], colors=['#fec44f', '#fff7bc'], alpha=0.5)
    axs[0].plot(years, cluster_adap, color='#d95f0e', linewidth=6)
    axs[0].set_title('Adaptation Funding', fontsize=14)
    axs[0].set_ylabel('Aggregated aid disbursements (billion USD)', labelpad=15, color='#333333', fontsize=12)
    axs[0].set_xlim(2000, 2023)
    axs[0].set_ylim(0, 23)
    axs[0].set_xticks(np.arange(2000, 2024, 2))
    
    # Annotate significant events
    events = [(2010, 'USD 100bn target'), (2015, 'Paris Agreement')]
    for year, event in events:
        axs[0].vlines(x=year, ymin=0, ymax=22, linestyle='--', color='#636363')
        axs[0].text(x=year, y=21, s=event, ha='center')

    # Plot for Mitigation
    axs[1].stackplot(years, [rio_miti2, rio_miti1], colors=['#addd8e', '#f7fcb9'], alpha=0.5)
    axs[1].plot(years, cluster_miti, color='#31a354', linewidth=6)
    axs[1].set_title('Mitigation Funding', fontsize=14)
    axs[1].set_ylabel('Aggregated aid disbursements (billion USD)', labelpad=15, color='#333333', fontsize=12)
    axs[1].set_xlim(2000, 2023)
    axs[1].set_ylim(0, 23)
    axs[1].set_xticks(np.arange(2000, 2024, 2))
    
    # Annotate significant events for Mitigation
    for year, event in events:
        axs[1].vlines(x=year, ymin=0, ymax=22, linestyle='--', color='#636363')
        axs[1].text(x=year, y=21, s=event, ha='center')
    
    # Add legend below the plot
    legend_list = [
        Patch(facecolor='#d95f0e', label='ClimateFinanceBERT (Adaptation)'),
        Patch(facecolor='#fec44f', label='Rio markers principal (Adaptation)'),
        Patch(facecolor='#fff7bc', label='Rio markers significant (Adaptation)'),
        Patch(facecolor='#31a354', label='ClimateFinanceBERT (Mitigation)'),
        Patch(facecolor='#addd8e', label='Rio markers principal (Mitigation)'),
        Patch(facecolor='#f7fcb9', label='Rio markers significant (Mitigation)')
    ]
    fig.legend(handles=legend_list, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize=14)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'combined_adaptation_mitigation_plot.png'), bbox_inches='tight')
    plt.show()


def analyze_climate_finance_trends(df, output_folder, finance_type='disbursement'):
    """
    Analyze and decompose climate finance trends for adaptation, mitigation, and environment funding.
    
    Parameters:
    df: DataFrame with yearly climate finance data
    output_folder: Path to save the output figures
    finance_type: 'disbursement' or 'commitment'
    """
    # Create date index
    date_range = pd.date_range(start=f'{df["effective_year"].min()}/1/1', 
                              end=f'{df["effective_year"].max()}/12/31', 
                              freq='Y')
    
    # Prepare data for each category
    categories = {
        'Adaptation': f'adaptation_{finance_type}',
        'Mitigation': f'mitigation_{finance_type}',
        'Environment': f'environment_{finance_type}'
    }
    
    colors = {
        'Adaptation': '#d95f0e',
        'Mitigation': '#2ca25f',
        'Environment': '#2b8cbe'
    }
    
    for category_name, column in categories.items():
        # Convert to billions and create time series
        time_series = pd.Series(df[column].values / 1000, index=date_range)
        
        # Handle any missing values
        time_series = time_series.fillna(method='ffill')
        time_series = time_series.dropna()
        
        # Decompose the time series
        decomposition = seasonal_decompose(time_series, period=2, model='additive')
        
        # Create the plot
        plt.figure(figsize=(15, 12))
        
        # Original Time Series
        plt.subplot(3, 1, 1)
        plt.plot(time_series, label=f'{category_name} Finance', color=colors[category_name], linewidth=2)
        plt.title(f'{category_name} Climate Finance Time Series ({finance_type.capitalize()})')
        plt.ylabel('Billion USD')
        plt.legend()
        
        # Trend Component
        plt.subplot(3, 1, 2)
        plt.plot(decomposition.trend, label='Trend', color='darkblue', linewidth=2)
        plt.title(f'{category_name} Climate Finance Trend')
        plt.ylabel('Billion USD')
        plt.legend()
        
        # Residual Component
        plt.subplot(3, 1, 3)
        plt.plot(decomposition.resid, label='Residuals', color='gray', linewidth=1)
        plt.title(f'{category_name} Climate Finance Residuals')
        plt.ylabel('Billion USD')
        plt.legend()
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_folder, f'time_series_decomposition_{category_name.lower()}_{finance_type}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate and print statistics
        print(f"\nStatistics for {category_name} Climate Finance ({finance_type.capitalize()}):")
        print(f"Average: {time_series.mean():.2f} billion USD")
        print(f"Growth Rate: {((time_series.iloc[-1] / time_series.iloc[0]) ** (1/len(time_series)) - 1) * 100:.2f}% per year")
        print(f"Total Change: {((time_series.iloc[-1] / time_series.iloc[0]) - 1) * 100:.2f}%")
        print(f"Volatility (std): {time_series.std():.2f} billion USD")

def analyze_combined_climate_finance(df_disbursement, df_commitment, output_folder):
    """
    Create combined analysis plots showing disbursements, commitments, and seasonal patterns
    for each climate category.
    
    Parameters:
    df_disbursement: DataFrame with disbursement data
    df_commitment: DataFrame with commitment data
    output_folder: Path to save the output figures
    """
    # Create date range
    date_range = pd.date_range(start=f'{df_disbursement["effective_year"].min()}/1/1', 
                              end=f'{df_disbursement["effective_year"].max()}/12/31', 
                              freq='Y')
    
    # Define categories and their properties
    categories = {
        'Adaptation': {
            'disbursement': 'adaptation_funding',
            'commitment': 'adaptation_commitment',
            'color_disb': '#d95f0e',
            'color_comm': '#fec44f'
        },
        'Mitigation': {
            'disbursement': 'mitigation_funding',
            'commitment': 'mitigation_commitment',
            'color_disb': '#2ca25f',
            'color_comm': '#addd8e'
        },
        'Environment': {
            'disbursement': 'environment_funding',
            'commitment': 'environment_commitment',
            'color_disb': '#2b8cbe',
            'color_comm': '#a6bddb'
        }
    }
    
    # Create the main figure with three rows and three columns
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Climate Finance Analysis: Disbursements, Commitments, and Seasonal Patterns', fontsize=16, y=0.95)
    
    for idx, (category, props) in enumerate(categories.items()):
        # Prepare time series data (convert to billions)
        disb_series = pd.Series(df_disbursement[props['disbursement']].values / 1000, index=date_range)
        comm_series = pd.Series(df_commitment[props['commitment']].values / 1000, index=date_range)
        
        # Handle missing values
        disb_series = disb_series.fillna(method='ffill').dropna()
        comm_series = comm_series.fillna(method='ffill').dropna()
        
        # Decompose both series
        decomp_disb = seasonal_decompose(disb_series, period=2, model='additive')
        decomp_comm = seasonal_decompose(comm_series, period=2, model='additive')
        
        # Plot original time series (left panel)
        axes[idx, 0].plot(disb_series, label='Disbursements', color=props['color_disb'], linewidth=2)
        axes[idx, 0].plot(comm_series, label='Commitments', color=props['color_comm'], linewidth=2)
        axes[idx, 0].set_title(f'{category} Climate Finance')
        axes[idx, 0].set_ylabel('Billion USD')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Plot trends (middle panel)
        axes[idx, 1].plot(decomp_disb.trend, label='Disbursements Trend', 
                         color=props['color_disb'], linewidth=2)
        axes[idx, 1].plot(decomp_comm.trend, label='Commitments Trend', 
                         color=props['color_comm'], linewidth=2)
        axes[idx, 1].set_title(f'{category} Trends')
        axes[idx, 1].set_ylabel('Billion USD')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, alpha=0.3)
        
        # Plot seasonal patterns (right panel)
        axes[idx, 2].plot(decomp_disb.seasonal, label='Disbursements Seasonal',
                         color=props['color_disb'], linewidth=2)
        axes[idx, 2].plot(decomp_comm.seasonal, label='Commitments Seasonal',
                         color=props['color_comm'], linewidth=2)
        axes[idx, 2].set_title(f'{category} Seasonal Patterns')
        axes[idx, 2].set_ylabel('Billion USD')
        axes[idx, 2].legend()
        axes[idx, 2].grid(True, alpha=0.3)
        
        # Format x-axis to show years
        for ax in axes[idx, :]:
            ax.set_xlabel('Year')
            plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=1)
    
    # Save figure
    output_path = os.path.join(output_folder, 'combined_climate_finance_analysis.png')
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    plt.close()
    
    # Print overall statistics
    print("\nOverall Climate Finance Statistics:")
    for category, props in categories.items():
        disb_series = pd.Series(df_disbursement[props['disbursement']].values / 1000, index=date_range)
        comm_series = pd.Series(df_commitment[props['commitment']].values / 1000, index=date_range)
        
        print(f"\n{category}:")
        print(f"Disbursements - Average: {disb_series.mean():.2f}B USD, Growth: {((disb_series.iloc[-1] / disb_series.iloc[0]) ** (1/len(disb_series)) - 1) * 100:.1f}% per year")
        print(f"Commitments - Average: {comm_series.mean():.2f}B USD, Growth: {((comm_series.iloc[-1] / comm_series.iloc[0]) ** (1/len(comm_series)) - 1) * 100:.1f}% per year")

def forecast_climate_finance(df_disbursement, output_folder, target_sum=85):
    """
    Forecast climate finance categories until their sum reaches target_sum (100B USD)
    using a robust forecasting approach that handles zero/negative values.
    
    Parameters:
    df_disbursement: DataFrame with disbursement data
    output_folder: Path to save the output figures
    target_sum: Target sum in billion USD (default 100)
    """
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    import matplotlib.pyplot as plt
    
    # Create date range for existing data
    date_range = pd.date_range(start=f'{df_disbursement["effective_year"].min()}/1/1', 
                              end=f'{df_disbursement["effective_year"].max()}/12/31', 
                              freq='YE')
    
    # Prepare the data
    categories = {
        'Adaptation': 'adaptation_funding',
        'Mitigation': 'mitigation_funding',
        'Environment': 'environment_funding'
    }
    
    colors = {
        'Adaptation': '#d95f0e',
        'Mitigation': '#2ca25f',
        'Environment': '#2b8cbe'
    }
    
    # Convert to billions and create time series
    series_dict = {}
    for category, column in categories.items():
        series_dict[category] = pd.Series(df_disbursement[column].values / 1000, index=date_range)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot historical data and calculate initial total
    current_total = 0
    for category, series in series_dict.items():
        plt.plot(series.index, series.values, label=f'Historical {category}', 
                color=colors[category], linewidth=2)
        current_total += series.iloc[-1]
    
    # Calculate required growth rate to reach target
    years_to_forecast = 200  # Maximum forecast horizon
    forecast_index = pd.date_range(start=date_range[-1] + pd.DateOffset(years=1), 
                                 periods=years_to_forecast, freq='YE')
    
    forecasts = {}
    confidence_intervals = {}
    
    for category, series in series_dict.items():
        try:
            # First attempt: Try multiplicative trend if data is strictly positive
            if (series > 0).all():
                model = ExponentialSmoothing(
                    series,
                    trend='multiplicative',
                    seasonal=None,
                    damped=True
                ).fit()
            else:
                # Fallback: Use additive trend for data with zeros/negatives
                model = ExponentialSmoothing(
                    series,
                    trend='additive',
                    seasonal='additive',
                    seasonal_periods=3,
                    damped=False
                ).fit()
            
            # Generate forecast
            forecast = model.forecast(years_to_forecast)
            
            # Ensure forecasted values are non-negative
            forecast = forecast.clip(lower=0)
            
        except Exception as e:
            print(f"Warning: Error fitting model for {category}, using simple trend extrapolation")
            # Fallback: Simple linear trend if exponential smoothing fails
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series.values, deg=1)
            
            # Generate forecast using linear trend
            future_x = np.arange(len(series), len(series) + years_to_forecast)
            forecast = pd.Series(
                np.polyval(coeffs, future_x).clip(min=0),
                index=forecast_index
            )
        
        # Store results
        forecasts[category] = forecast
        
        # Plot forecast
        plt.plot(forecast_index, forecast, '--', 
                color=colors[category], alpha=0.8,
                label=f'Forecast {category}')
    
    # Calculate year when target is reached
    total_forecast = pd.DataFrame(forecasts).sum(axis=1)
    target_year = None
    for year, value in zip(forecast_index, total_forecast):
        if value >= target_sum:
            target_year = year.year
            break
    
    # Draw a vertical line at the target year if found
    if target_year:
        plt.axvline(pd.Timestamp(f'{target_year}-12-31'), color='black', linestyle='--')
        plt.text(pd.Timestamp(f'{target_year}-12-31'), target_sum*0.3, 
                 f'100 bn $ disbursed Target reached by {target_year}', 
                 rotation=90, verticalalignment='bottom', horizontalalignment='right')
    
    # Formatting
    plt.title('Climate Finance Forecast to Reach 100B USD Target')
    plt.xlabel('Year')
    plt.ylabel('Billion USD')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)
    
    # Save and show plot
    plt.tight_layout()
    plt.savefig(f'{output_folder}/climate_finance_forecast.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print results
    print("\nForecast Results:")
    if target_year:
        print(f"Target of {target_sum}B USD projected to be reached by {target_year}")
    else:
        print(f"Target of {target_sum}B USD not reached within forecast horizon")
    
    print("\nCurrent values (latest year):")
    for category in categories:
        print(f"{category}: {series_dict[category].iloc[-1]:.2f}B USD")
    print(f"Total: {current_total:.2f}B USD")
    
    print("\nProjected values (end of forecast):")
    final_total = 0
    for category in categories:
        final_value = forecasts[category].iloc[-1]
        print(f"{category}: {final_value:.2f}B USD")
        final_total += final_value
    print(f"Total: {final_total:.2f}B USD")
    
    return forecasts, confidence_intervals

def forecast_climate_finance_sarima(df_disbursement, output_folder, target_sum=35):
    """
    Forecast climate finance categories until their sum reaches target_sum (100B USD)
    using SARIMA for forecasting.

    Parameters:
    df_disbursement: DataFrame with disbursement data
    output_folder: Path to save the output figures
    target_sum: Target sum in billion USD (default 100)
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Create date range for existing data
    date_range = pd.date_range(start=f'{df_disbursement["effective_year"].min()}/1/1',
                               end=f'{df_disbursement["effective_year"].max()}/12/31',
                               freq='YE')

    # Prepare the data
    categories = {
        'Adaptation': 'adaptation_funding',
        'Mitigation': 'mitigation_funding',
        'Environment': 'environment_funding'
    }

    colors = {
        'Adaptation': '#d95f0e',
        'Mitigation': '#2ca25f',
        'Environment': '#2b8cbe'
    }

    # Convert to billions and create time series
    series_dict = {}
    for category, column in categories.items():
        series_dict[category] = pd.Series(df_disbursement[column].values / 1000, index=date_range)

    # Create figure
    plt.figure(figsize=(15, 10))

    # Plot historical data
    for category, series in series_dict.items():
        plt.plot(series.index, series.values, label=f'Historical {category}',
                 color=colors[category], linewidth=2)

    # Set forecast parameters
    years_to_forecast = 18  # Maximum forecast horizon
    last_historical_date = date_range[-1]

    # Start forecasting a year earlier than the last historical date
    forecast_start_date = last_historical_date - pd.DateOffset(years=1)  # Start one year before
    forecast_index = pd.date_range(start=forecast_start_date, periods=years_to_forecast + 1, freq='YE')[1:]  # Exclude the first year

    forecasts = {}
    target_year = None

    # Forecast for each category
    for category, series in series_dict.items():
        try:
            # Fit SARIMA model
            model = SARIMAX(series, order=(4, 3, 1), seasonal_order=(0, 1, 0, 4)).fit(disp=False)
            # Generate forecast
            forecast = model.forecast(steps=years_to_forecast)

            # Replace the first forecasted value with the last historical value
            forecast[0] = series.iloc[-1]

        except Exception as e:
            print(f"Warning: Error fitting model for {category}, using simple trend extrapolation")
            # Fallback: Simple linear trend if SARIMA fails
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series.values, deg=1)
            future_x = np.arange(len(series), len(series) + years_to_forecast)
            forecast = pd.Series(np.polyval(coeffs, future_x).clip(min=0),
                                 index=forecast_index)
            # Replace the first forecasted value with the last historical value
            forecast[0] = series.iloc[-1]

        # Store results
        forecasts[category] = forecast

        # Plot forecasted series in dashed lines
        plt.plot(forecast_index, forecast, '--', color=colors[category], alpha=0.8,
                 label=f'Forecast {category}')

    # Calculate total forecast to find target year
    total_forecast = pd.DataFrame(forecasts).sum(axis=1)
    for year, value in zip(forecast_index, total_forecast):
        if value >= target_sum:
            target_year = year.year
            break

    # Draw a vertical line at the target year if found
    if target_year:
        plt.axvline(pd.Timestamp(f'{target_year}-12-31'), color='black', linestyle='--')
        plt.text(pd.Timestamp(f'{target_year}-12-31'), target_sum * 0.3,
                 f'100 bn $ disbursed Target reached by {target_year}',
                 rotation=90, verticalalignment='bottom', horizontalalignment='right')

    # Formatting
    plt.title('Climate Finance Forecast to Reach 100B USD Target')
    plt.xlabel('Year')
    plt.ylabel('Billion USD')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)

    # Save and show plot
    plt.tight_layout()
    plt.savefig(f'{output_folder}/climate_finance_forecast_sarima.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Print results
    print("\nForecast Results:")
    if target_year:
        print(f"Target of {target_sum}B USD projected to be reached by {target_year}")
    else:
        print(f"Target of {target_sum}B USD not reached within forecast horizon")

    print("\nCurrent values (latest year):")
    for category in categories:
        print(f"{category}: {series_dict[category].iloc[-1]:.2f}B USD")
    print(f"Total: {sum(series_dict[cat].iloc[-1] for cat in categories):.2f}B USD")

    print("\nProjected values (end of forecast):")
    final_total = 0
    for category in categories:
        final_value = forecasts[category].iloc[-1]
        print(f"{category}: {final_value:.2f}B USD")
        final_total += final_value
    print(f"Total: {final_total:.2f}B USD")

    return forecasts

    # Create lagged features from the series
    lagged_data = pd.DataFrame({f'lag_{i}': series.shift(i) for i in range(1, lag + 1)})
    lagged_data['value'] = series.values  # Adding the target variable
    return lagged_data

    # Create date range for existing data
    date_range = pd.date_range(start=f'{df_disbursement["effective_year"].min()}/1/1', 
                                end=f'{df_disbursement["effective_year"].max()}/12/31', 
                                freq='YE')
    
    # Prepare the data
    categories = {
        'Adaptation': 'adaptation_funding',
        'Mitigation': 'mitigation_funding',
        'Environment': 'environment_funding'
    }
    
    colors = {
        'Adaptation': '#d95f0e',
        'Mitigation': '#2ca25f',
        'Environment': '#2b8cbe'
    }
    
    # Create a DataFrame for the forecast
    forecast_df = pd.DataFrame(index=date_range, columns=categories.keys())
    
    # Fit Random Forest for each category
    for category, column in categories.items():
        # Create a Series for the current category
        series = pd.Series(df_disbursement[column].values / 1000, index=date_range)
        lagged_data = create_lagged_features(series, lag=3)
        lagged_data.dropna(inplace=True)
        
        # Define features and target
        X = lagged_data.drop(columns=['value'])
        y = lagged_data['value']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        
        # Generate forecasts
        forecast_index = pd.date_range(start=date_range[-1] + pd.DateOffset(years=1), 
                                        periods=150, freq='YE')
        future_data = pd.DataFrame(index=forecast_index, columns=X.columns.tolist() + ['predicted_value'])
        
        # Fill future_data with the latest available lagged values
        last_values = X.iloc[-1].values
        for i in range(len(future_data)):
            for lag in range(3):
                future_data.iloc[i, lag] = last_values[lag] if lag < len(last_values) else 0
            
            # Predict next value
            predicted_value = model.predict([last_values])[0]
            future_data.iloc[i, -1] = predicted_value  # Use the last column for predicted value
            last_values = np.roll(last_values, 1)
            last_values[0] = predicted_value
        
        # Store forecast in the forecast_df
        forecast_df[category] = future_data['predicted_value']

    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot historical data
    for category, column in categories.items():
        plt.plot(date_range, df_disbursement[column].values / 1000, label=f'Historical {category}', 
                 color=colors[category], linewidth=2)
    
    # Plot forecasted data
    for category in categories.keys():
        plt.plot(forecast_df.index, forecast_df[category], '--', 
                 color=colors[category], alpha=0.8, label=f'Forecast {category}')
    
    # Calculate total forecast and plot
    total_forecast = forecast_df.sum(axis=1)
    plt.plot(total_forecast.index, total_forecast, 'k--', label='Total Forecast', alpha=0.6)

    # Add target reference lines
    plt.axhline(y=target_sum / 3, color='gray', linestyle='--', alpha=0.5)
    plt.text(date_range[-1], target_sum / 3 + 1, 
             f'Target per category: {target_sum / 3:.1f}B USD', 
             verticalalignment='bottom')

    # Formatting
    plt.title('Climate Finance Forecast Using Random Forest')
    plt.xlabel('Year')
    plt.ylabel('Billion USD')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)

    # Save and show plot
    plt.tight_layout()
    plt.savefig(f'{output_folder}/climate_finance_forecast_rf.png', dpi=300, bbox_inches='tight')
    plt.close()

    return forecast_df


def main_analysis():
    # Set working directory
    import os
wd = os.getenv("PROJECT_DIR", ".")  # Use an environment variable or default to current dir
    
    # Import and prepare disbursement data
    df_disbursement = pd.read_csv(os.path.join(wd, "Data/timeline_disbursement.csv"))
    
    # Import and prepare commitment data
    df_commitment = pd.read_csv(os.path.join(wd, "Data/timeline_commitment.csv"))
    
    # Create output folder for analysis
    analysis_folder = os.path.join(wd, "Figures/trend_analysis")
    os.makedirs(analysis_folder, exist_ok=True)
    
    # Analyze disbursements
    print("\nAnalyzing Disbursement Trends...")
    analyze_climate_finance_trends(df_disbursement, analysis_folder, 'funding')
    
    # Analyze commitments
    print("\nAnalyzing Commitment Trends...")
    analyze_climate_finance_trends(df_commitment, analysis_folder, 'commitment')
    
def main_combined_analysis():
    # Set working directory
    import os
wd = os.getenv("PROJECT_DIR", ".")  # Use an environment variable or default to current dir
    
    # Import data
    df_disbursement = pd.read_csv(os.path.join(wd, "Data/timeline_disbursement.csv"))
    df_commitment = pd.read_csv(os.path.join(wd, "Data/timeline_commitment.csv"))
    
    # Create output folder
    analysis_folder = os.path.join(wd, "Figures/trend_analysis")
    os.makedirs(analysis_folder, exist_ok=True)
    
    # Run analysis
    analyze_combined_climate_finance(df_disbursement, df_commitment, analysis_folder)
    

def main_forecast():
    # Set working directory
    import os
wd = os.getenv("PROJECT_DIR", ".")  # Use an environment variable or default to current dir
    
    # Import disbursement data
    df_disbursement = pd.read_csv(os.path.join(wd, "Data/timeline_disbursement.csv"))
    
    # Create output folder
    forecast_folder = os.path.join(wd, "Figures/forecasts")
    os.makedirs(forecast_folder, exist_ok=True)
    
    # Run forecast
    forecasts, confidence_intervals = forecast_climate_finance(df_disbursement, forecast_folder)
    forecasts = forecast_climate_finance_sarima(df_disbursement, forecast_folder)

def create_comparison_timeline_by_donor(df_origin, df, output_folder):
    # List of specified countries
    selected_countries = [
        "Australia", "Austria", "Belgium", "Canada", "Czech Republic", "Denmark", "Finland", "France", "Germany", 
        "Greece", "Iceland", "Ireland", "Italy", "Japan", "Luxembourg", "Netherlands", "New Zealand", "Norway", 
        "Poland", "Portugal", "Slovak Republic", "Slovenia", "South Korea", "Spain", "Sweden", "Switzerland", 
        "United Kingdom", "United States"
    ]

    # Add DonorType classification for df_origin
    df_origin['DonorType'] = np.select(
        [
            df_origin.DonorCode < 807,
            df_origin.DonorCode > 1600,
            df_origin.DonorCode == 104,
            df_origin.DonorCode == 820
        ],
        ['Donor Country', 'Private Donor', 'Multilateral Donor Organization', 'Donor Country'],
        default='Multilateral Donor Organization'
    )

    # Filter to only include rows with Donor Country for both df_origin and df
    df_origin = df_origin[df_origin['DonorType'] == 'Donor Country'].reset_index(drop=True)
    df = df[df['DonorType'] == 'Donor Country'].reset_index(drop=True)

    # Filter for the specified countries only
    df_origin = df_origin[df_origin['DonorName'].isin(selected_countries)].reset_index(drop=True)
    df = df[df['DonorName'].isin(selected_countries)].reset_index(drop=True)

    # Filter for relevant funding categories (Mitigation and Adaptation) from df_origin
    df_origin['adaptation_funding_rio'] = np.where(df_origin['ClimateAdaptation'].isin([1, 2]), df_origin['USD_Disbursement_Defl'], 0)
    df_origin['mitigation_funding_rio'] = np.where(df_origin['ClimateMitigation'].isin([1, 2]), df_origin['USD_Disbursement_Defl'], 0)
    
    # Filter for years >= 2000
    df_origin = df_origin[df_origin['Year'] >= 2000].reset_index(drop=True)
    df = df[df['Year'] >= 2000].reset_index(drop=True)
    
    # Optional: Filter out rows with zero funding if you prefer
    df_origin = df_origin[(df_origin['adaptation_funding_rio'] > 0) | (df_origin['mitigation_funding_rio'] > 0)]

    # Filter relevant categories from df (ClimateFinanceBERT)
    df['adaptation_funding_bert'] = np.where(df['meta_category'] == 'Adaptation', df['USD_Disbursement_Defl'], 0)
    df['mitigation_funding_bert'] = np.where(df['meta_category'] == 'Mitigation', df['USD_Disbursement_Defl'], 0)
    
    # Group by year and donor for df_origin (Rio data)
    df_origin_grouped = df_origin.groupby(['DonorName', 'Year'])[['adaptation_funding_rio', 'mitigation_funding_rio']].sum().reset_index()

    # Group by year and donor for df (ClimateFinanceBERT)
    df_grouped = df.groupby(['DonorName', 'Year'])[['adaptation_funding_bert', 'mitigation_funding_bert']].sum().reset_index()

    # Merge the data from both sources (df_origin and df)
    comparison_df = pd.merge(df_origin_grouped, df_grouped, left_on=['DonorName', 'Year'], right_on=['DonorName', 'Year'], how='outer')

    # Optional: Handle missing data by filling NaNs with 0
    comparison_df.fillna(0, inplace=True)
    
    # Ensure Year is integer type after merge
    comparison_df['Year'] = comparison_df['Year'].astype(int)

     # Get the unique donors and calculate the number of rows and columns for the subplot grid
    unique_donors = comparison_df['DonorName'].unique()
    num_donors = len(unique_donors)
    cols = 6  # Keep 4 columns
    rows = (num_donors + cols - 1) // cols  # Number of rows, rounded up

    # Set font sizes
    plt.rcParams.update({
        'font.size': 14,          # Increased base font size
        'axes.titlesize': 16,     # Increased subplot title size
        'axes.labelsize': 14,     # Increased axis label size
        'legend.fontsize': 16     # Legend font size
    })

    # Define colors for adaptation and mitigation
    adaptation_color = '#d95f0e'  # Orange
    mitigation_color = '#2ca25f'  # Green

    # Increase figure size significantly
    fig = plt.figure(figsize=(24, 6 * rows))  # Increased width and height per row
    
    # Create subplot grid with appropriate spacing
    gs = fig.add_gridspec(rows, cols, 
                         hspace=0.4,    # Increased vertical space between plots
                         wspace=0.4,    # Added horizontal space between plots
                         bottom=0.1)    # Space at bottom for legend
    
    axes = []
    for i in range(rows):
        for j in range(cols):
            if i * cols + j < num_donors:
                axes.append(fig.add_subplot(gs[i, j]))
            else:
                axes.append(None)

    # Create a list to store the legend handles and labels
    handles = []
    labels = []

    for i, donor in enumerate(unique_donors):
        if axes[i] is not None:
            donor_data = comparison_df[comparison_df['DonorName'] == donor]
            ax = axes[i]  # Get the correct subplot

            # Plot adaptation lines with same color but different styles
            line1, = ax.plot(donor_data['Year'], donor_data['adaptation_funding_rio']/ 1000, 
                           color=adaptation_color, linestyle='--', label='Rio Adaptation',
                           linewidth=3)  # Increased line width
            line2, = ax.plot(donor_data['Year'], donor_data['adaptation_funding_bert']/ 1000, 
                           color=adaptation_color, linestyle='-', label='ClimateFinanceBERT Adaptation',
                           linewidth=3)  # Increased line width

            # Plot mitigation lines with same color but different styles
            line3, = ax.plot(donor_data['Year'], donor_data['mitigation_funding_rio']/ 1000, 
                           color=mitigation_color, linestyle='--', label='Rio Mitigation',
                           linewidth=3)  # Increased line width
            line4, = ax.plot(donor_data['Year'], donor_data['mitigation_funding_bert']/ 1000, 
                           color=mitigation_color, linestyle='-', label='ClimateFinanceBERT Mitigation',
                           linewidth=3)  # Increased line width

            # Add lines to the shared legend list (only add once)
            if i == 0:
                handles = [line1, line2, line3, line4]
                labels = [line1.get_label(), line2.get_label(), line3.get_label(), line4.get_label()]

            # Set the title and labels for each subplot
            ax.set_title(f'{donor}', pad=20)  # Added padding above title
            ax.set_xlabel('Year')
            ax.set_ylabel('Funding (USD in billions)')

            # Increase tick label sizes
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Create a single shared legend at the bottom with larger font
    fig.legend(handles, labels, 
              loc='center', 
              ncol=4, 
              bbox_to_anchor=(0.5, 0.06),
              fontsize=18,  # Increased legend font size
              frameon=True,
              borderaxespad=1)

    # Save the figure with extra bottom margin to include the legend
    plt.savefig(f"{output_folder}/combined_comparison_by_donor.png", 
                bbox_inches='tight',
                pad_inches=0.5,
                dpi=300)
    plt.close()

    return comparison_df

def create_ratio_comparison_timeline_by_donor(df_origin, df, output_folder):
    # List of specified countries (same as original)
    selected_countries = [
        "Australia", "Austria", "Belgium", "Canada", "Czech Republic", "Denmark", "Finland", "France", "Germany", 
        "Greece", "Iceland", "Ireland", "Italy", "Japan", "Luxembourg", "Netherlands", "New Zealand", "Norway", 
        "Poland", "Portugal", "Slovak Republic", "Slovenia", "South Korea", "Spain", "Sweden", "Switzerland", 
        "United Kingdom", "United States"
    ]

    # Add DonorType classification for df_origin
    df_origin['DonorType'] = np.select(
        [
            df_origin.DonorCode < 807,
            df_origin.DonorCode > 1600,
            df_origin.DonorCode == 104,
            df_origin.DonorCode == 820
        ],
        ['Donor Country', 'Private Donor', 'Multilateral Donor Organization', 'Donor Country'],
        default='Multilateral Donor Organization'
    )

    # Filter to only include rows with Donor Country for both df_origin and df
    df_origin = df_origin[df_origin['DonorType'] == 'Donor Country'].reset_index(drop=True)
    df = df[df['DonorType'] == 'Donor Country'].reset_index(drop=True)

    # Filter for the specified countries only
    df_origin = df_origin[df_origin['DonorName'].isin(selected_countries)].reset_index(drop=True)
    df = df[df['DonorName'].isin(selected_countries)].reset_index(drop=True)

    # Filter for relevant funding categories
    df_origin['adaptation_funding_rio'] = np.where(df_origin['ClimateAdaptation'].isin([1, 2]), df_origin['USD_Disbursement_Defl'], 0)
    df_origin['mitigation_funding_rio'] = np.where(df_origin['ClimateMitigation'].isin([1, 2]), df_origin['USD_Disbursement_Defl'], 0)
    
    # Filter for years >= 2000
    df_origin = df_origin[df_origin['Year'] >= 2010].reset_index(drop=True)
    df = df[df['Year'] >= 2010].reset_index(drop=True)

    # Filter relevant categories from df (ClimateFinanceBERT)
    df['adaptation_funding_bert'] = np.where(df['meta_category'] == 'Adaptation', df['USD_Disbursement_Defl'], 0)
    df['mitigation_funding_bert'] = np.where(df['meta_category'] == 'Mitigation', df['USD_Disbursement_Defl'], 0)
    
    # Group by year and donor
    df_origin_grouped = df_origin.groupby(['DonorName', 'Year'])[['adaptation_funding_rio', 'mitigation_funding_rio']].sum().reset_index()
    df_grouped = df.groupby(['DonorName', 'Year'])[['adaptation_funding_bert', 'mitigation_funding_bert']].sum().reset_index()

    # Merge the data
    comparison_df = pd.merge(df_origin_grouped, df_grouped, on=['DonorName', 'Year'], how='outer')
    comparison_df.fillna(0, inplace=True)
    comparison_df['Year'] = comparison_df['Year'].astype(int)

    # Calculate ratios with handling for division by zero
    epsilon = 1e-10  # Small number to avoid division by zero
    comparison_df['adaptation_ratio'] = np.where(
        comparison_df['adaptation_funding_rio'] > epsilon,
        comparison_df['adaptation_funding_bert'] / comparison_df['adaptation_funding_rio']*100,
        np.nan
    )
    comparison_df['mitigation_ratio'] = np.where(
        comparison_df['mitigation_funding_rio'] > epsilon,
        comparison_df['mitigation_funding_bert'] / comparison_df['mitigation_funding_rio']*100,
        np.nan
    )

    # Plotting setup
    unique_donors = comparison_df['DonorName'].unique()
    num_donors = len(unique_donors)
    cols = 6
    rows = (num_donors + cols - 1) // cols

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'legend.fontsize': 16
    })

    # Colors
    adaptation_color = '#d95f0e'  # Orange
    mitigation_color = '#2ca25f'  # Green

    fig = plt.figure(figsize=(24, 6 * rows))
    gs = fig.add_gridspec(rows, cols, hspace=0.4, wspace=0.4, bottom=0.15)  # Increased bottom margin for footnote
    
    axes = []
    for i in range(rows):
        for j in range(cols):
            if i * cols + j < num_donors:
                axes.append(fig.add_subplot(gs[i, j]))
            else:
                axes.append(None)

    handles = []
    labels = []

    for i, donor in enumerate(unique_donors):
        if axes[i] is not None:
            donor_data = comparison_df[comparison_df['DonorName'] == donor]
            ax = axes[i]

            # Plot ratio lines
            line1, = ax.plot(donor_data['Year'], donor_data['adaptation_ratio'], 
                           color=adaptation_color, label='Adaptation Ratio (BERT/Rio, %)',
                           linewidth=3)
            line2, = ax.plot(donor_data['Year'], donor_data['mitigation_ratio'], 
                           color=mitigation_color, label='Mitigation Ratio (BERT/Rio, %)',
                           linewidth=3)

            if i == 0:
                handles = [line1, line2]
                labels = [line1.get_label(), line2.get_label()]

            ax.set_title(f'{donor}', pad=20)
            ax.set_xlabel('Year')
            ax.set_ylabel('Ratio (BERT/Rio, %)')

            # Add horizontal line at ratio = 1
            ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
            
            # Set y-axis limits to focus on meaningful ratios
            ax.set_ylim(0, 120)
            
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Add legend
    fig.legend(handles, labels, 
              loc='center', 
              ncol=2, 
              bbox_to_anchor=(0.5, 0.08),
              fontsize=18,
              frameon=True,
              borderaxespad=1)

    # Add footnote
    footnote_text = (
        "Note: Ratio values show how Rio markers is declared compare to ClimateFinanceBERT classifications.\n"
        "Ratio = 100%: Perfect agreement between methods\n"
        "Ratio > 100%: Under declaration\n"
        "Ratio < 100%: Over declaration\n"
        "Missing values indicate periods where RIO classified no projects in that category"
    )
    
    fig.text(0.5, 0.02, footnote_text, 
             ha='center', va='bottom', 
             fontsize=14, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.savefig(f"{output_folder}/ratio_comparison_by_donor.png", 
                bbox_inches='tight',
                pad_inches=0.5,
                dpi=300)
    plt.close()

    return comparison_df




# Main function to run all operations
def main():
# Create disbursement timeline
    df_time_disbursement = create_funding_timeline(df, wd + "/Data/timeline_disbursement.csv")
    
    # Create commitment timeline
    df_time_commitment = create_commitment_timeline(df, wd + "/Data/timeline_commitment.csv")
    
    # Prepare Rio marker data
    rio_adapt1_disbursement, rio_adapt2_disbursement, rio_adapt1_commitment, rio_adapt2_commitment,rio_miti1_disbursement, rio_miti2_disbursement, rio_miti1_commitment, rio_miti2_commitment = prepare_rio_data(df_origin)

    # Create Rio stacked area plot for mitigation disbursements
    cluster = df_time_disbursement['mitigation_funding'] / 1000
    rio_principal = rio_miti2_disbursement
    rio_significant = rio_miti1_disbursement
    output_folder = wd + "/Figures/stacked_area_mitigation_disbursement.png"
    rio_colors = ['#f7fcb9', '#addd8e']  # Colors for mitigation
    cluster_color = '#31a354'  # Color for the cluster line
    max_y = max(max(rio_principal), max(rio_significant)) + 5

    rio_stacked_area(cluster, rio_principal, rio_significant, output_folder, rio_colors, cluster_color, "Mitigation", max_y, "disbursements")

    # Create Rio stacked area plot for mitigation commitments
    cluster = df_time_commitment['mitigation_commitment'] / 1000
    rio_principal = rio_miti2_commitment
    rio_significant = rio_miti1_commitment
    output_folder = wd + "/Figures/stacked_area_mitigation_commitment.png"
    max_y = max(max(rio_principal), max(rio_significant)) + 5

    rio_stacked_area(cluster, rio_principal, rio_significant, output_folder, rio_colors, cluster_color, "Mitigation", max_y, "commitments")

    # Create Rio stacked area plot for adaptation disbursements
    cluster = df_time_disbursement['adaptation_funding'] / 1000
    rio_principal = rio_adapt2_disbursement
    rio_significant = rio_adapt1_disbursement
    output_folder = wd + "/Figures/stacked_area_adaptation_disbursement.png"
    rio_colors = ['#fff7bc', '#fec44f']  # Colors for adaptation
    cluster_color = '#d95f0e'  # Color for the cluster line
    max_y = max(max(rio_principal), max(rio_significant)) + 5

    rio_stacked_area(cluster, rio_principal, rio_significant, output_folder, rio_colors, cluster_color, "Adaptation", max_y, "disbursements")

    # Create Rio stacked area plot for adaptation commitments
    cluster = df_time_commitment['adaptation_commitment'] / 1000
    rio_principal = rio_adapt2_commitment
    rio_significant = rio_adapt1_commitment
    output_folder = wd + "/Figures/stacked_area_adaptation_commitment.png"
    max_y = max(max(rio_principal), max(rio_significant)) + 5

    rio_stacked_area(cluster, rio_principal, rio_significant, output_folder, rio_colors, cluster_color, "Adaptation", max_y, "commitments")

    # Prepare stacked area plot for disbursements
    df_stack_disbursement = df_time_disbursement[['effective_year', 'adaptation_funding', 'mitigation_funding', 'environment_funding']] 
    stacked_area(df_stack_disbursement, output_folder=wd + '/Figures/stackplot_disbursement.png', input_colors=['#d95f0e', '#2ca25f', '#2b8cbe'], max_number=3, data_type="disbursements")
    
    # Prepare stacked area plot for commitments
    df_stack_commitment = df_time_commitment[['effective_year', 'adaptation_commitment', 'mitigation_commitment', 'environment_commitment']]
    stacked_area(df_stack_commitment, output_folder=wd + '/Figures/stackplot_commitment.png', input_colors=['#d95f0e', '#2ca25f', '#2b8cbe'], max_number=3, data_type="commitments")
    
    cluster_adap = df_time_disbursement['adaptation_funding'] / 1000
    cluster_miti = df_time_disbursement['mitigation_funding'] / 1000

    # Create combined adaptation and mitigation plot for disbursements
    combined_adaptation_mitigation_plot(
        cluster_adap,
        rio_adapt1_disbursement,
        rio_adapt2_disbursement,
        cluster_miti,
        rio_miti1_disbursement,
        rio_miti2_disbursement,
        wd + "/Figures")
    
    # Prepare the data for disbursements and commitments by country
    rio_adapt1_disbursement, rio_adapt2_disbursement, rio_adapt1_commitment, rio_adapt2_commitment, \
    rio_miti1_disbursement, rio_miti2_disbursement, rio_miti1_commitment, rio_miti2_commitment = prepare_rio_data_by_country(df_origin)

    # Assuming cluster_adap and cluster_miti are dataframes containing the necessary data
    # For example, if these are part of your dataset or computed elsewhere
    cluster_adap = df[['DonorCountry', 'Year', 'USD_Disbursement_Defl']]  # Example data for adaptation
    cluster_miti = df[['DonorCountry', 'Year', 'USD_Commitment_Defl']]  # Example data for mitigation


    # Call the plot function
    combined_adaptation_mitigation_plot_by_country(
        rio_adapt1_disbursement, 
        rio_adapt2_disbursement, 
        rio_miti1_disbursement, 
        rio_miti2_disbursement, 
        cluster_adap, 
        cluster_miti, 
        wd + "/Figures")
    
    comparison_by_donor = create_comparison_timeline_by_donor(df_origin, df, wd + "/Figures")
    comparison_by_donor = create_ratio_comparison_timeline_by_donor(df_origin, df, wd + "/Figures")



    print("All operations completed!")
# Run the main function
main()
