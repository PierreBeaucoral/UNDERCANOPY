import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.patches import Patch
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set working directory
wd = "...UNDERCANOPY/Climate finance estimation/"
os.chdir(wd)

# -------------------------
# Helper and Import Functions
# -------------------------
def csv_import(filepath, delimiter="|"):
    return pd.read_csv(filepath, encoding='utf8', delimiter=delimiter, dtype={'text': str, "USD_Disbursement": float})

# Import datasets
df_origin = pd.read_csv(os.path.join(wd,'/Data/DataPB.csv')), 
                        encoding='utf8', delimiter='|')
df = csv_import(os.path.join(wd, 'Data/climate_finance_total.csv'))

# -------------------------
# Timeline Creation Functions
# -------------------------
def create_funding_timeline(df, output_path):
    df_time = df.rename(columns={'Year': 'effective_year', 'USD_Disbursement_Defl': 'effective_funding'})
    df_time['adaptation_funding'] = np.where(df_time['meta_category'] == 'Adaptation', df_time['effective_funding'], 0)
    df_time['mitigation_funding'] = np.where(df_time['meta_category'] == 'Mitigation', df_time['effective_funding'], 0)
    df_time['environment_funding'] = np.where(df_time['meta_category'] == 'Environment', df_time['effective_funding'], 0)
    df_time = df_time[['effective_year', 'adaptation_funding', 'mitigation_funding', 'environment_funding']].groupby('effective_year').sum().reset_index()
    df_time.to_csv(output_path, index=False)
    return df_time

def create_commitment_timeline(df, output_path):
    df_time = df.rename(columns={'Year': 'effective_year', 'USD_Commitment_Defl': 'effective_commitment'})
    df_time['adaptation_commitment'] = np.where(df_time['meta_category'] == 'Adaptation', df_time['effective_commitment'], 0)
    df_time['mitigation_commitment'] = np.where(df_time['meta_category'] == 'Mitigation', df_time['effective_commitment'], 0)
    df_time['environment_commitment'] = np.where(df_time['meta_category'] == 'Environment', df_time['effective_commitment'], 0)
    df_time = df_time[['effective_year', 'adaptation_commitment', 'mitigation_commitment', 'environment_commitment']].groupby('effective_year').sum().reset_index()
    df_time.to_csv(output_path, index=False)
    return df_time

# -------------------------
# Rio Marker Preparation Functions
# -------------------------
def prepare_rio_data(df_origin):
    df_origin = df_origin[df_origin['Year'] >= 2000].reset_index(drop=True)
    df_origin['DonorType'] = np.select(
        [df_origin.DonorCode < 807, df_origin.DonorCode > 1600, df_origin.DonorCode == 104, df_origin.DonorCode == 820],
        ['Donor Country', 'Private Donor', 'Multilateral Donor Organization', 'Donor Country'],
        default='Multilateral Donor Organization'
    )
    df_origin = df_origin[df_origin['DonorType'] == 'Donor Country'].reset_index(drop=True)
    
    def group_marker(marker_col, funding_col, df, significant_value, principal_value, start_year):
        rio_data_significant = df[df[marker_col] == significant_value][[funding_col, 'Year']].groupby('Year').sum().reset_index()
        rio_data_principal = df[df[marker_col] == principal_value][[funding_col, 'Year']].groupby('Year').sum().reset_index()
        rio_data_filtered_significant = rio_data_significant[rio_data_significant['Year'] >= start_year][funding_col] / 1000
        rio_data_filtered_principal = rio_data_principal[rio_data_principal['Year'] >= start_year][funding_col] / 1000
        zero_padding = [0] * (start_year - 2000)
        return zero_padding + rio_data_filtered_significant.tolist(), zero_padding + rio_data_filtered_principal.tolist()

    # For adaptation (from 2008) and mitigation (from 2000)
    rio_adapt1_disbursement, rio_adapt2_disbursement = group_marker('ClimateAdaptation', 'USD_Disbursement_Defl', df_origin, 1, 2, 2008)
    rio_adapt1_commitment, rio_adapt2_commitment = group_marker('ClimateAdaptation', 'USD_Commitment_Defl', df_origin, 1, 2, 2008)
    rio_miti1_disbursement, rio_miti2_disbursement = group_marker('ClimateMitigation', 'USD_Disbursement_Defl', df_origin, 1, 2, 2000)
    rio_miti1_commitment, rio_miti2_commitment = group_marker('ClimateMitigation', 'USD_Commitment_Defl', df_origin, 1, 2, 2000)
    
    return (rio_adapt1_disbursement, rio_adapt2_disbursement, rio_adapt1_commitment, rio_adapt2_commitment,
            rio_miti1_disbursement, rio_miti2_disbursement, rio_miti1_commitment, rio_miti2_commitment)

# -------------------------
# Plot Functions
# -------------------------
def stacked_area(df, output_folder, input_colors, max_number, data_type):
    y = df.drop('effective_year', axis=1).T
    x = list(df['effective_year'])
    fig, ax = plt.subplots()
    ax.stackplot(x, y / 1000, colors=input_colors)
    ax.tick_params(labelsize=11)
    ylabel_text = f'Aggregated {"disbursements" if data_type=="disbursements" else "commitments"} (billion USD)'
    ax.set_ylabel(ylabel_text, labelpad=15, color='#333333', fontsize=11)
    xticks = np.arange(min(x), max(x)+1, 1.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks.astype(int), rotation=45)
    plt.xlim(2000, 2023)
    legend_list = [Patch(facecolor=input_colors[i], label=list(df.drop('effective_year', axis=1).columns)[i])
                   for i in range(max_number)]
    ax.legend(handles=legend_list, loc="upper left", prop={'size': 8}, frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    plt.savefig(output_folder, bbox_inches="tight", dpi=1200)
    plt.show()

def rio_stacked_area(cluster, rio_principal, rio_significant, output_folder, rio_colors, cluster_color, climate_type, max_y, data_type):
    years = list(range(2000, 2023))
    cluster_filtered = cluster[-len(years):]
    rio_principal_filtered = rio_principal[-len(years):]
    rio_significant_filtered = rio_significant[-len(years):]
    fig, ax = plt.subplots(figsize=(24, 8))
    ax.stackplot(years, [rio_principal_filtered, rio_significant_filtered],
                 colors=[rio_colors[1], rio_colors[0]], alpha=0.5)
    ax.plot(years, cluster_filtered, color=cluster_color, linewidth=6)
    events = [(2010, 'USD 100bn target'), (2015, 'Paris Agreement')]
    for year, event in events:
        ax.vlines(x=year, ymin=0, ymax=max_y-3, linestyle='--', color='#636363')
        ax.text(year, max_y-2, event, ha='center')
    ylabel_text = f'Aggregated aid {data_type} (billion USD)\nfor Climate Change {climate_type}'
    ax.set_ylabel(ylabel_text, labelpad=15, color='#333333', fontsize=12)
    plt.xticks(np.arange(2000,2023,2), rotation=45)
    plt.xlim(2000,2023)
    plt.ylim(0, max_y)
    legend_list = [
        Patch(facecolor=cluster_color, label='ClimateFinanceBERT'),
        Patch(facecolor=rio_colors[1], label='Rio markers principal'),
        Patch(facecolor=rio_colors[0], label='Rio markers significant')
    ]
    ax.legend(handles=legend_list, loc="upper left", prop={'size': 12}, frameon=True)
    plt.tight_layout()
    plt.savefig(output_folder, bbox_inches="tight", dpi=1200)
    plt.show()

def combined_adaptation_mitigation_plot(cluster_adap, rio_adapt1, rio_adapt2, cluster_miti, rio_miti1, rio_miti2, output_folder):
    years = list(range(2000, 2023))
    cluster_adap = cluster_adap[-len(years):]
    cluster_miti = cluster_miti[-len(years):]
    fig, axs = plt.subplots(2, 1, figsize=(24,12), sharey=True)
    axs[0].stackplot(years, [rio_adapt2, rio_adapt1], colors=['#fec44f','#fff7bc'], alpha=0.5)
    axs[0].plot(years, cluster_adap, color='#d95f0e', linewidth=6)
    axs[0].set_title('Adaptation Funding', fontsize=14)
    axs[0].set_ylabel('Aggregated aid disbursements (billion USD)', labelpad=15, fontsize=12)
    axs[0].set_xlim(2000,2023)
    axs[0].set_ylim(0,23)
    axs[0].set_xticks(np.arange(2000,2024,2))
    events = [(2010, 'USD 100bn target'), (2015, 'Paris Agreement')]
    for year, event in events:
        axs[0].vlines(year, 0, 22, linestyle='--', color='#636363')
        axs[0].text(year,21,event, ha='center')
    axs[1].stackplot(years, [rio_miti2, rio_miti1], colors=['#addd8e','#f7fcb9'], alpha=0.5)
    axs[1].plot(years, cluster_miti, color='#31a354', linewidth=6)
    axs[1].set_title('Mitigation Funding', fontsize=14)
    axs[1].set_ylabel('Aggregated aid disbursements (billion USD)', labelpad=15, fontsize=12)
    axs[1].set_xlim(2000,2023)
    axs[1].set_ylim(0,23)
    axs[1].set_xticks(np.arange(2000,2024,2))
    for year, event in events:
        axs[1].vlines(year, 0, 22, linestyle='--', color='#636363')
        axs[1].text(year,21,event, ha='center')
    legend_list = [
        Patch(facecolor='#d95f0e', label='ClimateFinanceBERT (Adaptation)'),
        Patch(facecolor='#fec44f', label='Rio markers principal (Adaptation)'),
        Patch(facecolor='#fff7bc', label='Rio markers significant (Adaptation)'),
        Patch(facecolor='#31a354', label='ClimateFinanceBERT (Mitigation)'),
        Patch(facecolor='#addd8e', label='Rio markers principal (Mitigation)'),
        Patch(facecolor='#f7fcb9', label='Rio markers significant (Mitigation)')
    ]
    fig.legend(handles=legend_list, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'combined_adaptation_mitigation_plot.png'), bbox_inches='tight')
    plt.show()

def analyze_combined_climate_finance(df_disbursement, df_commitment, output_folder):
    date_range = pd.date_range(start=f'{df_disbursement["effective_year"].min()}/1/1',
                               end=f'{df_disbursement["effective_year"].max()}/12/31', freq='Y')
    categories = {
        'Adaptation': {'disbursement': 'adaptation_funding', 'commitment': 'adaptation_commitment',
                       'color_disb': '#d95f0e', 'color_comm': '#fec44f'},
        'Mitigation': {'disbursement': 'mitigation_funding', 'commitment': 'mitigation_commitment',
                       'color_disb': '#2ca25f', 'color_comm': '#addd8e'},
        'Environment': {'disbursement': 'environment_funding', 'commitment': 'environment_commitment',
                        'color_disb': '#2b8cbe', 'color_comm': '#a6bddb'}
    }
    fig, axes = plt.subplots(3, 3, figsize=(18,15))
    fig.suptitle('Climate Finance Analysis: Disbursements, Commitments, and Seasonal Patterns', fontsize=16, y=0.95)
    for idx, (category, props) in enumerate(categories.items()):
        disb_series = pd.Series(df_disbursement[props['disbursement']].values/1000, index=date_range).fillna(method='ffill').dropna()
        comm_series = pd.Series(df_commitment[props['commitment']].values/1000, index=date_range).fillna(method='ffill').dropna()
        decomp_disb = seasonal_decompose(disb_series, period=2, model='additive')
        decomp_comm = seasonal_decompose(comm_series, period=2, model='additive')
        axes[idx,0].plot(disb_series, label='Disbursements', color=props['color_disb'], linewidth=2)
        axes[idx,0].plot(comm_series, label='Commitments', color=props['color_comm'], linewidth=2)
        axes[idx,0].set_title(f'{category} Climate Finance')
        axes[idx,0].set_ylabel('Billion USD')
        axes[idx,0].legend()
        axes[idx,0].grid(True, alpha=0.3)
        axes[idx,1].plot(decomp_disb.trend, label='Disbursements Trend', color=props['color_disb'], linewidth=2)
        axes[idx,1].plot(decomp_comm.trend, label='Commitments Trend', color=props['color_comm'], linewidth=2)
        axes[idx,1].set_title(f'{category} Trends')
        axes[idx,1].set_ylabel('Billion USD')
        axes[idx,1].legend()
        axes[idx,1].grid(True, alpha=0.3)
        axes[idx,2].plot(decomp_disb.seasonal, label='Disbursements Seasonal', color=props['color_disb'], linewidth=2)
        axes[idx,2].plot(decomp_comm.seasonal, label='Commitments Seasonal', color=props['color_comm'], linewidth=2)
        axes[idx,2].set_title(f'{category} Seasonal Patterns')
        axes[idx,2].set_ylabel('Billion USD')
        axes[idx,2].legend()
        axes[idx,2].grid(True, alpha=0.3)
        for ax in axes[idx, :]:
            ax.set_xlabel('Year')
            plt.setp(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    fig.subplots_adjust(top=1)
    output_path = os.path.join(output_folder, 'combined_climate_finance_analysis.png')
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    plt.close()

def forecast_climate_finance_sarima(df_disbursement, output_folder, target_sum=35):
    date_range = pd.date_range(start=f'{df_disbursement["effective_year"].min()}/1/1',
                               end=f'{df_disbursement["effective_year"].max()}/12/31', freq='YE')
    categories = {'Adaptation': 'adaptation_funding', 'Mitigation': 'mitigation_funding', 'Environment': 'environment_funding'}
    colors = {'Adaptation': '#d95f0e', 'Mitigation': '#2ca25f', 'Environment': '#2b8cbe'}
    series_dict = {}
    for category, column in categories.items():
        series_dict[category] = pd.Series(df_disbursement[column].values/1000, index=date_range)
    plt.figure(figsize=(15,10))
    for category, series in series_dict.items():
        plt.plot(series.index, series.values, label=f'Historical {category}', color=colors[category], linewidth=2)
    years_to_forecast = 18
    last_historical_date = date_range[-1]
    forecast_start_date = last_historical_date - pd.DateOffset(years=1)
    forecast_index = pd.date_range(start=forecast_start_date, periods=years_to_forecast+1, freq='YE')[1:]
    forecasts = {}
    target_year = None
    for category, series in series_dict.items():
        try:
            model = SARIMAX(series, order=(4,3,1), seasonal_order=(0,1,0,4)).fit(disp=False)
            forecast = model.forecast(steps=years_to_forecast)
            forecast[0] = series.iloc[-1]
        except Exception as e:
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series.values, deg=1)
            future_x = np.arange(len(series), len(series)+years_to_forecast)
            forecast = pd.Series(np.polyval(coeffs, future_x).clip(min=0), index=forecast_index)
            forecast[0] = series.iloc[-1]
        forecasts[category] = forecast
        plt.plot(forecast_index, forecast, '--', color=colors[category], alpha=0.8, label=f'Forecast {category}')
    total_forecast = pd.DataFrame(forecasts).sum(axis=1)
    for year, value in zip(forecast_index, total_forecast):
        if value >= target_sum:
            target_year = year.year
            break
    if target_year:
        plt.axvline(pd.Timestamp(f'{target_year}-12-31'), color='black', linestyle='--')
        plt.text(pd.Timestamp(f'{target_year}-12-31'), target_sum*0.3, f'100 bn $ disbursed Target reached by {target_year}', 
                 rotation=90, verticalalignment='bottom', horizontalalignment='right')
    plt.title('Climate Finance Forecast to Reach 100B USD Target')
    plt.xlabel('Year')
    plt.ylabel('Billion USD')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'climate_finance_forecast_sarima.png'), dpi=300, bbox_inches='tight')
    plt.close()


# -------------------------
# Comparison Functions by Donor (Original Code)
# -------------------------
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
    gs = fig.add_gridspec(rows, cols, hspace=0.4, wspace=0.4, bottom=0.15)
    
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
            ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylim(0, 120)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    fig.legend(handles, labels, 
              loc='center', 
              ncol=2, 
              bbox_to_anchor=(0.5, 0.08),
              fontsize=18,
              frameon=True,
              borderaxespad=1)
    fig.text(0.5, 0.02, 
             ("Note: Ratio values show how Rio markers is declared compare to ClimateFinanceBERT classifications.\n"
              "Ratio = 100%: Perfect agreement between methods\n"
              "Ratio > 100%: Under declaration\n"
              "Ratio < 100%: Over declaration\n"
              "Missing values indicate periods where RIO classified no projects in that category"),
             ha='center', va='bottom', fontsize=14, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    plt.savefig(f"{output_folder}/ratio_comparison_by_donor.png", bbox_inches='tight', pad_inches=0.5, dpi=300)
    plt.close()

    return comparison_df

# -------------------------
# Main Function: Generate Selected Graphs
# -------------------------
def main():
    figures_folder = os.path.join(wd, "Figures/Graphs/")
    os.makedirs(figures_folder, exist_ok=True)
    
    # Create timelines
    df_time_disbursement = create_funding_timeline(df, os.path.join(wd, "Data/timeline_disbursement.csv"))
    df_time_commitment = create_commitment_timeline(df, os.path.join(wd, "Data/timeline_commitment.csv"))
    
    # Prepare global Rio marker data
    (rio_adapt1_disbursement, rio_adapt2_disbursement, rio_adapt1_commitment, rio_adapt2_commitment,
     rio_miti1_disbursement, rio_miti2_disbursement, rio_miti1_commitment, rio_miti2_commitment) = prepare_rio_data(df_origin)
    
    # Stacked area plots for mitigation
    cluster = df_time_disbursement['mitigation_funding'] / 1000
    output_file = os.path.join(figures_folder, "stacked_area_mitigation_disbursement.png")
    rio_stacked_area(cluster, rio_miti2_disbursement, rio_miti1_disbursement, output_file,
                      rio_colors=['#f7fcb9','#addd8e'], cluster_color='#31a354', 
                      climate_type="Mitigation", max_y=max(max(rio_miti2_disbursement), max(rio_miti1_disbursement))+5, data_type="disbursements")
    cluster = df_time_commitment['mitigation_commitment'] / 1000
    output_file = os.path.join(figures_folder, "stacked_area_mitigation_commitment.png")
    rio_stacked_area(cluster, rio_miti2_commitment, rio_miti1_commitment, output_file,
                      rio_colors=['#f7fcb9','#addd8e'], cluster_color='#31a354', 
                      climate_type="Mitigation", max_y=max(max(rio_miti2_commitment), max(rio_miti1_commitment))+5, data_type="commitments")
    
    # Stacked area plots for adaptation
    cluster = df_time_disbursement['adaptation_funding'] / 1000
    output_file = os.path.join(figures_folder, "stacked_area_adaptation_disbursement.png")
    rio_stacked_area(cluster, rio_adapt2_disbursement, rio_adapt1_disbursement, output_file,
                      rio_colors=['#fff7bc','#fec44f'], cluster_color='#d95f0e', 
                      climate_type="Adaptation", max_y=max(max(rio_adapt2_disbursement), max(rio_adapt1_disbursement))+5, data_type="disbursements")
    cluster = df_time_commitment['adaptation_commitment'] / 1000
    output_file = os.path.join(figures_folder, "stacked_area_adaptation_commitment.png")
    rio_stacked_area(cluster, rio_adapt2_commitment, rio_adapt1_commitment, output_file,
                      rio_colors=['#fff7bc','#fec44f'], cluster_color='#d95f0e', 
                      climate_type="Adaptation", max_y=max(max(rio_adapt2_commitment), max(rio_adapt1_commitment))+5, data_type="commitments")
    
    # Stackplot graphs
    df_stack_disbursement = df_time_disbursement[['effective_year', 'adaptation_funding', 'mitigation_funding', 'environment_funding']]
    output_file = os.path.join(figures_folder, "stackplot_disbursement.png")
    stacked_area(df_stack_disbursement, output_file, input_colors=['#d95f0e','#2ca25f','#2b8cbe'], max_number=3, data_type="disbursements")
    df_stack_commitment = df_time_commitment[['effective_year', 'adaptation_commitment', 'mitigation_commitment', 'environment_commitment']]
    output_file = os.path.join(figures_folder, "stackplot_commitment.png")
    stacked_area(df_stack_commitment, output_file, input_colors=['#d95f0e','#2ca25f','#2b8cbe'], max_number=3, data_type="commitments")
    
    # Combined adaptation & mitigation plot (global)
    cluster_adap = df_time_disbursement['adaptation_funding'] / 1000
    cluster_miti = df_time_disbursement['mitigation_funding'] / 1000
    combined_adaptation_mitigation_plot(cluster_adap, rio_adapt1_disbursement, rio_adapt2_disbursement,
                                          cluster_miti, rio_miti1_disbursement, rio_miti2_disbursement, figures_folder)
    
    # Combined climate finance analysis
    analyze_combined_climate_finance(df_time_disbursement, df_time_commitment, os.path.join(figures_folder, "trend_analysis"))
    
    # SARIMA forecast plot
    forecast_folder = os.path.join(figures_folder, "forecasts")
    os.makedirs(forecast_folder, exist_ok=True)
    forecast_climate_finance_sarima(df_time_disbursement, forecast_folder)
    
    # Comparison plots by donor
    create_comparison_timeline_by_donor(df_origin, df, figures_folder)
    create_ratio_comparison_timeline_by_donor(df_origin, df, figures_folder)
    
    print("Selected graphs generated in folder:", figures_folder)

if __name__ == "__main__":
    main()
