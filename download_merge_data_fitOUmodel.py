import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from pandas.tseries.offsets import MonthEnd


# URLs for datasets
fhfa_url = "https://www.fhfa.gov/hpi/download/quarterly_datasets/hpi_at_us_and_census.csv"
fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
bls_url = "C:\\Users\\atrsr\\OneDrive - Widener University\\research\\Personal-Research\\bls.txt"

# Load FHFA House Price Index
fhfa = pd.read_csv(fhfa_url, header=None)
fhfa.columns=['Div', 'Year', 'Qtr', 'HPI']
fhfa = fhfa[fhfa['Div'] == 'USA']
# Map quarters to end-of-quarter months
quarter_to_month = {'1': '03', '2': '06', '3': '09', '4': '12'}
fhfa['Month'] = fhfa['Qtr'].astype(str).map(quarter_to_month)
fhfa['DATE'] = pd.to_datetime(fhfa['Year'].astype(str) + '-' + fhfa['Month'] + '-01')+MonthEnd(0)
# fhfa = fhfa[['DATE', 'Index (NSA)']].rename(columns={'Index (NSA)': 'HPI'})
fhfa.set_index('DATE', inplace=True)

# Load 10-Year Treasury Yield from FRED
fred = pd.read_csv(fred_url)
fred.columns = ['DATE', 'DGS10']
fred['DATE'] = pd.to_datetime(fred['DATE'])
fred['DGS10'] = pd.to_numeric(fred['DGS10'], errors='coerce')
fred = fred.set_index('DATE').resample('Q').mean().rename(columns={'DGS10': '10Y_Yield'})

# Load BLS Rent CPI
bls = pd.read_csv(bls_url, sep='\t', engine='python')
bls.columns = [col.strip() for col in bls.columns]
bls = bls[bls['series_id'].str.rstrip() == 'CUSR0000SEHA']
bls['DATE'] = pd.to_datetime(bls['year'].astype(str) + '-' + bls['period'].str[1:] + '-01')
bls = bls[['DATE', 'value']].rename(columns={'value': 'Rent_CPI'})
bls.set_index('DATE', inplace=True)
bls = bls.resample('Q').mean()

# Merge datasets
data = fhfa.join([bls,fred], how='inner')
data.dropna(inplace=True)

# Save to CSV
data.to_csv("calibration_data.csv")
print("Saved merged dataset as 'calibration_data.csv'")

data["Fundamental_Value"] = data["Rent_CPI"]/data["10Y_Yield"]
data["Log_HPI_deviation"] = np.log(data["HPI"]/data["Fundamental_Value"])

def ou_process(t, kappa, theta, x0):
    return theta+(x0-theta)*np.exp(-kappa*t)

def estimate_ou(series):
    series = series.dropna()
    x = series.values
    t = np.arange(len(x))
    x0=x[0]

    def model(t, kappa, theta):
        return ou_process(t,kappa, theta, x0)
    
    popt,_ = curve_fit(model, t, x, bounds=([0, -np.inf], [1,np.inf]), maxfev=10000)
    kappa, theta = popt
    print(popt)

    residuals = x - model(t, *popt)
    sigma=np.std(residuals)

    return kappa, theta, sigma

kappa_r, theta_r, sigma_r = estimate_ou(data["Rent_CPI"])
kappa_y, theta_y, sigma_y = estimate_ou(data["10Y_Yield"])
kappa_r, theta_r, sigma_r = estimate_ou(data["Log_HPI_deviation"])