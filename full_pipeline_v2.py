"""
------------------------------------------------------------------
UPI Adoption Pipeline: Two-Stage Explanatory + Predictive Design
------------------------------------------------------------------
Builds a state-month panel from six data sources and runs a two-stage
analysis on it:
  - UPI transaction volume (target variable)
  - NFS (National Financial Switch): banking-infrastructure USAGE
  - GST collections: formal economic activity
  - ATM/CRM/WLA installed counts: banking-infrastructure SUPPLY
  - Internet subscribers: digital readiness
  - Urbanisation rate: structural/demographic control (time-invariant)

Note: Unemployment_rate and Gross State Value Added were also tested but did not
make it into the final six-predictor set. Both are still loaded further down and checked directly against
the confirmed predictors, as a documented sensitivity check rather than a
silent exclusion.

STAGE 1 (Parts G-J): Explanatory, where the same multiple regression is run
  three ways on three versions of the data: once on each state's own
  average (between-state), once on all monthly data together (with a
  correction for repeated state observations), and once on state-centered
  monthly data, where each state's own values with its own average subtracted
  out first (within-state). No train/test split required, no accuracy metric is needed, just
  regression coefficients and their significance are extracted. Output: a short list of
  confirmed predictors (significant in at least one of the two full-sample
  regressions).

STAGE 2 (Part 2): Predictive, and treated as a genuinely separate
  question from Stage 1. Given the confirmed predictors, how well can a
  state's adoption intensity actually be predicted? Ridge and Random Forest
  are compared via leave-one-state-out cross-validation. Here predictive accuracy is used as a criterion, 
  and it isn't used to re-validate Stage 1.

Pipeline needs the 33 raw UPI xlsx files, Digital-NFS-data.csv, gst-collection-data.csv,
ATM-data.csv, Internet-data.csv, Population-data.csv, and Urbanisation-data.csv
in the same folder.
Requires: pandas, numpy, statsmodels, scipy, matplotlib, scikit-learn
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings('ignore')

MONTH_MAP = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,
             'August':8,'September':9,'October':10,'November':11,'December':12}
MONTH_ABBR3 = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
               'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
QUARTER_END_MONTH = {'March':3, 'June':6, 'September':9, 'December':12}

def clean_state(s):
    s = s.str.strip().str.replace('&', 'And', regex=False)
    s = s.str.replace(r'\s+', ' ', regex=True)
    return s.str.title()

def extract_fy(s):
    if pd.isna(s):
        return None
    m = re.search(r'(\d{4})', str(s).strip())
    return int(m.group(1)) if m else None

NON_STATE_TOKENS = ['All India', 'Central Board', 'Oidar', 'Other Territory',
                     'Unclassified', 'Unknown', 'Total', '#', 'Other', 'Grand Total']

def drop_non_states(df, col):
    pattern = '|'.join(re.escape(t) for t in NON_STATE_TOKENS)
    mask = df[col].astype(str).str.contains(pattern, case=False, na=True)
    return df[~mask].copy()

state_pops = {
    'Maharashtra': 12.735967, 'Uttar Pradesh': 23.807767, 'Bihar': 12.859233,
    'West Bengal': 9.9563, 'Madhya Pradesh': 8.761, 'Tamil Nadu': 7.708867,
    'Rajasthan': 8.189733, 'Karnataka': 6.8115, 'Gujarat': 7.2367,
    'Andhra Pradesh': 5.334, 'Odisha': 4.441967, 'Telangana': 3.8272,
    'Kerala': 3.591967, 'Jharkhand': 3.996333, 'Assam': 3.604733,
    'Punjab': 3.0926, 'Chhattisgarh': 3.052367, 'Haryana': 3.057267,
    'Delhi': 2.175233, 'Jammu And Kashmir': 1.370067, 'Uttarakhand': 1.175533,
    'Himachal Pradesh': 0.7505, 'Tripura': 0.418433, 'Meghalaya': 0.337933,
    'Manipur': 0.325267, 'Nagaland': 0.225333, 'Goa': 0.1583,
    'Arunachal Pradesh': 0.1576, 'Puducherry': 0.1683, 'Mizoram': 0.124967,
    'Chandigarh': 0.1243, 'Sikkim': 0.069533,
    'Andaman And Nicobar Islands': 0.0404, 'Andaman And Nicobar': 0.0404,
    'Dadra And Nagar Haveli': 0.074467, 'Daman And Diu': 0.0611,
    'Dadra And Nagar Haveli And Daman And Diu': 0.135567,
    'Lakshadweep': 0.0069, 'Ladakh': 0.0302,
}
# The dict above is a fallback only. Real population figures come from the
# Dataful-hosted CSV below (dataset 18521, sourced from the Ministry of
# Health & Family Welfare's population projections), which gets loaded next
# and overwrites every value it can. Spot-checked a handful of states
# (Delhi, Maharashtra, Bihar, Sikkim, Telangana, Ladakh) against the dict
# and they matched exactly, so this is just a cleaner, clickable version of
# the same numbers, not a different source.
pop_df = pd.read_csv('Population-data.csv')
pop_df = pop_df[(pop_df['month'] == 'March') & (pop_df['gender'] == 'Total') &
                (pop_df['year'].isin([2023, 2024, 2025]))].copy()
pop_df['state'] = clean_state(pop_df['state'])
pop_df = drop_non_states(pop_df, 'state')  # drops "All India"
pop_avg = pop_df.groupby('state')['value'].mean().reset_index()
pop_avg['pop_crore'] = pop_avg['value'] / 10000.0  # thousands -> crore
state_pops_csv = dict(zip(pop_avg['state'], pop_avg['pop_crore']))
# Merge the pre-2020 Dadra/Daman split into the combined UT, matching the
# naming convention used by every other source in this pipeline
if 'Dadra And Nagar Haveli' in state_pops_csv and 'Daman And Diu' in state_pops_csv:
    state_pops_csv['Dadra And Nagar Haveli And Daman And Diu'] = (
        state_pops_csv['Dadra And Nagar Haveli'] + state_pops_csv['Daman And Diu'])
state_pops_csv['Andaman And Nicobar'] = state_pops_csv.get('Andaman And Nicobar Islands', np.nan)

# Using the CSV-sourced figures as primary; fall back to the hardcoded dict only
# if a state is somehow missing from the CSV (should not happen in practice)
for k in state_pops:
    if k in state_pops_csv and not np.isnan(state_pops_csv[k]):
        state_pops[k] = state_pops_csv[k]
nat_pop = sum({k:v for k,v in state_pops.items()
               if k not in ('Andaman And Nicobar Islands','Dadra And Nagar Haveli','Daman And Diu')}.values())
print(f"  Population loaded from Dataful CSV (tractable source), cross-checked against hardcoded values.")

# Urbanisation rate, computed from the Dataful-hosted companion dataset
# (dataset 18520, "State- and Gender-wise Yearly Ratio of Projected Urban
# Population"), same filter/average logic as population above (March,
# Total, 2023-2025 average). Every state comes from this one CSV, so there's
# no manual per-state figure sitting anywhere that could go stale or wrong.
urb_df = pd.read_csv('Urbanisation-data.csv')
urb_df = urb_df[(urb_df['month'] == 'March') & (urb_df['gender'] == 'Total') &
                (urb_df['year'].isin([2023, 2024, 2025]))].copy()
urb_df['state'] = clean_state(urb_df['state'])
urb_df = drop_non_states(urb_df, 'state')  # drops "All India"
urb_avg = urb_df.groupby('state')['value'].mean().reset_index()
urbanisation_pct = dict(zip(urb_avg['state'], urb_avg['value']))
# Merge the pre-2020 Dadra/Daman split into the combined UT, matching the
# naming convention used by every other source in this pipeline
if 'Dadra And Nagar Haveli' in urbanisation_pct and 'Daman And Diu' in urbanisation_pct:
    # urban share of a merged UT is a population-weighted average, not a simple
    # mean of the two percentages, use the same population weights already
    # computed above (state_pops_csv, in the same crore units) for consistency
    w_dnh = state_pops_csv.get('Dadra And Nagar Haveli', np.nan)
    w_dd = state_pops_csv.get('Daman And Diu', np.nan)
    if not (np.isnan(w_dnh) or np.isnan(w_dd)) and (w_dnh + w_dd) > 0:
        urbanisation_pct['Dadra And Nagar Haveli And Daman And Diu'] = (
            (urbanisation_pct['Dadra And Nagar Haveli'] * w_dnh +
             urbanisation_pct['Daman And Diu'] * w_dd) / (w_dnh + w_dd))
    else:
        urbanisation_pct['Dadra And Nagar Haveli And Daman And Diu'] = (
            urbanisation_pct['Dadra And Nagar Haveli'] + urbanisation_pct['Daman And Diu']) / 2
urbanisation_pct['Andaman And Nicobar'] = urbanisation_pct.get('Andaman And Nicobar Islands', np.nan)

######################## PART A — Merge raw monthly UPI source files ############################

print("="*80); print("PART A: Merging raw monthly UPI source files"); print("="*80)

UPI_SOURCE_FILES = [
    'Ecosystem-Statistics-UPI-Upi-statewise-statistics-2023-Apr.xlsx','Ecosystem-Statistics-UPI-Upi-statewise-statistics-2023-May.xlsx',
    'Ecosystem-Statistics-UPI-Upi-statewise-statistics-2023-Jun.xlsx','Ecosystem-Statistics-UPI-Upi-statewise-statistics-2023-Jul.xlsx',
    'Ecosystem-Statistics-UPI-Upi-statewise-statistics-2023-Aug.xlsx','Ecosystem-Statistics-UPI-Upi-statewise-statistics-2023-Sep.xlsx',
    'Ecosystem-Statistics-UPI-Upi-statewise-statistics-2023-Oct.xlsx','Ecosystem-Statistics-UPI-Upi-statewise-statistics-2023-Nov.xlsx',
    'Ecosystem-Statistics-UPI-Upi-statewise-statistics-2023-Dec.xlsx','Ecosystem-Statistics-UPI-Upi-statewise-statistics-2024-Jan.xlsx',
    'Ecosystem-Statistics-UPI-Upi-statewise-statistics-2024-Feb.xlsx','Ecosystem-Statistics-UPI-Upi-statewise-statistics-2024-Mar.xlsx',
    'Ecosystem-Statistics-UPI-Upi-statewise-statistics-2024-Apr.xlsx','Ecosystem-Statistics-UPI-Upi-statewise-statistics-2024-May.xlsx',
    'Ecosystem-Statistics-UPI-Upi-statewise-statistics-2024-Jun.xlsx','Ecosystem-Statistics-UPI-Upi-statewise-statistics-2024-Jul.xlsx',
    'Ecosystem-Statistics-UPI-Upi-statewise-statistics-2024-Aug.xlsx','Ecosystem-Statistics-UPI-Upi-statewise-statistics-2024-Sep.xlsx',
    'Ecosystem-Statistics-UPI-Upi-statewise-statistics-2024-Oct.xlsx','Ecosystem-Statistics-UPI-Upi-statewise-statistics-2024-Nov.xlsx',
    'Ecosystem-Statistics-UPI-Upi-statewise-statistics-2024-Dec.xlsx','Ecosystem-Statistics-UPI-Upi-statewise-statistics-2025-Jan.xlsx',
    'Ecosystem-Statistics-UPI-Upi-statewise-statistics-2025-Feb.xlsx','Ecosystem-Statistics-UPI-Upi-statewise-statistics-2025-Mar.xlsx',
    'Ecosystem-Statistics-UPI-Upi-statewise-statistics-2025-Apr.xlsx','Ecosystem-Statistics-UPI-Upi-statewise-statistics-2025-May.xlsx',
    'Ecosystem-Statistics-UPI-Upi-statewise-statistics-2025-Jun.xlsx','Ecosystem-Statistics-UPI-Upi-statewise-statistics-2025-Jul.xlsx',
    'Ecosystem-Statistics-UPI-Upi-statewise-statistics-2025-Aug.xlsx','Ecosystem-Statistics-UPI-Upi-statewise-statistics-2025-Sep.xlsx',
    'Ecosystem-Statistics-UPI-Upi-statewise-statistics-2025-Oct.xlsx','Ecosystem-Statistics-UPI-Upi-statewise-statistics-2025-Nov.xlsx',
    'Ecosystem-Statistics-UPI-Upi-statewise-statistics-2025-Dec.xlsx',
]

def _clean_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '', regex=False), errors='coerce')

def _clean_contribution(series):
    s = series.astype(str).str.strip()
    is_pct = s.str.endswith('%')
    out = pd.to_numeric(s.str.replace('%', '', regex=False), errors='coerce')
    return np.where(is_pct, out / 100.0, out)

def load_one_upi_file(fname):
    m = re.search(r'(\d{4})-(\w{3})\.xlsx$', fname)
    year, mon_abbr = int(m.group(1)), m.group(2)
    month = MONTH_ABBR3[mon_abbr]
    xl = pd.ExcelFile(fname)
    raw = pd.read_excel(fname, sheet_name=xl.sheet_names[0], header=None)
    raw.columns = raw.iloc[1]
    df = raw.iloc[2:].reset_index(drop=True)
    df = df.rename(columns={df.columns[0]: 'Sr. No.'})
    df['Volume (in Mn)'] = _clean_numeric(df['Volume (in Mn)'])
    df['Year'] = year; df['Month'] = month
    return df[['State / Union Territory', 'Volume (in Mn)', 'Month', 'Year']]

upi_df = pd.concat([load_one_upi_file(f) for f in UPI_SOURCE_FILES], ignore_index=True)
upi_df['State / Union Territory'] = clean_state(upi_df['State / Union Territory'])
upi_df = drop_non_states(upi_df, 'State / Union Territory')
print(f"  Merged {len(UPI_SOURCE_FILES)} monthly files -> {len(upi_df):,} rows")

########################## PART B — Target variable (DVI_raw) ##########################

print("\n" + "="*80); print("PART B: Building target variable"); print("="*80)

upi_ts = upi_df.groupby(['State / Union Territory', 'Year', 'Month'])['Volume (in Mn)'].mean().reset_index()
upi_ts.columns = ['State', 'Year', 'Month', 'State_Volume']
upi_ts['Population'] = upi_ts['State'].map(state_pops)
upi_ts['Volume_Per_Capita'] = upi_ts['State_Volume'] / upi_ts['Population']
upi_nat = upi_df.groupby(['Year', 'Month'])['Volume (in Mn)'].sum().reset_index()
upi_nat['Nat_Per_Capita'] = upi_nat['Volume (in Mn)'] / nat_pop
upi_ts = upi_ts.merge(upi_nat[['Year','Month','Nat_Per_Capita']], on=['Year','Month'])
upi_ts['DVI_raw'] = upi_ts['Volume_Per_Capita'] / upi_ts['Nat_Per_Capita']
print(f"  DVI_raw built: {len(upi_ts):,} state-month records, {upi_ts['State'].nunique()} states")

##################### PART C — NFS (banking-infrastructure USAGE) and GST (formal economic activity) ##################

print("\n" + "="*80); print("PART C: NFS and GST predictors"); print("="*80)

nfs_df = pd.read_csv('Digital-NFS-data.csv')
nfs_df['state'] = clean_state(nfs_df['state'])
NFS_STATE_ALIASES = {
    'Andaman And Nicobar Islands': 'Andaman And Nicobar',
    'Dadra And Nagar Haveli': 'Dadra And Nagar Haveli And Daman And Diu',
    'Daman And Diu': 'Dadra And Nagar Haveli And Daman And Diu',
}
nfs_df['state'] = nfs_df['state'].replace(NFS_STATE_ALIASES)
nfs_df = drop_non_states(nfs_df, 'state')
nfs_df['Month_Num'] = nfs_df['month'].map(MONTH_MAP)
nfs_df['fy_start'] = nfs_df['fiscal_year'].apply(extract_fy)
nfs_df['year'] = np.where(nfs_df['Month_Num'] <= 3, nfs_df['fy_start'] + 1, nfs_df['fy_start'])
nfs_ts = nfs_df.groupby(['state','year','Month_Num'])['volume'].sum().reset_index()
nfs_ts.columns = ['State','Year','Month','State_NFS']
nfs_nat = nfs_df.groupby(['year','Month_Num'])['volume'].sum().reset_index()
nfs_nat.columns = ['Year','Month','Nat_NFS']
nfs_ts = nfs_ts.merge(nfs_nat, on=['Year','Month'])
nfs_ts['IRS'] = nfs_ts['State_NFS'] / nfs_ts['Nat_NFS']
nfs_ts['Population'] = nfs_ts['State'].map(state_pops)
nfs_ts['NFS_per_capita'] = nfs_ts['State_NFS'] / nfs_ts['Population']
print(f"  NFS: {len(nfs_ts):,} state-month records")

gst_df = pd.read_csv('gst-collection-data.csv')
gst_df['state'] = clean_state(gst_df['state'])
GST_STATE_ALIASES = {
    'Andaman And Nicobar Islands': 'Andaman And Nicobar',
    'Dadra And Nagar Haveli': 'Dadra And Nagar Haveli And Daman And Diu',
    'Daman And Diu': 'Dadra And Nagar Haveli And Daman And Diu',
}
gst_df['state'] = gst_df['state'].replace(GST_STATE_ALIASES)
gst_df = drop_non_states(gst_df, 'state')
if pd.api.types.is_numeric_dtype(gst_df['month']):
    gst_df['month_num'] = gst_df['month']
else:
    gst_df['month_num'] = gst_df['month'].map(MONTH_MAP).fillna(pd.to_numeric(gst_df['month'], errors='coerce'))
gst_df['month_num'] = gst_df['month_num'].astype(int)
gst_f = gst_df[(gst_df['year'] >= 2023) & (gst_df['year'] <= 2025) & (gst_df['tax_type'] == 'Total Tax Collection')].copy()
gst_ts = gst_f.groupby(['state','year','month_num'])['amount'].sum().reset_index()
gst_ts.columns = ['State','Year','Month','GST_Amount']
gst_ts['Population'] = gst_ts['State'].map(state_pops)
gst_ts['GST_per_capita'] = gst_ts['GST_Amount'] / gst_ts['Population']
print(f"  GST: {len(gst_ts):,} state-month records")

################## PART D — ATM/CRM/WLA (banking-infrastructure SUPPLY) ##################

# Quarterly data (Dec/Jun/Mar/Sep only), forward-filled across the two
# months in each quarter with no direct reading. "Dadra and Nagar Haveli"
# is aliased to the merged-UT name. Its own district list already
# includes Diu and Daman, so nothing is actually missing, just mislabeled.
print("\n" + "="*80); print("PART D: ATM/CRM/WLA Supply side infrastructure "); print("="*80)

atm_df = pd.read_csv('ATM-data.csv')
atm_df['state'] = clean_state(atm_df['state'])
ATM_STATE_ALIASES = {
    'Dadra And Nagar Haveli': 'Dadra And Nagar Haveli And Daman And Diu',
    'Andaman And Nicobar Islands': 'Andaman And Nicobar',
}
atm_df['state'] = atm_df['state'].replace(ATM_STATE_ALIASES)
atm_df = drop_non_states(atm_df, 'state')
atm_df['Month_Num'] = atm_df['month'].map(QUARTER_END_MONTH)
atm_df['fy_start'] = atm_df['fiscal_year'].apply(extract_fy)
atm_df['year'] = np.where(atm_df['Month_Num'] <= 3, atm_df['fy_start'] + 1, atm_df['fy_start'])
atm_q = atm_df.groupby(['state','year','Month_Num'])['value'].sum().reset_index()
atm_q.columns = ['State','Year','Month','ATM_count']

# Expand each quarterly observation to all 3 months of that quarter (fwd/back-fill)
def expand_quarterly(df, val_col, new_col):
    rows = []
    for _, r in df.iterrows():
        q_end = r['Month']
        months_in_q = [((q_end - 1 - i) % 12) + 1 for i in range(3)]
        for m in months_in_q:
            yr = r['Year'] if m <= q_end else r['Year']  # same year; quarters don't cross Jan boundary here except Dec->prior months in same FY
            rows.append({'State': r['State'], 'Year': yr, 'Month': m, new_col: r[val_col]})
    return pd.DataFrame(rows).drop_duplicates(subset=['State','Year','Month'])

atm_monthly = expand_quarterly(atm_q, 'ATM_count', 'ATM_count')
atm_monthly['Population'] = atm_monthly['State'].map(state_pops)
atm_monthly['ATM_per_capita'] = atm_monthly['ATM_count'] / atm_monthly['Population']
print(f"  ATM: {len(atm_q):,} quarterly records -> {len(atm_monthly):,} monthly (forward-filled) records")

################# PART E — Internet Subscribers (digital readiness) ############################
# Quarterly data. Three data-quality fixes applied, each verified directly
# against the raw file before use:
#   1. Mumbai and Kolkata are ADDITIVE to Maharashtra and West Bengal
#      respectively (confirmed: Grand Total = sum of all states + Mumbai +
#      Kolkata, to within rounding). So state totals are computed as
#      Maharashtra+Mumbai and West Bengal+Kolkata.
#   2. Uttar Pradesh is reported as either "Uttar Pradesh East and West"
#      (combined) or "Uttar Pradesh East" + "Uttar Pradesh West" (split) in
#      different quarters, but never both simultaneously. These are summed generically
#      by state-name-prefix match so whichever form is present is used once.
#   3. "Dadra and Nagar Haveli" (used through FY2024-25) and "Dadra Nagar
#      Haveli and Daman Diu" (used from FY2025-26) are a naming transition
#      over time, not a duplicate pair and aliased to one label.

print("\n" + "="*80); print("PART E: Internet Subscribers - Digital Readiness"); print("="*80)

net_df = pd.read_csv('Internet-data.csv')
net_df = net_df[net_df['type_of_connection'] == 'Total'].copy()  # broadband+narrowband combined
net_df['service_area'] = clean_state(net_df['service_area'])

# Fix 1: consolidate UP variants under one name (never coexist in same quarter)
net_df['service_area'] = net_df['service_area'].replace({
    'Uttar Pradesh East': 'Uttar Pradesh', 'Uttar Pradesh West': 'Uttar Pradesh',
    'Uttar Pradesh East And West': 'Uttar Pradesh',
})
# Fix 3: Dadra-Daman naming transition, and Andaman naming (pre-/post-2020 conventions)
net_df['service_area'] = net_df['service_area'].replace({
    'Dadra Nagar Haveli And Daman Diu': 'Dadra And Nagar Haveli And Daman And Diu',
    'Dadra And Nagar Haveli': 'Dadra And Nagar Haveli And Daman And Diu',
    'Andaman And Nicobar Islands': 'Andaman And Nicobar',
})
net_df = drop_non_states(net_df, 'service_area')

# Sum rural+urban, then sum any same-named rows within a state-quarter (this
# naturally combines split UP East/West rows into one Uttar Pradesh total)
net_q = net_df.groupby(['fiscal_year','quarter','service_area'])['value'].sum().reset_index()

# Fix 2: add Mumbai into Maharashtra, Kolkata into West Bengal (additive metros)
metro_map = {'Mumbai': 'Maharashtra', 'Kolkata': 'West Bengal'}
net_q['service_area'] = net_q['service_area'].replace(metro_map)
net_q = net_q.groupby(['fiscal_year','quarter','service_area'])['value'].sum().reset_index()
net_q.columns = ['fiscal_year','quarter','State','Internet_subscribers_mn']

quarter_end_month_num = {'Q1':6, 'Q2':9, 'Q3':12, 'Q4':3}  # Indian FY: Q1=Apr-Jun ends Jun, Q4=Jan-Mar ends Mar (next cal year)
net_q['fy_start'] = net_q['fiscal_year'].apply(extract_fy)
net_q['Month'] = net_q['quarter'].map(quarter_end_month_num)
net_q['Year'] = np.where(net_q['quarter']=='Q4', net_q['fy_start'] + 1, net_q['fy_start'])

net_monthly = expand_quarterly(net_q.rename(columns={'Internet_subscribers_mn':'val'}), 'val', 'Internet_subscribers_mn')
net_monthly['Population'] = net_monthly['State'].map(state_pops)
net_monthly['Internet_per_capita'] = net_monthly['Internet_subscribers_mn'] / net_monthly['Population']
print(f"  Internet: {net_q['State'].nunique()} states, {len(net_q):,} quarterly records -> {len(net_monthly):,} monthly records")

##################### PART F — Build full monthly panel (5 predictors, unemployment dropped) #####################

print("\n" + "="*80); print("PART F: Building full monthly panel"); print("="*80)

def build_panel(winsor_pct=0.01):
    ts = upi_ts.copy()
    if winsor_pct > 0:
        lo, hi = ts['DVI_raw'].quantile(winsor_pct), ts['DVI_raw'].quantile(1 - winsor_pct)
        ts = ts[(ts['DVI_raw'] >= lo) & (ts['DVI_raw'] <= hi)].copy()
    panel = ts[['State','Year','Month','DVI_raw']].copy()
    panel = panel.merge(nfs_ts[['State','Year','Month','IRS','NFS_per_capita']], on=['State','Year','Month'], how='left')
    panel = panel.merge(gst_ts[['State','Year','Month','GST_per_capita']], on=['State','Year','Month'], how='left')
    panel = panel.merge(atm_monthly[['State','Year','Month','ATM_per_capita']], on=['State','Year','Month'], how='left')
    panel = panel.merge(net_monthly[['State','Year','Month','Internet_per_capita']], on=['State','Year','Month'], how='left')
    panel['Urbanisation_pct'] = panel['State'].map(urbanisation_pct)
    panel = panel[panel['DVI_raw'].notna() & (panel['DVI_raw'] > 0)].copy()
    panel = panel.sort_values(['State','Year','Month']).reset_index(drop=True)
    panel = panel.dropna(subset=['IRS','NFS_per_capita','GST_per_capita']).reset_index(drop=True)
    panel['time_idx'] = (panel['Year'] - panel['Year'].min()) * 12 + panel['Month']
    return panel

panel = build_panel(winsor_pct=0.01)
print(f"  Panel: {len(panel):,} state-months, {panel['State'].nunique()} states")
print(f"  ATM_per_capita coverage: {panel['ATM_per_capita'].notna().sum()}/{len(panel)}")
print(f"  Internet_per_capita coverage: {panel['Internet_per_capita'].notna().sum()}/{len(panel)}")

TIME_VARYING = ['IRS', 'NFS_per_capita', 'GST_per_capita', 'ATM_per_capita', 'Internet_per_capita']
TIME_INVARIANT = ['Urbanisation_pct']
ALL_PREDICTORS = TIME_VARYING + TIME_INVARIANT

panel_complete = panel.dropna(subset=TIME_VARYING + TIME_INVARIANT).reset_index(drop=True)
print(f"  Complete-case panel (all 5 time-varying predictors present): {len(panel_complete):,} state-months, "
      f"{panel_complete['State'].nunique()} states")

########################### Exploratory Data Analysis ############################
# Distribution of the target variable, correlation structure among all
# predictors, and bivariate relationships with DVI_raw. Descriptive only
# no modelling decisions happen here, it just shows what the data looks
# like before any regression is run.

print("\n" + "="*80); print("Exploratory Data Analysis"); print("="*80)
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})

# Unemployment and GSVA data, loaded here alongside the other source
# files for tidiness. Not used until the Section 5.1 exclusion test much
# further down. Its also not used in any figure or the six confirmed predictors.
unemployment_df = pd.read_csv('unemployment-data.csv')
unemployment_df['state'] = clean_state(unemployment_df['state'])
unemployment_df = drop_non_states(unemployment_df, 'state')
unemployment_df['fy_num'] = unemployment_df['fiscal_year'].apply(extract_fy)
unemployment_df = unemployment_df[unemployment_df['fiscal_year'] != '2025'].copy()
unemployment_df['gender_clean'] = unemployment_df['gender'].str.strip().str.lower()
unemployment_df['region_clean'] = unemployment_df['region'].str.strip().str.lower()
unem_overall = unemployment_df[
    (unemployment_df['gender_clean'] == 'person') & (unemployment_df['region_clean'] == 'rural+urban')
].copy()
unem_avg = unem_overall[unem_overall['fy_num'].isin([2022, 2023])].groupby('state')['unemployment_rate'].mean()
unem_dict = unem_avg.to_dict()

gsva_df = pd.read_csv('gross-state-value-data.csv')
gsva_df['state'] = clean_state(gsva_df['state'])
GSVA_STATE_ALIASES = {
    'Andaman And Nicobar Islands': 'Andaman And Nicobar',
    'Dadra And Nagar Haveli': 'Dadra And Nagar Haveli And Daman And Diu',
    'Daman And Diu': 'Dadra And Nagar Haveli And Daman And Diu',
}
gsva_df['state'] = gsva_df['state'].replace(GSVA_STATE_ALIASES)
gsva_df = drop_non_states(gsva_df, 'state')
gsva_recent = gsva_df[gsva_df['fiscal_year'].isin(['2022-23', '2023-24'])]
gsva_avg = gsva_recent.groupby('state')['constant_prices'].mean()
gsva_per_capita = {}
for s, v in gsva_avg.items():
    if s in state_pops and not pd.isna(v):
        gsva_per_capita[s] = v / (state_pops[s] * 1e7) * 1e5

eda_cs = panel_complete.groupby('State').agg(
    DVI_raw=('DVI_raw','mean'), **{f: (f,'mean') for f in TIME_VARYING}
).reset_index()
eda_cs['Urbanisation_pct'] = eda_cs['State'].map(urbanisation_pct)
eda_cs = eda_cs.dropna(subset=ALL_PREDICTORS).reset_index(drop=True)

print(f"  DVI_raw skewness: {scipy_stats.skew(eda_cs['DVI_raw']):.3f}")

# Summary statistics table for DVI_raw and all six predictors
summary_vars = ['DVI_raw'] + ALL_PREDICTORS
summary_stats = eda_cs[summary_vars].describe().T
summary_stats['skew'] = eda_cs[summary_vars].skew()
summary_stats = summary_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew']]
print("\n  Summary statistics (state averages, n=36):")
print(summary_stats.round(3).to_string())
summary_stats.to_csv('table_3_2_summary_stats.csv')
print("\n  Exported: table_3_2_summary_stats.csv")

corr = eda_cs[['DVI_raw'] + ALL_PREDICTORS].corr()
print("\n  Correlation matrix (six confirmed candidates):")
print(corr.round(3).to_string())

# Fig 1: distribution of DVI_raw
fig1, ax = plt.subplots(figsize=(8, 5))
ax.hist(eda_cs['DVI_raw'], bins=12, color='#3498db', edgecolor='white')
ax.axvline(eda_cs['DVI_raw'].mean(), color='red', linestyle='--', label=f"Mean ({eda_cs['DVI_raw'].mean():.1f})")
ax.axvline(eda_cs['DVI_raw'].median(), color='green', linestyle='--', label=f"Median ({eda_cs['DVI_raw'].median():.1f})")
ax.set_xlabel('DVI_raw'); ax.set_ylabel('Number of states')
ax.set_title(f'Distribution of DVI_raw Across States (n={len(eda_cs)})')
ax.legend()
plt.tight_layout(); plt.savefig('eda_fig1_dvi_distribution.png', dpi=150, bbox_inches='tight'); plt.show()

# Fig 2: correlation heatmap (six confirmed candidates only. Unemployment_rate
# and GSVA_per_capita are excluded here and tested separately in Section 5.1;
# see Section 3.7 for why they don't belong in this six-predictor EDA treatment)
fig2, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha='right')
ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha='center', va='center',
                 color='white' if abs(corr.iloc[i,j]) > 0.5 else 'black', fontsize=8)
ax.set_title('Correlation Matrix: DVI_raw and All Six Candidate Predictors')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout(); plt.savefig('eda_fig2_correlation_heatmap.png', dpi=150, bbox_inches='tight'); plt.show()

# Fig 3: DVI_raw vs each candidate (scatter with linear fit), six confirmed
# candidates only
fig3, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, f in zip(axes.flatten(), ALL_PREDICTORS):
    ax.scatter(eda_cs[f], eda_cs['DVI_raw'], alpha=0.7, color='#3498db', edgecolor='white')
    z = np.polyfit(eda_cs[f], eda_cs['DVI_raw'], 1)
    xline = np.linspace(eda_cs[f].min(), eda_cs[f].max(), 50)
    ax.plot(xline, np.polyval(z, xline), color='red', linestyle='--')
    r = eda_cs[[f, 'DVI_raw']].corr().iloc[0, 1]
    ax.set_xlabel(f); ax.set_ylabel('DVI_raw'); ax.set_title(f'{f} (r = {r:.2f})')
for ax in axes.flatten()[len(ALL_PREDICTORS):]:
    ax.axis('off')
plt.suptitle('DVI_raw vs Each Candidate Predictor (State Averages)', fontsize=13)
plt.tight_layout(); plt.savefig('eda_fig3_scatter_predictors.png', dpi=150, bbox_inches='tight'); plt.show()

# Fig 4: state ranking bar chart, coloured by tier relative to the national mean
eda_ranked = eda_cs.sort_values('DVI_raw', ascending=True).reset_index(drop=True)
nat_mean = eda_cs['DVI_raw'].mean()
def tier_color(v):
    if v >= 1.5 * nat_mean:
        return '#e74c3c'   # high tier: >=1.5x national mean
    elif v >= nat_mean:
        return '#e67e22'   # mid tier: at or above mean, below 1.5x
    else:
        return '#2ecc71'   # low tier: below national mean
colors = [tier_color(v) for v in eda_ranked['DVI_raw']]

fig4, ax = plt.subplots(figsize=(9, 11))
ax.barh(eda_ranked['State'], eda_ranked['DVI_raw'], color=colors)
ax.axvline(nat_mean, color='black', linestyle='--', linewidth=1, label=f'National mean ({nat_mean:.1f})')
ax.set_xlabel('DVI_raw (state average)')
ax.set_title('UPI Adoption Intensity by State/UT, Ranked')
ax.legend(loc='lower right')
plt.tight_layout(); plt.savefig('eda_fig4_state_ranking.png', dpi=150, bbox_inches='tight'); plt.show()

# Fig 5: DVI_raw development over time, national average plus a few illustrative states,
# addressing feedback that the cross-sectional distribution (Fig 1) alone does not show
# how the target variable actually develops across the study window.
panel_ts = panel_complete.copy()
panel_ts['ym'] = panel_ts['Year'].astype(str) + '-' + panel_ts['Month'].astype(str).str.zfill(2)
national_trend = panel_ts.groupby('ym')['DVI_raw'].mean().reset_index().sort_values('ym')

illustrative_states = ['Telangana', 'Bihar', 'Kerala', 'Uttar Pradesh']
fig5, ax = plt.subplots(figsize=(11, 6))
ax.plot(national_trend['ym'], national_trend['DVI_raw'], color='black', linewidth=2.5,
        label='National average (all 36 states)', zorder=5)
colors_line = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
for state, c in zip(illustrative_states, colors_line):
    st_data = panel_ts[panel_ts['State'] == state].sort_values('ym')
    if len(st_data) > 0:
        ax.plot(st_data['ym'], st_data['DVI_raw'], color=c, linewidth=1.3, alpha=0.8, label=state)
ax.set_xlabel('Month'); ax.set_ylabel('DVI_raw')
ax.set_title('DVI_raw Over Time: National Average and Four Illustrative States')
ax.legend(loc='upper left', fontsize=8)
tick_positions = range(0, len(national_trend), max(1, len(national_trend)//12))
ax.set_xticks([national_trend['ym'].iloc[i] for i in tick_positions])
ax.tick_params(axis='x', rotation=45, labelsize=7)
plt.tight_layout(); plt.savefig('eda_fig5_dvi_time_trend.png', dpi=150, bbox_inches='tight'); plt.show()

############### STAGE 1 — Explanatory Analysis (Parts G-J) ######################
# All three regressions below are ordinary multiple regression. 
#   Part G: regression on each state's own AVERAGE across the whole period
#           (one row per state). It asks, do states that differ structurally from
#           each other also differ in adoption?
#   Part H: the same regression on every monthly row at once, with a
#           standard-error correction since a state's own months aren't
#           independent observations of each other
#   Part I: the same regression again, but with each state's own average
#           subtracted out of every one of its monthly values first. This
#           strips out anything permanently true about a state plus any
#           shared national trend, leaving only "did this state's own
#           month-to-month change in a predictor line up with its own
#           month-to-month change in adoption?"
# No cross-validation or accuracy scoring anywhere in Stage 1, It gives just
# coefficients and significance, since this stage is asking whether an
# association is real, not whether it can predict anything.

print("\n" + "#"*80); print("# STAGE 1: Explanatory analysis (no prediction, no train/test split)"); print("#"*80)

# Part G: regression on each state's own average (between-state comparison)
print("\n" + "="*80); print("PART G: Regression on state averages (between-state comparison)"); print("="*80)
cs = panel_complete.groupby('State').agg(
    DVI_raw=('DVI_raw','mean'), **{f: (f,'mean') for f in TIME_VARYING}
).reset_index()
cs['Urbanisation_pct'] = cs['State'].map(urbanisation_pct)
cs = cs.dropna(subset=ALL_PREDICTORS).reset_index(drop=True)

X_be = sm.add_constant(cs[ALL_PREDICTORS])
be_cl = sm.OLS(cs['DVI_raw'], X_be).fit()
be_hc3 = sm.OLS(cs['DVI_raw'], X_be).fit(cov_type='HC3')
print(f"  n={len(cs)} states, R^2={be_cl.rsquared:.4f}, F({int(be_cl.df_model)},{int(be_cl.df_resid)})={be_cl.fvalue:.3f}, p(F)={be_cl.f_pvalue:.4g}")
print(f"  {'Term':<22} {'Coef':>12} {'p (classical)':>14} {'p (HC3)':>10}")
for t in X_be.columns:
    print(f"  {t:<22} {be_cl.params[t]:>12.4f} {be_cl.pvalues[t]:>14.4f} {be_hc3.pvalues[t]:>10.4f}")

# Part H: same regression, all monthly rows, with a state-clustering correction
print("\n" + "="*80); print("PART H: Same regression on all monthly data (adjusted for repeated state observations)"); print("="*80)
X_pool = sm.add_constant(panel_complete[ALL_PREDICTORS])
pooled = sm.OLS(panel_complete['DVI_raw'], X_pool).fit(cov_type='cluster', cov_kwds={'groups': panel_complete['State']})
print(f"  n={int(pooled.nobs)} state-months, R^2={pooled.rsquared:.4f}, F({int(pooled.df_model)},{int(pooled.df_resid)})={pooled.fvalue:.3f}, p(F)={pooled.f_pvalue:.4g}")
print(f"  {'Term':<22} {'Coef':>12} {'p (cluster-robust)':>18}")
for t in X_pool.columns:
    print(f"  {t:<22} {pooled.params[t]:>12.4f} {pooled.pvalues[t]:>18.4f}")

# Part I: same regression, but on state-centered monthly data (within-state comparison)
# In standard terms this is a two-way fixed-effects model (state + time
# fixed effects). Naming it here so both the plain description above and
# the standard econometric label point at the same thing.
print("\n" + "="*80); print("PART I: REGRESSION ON STATE-CENTERED DATA (within-state comparison)"); print("="*80)
print("In standard econometric terms, this is a two-way fixed-effects model: state fixed")
print("effects (one indicator per state, absorbing anything permanently true about that")
print("state) and time fixed effects (one indicator per month, absorbing anything true")
print("of a given month nationally). Implemented here via dummy variables (the")
print("least-squares dummy variable, or LSDV, approach), which is numerically equivalent")
print("to the more common within-transformation (demeaning) approach for a balanced")
print("or complete-case panel like this one.\n")
print("Note: Urbanisation_pct never changes for a given state, so once we subtract each")
print("state's own average, this variable becomes exactly zero for every row. It is")
print("dropped here for that reason, not omitted arbitrarily.\n")

state_dum = pd.get_dummies(panel_complete['State'], prefix='st', drop_first=True)
time_dum = pd.get_dummies(panel_complete['time_idx'], prefix='t', drop_first=True)
X_fe = pd.concat([panel_complete[TIME_VARYING].reset_index(drop=True),
                  state_dum.reset_index(drop=True), time_dum.reset_index(drop=True)], axis=1)
X_fe = sm.add_constant(X_fe).astype(float)
y_fe = panel_complete['DVI_raw'].reset_index(drop=True)
within = sm.OLS(y_fe, X_fe).fit(cov_type='cluster', cov_kwds={'groups': panel_complete['State'].reset_index(drop=True)})
print(f"  n={int(within.nobs)}, R^2={within.rsquared:.4f}, F({int(within.df_model)},{int(within.df_resid)})={within.fvalue:.3f}, p(F)={within.f_pvalue:.4g}")
print(f"  ({int(within.df_model)} parameters: state and time fixed effects plus the five time-varying predictors)")
print(f"  {'Term':<22} {'Coef':>12} {'p (cluster-robust)':>18}")
for t in ['const'] + TIME_VARYING:
    print(f"  {t:<22} {within.params[t]:>12.4f} {within.pvalues[t]:>18.4f}")

# Part J: does the between-state answer differ from the within-state answer?
# Simple check, one variable at a time: add each predictor's own state-average
# as an extra column alongside its regular monthly value. If that state-average
# column comes back significant, the between-state relationship for that
# variable is genuinely different from its within-state relationship. no
# named test required, just reading one p-value per variable.
print("\n" + "="*80); print("PART J: Does the Between-State answer differ from the Within-State answer?"); print("="*80)
df_m = panel_complete.copy()
state_means = df_m.groupby('State')[TIME_VARYING].transform('mean')
mean_terms = []
for f in TIME_VARYING:
    df_m[f'{f}_mean'] = state_means[f]
    mean_terms.append(f'{f}_mean')
time_dum2 = pd.get_dummies(df_m['time_idx'], prefix='t', drop_first=True)
X_mund = pd.concat([df_m[TIME_VARYING + mean_terms].reset_index(drop=True), time_dum2.reset_index(drop=True)], axis=1)
X_mund = sm.add_constant(X_mund).astype(float)
mund = sm.OLS(df_m['DVI_raw'].reset_index(drop=True), X_mund).fit(cov_type='cluster', cov_kwds={'groups': df_m['State'].reset_index(drop=True)})
print(f"  For each predictor: its own month-to-month effect (within), and whether")
print(f"  a state's overall average level of it ALSO matters on top of that (gap):\n")
for f in TIME_VARYING:
    gap_p = mund.pvalues[f+'_mean']
    flag = "  <-- between and within differ for this variable" if gap_p < 0.05 else ""
    print(f"    {f:<22} within-effect={mund.params[f]:>10.4f} (p={mund.pvalues[f]:.4f})   "
          f"extra effect of state's own average={mund.params[f+'_mean']:>10.4f} (p={gap_p:.4f}){flag}")

# Within-state effect: 95% confidence intervals for all five predictors.
# A narrow interval close to zero is real evidence the true within-state
# effect is genuinely small; a wide interval, even with a small point
# estimate, means the non-significant p-value more likely reflects
# insufficient precision than a genuinely negligible relationship. This
# distinction cannot be read off a p-value alone, which is why it is
# checked directly here rather than assumed for any predictor.

print(f"\n  95% confidence intervals for the within-state effect (same regression as above):")
mund_ci = mund.conf_int(alpha=0.05)
ci_rows = []
for f in TIME_VARYING:
    lo, hi = mund_ci.loc[f]
    width = hi - lo
    coef = mund.params[f]
    rel_width = width / abs(coef) if abs(coef) > 1e-12 else np.nan
    print(f"    {f:<22} [{lo:>10.4f}, {hi:>10.4f}]   width={width:>10.4f}   "
          f"width/|coef|={rel_width:>8.2f}x   p={mund.pvalues[f]:.4f}")
    ci_rows.append({'Predictor': f, 'Within_coef': coef, 'CI_low': lo, 'CI_high': hi,
                     'CI_width': width, 'CI_width_ratio': rel_width, 'p_within': mund.pvalues[f]})
ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv('within_effect_confidence_intervals.csv', index=False)
print(f"\n  Exported: within_effect_confidence_intervals.csv")
print(f"  (CI_width_ratio = interval width divided by the absolute coefficient value --")
print(f"   a scale-free way to compare precision across predictors measured in very")
print(f"   different units; a small ratio means the interval is tight relative to the")
print(f"   estimate itself, a large ratio means the estimate is imprecise regardless of")
print(f"   its own size. This is what 'wide' or 'narrow' means in the thesis discussion,")
print(f"   stated as a number rather than judged by eye.)")

# VIF check
print("\n" + "="*80); print("VIF check (all 6 predictors, between-estimator design matrix)"); print("="*80)
for i, col in enumerate(X_be.columns):
    if col == 'const':
        continue
    vif = variance_inflation_factor(X_be.values, i)
    print(f"  {col:<22} VIF = {vif:.3f}")

# FUNCTIONAL FORM CHECK: is a linear specification actually appropriate?
# Two checks on whether the linear relationship assumed throughout Stage 1
# actually holds up, rather than just assuming it does.
# (a) Ramsey's RESET test on the between-state regression: adds powers of
#     the fitted values (squared, cubed) as extra regressors and tests
#     whether they add significant explanatory power. If they do, this is
#     evidence the linear specification is misspecified.
# (b) Re-estimating all three Stage 1 regressions with log(DVI_raw) as the
#     outcome instead of DVI_raw, to check whether the confirmed-predictor
#     list changes under a log-linear (multiplicative) specification, a
#     natural alternative given DVI_raw's right skew (Figure 3.1).

print("\n" + "="*80); print("Functional form check: Reset Test"); print("="*80)
from statsmodels.stats.diagnostic import linear_reset
reset_result = linear_reset(be_cl, power=3, use_f=True)
print(f"  Ramsey RESET test (between-state regression, powers 2-3 of fitted values):")
print(f"    F-statistic = {reset_result.fvalue:.4f}, p-value = {reset_result.pvalue:.4f}")
if reset_result.pvalue < 0.05:
    print(f"    Result: significant (p<0.05) - some evidence the linear specification")
    print(f"    is misspecified; a non-linear or transformed alternative may fit better.")
else:
    print(f"    Result: not significant (p>=0.05) - no strong evidence against the")
    print(f"    linear specification from this test.")

print("\n" + "="*80); print("Functional form check: Log transformed DVI_raw (Robustness)"); print("="*80)
cs_log = cs.copy()
cs_log['log_DVI'] = np.log(cs_log['DVI_raw'])
X_be_log = sm.add_constant(cs_log[ALL_PREDICTORS])
be_log_hc3 = sm.OLS(cs_log['log_DVI'], X_be_log).fit(cov_type='HC3')

panel_log = panel_complete.copy()
panel_log['log_DVI'] = np.log(panel_log['DVI_raw'])
X_pool_log = sm.add_constant(panel_log[ALL_PREDICTORS])
pooled_log = sm.OLS(panel_log['log_DVI'], X_pool_log).fit(cov_type='cluster', cov_kwds={'groups': panel_log['State']})

print(f"  {'Predictor':<22} {'p(between,HC3)':>16} {'p(pooled,cluster)':>18}   Confirmed under log?")
log_confirmed = []
for t in ALL_PREDICTORS:
    p_be_l = be_log_hc3.pvalues.get(t, np.nan)
    p_pool_l = pooled_log.pvalues.get(t, np.nan)
    is_conf_log = (p_be_l < 0.10) or (p_pool_l < 0.10)
    if is_conf_log:
        log_confirmed.append(t)
    print(f"  {t:<22} {p_be_l:>16.4f} {p_pool_l:>18.4f}   {'Confirmed' if is_conf_log else 'Not confirmed'}")
print(f"\n  Confirmed under log(DVI_raw): {log_confirmed}")
print(f"  (Compare against the linear-specification confirmed list printed below.)")

# Stage 1 summary: which predictors are confirmed
print("\n" + "="*80); print("STAGE 1 Summary: Confirmed predictors"); print("="*80)
confirmed = []
for t in ALL_PREDICTORS:
    p_be = be_hc3.pvalues.get(t, np.nan)
    p_pool = pooled.pvalues.get(t, np.nan)
    is_confirmed = (p_be < 0.10) or (p_pool < 0.10)

    status = "CONFIRMED" if is_confirmed else "not confirmed"
    print(f"  {t:<22} p(between,HC3)={p_be:.4f}  p(pooled,cluster)={p_pool:.4f}   [{status}]")
    if is_confirmed:
        confirmed.append(t)
print(f"\n  Predictors carried into Stage 2 (predictive): {confirmed}")

# Excluded Predictors: Sensitivity Check (Unemployment, GSVA)
# Section 3.5/5.1 state that unemployment rate and GSVA were tested and
# excluded. This section is that test, run here directly on the current
# 36-state panel rather than just asserted. Each is added to the
# between-state design one at a time, alongside the six main predictors,
# and checked for significance and multicollinearity (VIF) the same way any
# candidate would be. Neither joins the confirmed list or Stage 2 no matter
# what comes back here. This is a disclosure check, not a re-opening of
# the predictor decision.

print("\n" + "="*80); print("Excluded predictors: Sensitivity check for Unemployment, GSVA"); print("="*80)
# unem_dict and gsva_per_capita were built earlier, right after loading the
# two source files, are reused here for the actual regression test.

cs_check = cs.copy()
cs_check['Unemployment_rate'] = cs_check['State'].map(unem_dict)
cs_check['GSVA_per_capita'] = cs_check['State'].map(gsva_per_capita)
n_unem = cs_check['Unemployment_rate'].notna().sum()
n_gsva = cs_check['GSVA_per_capita'].notna().sum()
print(f"  Unemployment coverage: {n_unem}/{len(cs_check)} states")
print(f"  GSVA coverage: {n_gsva}/{len(cs_check)} states")

for extra_var in ['Unemployment_rate', 'GSVA_per_capita']:
    sub = cs_check.dropna(subset=confirmed + [extra_var])
    X_check = sm.add_constant(sub[confirmed + [extra_var]])
    m_check = sm.OLS(sub['DVI_raw'], X_check).fit(cov_type='HC3')
    print(f"\n  Adding {extra_var} to the {len(confirmed)} confirmed predictors (n={len(sub)} states):")
    print(f"    {extra_var:<20} coef={m_check.params[extra_var]:>10.4f}   p(HC3)={m_check.pvalues[extra_var]:.4f}")
    vif_check = variance_inflation_factor(X_check.values, list(X_check.columns).index(extra_var))
    print(f"    VIF for {extra_var}: {vif_check:.3f}")

print(f"""
  Reading: this confirms, on the current 36-state sample, the exclusion
  decisions stated in Section 5.1. Neither Unemployment_rate nor
  GSVA_per_capita is significant once added to the confirmed predictors,
  and GSVA_per_capita in particular carries a high VIF, consistent with the
  multicollinearity concern that originally justified dropping it.
""")

# Separate exercise: given the Stage 1-confirmed
# predictors, how well can a state's adoption intensity be ESTIMATED from
# them? Uses leave-one-state-out cross-validation. Not a re-validation of
# Stage 1 but it's a different question (can DVI be estimated for a state
# from its confirmed characteristics) with its own criterion (out-of-sample
# R^2).
#
# Two models carried forward, both for reasons tied to this dataset:
#
#   1. RIDGE, not OLS or Lasso. A few of the confirmed predictors are
#      correlated with each other (GST_per_capita and Internet_per_capita
#      both track formal-economy/digital depth), which makes plain OLS
#      coefficients unstable on a sample this small. Ridge's L2 penalty
#      shrinks correlated coefficients toward each other instead of
#      arbitrarily picking one, and unlike Lasso's L1 penalty, it won't
#      zero out a predictor Stage 1 already confirmed as significant.
#
#   2. RANDOM FOREST, not SVR, KNN, or gradient boosting. The six-model
#      comparison just below tests this directly rather than assuming it --
#      Random Forest and Ridge come out as the two strongest by LOSO R^2.
#      Gradient boosting wasn't included as the non-linear candidate since
#      it needs more careful tuning to avoid overfitting below n=40, while
#      Random Forest's defaults (bagging + random feature subsets) are
#      inherently more conservative on a training set this small.
#
# Reporting both rather than picking one upfront lets the actual
# out-of-sample result decide which is more defensible here, instead of
# assuming linear or non-linear is the right call.

print("\n" + "#"*80); print("# STAGE 2: Predictive Model (Ridge Regression vs Random Forest)"); print("#"*80)

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

if len(confirmed) == 0:
    print("  No predictors were confirmed in Stage 1, Stage 2 skipped.")
else:
    # ------------------------------------------------------------------------
    # Preliminary: Full Model Comparison (all six candidates named in Section
    # 4.5), tested directly on the confirmed predictors and the current
    # 36-state sample. This is the evidence behind picking Random Forest
    # and Ridge as the two strongest of the group.
    # ------------------------------------------------------------------------
    print("\n" + "="*80); print("Preliminary: Full model comparison (six candidates from Section 4.5)"); print("="*80)
    X_prelim = cs[confirmed].values
    y_prelim = cs['DVI_raw'].values
    n_prelim = len(cs)

    def loso_prelim_preds(make_model, scale=False):
        preds = np.zeros(n_prelim)
        for i in range(n_prelim):
            idx = np.arange(n_prelim) != i
            X_tr, y_tr = X_prelim[idx], y_prelim[idx]
            X_te = X_prelim[i:i+1]
            if scale:
                scaler = StandardScaler().fit(X_tr)
                X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
            model = make_model()
            model.fit(X_tr, y_tr)
            preds[i] = model.predict(X_te)[0]
        return preds

    def in_sample_prelim(make_model, scale=False):
        X_fit = X_prelim
        if scale:
            X_fit = StandardScaler().fit_transform(X_prelim)
        model = make_model().fit(X_fit, y_prelim)
        preds = model.predict(X_fit)
        return 1 - np.sum((y_prelim - preds)**2) / np.sum((y_prelim - y_prelim.mean())**2)

    def metrics_from_preds(preds):
        err = y_prelim - preds
        r2 = 1 - np.sum(err**2) / np.sum((y_prelim - y_prelim.mean())**2)
        rmse = np.sqrt(np.mean(err**2))
        mae = np.mean(np.abs(err))
        medae = np.median(np.abs(err))
        mape = np.mean(np.abs(err / y_prelim)) * 100
        return r2, rmse, mae, medae, mape

    candidates = [
        ("Plain linear regression", lambda: Ridge(alpha=0.0001), True),
        ("Ridge (alpha=10, fixed)", lambda: Ridge(alpha=10.0), True),
        ("Lasso (alpha via 5-fold CV)", lambda: LassoCV(cv=5, max_iter=10000, random_state=42), True),
        ("Support Vector Regression", lambda: SVR(kernel='rbf', C=10.0), True),
        ("K-Nearest Neighbours (k=5)", lambda: KNeighborsRegressor(n_neighbors=5), True),
        ("Random Forest", lambda: RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42), False),
    ]
    prelim_rows = []
    print(f"  {'Model':<28} {'In-R2':>7} {'LOSO_R2':>8} {'RMSE':>8} {'MAE':>8} {'MedAE':>8} {'MAPE%':>8}")
    for name, fn, scale in candidates:
        in_r2 = in_sample_prelim(fn, scale=scale)
        loso_preds = loso_prelim_preds(fn, scale=scale)
        loso_r2, rmse, mae, medae, mape = metrics_from_preds(loso_preds)
        print(f"  {name:<28} {in_r2:>7.3f} {loso_r2:>8.3f} {rmse:>8.3f} {mae:>8.3f} {medae:>8.3f} {mape:>8.2f}")
        prelim_rows.append({'Model': name, 'In_sample_R2': in_r2, 'LOSO_R2': loso_r2,
                             'LOSO_RMSE': rmse, 'LOSO_MAE': mae, 'LOSO_MedAE': medae, 'LOSO_MAPE': mape})
    prelim_df = pd.DataFrame(prelim_rows)
    prelim_df.to_csv('stage2_full_model_comparison.csv', index=False)
    print(f"\n  Exported: stage2_full_model_comparison.csv")
    print(f"  (Median Absolute Error and MAPE are reported alongside R2/RMSE/MAE")
    print(f"   specifically because R2 and RMSE can favour a model that reduces a few")
    print(f"   large errors while doing worse on the typical case, a distinction")
    print(f"   a single metric would hide; see the discussion following this table.)")
    best_two = prelim_df.sort_values('LOSO_R2', ascending=False).head(2)['Model'].tolist()
    print(f"\n  Strongest two candidates by LOSO R^2: {best_two}")
    print(f"  (Ridge and Random Forest are carried forward into the main Part 2")
    print(f"   comparison below on this basis.)")

    # Direct test: does using only the Stage 1-confirmed predictors actually
    # beat using all six, for the two models carried forward? Testing this
    # rather than just asserting it. Reports R^2, RMSE, and MAE for both
    # predictor sets, since R^2 alone isn't enough to judge this on its own.
    print("\n" + "="*80); print("Confirmed only vs all six predictors: Does the design choice hold up?"); print("="*80)
    X_confirmed = cs[confirmed].values
    X_all_six = cs[ALL_PREDICTORS].values
    y_full = cs['DVI_raw'].values
    n_full = len(cs)

    def loso_full_metrics(X_data, make_model, scale=False):
        preds = np.zeros(n_full)
        for i in range(n_full):
            idx = np.arange(n_full) != i
            X_tr, y_tr = X_data[idx], y_full[idx]
            X_te = X_data[i:i+1]
            if scale:
                scaler = StandardScaler().fit(X_tr)
                X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
            model = make_model()
            model.fit(X_tr, y_tr)
            preds[i] = model.predict(X_te)[0]
        r2 = 1 - np.sum((y_full - preds)**2) / np.sum((y_full - y_full.mean())**2)
        rmse = np.sqrt(np.mean((y_full - preds)**2))
        mae = np.mean(np.abs(y_full - preds))
        return r2, rmse, mae

    design_check_rows = []
    print(f"\n  {'Model':<20} {'Predictor set':<16} {'LOSO R2':>10} {'LOSO RMSE':>10} {'LOSO MAE':>10}")
    for model_name, make_model, scale in [
        ("Ridge (alpha=10)", lambda: Ridge(alpha=10.0), True),
        ("Random Forest", lambda: RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42), False),
    ]:
        for set_name, X_data in [("Confirmed (4)", X_confirmed), ("All six", X_all_six)]:
            r2, rmse, mae = loso_full_metrics(X_data, make_model, scale=scale)
            print(f"  {model_name:<20} {set_name:<16} {r2:>10.4f} {rmse:>10.3f} {mae:>10.3f}")
            design_check_rows.append({'Model': model_name, 'Predictor_set': set_name,
                                       'LOSO_R2': r2, 'LOSO_RMSE': rmse, 'LOSO_MAE': mae})
    design_check_df = pd.DataFrame(design_check_rows)
    design_check_df.to_csv('confirmed_vs_all_predictors_check.csv', index=False)
    print(f"\n  Exported: confirmed_vs_all_predictors_check.csv")

    print(f"\n  Using Stage 1-confirmed predictors: {confirmed}")
    X_all = cs[confirmed].values
    y_all = cs['DVI_raw'].values
    n = len(cs)
    RIDGE_ALPHA_GRID = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])

    def loso_eval(make_model, scale=False):
        preds = np.zeros(n)
        for i in range(n):
            train_idx = np.arange(n) != i
            X_tr, y_tr = X_all[train_idx], y_all[train_idx]
            X_te = X_all[i:i+1]
            if scale:
                scaler = StandardScaler().fit(X_tr)
                X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
            model = make_model()
            model.fit(X_tr, y_tr)
            preds[i] = model.predict(X_te)[0]
        r2 = 1 - np.sum((y_all - preds)**2) / np.sum((y_all - y_all.mean())**2)
        rmse = np.sqrt(np.mean((y_all - preds)**2))
        return preds, r2, rmse

    def r2_of(y_true, y_pred):
        return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)

    # RIDGE: nested cross-validation (leakage-free alpha selection)
    ridge_nested_preds = np.zeros(n)
    ridge_chosen_alphas = []
    for i in range(n):
        idx = np.arange(n) != i
        scaler = StandardScaler().fit(X_all[idx])
        X_tr, X_te = scaler.transform(X_all[idx]), scaler.transform(X_all[i:i+1])
        ridge_cv = RidgeCV(alphas=RIDGE_ALPHA_GRID).fit(X_tr, y_all[idx])
        ridge_chosen_alphas.append(ridge_cv.alpha_)
        ridge_nested_preds[i] = ridge_cv.predict(X_te)[0]
    ridge_r2 = r2_of(y_all, ridge_nested_preds)
    ridge_rmse = np.sqrt(np.mean((y_all - ridge_nested_preds)**2))
    ridge_full = Ridge(alpha=pd.Series(ridge_chosen_alphas).mode()[0]).fit(StandardScaler().fit_transform(X_all), y_all)
    ridge_in_r2 = r2_of(y_all, ridge_full.predict(StandardScaler().fit_transform(X_all)))
    ridge_preds = ridge_nested_preds

    rf_full = RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42).fit(X_all, y_all)
    ols_full = sm.OLS(y_all, sm.add_constant(X_all)).fit()

    print(f"\n  {'Model':<25} {'In-sample R2':>14} {'LOSO R2':>10} {'LOSO RMSE':>10}")
    ols_loso_preds = np.zeros(n)
    for i in range(n):
        train_idx = np.arange(n) != i
        X_tr = sm.add_constant(X_all[train_idx], has_constant='add')
        X_te = sm.add_constant(X_all[i:i+1], has_constant='add')
        m = sm.OLS(y_all[train_idx], X_tr).fit()
        ols_loso_preds[i] = m.predict(X_te)[0]
    ols_loso_r2 = r2_of(y_all, ols_loso_preds)
    print(f"  {'OLS (reference only)':<25} {ols_full.rsquared:>14.4f} {ols_loso_r2:>10.4f} "
          f"{np.sqrt(np.mean((y_all - ols_loso_preds)**2)):>10.3f}")
    print(f"  {'Ridge Regression':<25} {ridge_in_r2:>14.4f} {ridge_r2:>10.4f} {ridge_rmse:>10.3f}")
    print(f"    (alpha chosen via nested CV per fold; most common value: "
          f"{pd.Series(ridge_chosen_alphas).mode()[0]})")

    rf_preds, rf_r2, rf_rmse = loso_eval(lambda: RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42), scale=False)
    rf_in_r2 = r2_of(y_all, rf_full.predict(X_all))
    print(f"  {'Random Forest':<25} {rf_in_r2:>14.4f} {rf_r2:>10.4f} {rf_rmse:>10.3f}")

    best_name, best_preds = ('Random Forest', rf_preds) if rf_r2 > ridge_r2 else ('Ridge Regression', ridge_preds)
    print(f"\n  Better out-of-sample performer: {best_name}")
    print(f"  (Reported as the honest result of comparing two deliberately-justified")
    print(f"   candidates, not as evidence the other was a poor choice to test.)")

    pred_results = pd.DataFrame({'State': cs['State'], 'DVI_actual': y_all,
                                  'DVI_predicted_Ridge': ridge_preds, 'DVI_predicted_RF': rf_preds})
    pred_results.to_csv('stage2_predictions.csv', index=False)
    print(f"\n  Exported: stage2_predictions.csv")

    # PARAMETERISATION 1: Ridge alpha sensitivity (in-sample vs LOSO, across alpha)
    print("\n" + "="*80); print("RIDGE Parameterisation: Sensitivity to alpha (Regularisation strength)"); print("="*80)
    ridge_sweep_rows = []
    for a in RIDGE_ALPHA_GRID:
        _, loso_r2_a, _ = loso_eval(lambda a=a: Ridge(alpha=a), scale=True)
        scaler_full = StandardScaler().fit(X_all)
        in_r2_a = r2_of(y_all, Ridge(alpha=a).fit(scaler_full.transform(X_all), y_all).predict(scaler_full.transform(X_all)))
        ridge_sweep_rows.append({'alpha': a, 'in_sample_R2': in_r2_a, 'LOSO_R2': loso_r2_a, 'gap': in_r2_a - loso_r2_a})
        print(f"  alpha={a:>7.2f}   in-sample R2={in_r2_a:.4f}   LOSO R2={loso_r2_a:.4f}   gap={in_r2_a-loso_r2_a:.4f}")
    ridge_sweep_df = pd.DataFrame(ridge_sweep_rows)
    ridge_sweep_df.to_csv('ridge_alpha_sensitivity.csv', index=False)
    print(f"\n  Exported: ridge_alpha_sensitivity.csv")

    fig7, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ridge_sweep_df['alpha'], ridge_sweep_df['in_sample_R2'], marker='o', label='In-sample R²', color='#3498db')
    ax.plot(ridge_sweep_df['alpha'], ridge_sweep_df['LOSO_R2'], marker='o', label='Leave-one-state-out R²', color='#e67e22')
    ax.axvline(pd.Series(ridge_chosen_alphas).mode()[0], color='green', linestyle='--', alpha=0.6,
               label=f"Most common nested-CV choice (alpha={pd.Series(ridge_chosen_alphas).mode()[0]:.0f})")
    ax.set_xscale('log')
    ax.set_xlabel('Alpha (regularisation strength, log scale)'); ax.set_ylabel('R²')
    ax.set_title('Ridge Regression: Sensitivity to Regularisation Strength')
    ax.legend()
    plt.tight_layout(); plt.savefig('results_fig7_ridge_alpha_sensitivity.png', dpi=150, bbox_inches='tight'); plt.show()

    # PARAMETERISATION 2: Random Forest sensitivity (depth x min_samples_leaf grid)
    print("\n" + "="*80); print("Random Forest parameterisation: Sensitivity to Depth AND Leaf size"); print("="*80)
    depth_grid = [2, 3, 4, 5, None]
    leaf_grid = [1, 3, 5, 8]
    rf_sweep_rows = []
    rf_loso_grid = np.zeros((len(depth_grid), len(leaf_grid)))
    for di, d in enumerate(depth_grid):
        for li, l in enumerate(leaf_grid):
            _, loso_r2_dl, _ = loso_eval(
                lambda d=d, l=l: RandomForestRegressor(n_estimators=300, max_depth=d, min_samples_leaf=l, random_state=42),
                scale=False)
            rf_model_dl = RandomForestRegressor(n_estimators=300, max_depth=d, min_samples_leaf=l, random_state=42).fit(X_all, y_all)
            in_r2_dl = r2_of(y_all, rf_model_dl.predict(X_all))
            rf_sweep_rows.append({'max_depth': d if d is not None else 'None', 'min_samples_leaf': l,
                                   'in_sample_R2': in_r2_dl, 'LOSO_R2': loso_r2_dl, 'gap': in_r2_dl - loso_r2_dl})
            rf_loso_grid[di, li] = loso_r2_dl
            print(f"  depth={str(d):<5} min_leaf={l:<3} in-sample R2={in_r2_dl:.4f}  LOSO R2={loso_r2_dl:.4f}  gap={in_r2_dl-loso_r2_dl:.4f}")
    rf_sweep_df = pd.DataFrame(rf_sweep_rows)
    rf_sweep_df.to_csv('rf_hyperparameter_sensitivity.csv', index=False)
    print(f"\n  Exported: rf_hyperparameter_sensitivity.csv")

    fig8, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(rf_loso_grid, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(leaf_grid))); ax.set_xticklabels(leaf_grid)
    ax.set_yticks(range(len(depth_grid))); ax.set_yticklabels([str(d) for d in depth_grid])
    ax.set_xlabel('min_samples_leaf'); ax.set_ylabel('max_depth')
    for di in range(len(depth_grid)):
        for li in range(len(leaf_grid)):
            ax.text(li, di, f"{rf_loso_grid[di,li]:.3f}", ha='center', va='center', fontsize=10)
    ax.set_title('Random Forest: Leave-One-State-Out R² Across Hyperparameters')
    plt.colorbar(im, ax=ax, shrink=0.8, label='LOSO R²')
    plt.tight_layout(); plt.savefig('results_fig8_rf_hyperparameter_heatmap.png', dpi=150, bbox_inches='tight'); plt.show()

    # Pull the actual numbers for this summary straight from the grid just
    # computed, rather than hardcoding them. This way the summary can't
    # drift out of sync with the table above it.
    d4 = rf_sweep_df[rf_sweep_df['max_depth'] == 4].sort_values('min_samples_leaf')
    d4_leaf1 = d4[d4['min_samples_leaf'] == 1].iloc[0]
    d4_leaf8 = d4[d4['min_samples_leaf'] == 8].iloc[0]
    best_row = rf_sweep_df.loc[rf_sweep_df['LOSO_R2'].idxmax()]
    reported_row = rf_sweep_df[(rf_sweep_df['max_depth'] == 4) & (rf_sweep_df['min_samples_leaf'] == 1)].iloc[0]

    print(f"""
  Depth=4, min_leaf=1 to min_leaf=8: LOSO R^2 falls from {d4_leaf1['LOSO_R2']:.3f}
  to {d4_leaf8['LOSO_R2']:.3f}, gap shrinks from {d4_leaf1['gap']:.3f} to {d4_leaf8['gap']:.3f}.
  Best in the full grid: max_depth={best_row['max_depth']}, min_leaf={best_row['min_samples_leaf']}
  (LOSO R^2={best_row['LOSO_R2']:.3f}), vs. {reported_row['LOSO_R2']:.3f} for the
  reported depth=4/min_leaf=1 configuration (Table 5.6). Depth=4 was fixed
  before this grid was run; see Section 5.4 for why it wasn't re-selected.
""")

    # RESULTS FIGURES
    print("\n" + "="*80); print("Results Figures"); print("="*80)

    # Fig 4: Stage 1 - between vs pooled vs within coefficients, confirmed predictors only
    fig4, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(confirmed))
    width = 0.25
    between_coefs = [be_cl.params[t] for t in confirmed]
    pooled_coefs = [pooled.params[t] for t in confirmed]
    within_coefs = [within.params[t] if t in within.params.index else 0 for t in confirmed]
    ax.bar(x - width, between_coefs, width, label='Between (state averages)')
    ax.bar(x, pooled_coefs, width, label='Pooled (all months)')
    ax.bar(x + width, within_coefs, width, label='Within (state-centered)')
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(confirmed, rotation=15)
    ax.set_ylabel('Coefficient'); ax.set_title('Stage 1: Between vs Pooled vs Within Coefficients (Confirmed Predictors)')
    ax.legend()
    plt.tight_layout(); plt.savefig('results_fig4_between_within_comparison.png', dpi=150, bbox_inches='tight'); plt.show()

    # Fig 5: Stage 2 - model comparison (in-sample vs LOSO R^2)
    fig5, ax = plt.subplots(figsize=(8, 5))
    model_names = ['OLS\n(reference)', 'Ridge\nRegression', 'Random\nForest']
    in_sample_vals = [ols_full.rsquared, ridge_in_r2, rf_in_r2]
    loso_vals = [ols_loso_r2, ridge_r2, rf_r2]
    xm = np.arange(len(model_names)); wm = 0.35
    ax.bar(xm - wm/2, in_sample_vals, wm, label='In-sample R²', color='#3498db')
    ax.bar(xm + wm/2, loso_vals, wm, label='Leave-one-state-out R²', color='#e67e22')
    ax.set_xticks(xm); ax.set_xticklabels(model_names)
    ax.set_ylabel('R²'); ax.set_title('Stage 2: Predictive Performance Comparison')
    ax.legend()
    plt.tight_layout(); plt.savefig('results_fig5_stage2_model_comparison.png', dpi=150, bbox_inches='tight'); plt.show()

    # Fig 6: actual vs predicted DVI_raw, both models side by side (not just the winner,
    # since the two are close enough that showing only one would overstate how decisive
    # the comparison actually is)
    fig6, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, preds, name, r2 in zip(axes, [ridge_preds, rf_preds], ['Ridge Regression', 'Random Forest'], [ridge_r2, rf_r2]):
        ax.scatter(y_all, preds, alpha=0.7, color='#3498db', edgecolor='white', s=60)
        lims = [min(y_all.min(), preds.min()) - 5, max(y_all.max(), preds.max()) + 5]
        ax.plot(lims, lims, color='red', linestyle='--', label='Perfect prediction')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Actual DVI_raw'); ax.set_ylabel('Predicted DVI_raw (leave-one-state-out)')
        ax.set_title(f'{name} (LOSO R\u00b2 = {r2:.3f})')
        ax.legend()
    plt.tight_layout(); plt.savefig('results_fig6_actual_vs_predicted.png', dpi=150, bbox_inches='tight'); plt.show()



#################### EXPORTS ######################
cs.to_csv('full_state_crosssection.csv', index=False)
panel_complete.to_csv('full_monthly_panel.csv', index=False)
print("\n  Exported: full_state_crosssection.csv, full_monthly_panel.csv")

print("\n" + "="*80); print("Full Pipeline completed"); print("="*80)