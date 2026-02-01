# Cyber-insurance-risk-assesment-tool
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("Large_Sample_Claim_Request_and_Calculation_Dataset.csv")

# Replace NaN in `Final_Payout` with 0
df['Final_Payout'] = df['Claim Fee (Final Payout)'].fillna(0)

# Standardize company names
df['Company_Name'] = df['Company Name'].str.strip().str.lower()

# Ensure Policy Numbers are unique
duplicates = df[df.duplicated('Policy Number')]

# Check all column names
print(df.columns.tolist())

#2.1 Total rows and uniqueness
total_rows = len(df)
unique_companies = df['Company Name'].nunique()
unique_policies = df['Policy Number'].nunique()

# 2.2 Missing values
missing_data = df[['Incurred Loss Amount', 'Claim Fee (Final Payout)', 'Description of Incident']].isnull().sum()

# 2.3 Distribution of Incurred L oss
summary_stats = df['Incurred Loss Amount'].agg(['min', 'max', 'mean', 'median'])

# 2.4 Time-Series Trends
df['Date of Incident'] = pd.to_datetime(df['Date of Incident'])
df['Year'] = df['Date of Incident'].dt.year
yearly_trends = df.groupby('Year').agg(
    total_claims=('Policy Number', 'count'),
    total_payout=('Claim Fee (Final Payout)', 'sum')
).reset_index()

# 3.1 Claim Dates
claim_dates = df['Date of Incident'].agg(['min', 'max'])

# 3.2 Top 5 Companies with Most Claims
top5_companies = df['Company Name'].value_counts().head(5)

# 3.3 Incurred Loss Summary by Company
loss_by_company = df.groupby('Company Name')['Incurred Loss Amount'].agg(['mean', 'max', 'min']).reset_index()

# 3.4 Final Payouts vs. Incurred Loss
df['Payout_Status'] = np.where(df['Claim Fee (Final Payout)'] > df['Incurred Loss Amount'], 'Need Attention', 'Sorted')

# 3.5 Top 5% Outlier Companies (by Median Payout)
median_by_company = df.groupby('Company Name')['Claim Fee (Final Payout)'].median()
threshold_95 = np.percentile(df['Claim Fee (Final Payout)'], 95)
outliers = median_by_company[median_by_company > threshold_95].sort_values(ascending=False)

# Classify threats
def classify_threat(desc):
    if isinstance(desc, str):
        desc = desc.lower()
        if 'ransomware' in desc:
            return 'Ransomware'
        elif 'phishing' in desc:
            return 'Phishing'
        elif 'data breach' in desc:
            return 'Data Breach'
    return 'Other'

df['Threat Type'] = df['Description of Incident'].apply(classify_threat)

# 4.1 Common Threats
top_threats = df['Threat Type'].value_counts().head(3)

# 4.2 Threats Over Time
threats_over_time = df.groupby(['Year', 'Threat Type']).size().reset_index(name='Threat Count')

# 5.1 Losses by Threat Type and Year
loss_by_threat_year = df.groupby(['Threat Type', 'Year'])['Incurred Loss Amount'].sum().reset_index()

# 5.2 Top 10 Inflated Claims
df['Inflated Loss'] = df['Incurred Loss Amount'] - df['Verified Incurred Loss Amount']
inflated_avg = df.groupby('Company Name')['Inflated Loss'].mean().reset_index()
inflated_top10 = inflated_avg[inflated_avg['Inflated Loss'] > np.percentile(inflated_avg['Inflated Loss'], 90)]

# 5.3 Underreported Claims
underreported = inflated_avg[inflated_avg['Inflated Loss'] < np.percentile(inflated_avg['Inflated Loss'], 10)]

# 5.4 Threat Discrepancies
discrepancy_by_threat = df.groupby('Threat Type')['Inflated Loss'].mean().reset_index()

# 5.5 Repeated Claims by Threat and Year
repeated_claims = df.groupby(['Company Name', 'Threat Type', 'Year']).size().reset_index(name='Claim_Count').sort_values('Claim_Count', ascending=False)

# 5.6 Claim Frequency Change Over Time
claim_change = repeated_claims.copy()
claim_change['Previous_Year_Claim_Count'] = claim_change.groupby(['Company Name', 'Threat Type'])['Claim_Count'].shift(1)
claim_change['Claim_Change'] = claim_change['Claim_Count'] - claim_change['Previous_Year_Claim_Count']

# 5.7 Frequency vs Financial Impact
impact = df.groupby('Year').agg(
    total_claims=('Policy Number', 'count'),
    total_payout=('Claim Fee (Final Payout)', 'sum'),
    avg_payout=('Claim Fee (Final Payout)', 'mean')
).reset_index()
impact[['Prev_Claims', 'Prev_Payout', 'Prev_Avg']] = impact[['total_claims', 'total_payout', 'avg_payout']].shift(1)
impact['Claim_Change'] = impact['total_claims'] - impact['Prev_Claims']
impact['Payout_Change'] = impact['total_payout'] - impact['Prev_Payout']
impact['Avg_Payout_Change'] = impact['avg_payout'] - impact['Prev_Avg']

# Correlation between Deductible and Payout
corr_deductible_payout = df[['Deductible', 'Claim Fee (Final Payout)']].corr().iloc[0,1]

# Correlation between Incurred Loss and Final Payout
corr_loss_payout = df[['Incurred Loss Amount', 'Claim Fee (Final Payout)']].corr().iloc[0,1]

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.histplot(df['Claim Fee (Final Payout)'], kde=True, bins=30)
plt.title('Distribution of Final Payout Amounts')
plt.xlabel('Final Payout')
plt.ylabel('Number of Claims')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Verified Incurred Loss Amount', y='Claim Fee (Final Payout)', data=df)
plt.title('Final Payout vs. Verified Incurred Loss Amount')
plt.xlabel('Verified Incurred Loss')
plt.ylabel('Final Payout')
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
df['Company Name'].value_counts().plot(kind='bar')
plt.title('Number of Claims per Company')
plt.xlabel('Company')
plt.ylabel('Claim Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.xlabel('Company')
plt.ylabel('Claim Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df['Date of Incident'] = pd.to_datetime(df['Date of Incident'])

df.set_index('Date of Incident', inplace=True)
claims_per_month = df.resample('M').size()

plt.figure(figsize=(10, 6))
claims_per_month.plot()
plt.title('Claims Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Claims')
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("Large_Sample_Claim_Request_and_Calculation_Dataset.csv")

# Replace NaN in `Final_Payout` with 0
df['Final_Payout'] = df['Claim Fee (Final Payout)'].fillna(0)

# Standardize company names
df['Company_Name'] = df['Company Name'].str.strip().str.lower()

# Ensure Policy Numbers are unique
duplicates = df[df.duplicated('Policy Number')]


#explanation
A cyber insurance risk assessment tool is used to figure out how likely a company is to face cyberattacks and how serious the damage could be.​
● It checks how exposed a company is to online threats.​
● This helps insurance companies decide what to cover, how much to charge, and what to leave​
● The tool looks at technology, company policies, and how people handle security.​
● It gathers details like how the network is set up, how sensitive the data is, and who has access.​
● It also checks past cyber incidents, current defenses, and whether the company follows rules and regulations.​
● A score is given to show how risky the situation is, including possible financial loss.
● It often follows well-known standards like NIST, ISO 27001, or FAIR.​
● The tool may run scans to find weak spots or mistakes in the system.
● It checks if outside vendors or partners add extra risk.​
● It also looks at how well employees are trained and if the company is ready to respond to attacks.​
● The results are shown in reports and dashboards for decision-makers.​
● Some tools use AI to predict future risks better.​
● Companies can run the assessment regularly to keep up with changes.
● This helps connect cybersecurity with financial planning.​
● In the end, it helps insurance companies make smarter choices and helps businesses stay safer.
