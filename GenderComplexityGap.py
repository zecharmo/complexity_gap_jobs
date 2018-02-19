
# coding: utf-8

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# Read in multiple choice and free response datasets
mult = pd.read_csv("C:\Data\kaggle_ds\multipleChoiceResponses.csv", encoding="ISO-8859-1")
free = pd.read_csv("C:\Data\kaggle_ds\\freeformResponses.csv", encoding="ISO-8859-1")


# Join data to combine free response answers with demographics from main dataset
data = mult.join(free)

# Remove respondents who do not identify as male or female for cleaner profiling
data = data[(data['GenderSelect'] == 'Male') | (data['GenderSelect'] == 'Female')]

# Remove respondents who are not of working age
data=data[(data['Age'] > 17) | (data['Age'] <= 70)]


# Looking at education and differences between genders
data['FormalEducation'].value_counts()
data['MajorSelect'].value_counts()
data['Tenure'].value_counts()


# Differences in formal education
pd.crosstab(data['GenderSelect'],data['FormalEducation'], margins=True)

# Women
print(731/2439) #30% - Bachelor's
print(380/2439) #16% - Doctoral
print(1122/2439) #46% - Master's
print(67/2439) #3% - Professional
print(129/2439) #5% - less than Bachelor's

# Men
print(3966/12141) #33% 
print(1882/12141) #16%
print(4968/12141) #41%
print(372/12141) #3%
print(881/12141) #7% 
# Men and women have very similar formal education levels, with women having slightly more advanced education

# Differences in area of study
pd.crosstab(data['GenderSelect'],data['MajorSelect'],margins=True)

# Women
print(622/2144) #29% - CS
print(471/2144) #22% - Math

# Men
print(3651/10761) #34%
print(1694/10761) #16%
# Women lean slightly towards math while men lean slightly towards computer science
# Overall about 50% of both genders have directly applicable education experience


# Differences in job tenure
pd.crosstab(data['GenderSelect'],data['Tenure'],margins=True)

# Women
print(639/2153) #30% - 1-2 years
print(520/2153) #24% - 3-5 years
print(236/2153) #11% - 6-10 years
print(120/2153) #6% - none
print(425/2153) #20% - <1 years
print(213/2153) #10% - >10 years

# Men
print(2714/11006) #25%
print(2748/11006) #25%
print(1431/11006) #13%
print(483/11006) #4%
print(1910/11006) #17%
print(1720/11006) #16%
# A larger share of men have 10+ years in the industry, but overall experience levels are pretty similar


# Create a new dataframe for job title analysis
# Job titles are stored as a list in a single variable, need to split into separate variables to analyze
df = data.join(data['PastJobTitlesSelect'].str.get_dummies(','))

# Remove responses for current students
df = df[df["I haven't started working yet"] != 1]

df.iloc[0:5,290:]


# List of past job titles respondents could pick from
past_jobs_list = ["Business Analyst","Computer Scientist","DBA/Database Engineer","Data Analyst","Data Miner","Data Scientist","Engineer","Machine Learning Engineer","Operations Research Practitioner","Other","Predictive Modeler","Programmer","Researcher","Software Developer/Software Engineer"]


# Percentage of female and male respondents who held each job in the past
f_past_jobs = [0.1553398058252427,0.06247361756015196,0.043056141831996624,0.24271844660194175,0.05065428450823132,0.13127902068383285,0.11355002110595187,0.05149852258336851,0.019839594765723934,0.19164204305614183,0.05149852258336851,0.16124947235120304,0.2760658505698607,0.17602363866610385]
m_past_jobs = [0.15437012720558063,0.10537546163315552,0.0727944193680755,0.20853508411981944,0.06425933524825606,0.16372589249076733,0.1713582273286828,0.07804677882642594,0.0254411161263849,0.15190808370947886,0.062453836684448094,0.24743537135822732,0.22798522773902338,0.2880590890439064]


# Compute t-test to determine if job histories are significantly different by gender
from scipy.stats import ttest_ind
ttest_ind(f_past_jobs, m_past_jobs)
# Ttest_indResult(statistic=-0.69725755715044047, pvalue=0.49182986237589466)
# Percentage of male/female respondents who previously held each job title is not signficantly different
# While men and women have similarly relevant levels of education, similar tenures, and similar job histories, we still observe signficantly different
# levels of usage of analytical methods and machine learning tools between the genders


# Create visualization of job histories by gender
past_jobs_items = [('JobTitle', past_jobs_list), ('Female',f_past_jobs), ('Male', m_past_jobs),]
past_jobs_df = pd.DataFrame.from_items(past_jobs_items)
past_jobs_df['diff'] = abs(past_jobs_df['Male']-past_jobs_df['Female'])

plt.subplot(1,2,1)
plt.bar(past_jobs_df['JobTitle'],past_jobs_df['Female'], color='darkred')

plt.subplot(1,2,2)
plt.bar(past_jobs_df['JobTitle'],past_jobs_df['Male'], color='darkblue')

plt.show()

plt.bar(past_jobs_df['JobTitle'],past_jobs_df['diff'], color='darkblue')
plt.show()


# Look at differences in how appropriately respondents believe their title fits their job
pd.crosstab(data['GenderSelect'],data['TitleFit'])

# Female
print(1154/2714) #43% - Fine
print(341/2714) #13% - Perfectly
print(234/2714) #9% - Poorly

# Male
print(6138/13427) #46%
print(1907/13427) #14%
print(1407/13427) #10%
# Not much difference between genders


# Looking at the differences in how workers spend their time, percentage performing each task
data['TimeGatheringData'].fillna(0, inplace=True)
data['TimeModelBuilding'].fillna(0, inplace=True)
data['TimeProduction'].fillna(0, inplace=True)
data['TimeVisualizing'].fillna(0, inplace=True)
data['TimeFindingInsights'].fillna(0, inplace=True)
data['TimeOtherSelect'].fillna(0, inplace=True)


# Create dataframe of task percentages
df2 = data[['GenderSelect','CurrentJobTitleSelect','TimeGatheringData','TimeModelBuilding','TimeProduction','TimeVisualizing','TimeFindingInsights','TimeOtherSelect']]
df2.describe()


# Looking for differences between genders
pd.pivot_table(df,index=['GenderSelect'],values=['TimeGatheringData','TimeModelBuilding','TimeProduction','TimeVisualizing','TimeFindingInsights'],aggfunc=[np.mean,np.std])


# Looking for differences between geneders by job title - presumably different jobs require different tasks/ratios of tasks
t1= pd.pivot_table(df,index=['GenderSelect','CurrentJobTitleSelect'],values=['TimeGatheringData','TimeModelBuilding','TimeProduction','TimeVisualizing','TimeFindingInsights'],aggfunc=[np.mean])
# There are observable differences between job titles, but means and standard deviations are fairly similar for both genders


# Looking at compensation. Split dataset to look only at US respondents
# This allows us to only look at compensation in US dollars (no exchange rate conversions)
# This also allows us to avoid any differences in work type or work status between countries
usa = data[data['Country']=='United States']


# Compensation is stored as an object where numerical values have commas to separate thousands
usa['CompensationAmount'].str.replace(",","").fillna(0, inplace=True)
pd.to_numeric(usa['CompensationAmount'],errors='coerce')


# Look at differences between genders for each job title
pd.pivot_table(usa,index=['GenderSelect','CurrentJobTitleSelect'],values=['CompensationAmount'],aggfunc=[np.mean,np.std])