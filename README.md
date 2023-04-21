# covid_19-analysis
This is a project based on covid-19 analysis using python and machine learning algorithms.

#loading dataset
import pandas as pd
import numpy as np
#visualisation
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
#EDA
from collections import Counter
import pandas_profiling as pp
# data preprocessing
from sklearn.preprocessing import StandardScaler
# data splitting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # for scaling
from sklearn.pipeline import make_pipeline # for create classifier with preprocessing
from sklearn.tree import DecisionTreeClassifier, plot_tree # for building model and draw (plo
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split # for cross validation,
# data modeling
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#ensembling
from mlxtend.classifier import StackingCVClassifier
url = 'https://raw.githubusercontent.com/SohaMohajeri/Covid-19-Analysis-Visualization-and-For
df = pd.read_csv(url)
# Dataset is now stored in a Pandas Dataframe
df.head(2)
id case_in_country reporting
date summary location country gender age symptom
0 765 15.0 02-10-20
new
confirmed
COVID-19
patient in
Vietnam: 3
m...
Vinh Phuc Vietnam NaN 0.25
1 477 27.0 02-05-20
new
confirmed
COVID-19
patient in
Singapore:
m...
Singapore Singapore male 0.50
df.shape
(1085, 20)
df.dtypes
id int64
case_in_country float64
reporting date object
summary object
location object
country object
gender object
age float64
symptom_onset object
If_onset_approximated float64
hosp_visit_date object
exposure_start object
exposure_end object
visiting Wuhan int64
from Wuhan int64
death int64
recovered int64
symptom object
source object
link object
dtype: object
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1085 entries, 0 to 1084
Data columns (total 20 columns):
 # Column Non-Null Count Dtype
--- ------ -------------- -----
 0 id 1085 non-null int64
 1 case_in_country 888 non-null float64
 2 reporting date 1084 non-null object
 3 summary 1080 non-null object
 4 location 1085 non-null object
 5 country 1085 non-null object
 6 gender 902 non-null object
 7 age 843 non-null float64
 8 symptom_onset 563 non-null object
 9 If_onset_approximated 560 non-null float64
 10 hosp_visit_date 507 non-null object
 11 exposure_start 128 non-null object
 12 exposure_end 341 non-null object
 13 visiting Wuhan 1085 non-null int64
 14 from Wuhan 1085 non-null int64
 15 death 1085 non-null int64
 16 recovered 1085 non-null int64
 17 symptom 270 non-null object
 18 source 1085 non-null object
 19 link 1085 non-null object
dtypes: float64(3), int64(5), object(12)
memory usage: 169.7+ KB
df.drop(['id','case_in_country','summary','symptom_onset', 'If_onset_approximated', 'hosp_vis
'exposure_end', 'symptom', 'source', 'link'],axis=1,inplace=True)
100*df.isnull().sum()/df.shape[0]
reporting date 0.092166
location 0.000000
country 0.000000
gender 16.866359
age 22.304147
visiting Wuhan 0.000000
from Wuhan 0.000000
death 0.000000
recovered 0.000000
dtype: float64
df['age']= df['age']. fillna(df['age'].mean())
df_dum=pd.get_dummies(df['gender'].dropna(), drop_first=True)
df_dum['male'].median()
1.0
df['gender']= df['gender']. fillna('male')
df.dropna(inplace=True)
df.isnull().sum()
reporting date 0
location 0
country 0
gender 0
age 0
visiting Wuhan 0
from Wuhan 0
death 0
recovered 0
dtype: int64
# plot histograms for each variable
df.hist(figsize = (12, 12))
plt.show()
df.columns=df.columns.str.lower().str.replace(' ','_')
df['reporting_date']=pd.to_datetime(df['reporting_date'])
df['year']=df['reporting_date'].apply(lambda x:x.year)
df['month']=df['reporting_date'].apply(lambda x:x.month)
df['month'].unique()
df.drop(['reporting_date', 'year'], axis=1, inplace=True)
df.head(2)
location country gender age visiting_wuhan from_wuhan death recovered month
0 Vinh Phuc Vietnam male 0.25 0 0 0 1 2
1 Singapore Singapore male 0.50 0 0 0 
