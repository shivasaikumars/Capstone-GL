#!/usr/bin/env python
# coding: utf-8

# In[978]:


import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_auc_score, roc_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[979]:


from matplotlib import rcParams

rcParams['figure.figsize'] = (10,5) # figure size in inches
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


# ## Index:
# * [Checking Data Types](#first-bullet)
# * [Univariate Analysis](#second-bullet)
# * [Bivariate Analysis](#third-bullet)
# * [Multivariate Analysis](#fourth-bullet)
# * [Missing Values](#fifth-bullet)
# * [Outliers](#sixth-bullet)
# * [Model Building](#seventh-bullet)

# ## Reading The DataSet

# In[980]:


df = pd.read_csv('patient-survival.csv')
df.shape


# In[981]:


df.head()


# ## Data Types

# In[982]:


df.info()


# ## Data Summary

# In[983]:


df.describe(include='all')


# ## Removing unnecessary columns

# In[984]:


df['patient_id'].duplicated().sum()


# In[985]:


df['encounter_id'].duplicated().sum()


# In[986]:


df['icu_id'].duplicated().sum()


# In[987]:


df['hospital_id'].duplicated().sum()


# In[988]:


sns.countplot(data=df, x='hospital_id')


# In[989]:


# Since these columns are arbitrary and are only used for identification, we will drop them


# In[990]:


df = df.drop(columns=['Unnamed: 83', 'patient_id', 'encounter_id', 'hospital_id', 'icu_id'])


# In[991]:


# New shape
df.shape


# In[992]:


# duplicate rows
df.duplicated().sum()


# ## Checking Data Types <a class="anchor" id="first-bullet"></a>

# In[993]:


df_cat = df.select_dtypes('object')
df_cat.shape


# In[994]:


df_num = df.select_dtypes('number')
df_num.shape


# In[995]:


# No of categories in categorical variables


# In[996]:


for col in df_cat.columns:
    print(f'{col:>20}: ',df_cat[col].value_counts().count())


# In[997]:


# some numerical columns have very few unique values, they are possibly categorical
categorical_columns = []
for col in df_num.columns:
    cats = df_num[col].value_counts()
    if cats.count() <= 10: 
        categorical_columns.append(col)
        print(f'{col:>30}: ',cats.index.to_list())


# In[998]:


# All these variables need to be converted to categorical


# In[999]:


# separate numerical and categorical again


# In[1000]:


for col in categorical_columns:
    df[col] = df[col].astype('object')


# In[1001]:


df_num = df.select_dtypes('number')
numerical_columns = df_num.columns
df_num.shape


# In[1002]:


df_cat = df.select_dtypes('object')
categorical_columns = df_cat.columns
df_cat.shape


# In[1003]:


# Display categories of multicategory variables
for col in categorical_columns:
    if df_cat[col].value_counts().count()>2:
        print('\n')
        print(df_cat[col].value_counts(normalize=True)*100)


# In[1004]:


# Undefined Diagnoses repeats two times in apache_2_bodysystem, we need to replace it
df['apache_2_bodysystem'] = df['apache_2_bodysystem'].replace('Undefined Diagnoses', 'Undefined diagnoses')


# In[1005]:


# Total no of categories across all variables
df[categorical_columns].nunique().sort_values(ascending=False).sum()


# ## Univariate Analysis <a class="anchor" id="second-bullet"></a>

# ### Target Variable - hospital_death

# In[206]:


df['hospital_death'].value_counts(normalize=True)


# In[207]:


df['hospital_death'].value_counts().plot(kind='pie', autopct='%0.2f%%')


# In[208]:


df['cirrhosis'].value_counts().plot(kind='pie', autopct='%0.2f%%')


# In[47]:


df[categorical_columns].describe()


# #### We observe that our target variable - hospital_death is moderately unbalanced

# In[56]:


# some categorical variables have many categories, we will plot them separately
many_cat_cols = []

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(20,20))
i=0
for col in categorical_columns:
    
    if df[col].value_counts().count()>2:
        many_cat_cols.append(col)
        continue
    
    # row_index and col_index to use for subplot
    r = i//4
    c = i%4
    i+=1
    sns.countplot(data=df, x=col, ax=ax[r][c])


# In[35]:


# plot multi category columns separately
print(many_cat_cols)


# In[36]:


plt.figure(figsize=(20,6))
sns.countplot(data=df, x='ethnicity')


# In[37]:


plt.figure(figsize=(15,6))
sns.countplot(data=df, x='icu_admit_source')


# In[38]:


sns.countplot(data=df, x='icu_stay_type')


# In[39]:


sns.countplot(data=df, x='icu_type')


# In[40]:


sns.countplot(data=df, x='gcs_eyes_apache')


# In[41]:


sns.countplot(data=df, x='gcs_motor_apache')


# In[42]:


sns.countplot(data=df, x='gcs_verbal_apache')


# In[43]:


plt.figure(figsize=(20,6))
sns.countplot(data=df, x='apache_3j_bodysystem')


# In[44]:


plt.figure(figsize=(20,6))
sns.countplot(data=df, x='apache_2_bodysystem')


# In[57]:


df[numerical_columns].describe()


# In[45]:


fig, ax = plt.subplots(nrows=10, ncols=6, figsize=(30,50))

i=0
for col in df_num.columns:   
    r = i//6
    c = i%6
    i+=1
    sns.kdeplot(data=df, x=col, ax=ax[r][c])


# In[137]:


sns.kdeplot(df[(df['pre_icu_los_days']>=0)&(df['pre_icu_los_days']<26)]['pre_icu_los_days'])


# ## Bivariate Analysis <a class="anchor" id="third-bullet"></a>

# In[29]:


def show_bivariate(df, col):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25,6))
    ax1 = sns.kdeplot(df.loc[df['hospital_death']==1, col], color='Red', shade=True, ax= ax[0])
    ax1 = sns.kdeplot(df.loc[df['hospital_death']==0, col], color='Green', shade=True, ax= ax[0])
    ax1.legend(['Death', 'Not Death'], loc='upper right')
    ax1.set_ylabel('Density')
    ax1.set_xlabel(col)
    ax1.set_title('Distribution of variable by death')
    sns.boxplot(x=df['hospital_death'], y=df[col], ax=ax[1])


# In[30]:


show_bivariate(df, 'age')


# In[31]:


df_filt = df[(df['pre_icu_los_days']>0) & (df['pre_icu_los_days']<7)]
show_bivariate(df_filt, 'pre_icu_los_days')


# In[32]:


# df_filt = df[(df['pre_icu_los_days']>0) & (df['pre_icu_los_days']<7)]
show_bivariate(df, 'd1_glucose_max')


# In[33]:


show_bivariate(df, 'd1_sysbp_min')


# In[53]:


df_filt = df[(df['h1_spo2_max']>80) & (df['h1_spo2_max']<120)]
show_bivariate(df_filt, 'h1_spo2_max')


# In[ ]:


df['h1_spo2_min


# In[54]:


show_bivariate(df, 'h1_resprate_max')


# In[55]:


show_bivariate(df, 'h1_sysbp_min')


# In[56]:


show_bivariate(df, 'h1_diasbp_min')


# In[57]:


show_bivariate(df, 'd1_diasbp_min')


# In[164]:


show_bivariate(df, 'apache_3j_diagnosis')


# In[59]:


sns.countplot(data=df, x='icu_type', hue='hospital_death')


# In[91]:


# analyze the apache death probabilities with target


# In[86]:


threshold = 0.2
apache_death_prob = df['apache_4a_hospital_death_prob'].apply(lambda x: 1 if x >=threshold else 0)
apache_death_prob.value_counts(normalize=True)


# In[88]:


pd.crosstab(df['hospital_death'], apache_death_prob, normalize=True)*100


# In[62]:


# We observe apache mortality prediction does not have a good performance in case of class 1 (death), when threshhold was 0.5, but performs better at 0.2


# In[84]:


apache_death_prob_icu = df['apache_4a_icu_death_prob'].apply(lambda x: 1 if x >=0.2 else 0)
apache_death_prob_icu.value_counts(normalize=True)


# In[90]:


pd.crosstab(df['hospital_death'], apache_death_prob_icu, normalize=True)*100


# ## Multi Variate Analysis <a class="anchor" id="fourth-bullet"></a>

# In[65]:


plt.figure(figsize=(20,20))
_=sns.heatmap(df[numerical_columns].corr(),cmap='PiYG')


# In[92]:


plt.figure(figsize=(25,25))
corr = df[numerical_columns].corr()
pearsonmap=sns.heatmap(corr[corr>=0.7],cmap='Blues',annot=True)


# ## Outliers <a class="anchor" id="sixth-bullet"></a>

# In[775]:


df.describe(percentiles=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999])


# In[776]:


plt.figure(figsize=(15,8))
df_num.loc[:,['age', 'bmi', 'height']].boxplot()


# In[55]:


plt.figure(figsize=(15,8))
df_num.loc[:,['apache_2_diagnosis','apache_3j_diagnosis']].boxplot()


# In[56]:


plt.figure(figsize=(15,8))
df_num.loc[:,['heart_rate_apache','map_apache','resprate_apache','d1_diasbp_max','d1_diasbp_min','d1_diasbp_noninvasive_max','d1_diasbp_noninvasive_min','d1_heartrate_max','d1_heartrate_min']].boxplot()


# In[57]:


plt.figure(figsize=(15,8))
df_num.loc[:,['d1_mbp_max','d1_mbp_min','d1_mbp_noninvasive_max','d1_mbp_noninvasive_min','d1_resprate_max','d1_resprate_min','d1_spo2_min','d1_sysbp_max']].boxplot()


# In[58]:


plt.figure(figsize=(15,8))
df_num.loc[:,['d1_sysbp_min','d1_sysbp_noninvasive_max', 'd1_sysbp_noninvasive_min']].boxplot()


# In[59]:


plt.figure(figsize=(15,8))
df_num.loc[:,['d1_temp_max','d1_temp_min']].boxplot()


# In[60]:


plt.figure(figsize=(15,8))
df_num.loc[:,['h1_diasbp_max','h1_diasbp_min','h1_diasbp_noninvasive_max',
              'h1_diasbp_noninvasive_min','h1_heartrate_max']].boxplot()


# In[61]:


plt.figure(figsize=(15,8))
df_num.loc[:,['h1_heartrate_min', 'h1_mbp_max', 'h1_mbp_min',
       'h1_mbp_noninvasive_max', 'h1_mbp_noninvasive_min', 'h1_resprate_max',
       'h1_resprate_min']].boxplot()


# In[62]:


plt.figure(figsize=(15,8))
df_num.loc[:,['h1_spo2_max', 'h1_spo2_min']].boxplot()


# In[63]:


plt.figure(figsize=(15,8))
df_num.loc[:,['h1_sysbp_max','h1_sysbp_min', 'h1_sysbp_noninvasive_max', 'h1_sysbp_noninvasive_min',
       'd1_glucose_max', 'd1_glucose_min']].boxplot()


# In[64]:


plt.figure(figsize=(15,8))
df_num.loc[:,['d1_potassium_max','d1_potassium_min']].boxplot()


# In[65]:


plt.figure(figsize=(15,8))
df_num.loc[:,['apache_4a_hospital_death_prob','apache_4a_icu_death_prob']].boxplot()


# In[1006]:


from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold


# In[1007]:


df_normalized = pd.DataFrame(MinMaxScaler().fit_transform(df[numerical_columns]), columns=numerical_columns)


# In[1008]:


df_normalized.std().sort_values()


# In[1009]:


# pre_icu_los_days, spo2_max, resperate_min, spo2_min have very low variance, possibly due to outliers


# In[1010]:


df.loc[np.abs(df['pre_icu_los_days'])>26, 'hospital_death'].value_counts(normalize=True)


# In[1011]:


# high value of pre_icu_los_days, leads to more deaths > 20% (compared to 8% overall )


# In[1012]:


df_original = df.copy()


# In[1013]:


(df['pre_icu_los_days']<0).sum()


# In[1014]:


(df['apache_4a_hospital_death_prob']<0).sum()


# In[1015]:


(df['apache_4a_icu_death_prob']<0).sum()


# In[ ]:





# In[1016]:


# As probabilities acnnot be negative, we can remove the -ve value data points


# In[1017]:


# mark Invalid values as null
# df['pre_icu_los_days'] = np.abs(df['pre_icu_los_days'])     # maybe they marked as -ve because 
df.loc[df['pre_icu_los_days']<0, 'pre_icu_los_days'] = np.nan
df.loc[df['apache_4a_hospital_death_prob']<0, 'apache_4a_hospital_death_prob'] = np.nan
df.loc[df['apache_4a_icu_death_prob']<0, 'apache_4a_icu_death_prob'] = np.nan


# In[1020]:


def treat_outliers_capping(df):
    x = df['pre_icu_los_days'].copy()
    x.loc[x >  10] = 10
    df['pre_icu_los_days']=x
    
    x1 = df['d1_resprate_min'].copy()
    x1.loc[x1 >  47.734600] = 47.734600
    df['d1_resprate_min']=x1
    
    x2  = df['d1_spo2_max'].copy()
    x2.loc[x2<87] = 87
    df['d1_spo2_max']=x2
    
    x3 = df['d1_spo2_min'].copy()
    x3[x3<43]=43
    df['d1_spo2_min'] = x3
    
    x4 = df['h1_resprate_min'].copy()
    x4.loc[x4>50] = 50
    df['h1_resprate_min']=x4
    
    x5 = df['h1_spo2_max'].copy()
    x5.loc[x5<60] = 60
    df['h1_spo2_max'] = x5
    
    x6 = df['h1_spo2_min'].copy()
    x6.loc[x6<60] = 60
    df['h1_spo2_min'] = x6
    
    return df


# In[1021]:


df = treat_outliers_capping(df)


# In[1022]:


fig, ax = plt.subplots(nrows=10, ncols=6, figsize=(30,50))

i=0
for col in df_num.columns:   
    r = i//6
    c = i%6
    i+=1
    sns.kdeplot(data=df, x=col, ax=ax[r][c])


# In[77]:


# we see using z score method to remove outliers is not viable as it leads to too much data loss


# In[1023]:


# We will categorize some variables to mitigate the effect of outliers


# In[1024]:


df['pre_icu_los_days'] = pd.cut(df['pre_icu_los_days'], bins=[0,1, 7, df['pre_icu_los_days'].max()], labels=[0, 1, 2], include_lowest=True).astype(object)


# In[1025]:


df['d1_spo2_max'] = pd.cut(df['d1_spo2_max'], bins=[0,94,df['d1_spo2_max'].max()], labels=[0,1], include_lowest=True).astype(object)


# In[1026]:


df['d1_spo2_min'] = pd.cut(df['d1_spo2_min'], bins=[0,70, 85, df['d1_spo2_min'].max()], labels=[0, 1, 2], include_lowest=True).astype(object)


# In[1027]:


df['h1_spo2_max'] = pd.cut(df['h1_spo2_max'], bins=[0,94,df['h1_spo2_max'].max()], labels=[0, 1], include_lowest=True).astype(object)


# In[1028]:


df['h1_spo2_min'] = pd.cut(df['h1_spo2_min'], bins=[0,70, 85, df['h1_spo2_min'].max()], labels=[0, 1, 2], include_lowest=True)


# In[1029]:


numerical_columns = df.select_dtypes('number').columns


# In[1030]:


categorical_columns = df.select_dtypes(exclude='number').columns


# In[1031]:


len(numerical_columns), len(categorical_columns)


# ## Missing Values <a class="anchor" id="fifth-bullet"></a>

# ### Check Missing Values

# In[1033]:


df_before_missing = df.copy()


# In[1034]:


import missingno as msno


# In[1035]:


msno.matrix(df[numerical_columns], figsize=(50,20), labels=True, label_rotation=90)


# In[1036]:


msno.matrix(df[categorical_columns], figsize=(50,20), labels=True, label_rotation=90)


# In[51]:


#relationship between null valued columns based on nullity correlation. 1 - > if one col has null, other col will also have null. for that row.


# In[1037]:


msno.heatmap(df, figsize=(50,50))


# In[1038]:


((df[numerical_columns].isnull().sum()/len(df)).sort_values(ascending=False)*100)


# In[1039]:


((df[categorical_columns].isnull().sum()/len(df)).sort_values(ascending=False)*100)


# ###

# ### Highly Correlated columns
# 
# Many columns are highly correlated . The threshold for correlation is being considered as 0.98 or 98% for this dataset.
# Accordingly there will be 12 highly coorelated columns which will be dropped from  " df_full "

# In[1040]:


corr_matrix = df.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.98
to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
print(len(to_drop), to_drop)


# In[1041]:


df.drop(to_drop,axis=1,inplace=True)


# In[1042]:


df.shape


# In[1043]:


numerical_columns = df.select_dtypes('number').columns


# ###

# ### Splitting the dataset

# In[1044]:


def split_dataset(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=test_size, random_state=100)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test


# In[1045]:


X = df.drop(columns='hospital_death')
y = df['hospital_death']
X_train, X_test, y_train, y_test = split_dataset(X, y, 0.3)

print('Total no. of samples: Training and Testing dataset separately')
print('X_train:', np.shape(X_train))
print('y_train:', np.shape(y_train))
print('X_test:', np.shape(X_test))
print('y_test:', np.shape(y_test))


# In[1046]:


df.shape


# ##

# ### Encoding categorical variables

# In[1047]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


# In[1048]:


# types of categorical columns
ordinal_cols = ['gcs_eyes_apache', 'gcs_motor_apache', 'gcs_verbal_apache', 'd1_spo2_min', 'h1_spo2_min', 'pre_icu_los_days']
nominal_cols = ['ethnicity', 'icu_admit_source', 'icu_stay_type', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem']
binary_cols = list( set(categorical_columns).difference( set(ordinal_cols + nominal_cols) ))
binary_cols.remove('hospital_death')
print('ordinal_columns:\n', len(ordinal_cols), ordinal_cols)
print('nominal_columns:\n', len(nominal_cols), nominal_cols)
print('binary_columns:\n', len(binary_cols), binary_cols)


# In[ ]:





# In[1049]:


oe = OrdinalEncoder()
# these columns are either ordered or binary, hence we can use ordinal
oe.fit(X_train[ordinal_cols+binary_cols])


# In[1050]:


x1 = pd.DataFrame(oe.transform(X_train[ordinal_cols+binary_cols]), columns=ordinal_cols+binary_cols)


# In[1051]:


ohe = OneHotEncoder(drop='first', sparse=False)
# nulls are assigned as new subcategory
ohe.fit(X_train[nominal_cols])


# In[1052]:


x2 = pd.DataFrame(ohe.transform(X_train[nominal_cols]), columns=ohe.get_feature_names_out())


# In[1053]:


X_train = pd.concat([X_train[numerical_columns], x1, x2], axis=1)


# In[1054]:


X_train.shape


# In[1055]:


X_train.head()


# In[1056]:


X_train.dtypes.unique()


# In[1057]:


X_columns = X_train.columns


# In[1058]:


# All columns are numerical now.


# In[1059]:


X_train.isna().sum().sum()


# In[1060]:


x1_test = pd.DataFrame(oe.transform(X_test[ordinal_cols+binary_cols]), columns=ordinal_cols+binary_cols)


# In[1061]:


x2_test = pd.DataFrame(ohe.transform(X_test[nominal_cols]), columns=ohe.get_feature_names_out())


# In[1062]:


X_test = pd.concat([X_test[numerical_columns], x1_test, x2_test], axis=1)


# In[1063]:


X_test.shape


# In[1064]:


X_test.head()


# In[1065]:


le = LabelEncoder()
le.fit(y_train)


# In[1066]:


y_train = le.transform(y_train)


# In[1067]:


y_test = le.transform(y_test)


# In[1068]:


pd.Series(y_train).value_counts(normalize=True)


# In[1069]:


pd.Series(y_test).value_counts(normalize=True)


# ##

# ## Imputation

# In[1070]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier


# ### Simple Imputation
# Filling missing values using median and mode

# In[1071]:


def perform_imputation_simple(X_train, X_test, numerical_columns, ordinal_cols, binary_cols):
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()
    
    # median imputation for numerical columns
    imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_num.fit(X_train[numerical_columns])
    X_train_imputed[numerical_columns] = imp_num.transform(X_train[numerical_columns])
    X_test_imputed[numerical_columns] = imp_num.transform(X_test[numerical_columns])
    
    # Mode imputation for categorical variables (nominal columns already include new column for missing)
    imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp_cat.fit(X_train[ordinal_cols+binary_cols])
    X_train_imputed[ordinal_cols+binary_cols] = imp_cat.transform(X_train[ordinal_cols+binary_cols])
    X_test_imputed[ordinal_cols+binary_cols] = imp_cat.transform(X_test[ordinal_cols+binary_cols])
    
    return X_train_imputed, X_test_imputed


# ### Multiple Imputer
# A strategy for imputing missing values by modeling each feature with
# missing values as a function of other features in a round-robin fashion

# In[1072]:


def perform_imputation_mice(X_train, X_test, numerical_columns, ordinal_cols, binary_cols):
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()
    
    # Imputing numerical features with Linear Regression as estimator for MICE
    lr = LinearRegression()
    imp_num = IterativeImputer(estimator=lr, missing_values=np.nan, max_iter=10, verbose=2, random_state=0)
    imp_num.fit(X_train[numerical_columns])
    X_train_imputed[numerical_columns] = imp_num.transform(X_train[numerical_columns])
    X_test_imputed[numerical_columns] = imp_num.transform(X_test[numerical_columns])
    
    # Imputing categorical features with Decision Tree as estimator for MICE (nominal columns already include new column for missing)
    dt = DecisionTreeClassifier(max_depth=10)
    imp_cat = IterativeImputer(estimator=dt, missing_values=np.nan, max_iter=10, verbose=2, random_state=0)
    imp_cat.fit(X_train[ordinal_cols+binary_cols])
    X_train_imputed[ordinal_cols+binary_cols] = imp_cat.transform(X_train[ordinal_cols+binary_cols])
    X_test_imputed[ordinal_cols+binary_cols] = imp_cat.transform(X_test[ordinal_cols+binary_cols])
    
    return X_train_imputed, X_test_imputed


# ### KNN Imputer
# 
# Since KNN uses nearest neighbor and imputes the missing values and it doesnt depend
# on other columns like in multivariate approach.
# Second it uses Euclidean distance to form the neighbors and fillinf missing values 
# is not random or artificial unlike in Simple Imputations

# In[1073]:


def perform_imputation_knn(X_train, X_test, numerical_columns, ordinal_cols, binary_cols):
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Scaling numerical columns before applying KNN impute
    sc = StandardScaler()
    sc.fit(X_train[numerical_columns])
    X_train_scaled[numerical_columns] = sc.transform(X_train[numerical_columns])
    X_test_scaled[numerical_columns] = sc.transform(X_test[numerical_columns])
    
    # Imputing numerical features with KNN
    impute_knn=KNNImputer(n_neighbors=4)
    impute_knn.fit(X_train_scaled[numerical_columns])
    X_train_scaled[numerical_columns] = impute_knn.transform(X_train_scaled[numerical_columns])
    X_test_scaled[numerical_columns] = impute_knn.transform(X_test_scaled[numerical_columns])
    
#     # Mode imputation for categorical variables (nominal columns already include new column for missing)
#     imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
#     imp_cat.fit(X_train_scaled[ordinal_cols+binary_cols])
#     X_train_scaled[ordinal_cols+binary_cols] = imp_cat.transform(X_train_scaled[ordinal_cols+binary_cols])
#     X_test_scaled[ordinal_cols+binary_cols] = imp_cat.transform(X_test_scaled[ordinal_cols+binary_cols])
    
    # Imputing categorical features with Decision Tree as estimator for MICE (nominal columns already include new column for missing)
    dt = DecisionTreeClassifier(max_depth=10)
    imp_cat = IterativeImputer(estimator=dt, missing_values=np.nan, max_iter=10, verbose=2, random_state=0)
    imp_cat.fit(X_train_scaled[ordinal_cols+binary_cols])
    X_train_scaled[ordinal_cols+binary_cols] = imp_cat.transform(X_train_scaled[ordinal_cols+binary_cols])
    X_train_scaled[ordinal_cols+binary_cols] = imp_cat.transform(X_train_scaled[ordinal_cols+binary_cols])
    
    # inverse transform the numerical values to get to to original scale before returning
    X_train_scaled[numerical_columns] = sc.inverse_transform(X_train[numerical_columns])
    X_test_scaled[numerical_columns] = sc.inverse_transform(X_test[numerical_columns])
    
    return X_train_scaled, X_test_scaled


# In[1074]:


# there are no missing values in target
pd.isna(y_train).sum(), pd.isna(y_test).sum()


# In[1075]:


X_train, X_test = perform_imputation_mice(X_train, X_test, numerical_columns, ordinal_cols, binary_cols)
# X_train, X_test = perform_imputation_knn(X_train, X_test, numerical_columns, ordinal_cols, binary_cols)
# X_train, X_test = perform_imputation_simple(X_train, X_test, numerical_columns, ordinal_cols, binary_cols)


# In[1076]:


X_train.isna().sum().sum(), X_test.isna().sum().sum()


# In[1077]:


# Save the imputed data, as imputation is computationally expensive. We can retreive this back after any more changes during model building
X_train_copy = X_train.copy()
X_test_copy = X_test.copy()
y_train_copy = y_train.copy()
y_test_copy = y_test.copy()


# ##

# ## Statistical Tests for feature selection

# In[1078]:


from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif


# In[1079]:


def select_features_chi2(X_train, y_train, X_test, k='all'):
    fs = SelectKBest(score_func=chi2, k=k)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


# In[1080]:


def select_features_mi(X_train, y_train, X_test, k='all'):
    fs = SelectKBest(score_func=mutual_info_classif, k=k)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


# In[1081]:


encoded_categorical_columns = list(set(X_columns).difference(set(numerical_columns)))


# In[1082]:


len(encoded_categorical_columns)


# In[1083]:


# Using chi-squared test for feature selection
X_train_fs, X_test_fs, fs_chi2 = select_features_chi2(X_train[encoded_categorical_columns], y_train, X_test[encoded_categorical_columns], k=15)


# In[1084]:


df_best_features_chi2 = pd.Series(fs_chi2.scores_, index=encoded_categorical_columns, name='score').sort_values(ascending=False)
df_best_features_chi2.head(15).index


# In[1085]:


plt.figure(figsize=(15,8))
plt.barh(df_best_features_chi2.head(15).index, df_best_features_chi2.head(15).values)
# plt.xticks(rotation=90)
plt.show()


# In[1086]:


# Using Mutual Information for feature selection
X_train_fs, X_test_fs, fs_mi = select_features_mi(X_train[encoded_categorical_columns], y_train, X_test[encoded_categorical_columns], k=15)


# In[1087]:


df_best_features_mi = pd.Series(fs_mi.scores_, index=encoded_categorical_columns, name='score').sort_values(ascending=False)
df_best_features_mi[:15].index


# In[1088]:


plt.figure(figsize=(15,8))
plt.barh(df_best_features_mi.head(15).index, df_best_features_mi.head(15).values)
# plt.xticks(rotation=90)
plt.show()


# In[1089]:


# Using ANOVA for feature selection from numerical columns


# In[1090]:


from sklearn.feature_selection import f_classif


# In[1091]:


fs = SelectKBest(score_func=f_classif, k=20)
# apply feature selection


# In[1092]:


X_selected = fs.fit_transform(X_train[numerical_columns], y_train)


# In[1093]:


best_features_num = pd.Series(fs.scores_, index=numerical_columns).sort_values(ascending=False)
best_features_num.head(20).index


# In[1094]:


plt.figure(figsize=(15,8))
plt.barh(best_features_num.head(20).index, best_features_num.head(20).values)
# plt.xticks(rotation=90)
plt.show()


# In[1138]:


top_features = ['gcs_motor_apache', 'gcs_verbal_apache', 'gcs_eyes_apache', 'ventilated_apache', 'intubated_apache', 'elective_surgery', 'icu_admit_source_Operating Room / Recovery', 'pre_icu_los_days', 'apache_3j_bodysystem_Metabolic', 'apache_3j_bodysystem_Sepsis', 'apache_post_operative', 'd1_spo2_min', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'd1_heartrate_max', 'd1_mbp_min', 'd1_temp_min', 'd1_sysbp_min', 'd1_diasbp_min', 'temp_apache']


# ## Model Building <a class="anchor" id="seventh-bullet"></a>

# In[1095]:


from sklearn.metrics import fbeta_score, make_scorer


# In[1096]:


X_train.shape


# In[1097]:


def plot_confusion_matrix(cm, norm=False): 
    conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], 
                               index = ['Actual:0','Actual:1'])
    fmt = '0.2f' if norm else 'd'
    sns.heatmap(conf_matrix, annot = True, fmt = fmt, cmap = ListedColormap(['lightskyblue']), cbar = False, 
                linewidths = 0.1, annot_kws = {'size':25})    
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.show()


# In[1098]:


def plot_roc(fpr, tpr, thresholds, score):
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot([0, 1], [0, 1],'r--')
    plt.title('ROC curve ', fontsize = 15)
    plt.xlabel('False positive rate (1-Specificity)', fontsize = 15)
    plt.ylabel('True positive rate (Sensitivity)', fontsize = 15)
    plt.text(x = 0.02, y = 0.9, s = ('AUC Score:',round(score)))
    plt.grid(True)


# In[1099]:


def display_model_results(y_true, y_predicted, y_predicted_prob, norm=False):
    accuracy = accuracy_score(y_true, y_predicted)*100
    auc_roc = roc_auc_score(y_true, y_predicted_prob)*100
    # PR AUC - precision recall ROC curve area
    avg_prec_score = average_precision_score(y_true, y_predicted_prob)
    cm = confusion_matrix(y_true, y_predicted)
    cm_norm = cm*100/cm.sum()
    print('ROC AUC: %.4f' % auc_roc)
    print('Accuracy: %.4f %%' % accuracy)
    print(f'Avg Precision Score: {avg_prec_score:.2f}')
    print()
    print(classification_report(y_true, y_predicted))
    if norm:
        plot_confusion_matrix(cm_norm, True)
    else:
        plot_confusion_matrix(cm)
        
    return accuracy, auc_roc, cm[0][1]/(cm[1][1]+cm[1][0])


# In[1100]:


# Scaling numerical columns before applying logistic regression
sc = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

sc.fit(X_train[numerical_columns])
X_train_scaled[numerical_columns] = sc.transform(X_train[numerical_columns])
X_test_scaled[numerical_columns] = sc.transform(X_test[numerical_columns])


# ## Logistic Regression

# In[1101]:


X_train_scaled.shape, y_train.shape


# In[1102]:


X_test_scaled.shape, y_test.shape


# In[1103]:


classifier = LogisticRegression(random_state=0, max_iter=300)
classifier.fit(X_train_scaled, y_train)


# In[1104]:


y_train_hat = classifier.predict(X_train_scaled)
y_train_hat_probs = classifier.predict_proba(X_train_scaled)[:,1]


# In[1105]:


display_model_results(y_train, y_train_hat, y_train_hat_probs, True)


# In[1106]:


y_test_hat = classifier.predict(X_test_scaled)
y_test_hat_probs = classifier.predict_proba(X_test_scaled)[:,1]
display_model_results(y_test, y_test_hat, y_test_hat_probs, True)


# In[1107]:


fbeta_score(y_test, y_test_hat, beta=2)


# In[ ]:





# In[1108]:


# Wrong to Right ratio for class 1
cm = confusion_matrix(y_test, y_test_hat)
cm[0,1]/cm[1,1]


# ### Threshold tuning

# In[1109]:


test_auc_roc = roc_auc_score(y_test, y_test_hat_probs)*100
fpr, tpr, thresholds = roc_curve(y_test, y_test_hat_probs)
plot_roc(fpr, tpr, thresholds, test_auc_roc)


# In[ ]:





# In[1110]:


from sklearn.metrics import precision_recall_curve


# In[1111]:


precision_, recall_, thresholds_ = precision_recall_curve(y_test, y_test_hat)
plt.xlabel('precision')
plt.ylabel('recall')
plt.plot(precision_, recall_)


# In[ ]:





# In[1112]:


from sklearn.metrics import precision_recall_fscore_support


# In[1113]:


thesholds = [0.05*i for i in range(1,11)]

accuracies = []
recalls = []
fbetas = []
for thresh in thresholds:
    y_test_hat2 = np.where(y_test_hat_probs>=thresh, 1, 0)
    accuracy = accuracy_score(y_test, y_test_hat2)
    prec, recall, fbeta, support  = precision_recall_fscore_support(y_test, y_test_hat2, beta=2, zero_division=0)
    accuracies.append(accuracy)
    recalls.append(recall[1])
    fbetas.append(fbeta[1])


# In[1114]:


plt.xlim(0,1)
sns.lineplot(x=thresholds, y=accuracies, label='acc')
sns.lineplot(x=thresholds, y=recalls, label='rec')
sns.lineplot(x=thresholds, y=fbetas, label='f2')
    


# In[1115]:


# we are seeing tradeoff between precision and recall. 


# In[1116]:


# Youdens index to select cutoff
youdens_table = pd.DataFrame({'TPR': tpr,
                             'FPR': fpr,
                             'Threshold': thresholds})

# calculate the difference between TPR and FPR for each threshold and store the values in a new column 'Difference'
youdens_table['Difference'] = youdens_table.TPR - youdens_table.FPR
youdens_table = youdens_table.sort_values('Difference', ascending = False).reset_index(drop = True)
# print the first five observations
youdens_table.head(5)


# In[1117]:


y_test_hat2 = np.where(y_test_hat_probs>=0.1, 1, 0)


# In[1120]:


fbeta_score(y_test, y_test_hat2, beta=2)


# In[1121]:


acc, roc, recall = display_model_results(y_test, y_test_hat2, y_test_hat_probs, True)


# In[1122]:


# Although we get better scores using low threshold of 0.1, but it is not suitable as it decreases the confidence in our models prediction power. 
# Hence we evaluate other methods to improve the model


# ### Cross Validation

# In[1123]:


fbeta_score(y_test, y_test_hat, beta=2)


# In[1124]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from warnings import filterwarnings
filterwarnings('ignore')


# In[1125]:


ftwo_scorer = make_scorer(fbeta_score, beta=2)


# In[1126]:


scores = cross_val_score(estimator = LogisticRegression(), 
                         X = X_train_scaled, 
                         y = y_train, 
                         cv = 5, 
                         scoring = ftwo_scorer)


# In[1127]:


print(scores)


# In[1128]:


# We can observe that the model is performing good with all subsets of data


# In[1129]:


# Comparing Apache death probability and death probability of our model


# In[1130]:


predictions = pd.concat([pd.Series(y_test, name='actual'), pd.Series(y_test_hat_probs, name='predicted_probs'), X_test['apache_4a_hospital_death_prob']], axis=1)


# In[1131]:


predictions['actual'].value_counts()


# In[1132]:


sns.kdeplot(predictions.loc[predictions['actual']==1, 'apache_4a_hospital_death_prob'], label='Apache death probability')
sns.kdeplot(predictions.loc[predictions['actual']==1, 'predicted_probs'], label = 'Model death Probability')
plt.legend()


# In[1133]:


sns.kdeplot(pd.Series(y_test_hat_probs)[y_test==1], label='prob when patients died')
sns.kdeplot(pd.Series(y_test_hat_probs)[y_test==0], label='prob when patients survived')
plt.legend()


# In[1134]:


# Coefficients


# In[1135]:


classifier.intercept_


# In[1136]:


coeff = pd.Series(classifier.coef_[0], index=X_train_scaled.columns)
coeff.sort_values(ascending=False)[:20]


# In[1137]:


coeff.sort_values(ascending=False)[:-20:-1]


# In[ ]:





# ## PCA for Dimention reduction

# In[1187]:


from sklearn.decomposition import PCA


# In[1188]:


len(numerical_columns)


# In[1189]:


n_pca = 20
pca = PCA(n_components=n_pca)


# In[1190]:


pca_train = pca.fit_transform(X_train_scaled[numerical_columns])
pca_test = pca.transform(X_test_scaled[numerical_columns])


# In[1191]:


pca.explained_variance_ratio_.sum()


# In[1192]:


pca_train_df = pd.DataFrame(data = pca_train, columns = ['PC'+str(i) for i in range(1, n_pca+1)])
pca_test_df = pd.DataFrame(data = pca_test, columns = ['PC'+str(i) for i in range(1, n_pca+1)])


# In[1193]:


X_train_pca = pd.concat([pca_train_df, X_train[encoded_categorical_columns]], axis = 1)
X_test_pca = pd.concat([pca_test_df, X_test[encoded_categorical_columns]], axis = 1)


# In[ ]:





# In[1194]:


model = LogisticRegression(random_state = 100, max_iter=200)


# In[1195]:


model.fit(X_train_pca, y_train)


# In[1196]:


y_test_pred = model.predict(X_test_pca)
y_test_pred_prob = model.predict_proba(X_test_pca)[:,1]


# In[1197]:


display_model_results(y_test, y_test_pred, y_test_pred_prob)


# ## AutoML
import autosklearn.classificationautoml = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=36000,
    per_run_time_limit=3600,
    tmp_folder='/tmp/autosklearn_classification_example_tmp',
)
automl.fit(X_train, y_train, X_test, y_test, dataset_name='patient-survival')y_pred = automl.predict(X_test)
print("Accuracy score:", accuracy_score(y_test, y_pred))cm = confusion_matrix(y_test, y_pred)
print(cm)print(classification_report(y_test, y_pred))automl.performance_over_time_.plot(
        x='Timestamp',
        kind='line',
        legend=True,
        title='Auto-sklearn accuracy over time',
        grid=True,
    )
plt.show()print(automl.sprint_statistics())automl.leaderboard(ensemble_only=False)df_res = pd.DataFrame(automl.cv_results_)
# In[ ]:





# ## Model with top features

# In[1139]:


classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train_scaled[top_features], y_train)


# In[1140]:


y_test_hat2 = classifier2.predict(X_test_scaled[top_features])
y_test_hat_probs2 = classifier2.predict_proba(X_test_scaled[top_features])[:,1]
display_model_results(y_test, y_test_hat2, y_test_hat_probs2, True)


# In[1141]:


# Wrong to Right ratio for class 1
cm = confusion_matrix(y_test, y_test_hat2)
cm[0,1]/cm[1,1]


# In[1142]:


fbeta_score(y_test, y_test_hat2, beta=2)


# In[ ]:





# ## Decision Tree

# In[1143]:


model_dt = DecisionTreeClassifier(max_depth=20, random_state = 100)


# In[1144]:


model_dt.fit(X_train, y_train)


# In[1145]:


y_train_hat = model_dt.predict(X_train)
y_train_hat_probs = model_dt.predict_proba(X_train)[:,1]


# In[1146]:


display_model_results(y_train, y_train_hat, y_train_hat_probs, True)


# In[1147]:


y_test_hat = model_dt.predict(X_test)
y_test_hat_probs = model_dt.predict_proba(X_test)[:,1]
display_model_results(y_test, y_test_hat, y_test_hat_probs, True)


# In[1148]:


fbeta_score(y_test, y_test_hat, beta=2)


# In[1149]:


pd.Series(model_dt.feature_importances_, index=X_test.columns).sort_values(ascending=False)[:20]


# ## Random Forest

# In[1150]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier


# In[1151]:


model_rf = RandomForestClassifier(max_depth=20, random_state = 100)


# In[1152]:


model_rf.fit(X_train, y_train)


# In[1153]:


y_train_hat = model_rf.predict(X_train)
y_train_hat_probs = model_rf.predict_proba(X_train)[:,1]


# In[1154]:


display_model_results(y_train, y_train_hat, y_train_hat_probs, True)


# In[1155]:


y_test_hat = model_rf.predict(X_test)
y_test_hat_probs = model_rf.predict_proba(X_test)[:,1]
display_model_results(y_test, y_test_hat, y_test_hat_probs, True)


# In[1156]:


fbeta_score(y_test, y_test_hat, beta=2)


# In[1157]:


pd.Series(model_rf.feature_importances_, index=X_test.columns).sort_values(ascending=False)[:20]


# In[ ]:





# ## XGBoost

# In[1158]:


from xgboost import XGBClassifier


# In[1159]:


xgb_model = XGBClassifier( use_label_encoder=False, scale_pos_weight=10)
xgb_model.fit(X_train, y_train, eval_metric=ftwo_scorer)


# In[ ]:





# In[1163]:


y_test_hat = xgb_model.predict(X_test)
y_test_hat_probs = xgb_model.predict_proba(X_test)[:,1]


# In[1164]:


display_model_results(y_test, y_test_hat, y_test_hat_probs, True)


# In[1165]:


fbeta_score(y_test, y_test_hat, beta=2)


# In[1166]:


sns.kdeplot(pd.Series(y_test_hat_probs)[y_test==1], label='prob when patients died')
sns.kdeplot(pd.Series(y_test_hat_probs)[y_test==0], label='prob when patients survived')
plt.legend()


# In[1167]:


pd.Series(xgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=False).iloc[:20]


# ## Hybridization: SMOTE + TOMEK LINKS
# ### for Target since target is imbalanced  

# SMOTE+TOMEK is such a hybrid technique that aims to clean overlapping data points for each of the classes distributed in sample space. After the oversampling is done by SMOTE, the class clusters may be invading each other’s space. As a result, the classifier model will be overfitting. Now, Tomek links are the opposite class paired samples that are the closest neighbors to each other. Therefore the majority of class observations from these links are removed as it is believed to increase the class separation near the decision boundaries. Now, to get better class clusters, Tomek links are applied to oversampled minority class samples done by SMOTE. Thus instead of removing the observations only from the majority class, we generally remove both the class observations from the Tomek links.
# Reference:
# 
# https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/

# In[724]:


from imblearn.combine import SMOTETomek
from collections import Counter


# In[725]:


# SMOTE + TOMEK for train data
counter = Counter(y_train)
print('Before SMOTE : ', counter)

smt = SMOTETomek(random_state=100)
X_train_smt, y_train_smt = smt.fit_resample(X_train, y_train)

counter = Counter(y_train_smt)
print('After SMOTE : ', counter)


# In[726]:


print('data shape before smote')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[729]:


print('data shape after smote')
print(X_train_smt.shape, X_test.shape, y_train_smt.shape, y_test.shape)


# # Logistic regression SMOTE variables

# In[748]:


model_rfs = LogisticRegression(random_state = 100)
model_rfs.fit(X_train_smt[top_features], y_train_smt)


# In[752]:


y_test_hat = model_rfs.predict(X_test[top_features])
y_test_hat_probs = model_rfs.predict_proba(X_test[top_features])[:,1]


# In[753]:


fbeta_score(y_test, y_test_hat, beta=2)


# In[754]:


display_model_results(y_test, y_test_hat, y_test_hat_probs, True)


# In[ ]:





# In[755]:


sns.kdeplot(pd.Series(y_test_hat_probs)[y_test==1], label='prob when patients died')
sns.kdeplot(pd.Series(y_test_hat_probs)[y_test==0], label='prob when patients survived')
plt.legend()


# In[ ]:





# ## HyperParameter Tuning

# In[738]:


from sklearn.model_selection import GridSearchCV, GroupKFold


# In[778]:


tuned_paramaters = [{'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                     'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                     'C': [100, 10, 1.0, 0.1, 0.01],
                     'class_weight': ['balanced', None],
                     'max_iter': [200]}]
 
log_reg = LogisticRegression(random_state = 10)

logreg_grid = GridSearchCV(estimator = log_reg, 
                       param_grid = tuned_paramaters, 
                       scoring=ftwo_scorer,
                       cv = 5)


# In[779]:


# use fit() to fit the model on the train set
logreg_grid_model = logreg_grid.fit(X_train_scaled[top_features], y_train)


# In[780]:


# get the best parameters
print('Best parameters for Logistic Regression classifier: ', logreg_grid_model.best_params_, '\n')


# In[348]:


# Save the model and reload


# In[784]:


from joblib import dump, load


# In[785]:


dump(logreg_grid_model, 'logreg_grid_model.joblib') 


# In[786]:


logreg_grid_model2 = load('logreg_grid_model.joblib')


# In[790]:


res = logreg_grid_model.cv_results_


# In[ ]:





# In[796]:


ranks = pd.Series(res['rank_test_score'])


# In[797]:


params = pd.Series(res['params'])


# In[800]:


scores0 = pd.Series(res['split0_test_score'])


# In[801]:


scores_mean = pd.Series(res['mean_test_score'])


# In[814]:


params_exp = params.apply(pd.Series)


# In[825]:


res_comb = pd.concat([ranks, scores_mean, scores0, params, params_exp], axis=1)
res_comb.columns = ['ranks', 'scores_mean', 'scores0', 'params'] + params_exp.columns.tolist()


# In[866]:


indexes_ranked = res_comb.sort_values(by=['ranks']).index
res_comb.sort_values(by=['ranks'])[:20]


# In[875]:


def try_params_logistic(params):
    params2 = {'random_state': 100}
    params2.update(params)
    print('parameters:', params2)
    model_temp = LogisticRegression(**params2)
    model_temp.fit(X_train_scaled[top_features], y_train)
    y_test_hat = model_temp.predict(X_test_scaled[top_features])
    y_test_hat_probs = model_temp.predict_proba(X_test_scaled[top_features])[:,1]
    print('f2 score:', fbeta_score(y_test, y_test_hat, beta=2))
    display_model_results(y_test, y_test_hat, y_test_hat_probs, True)
    print(pd.Series(model_temp.coef_[0], index=top_features).sort_values())


# In[876]:


try_params_logistic(res_comb.loc[indexes_ranked[0], 'params'])


# In[ ]:





# ### Hyperparameter tuning for decision tree

# In[873]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[1222]:


tuned_paramaters = [{'criterion' : ["gini", "entropy"],
                     'max_depth': [5, 7, 9, 12],
                     'min_samples_split': [2,5,10,20],
                     'min_samples_leaf': [5,10,20,30],
                     'class_weight': ['balanced', None]}]

dt_est = DecisionTreeClassifier(random_state = 10)

dt_grid = GridSearchCV(estimator = dt_est, 
                       param_grid = tuned_paramaters, 
                       scoring=ftwo_scorer,
                       cv = 5)


# In[1223]:


# use fit() to fit the model on the train set
dt_grid_model = dt_grid.fit(X_train[top_features], y_train)


# In[1224]:


# get the best parameters
print('Best parameters for random forest classifier: ', dt_grid_model.best_params_, '\n')


# In[1225]:


# Save the model and reload


# In[1226]:


from joblib import dump, load


# In[1227]:


dump(dt_grid_model, 'dt_grid_model1.joblib') 


# In[1228]:


res = dt_grid_model.cv_results_
ranks = pd.Series(res['rank_test_score'])
params = pd.Series(res['params'])
scores0 = pd.Series(res['split0_test_score'])
scores_mean = pd.Series(res['mean_test_score'])
params_exp = params.apply(pd.Series)


# In[1229]:


res_comb = pd.concat([ranks, scores_mean, scores0, params, params_exp], axis=1)
res_comb.columns = ['ranks', 'scores_mean', 'scores0', 'params'] + params_exp.columns.tolist()


# In[1230]:


res_comb.sort_values(by=['ranks'])[:3]


# In[1231]:


indexes_ranked = res_comb.sort_values(by=['ranks'])[:20].index


# In[1232]:


def try_params_decisiontree2(params):
    params2 = {'random_state': 100}
    params2.update(params)
    model_temp = DecisionTreeClassifier(**params2)
    model_temp.fit(X_train[top_features], y_train)
    y_test_hat = model_temp.predict(X_test[top_features])
    y_test_hat_probs = model_temp.predict_proba(X_test[top_features])[:,1]
    f2score = fbeta_score(y_test, y_test_hat, beta=2)
    auc_roc = roc_auc_score(y_test, y_test_hat)*100
    avg_prec_score = average_precision_score(y_test, y_test_hat_probs)
    rep = classification_report(y_test, y_test_hat, output_dict=True)
    prec = rep['1']['precision']
    recall = rep['1']['recall']
    f1score = rep['1']['f1-score']
    accuracy = rep['accuracy']
    return pd.Series([f2score, auc_roc, accuracy, prec, recall, f1score])


# In[1233]:


results_ = res_comb['params'].apply(try_params_decisiontree2)


# In[1234]:


results_.columns = ['f2score', 'auc_roc', 'accuracy', 'prec', 'recall', 'f1score']
results_comb2 = pd.concat([res_comb, results_], axis=1)


# In[1235]:


results_comb2.sort_values('ranks')[:10]


# In[ ]:





# In[1236]:


# Model builing with tuned parameters
model_dt_tuned = DecisionTreeClassifier(random_state=100, **dt_grid_model.best_params_)
model_dt_tuned.fit(X_train[top_features], y_train)
y_test_hat = model_dt_tuned.predict(X_test[top_features])
y_test_hat_probs = model_dt_tuned.predict_proba(X_test[top_features])[:,1]
print('f2 score:', fbeta_score(y_test, y_test_hat, beta=2))
display_model_results(y_test, y_test_hat, y_test_hat_probs, True)


# In[1237]:


from sklearn.tree import plot_tree


# In[1238]:


plt.figure(figsize=(50,20))
plot_tree(model_dt_tuned, filled=True, max_depth=3, feature_names=top_features, fontsize=20)
plt.title("Decision tree trained on top 20 features")
plt.show()


# In[1239]:


important_features = pd.Series(model_dt_tuned.feature_importances_, index=model_dt_tuned.feature_names_in_).sort_values(ascending=False)
important_features


# In[1243]:


plt.figure(figsize=(15,8))
plt.barh(important_features.index, important_features.values)
plt.xlabel('importance')
plt.title('Feature importance')
plt.show()


# In[ ]:




