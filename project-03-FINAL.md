---
jupyter:
  jupytext:
    formats: ipynb,md,py:light
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 15px; height: 80px">

# Project 3

### Regression and Classification with the Ames Housing Data

---

You have just joined a new "full stack" real estate company in Ames, Iowa. The strategy of the firm is two-fold:
- Own the entire process from the purchase of the land all the way to sale of the house, and anything in between.
- Use statistical analysis to optimize investment and maximize return.

The company is still small, and though investment is substantial the short-term goals of the company are more oriented towards purchasing existing houses and flipping them as opposed to constructing entirely new houses. That being said, the company has access to a large construction workforce operating at rock-bottom prices.

This project uses the [Ames housing data recently made available on kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

```python
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style('whitegrid')

%config InlineBackend.figure_format = 'retina'
%matplotlib inline
```

<img src="http://imgur.com/l5NasQj.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 1. Estimating the value of homes from fixed characteristics.

---

Your superiors have outlined this year's strategy for the company:
1. Develop an algorithm to reliably estimate the value of residential houses based on *fixed* characteristics.
2. Identify characteristics of houses that the company can cost-effectively change/renovate with their construction team.
3. Evaluate the mean dollar value of different renovations.

Then we can use that to buy houses that are likely to sell for more than the cost of the purchase plus renovations.

Your first job is to tackle #1. You have a dataset of housing sale data with a huge amount of features identifying different aspects of the house. The full description of the data features can be found in a separate file:

    housing.csv
    data_description.txt
    
You need to build a reliable estimator for the price of the house given characteristics of the house that cannot be renovated. Some examples include:
- The neighborhood
- Square feet
- Bedrooms, bathrooms
- Basement and garage space

and many more. 

Some examples of things that **ARE renovate-able:**
- Roof and exterior features
- "Quality" metrics, such as kitchen quality
- "Condition" metrics, such as condition of garage
- Heating and electrical components

and generally anything you deem can be modified without having to undergo major construction on the house.

---

**Your goals:**
1. Perform any cleaning, feature engineering, and EDA you deem necessary.
- Be sure to remove any houses that are not residential from the dataset.
- Identify **fixed** features that can predict price.
- Train a model on pre-2010 data and evaluate its performance on the 2010 houses.
- Characterize your model. How well does it perform? What are the best estimates of price?

> **Note:** The EDA and feature engineering component to this project is not trivial! Be sure to always think critically and creatively. Justify your actions! Use the data description file!

```python
# Load the data
house = pd.read_csv('./housing.csv')
```

### Grabing relvaent data for fixed characteristic house model

```python
fixed_cols = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
              'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 
              'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
              'YearBuilt', 'YearRemodAdd', 'Foundation', 'BsmtQual', '1stFlrSF', 
              '2ndFlrSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
              'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageType', 'GarageYrBlt', 
              'GarageCars', 'GarageArea', 'MiscFeature', 'SalePrice','MoSold','YrSold']
#     List of fixed house characteristic features and other relevant data

df_fixed = house[fixed_cols]
#     Creating fixed characteristics dataframe

df_fixed = df_fixed[(df_fixed['MSZoning']=='FV') | (df_fixed['MSZoning']=='RH') 
              | (df_fixed['MSZoning']=='RL') | (df_fixed['MSZoning']=='RP') | (df_fixed['MSZoning']=='RM')]
#     Keeping all houses belonging only to residental zones
```

```python
# Confirm
df_fixed.head().T
```

### DATA CLEANING AND MUNGING

```python
df_fixed.shape
```

```python
df_fixed.info()
```

```python
# LotFrontage, Alley, BsmtQual, GarageType, GarageYrBlt, GarageCars, GarageArea, MiscFeature all have 
# Missing Values. 

# With the amount of unique data points of MSSubClass, better to convert the numbers to categories 
# (Most models won't intrepret this column as a category)
```

#### Dealing with missing values

LotFrontage, Alley, BsmtQual, GarageType, GarageYrBlt, GarageCars, GarageArea, MiscFeature all have 
Missing Values. 

With the amount of unique data points of MSSubClass, better to convert the numbers to categories 
(Most models won't intrepret this column as a category)


##### MiscFeatures

```python
df_fixed['MiscFeature'].value_counts()
```

```python
# Drop this column but encode if the house has a shed first 
# (Sufficient data points for possible usuability)
df_fixed['Has_Shed'] = df_fixed['MiscFeature'].apply(lambda x : 1 if x=='Shed' else 0)
#     Encode 'Has_Shed'
df_fixed.drop(labels='MiscFeature',axis=1,inplace=True)
#     Drop MiscFeature
```

```python
# Print Check for proper encoding and dropping of column
print(df_fixed['Has_Shed'].value_counts())

try:
    df_fixed['MiscFeature']
except:
    print('\n Error: There is No column named MiscFeature')
```

##### Dealing with the Garage Varaibles
Oddities
- GarageType is null if there is no garage. 
- There are less GarageCars and GarageArea null values (???)

```python
# Investigating the GarageCars and GarageArea columns where GarageType is null
df_fixed[df_fixed['GarageType'].isnull()==True][['GarageType','GarageYrBlt','GarageCars','GarageArea']]
```

```python
for cols in df_fixed[df_fixed['GarageType'].isnull()==True]\
[['GarageType','GarageYrBlt','GarageCars','GarageArea']].columns: 
    print(df_fixed[df_fixed['GarageType'].isnull()==True][f"{cols}"].value_counts())
#     This Block of code prints out the value_counts for each column
#     It looks awkward from the \ code continuation character

print(df_fixed['GarageCars'].value_counts()[0])
print(df_fixed['GarageArea'].value_counts()[0])
#     This code helps check if ONLY the null houses have zero's
```

- **Conclusion from Above** : All GarageTypes == Null are also Null in GarageYrBlt and 0 in GarageCars and GarageArea. This is important because we are certain there are NO NON-NULL GarageType houses with NULL or 0's in the other columns. (Thank Goodness)

```python
# Let's fillna GarageType and GarageYrBlt
df_fixed['GarageType'] = df_fixed['GarageType'].fillna(value='NoG')

# Hang On. What value to impute into a numerical column...let's go with zero(for now..we're just dealing with
# missing values)
df_fixed['GarageYrBlt'] = df_fixed['GarageYrBlt'].fillna(value=0)

# Print Check
print(df_fixed['GarageType'].value_counts())
print(df_fixed['GarageYrBlt'].value_counts())
```

##### BsmtQual

```python
# BsmtQual name annoys me. Change to name 'BsmtHeight'
df_fixed.rename(columns={'BsmtQual':'BsmtHeight'},inplace=True)
```

```python
# Filling NA's
df_fixed['BsmtHeight'] = df_fixed['BsmtHeight'].fillna(value='NO')
# Print Check Proper renaming of BsmtHeight
print(df_fixed['BsmtHeight'].value_counts())
```

##### LotFrontage

```python
# Around 200-300 houses have null houses. I suspect because the data entry gave apartments a null instead of 
# A zero. Implying there may be zero values already
```

```python
df_fixed['LotFrontage'].min()
#     Wrong....
```

```python
# Null values are to be imputed with zero's. Further, encoding a new feature 'has_lotfrontage' may be
# Useful 

df_fixed['LotFrontage'] = df_fixed['LotFrontage'].fillna(value=0)
#     Replace null to 0

df_fixed['has_LotFrontage'] = df_fixed['LotFrontage'].apply(lambda x: 0 if x==0 else 1)
#     Create encoding feature
```

```python
# Check
print(df_fixed['LotFrontage'].min())
print(df_fixed['has_LotFrontage'].value_counts())
#     Nice
```

##### Alley

```python
# Replace nulls with 'No Alley'
df_fixed['Alley'] = df_fixed['Alley'].fillna(value='No Alley')

# Print Check
print(df_fixed['Alley'].value_counts())
```

```python
# Amount of houses with alleys is low. Maybe worthwhile to encode "Has_Alley"
df_fixed['Has_Alley'] = df_fixed['Alley'].apply(lambda x: 0 if x=='No Alley' else 1)

# Print Check Encoding is correct
print(df_fixed['Has_Alley'].value_counts())
```

#### Rewriting the MSSubClass Column

```python
# Rewriting the category to also improve readability
# Categories are changed from numbers into strings
# This will also ensure that this column will be interpreted as a category by models
df_fixed['MSSubClass_1'] = df_fixed['MSSubClass'].apply(lambda x : "1S 46_NEWER" if x == 20
                                                       else "1_S 45_OLDER" if x == 30
                                                       else "1_S WITH_ATTIC" if x == 40
                                                       else "1.5_S NOT_FIN" if x == 45
                                                       else "1.5_S FIN" if x == 50
                                                       else "2_S 46_NEWER" if x == 60
                                                       else "2_S 45_OLDER" if x == 70
                                                       else "2.5_S" if x == 75
                                                       else "M_S or_split" if x == 80
                                                       else "SPLIT" if x == 85
                                                       else "DUPLEX" if x == 90
                                                       else "PUD 1_S 46_NEWER" if x == 120
                                                       else "PUD 1.5_S" if x == 150
                                                       else "PUD 2_S" if x == 160
                                                       else "PUD M_S" if x == 180
                                                       else "FAM_CON_2" if x == 190
                                                       else np.nan)
#     Create a new column that replaces the the SubClass Number to a SubClass String
```

```python
# Print Check for successful rewriting by comparing value counts to original column
print(df_fixed['MSSubClass_1'].value_counts())
print(df_fixed['MSSubClass'].value_counts())
```

**LOOKS GOOD**


#### CONCLUSION

```python
df_fixed.info()
```


_**PERFECT**_


### EDA and FEATURE SELECTION/ENGINEERING

<!-- #region toc-hr-collapsed=true toc-nb-collapsed=true -->
#### Using Only pre-2010 housing data
<!-- #endregion -->

```python
# For unbias EDA Exploration
df_fixed_pre10 = df_fixed[df_fixed['YrSold']<2010]

# Print Check for proper extraction
print(df_fixed_pre10['YrSold'].max())
```

<!-- #region toc-hr-collapsed=true toc-nb-collapsed=true -->
#### CORRELATION MATRIX AND PPSCORE MATRIX
- Correlation matrix shows linear relationships
- PPSCORE is a library that shows how good a feature can predict the other using RandomForestClassifier and RandomForestRegression. Can intrepet non-linear relationships and understands asymetry (A prediction ability of B is not equal to the vice versa). Non Numerical features are also intrepeted.
<!-- #endregion -->

**CORRELATION MATRIX**

```python
# Create Matrix using a Heatmap of viewable size
plt.figure(figsize=(15,9))
sns.heatmap(data=df_fixed_pre10.select_dtypes(include='number').corr(),cmap='viridis',annot=True)
plt.show()
```

**PPSCORE MATRIX**

```python jupyter={"outputs_hidden": true}
import ppscore as pp
df_fixed_pre10_pp = pp.matrix(df=df_fixed_pre10)
#     pp.matrix creates a df of ppscores. We assign the ouput to a variable for further analysis
```

```python
# Created a Heatmap to easier understand the df
plt.figure(figsize=(13,9))
sns.heatmap(data=df_fixed_pre10_pp,cmap='viridis')
```

**We Interpret the heatmap as follows**:
A feature in the columns (bottom) predicts a feature in the rows (left) with such accuracy (ppscore, higher is better)


**FROM OUR MATRIX'S LET INVESTIGATE FEATURES THAT ARE POOR PREDICTORS OF SALES PRICE**

<!-- #region toc-hr-collapsed=true toc-nb-collapsed=true -->
#### EXPLORING POOR PREDICTIVE FEATURES
<!-- #endregion -->

**Poor Features Are**:  
**Corr < 0.5**
- MSSubClass
- LotFrontage
- LotArea
- 2ndFlrSF
- BsmtFullBath
- BsmtHalfBath
- HalfBath
- BedroomAbvGr
- KitchenAbvGr
- GarageYrBlt



##### MSSubClass

```python
# Visualising Relationship between LotFrontage and SalePrice
sns.set_style(style='white')
plt.figure(figsize=(10,4))
sns.scatterplot(x=df_fixed_pre10['SalePrice'],y=df_fixed_pre10['MSSubClass_1'])
plt.title("MSSubClass vs SalePrice")
plt.tight_layout()
```

Useful Observations in the Data
- 46_NEWER SalePrice's are higher than 45_OLDER SalePrice's
- 2_S SalePrice's are higher than 1_S SalePrice's
- PUD 1_S is more expensive than PUD 2_S

Therefore this column is useful (correberated with decent ppscore)


###### Potentially Useful Feature Engineering


Catergory for 1_S Houses

```python
one_story_cat = ['1_S WITH_ATTIC', '1_S 45_OLDER','PUD 1_S 46_NEWER','1S 46_NEWER']

df_fixed_pre10['Has_1_S'] = df_fixed_pre10['MSSubClass_1'].apply(lambda x: 1 if x in one_story_cat
                                                                else 0)
#     Apply A 1 if the house is 1 story, else a 0
```

Catergory for Old Houses

```python
old_house_cat = ['2_S 45_OLDER','1_S 45_OLDER']

df_fixed_pre10['Has_OLDER'] = df_fixed_pre10['MSSubClass_1'].apply(lambda x: 1 if x in old_house_cat
                                                                else 0)
#     Apply a 1 if the house is older than 1945 else 0
```

Checking PPSCORES OF NEW FEATURES

```python
print(f""" 
PPSCORES
MSSubClass_1: {pp.score(df_fixed_pre10,x='MSSubClass_1',y='SalePrice',task='classification')['ppscore']}
Has_1_S: {pp.score(df_fixed_pre10,x='Has_1_S',y='SalePrice',task='classification')['ppscore']}
Has_OLDER: {pp.score(df_fixed_pre10,x='Has_OLDER',y='SalePrice',task='classification')['ppscore']}
""")
```

**CONCLUSION**: PPSCORE of MSSubClass_1 is still better than engineered features. We keep this feature


##### LotFrontage

```python
# Visualising Relationship between LotFrontage and SalePrice
sns.set_style(style='white')
plt.figure(figsize=(10,4))
sns.scatterplot(x=df_fixed_pre10['LotFrontage'],y=df_fixed_pre10['SalePrice'])
plt.title("LotFrontage vs SalePrice")
plt.tight_layout()
```

Remove the Outliers for Improved Linear Model (LotFrontage > 150)

```python
outliers = df_fixed_pre10[df_fixed_pre10['LotFrontage']>150].index
#     Get index of all outlier houses

df_fixed_pre10 = df_fixed_pre10.drop(index=outliers,inplace=False)
```

<!-- #region toc-hr-collapsed=true toc-nb-collapsed=true -->
##### 2ndFlrSF
<!-- #endregion -->

```python
# Visualising Relationship between 2ndFlrSF and SalePrice
sns.set_style(style='white')
plt.figure(figsize=(10,4))
sns.scatterplot(x=df_fixed_pre10['2ndFlrSF'],y=df_fixed_pre10['SalePrice'])
plt.title("2ndFlrSF vs SalePrice")
plt.tight_layout()
```

```python
# 2ndFlrSF = 0 if property doesn't have it
# We keep the feature and encode "Has_2ndFlr"

df_fixed_pre10['Has_2ndFlr'] = df_fixed_pre10['2ndFlrSF'].apply(lambda x: 1 if x==0 else 0)
```

##### BsmtFullBath and BsmtHalfBath

```python
# Visualising Relationship between BsmtFullBath and SalePrice
sns.set_style(style='white')
plt.figure(figsize=(10,4))
sns.scatterplot(x=df_fixed_pre10['BsmtFullBath'],y=df_fixed_pre10['SalePrice'])
plt.title("BsmtFullBat vs SalePrice")
plt.tight_layout()
```

```python
# Visualising Relationship between BsmtHalfBath and SalePrice
sns.set_style(style='white')
plt.figure(figsize=(10,4))
sns.scatterplot(x=df_fixed_pre10['BsmtHalfBath'],y=df_fixed_pre10['SalePrice'])
plt.title("BsmtHalfBath vs SalePrice")
plt.tight_layout()
```

```python
print(df_fixed_pre10['BsmtFullBath'].value_counts())
#     House many houses have no basement  full bathroom

print(df_fixed_pre10['BsmtHalfBath'].value_counts())
#     House many houses have no basement  half bathroom
```

```python
# Let's encode 'BsmtBath' = total amount of Basement Bathrooms
# And 'Has_BsmtBath'
df_fixed_pre10['BsmtBath'] = df_fixed_pre10['BsmtFullBath'] + df_fixed_pre10['BsmtHalfBath']
#     Add up basement bath's for each property

df_fixed_pre10['Has_BsmtBath'] = df_fixed_pre10['BsmtBath'].apply(lambda x: 0 if x==0 else 1)
#     Encode if the property has a basement bath
```

##### BedroomAbvGr and KitchenAbvGr

```python
# Visualising Relationship between BedroomAbvGr and SalePrice
sns.set_style(style='white')
plt.figure(figsize=(10,4))
sns.scatterplot(x=df_fixed_pre10['BedroomAbvGr'],y=df_fixed_pre10['SalePrice'])
plt.title("BedroomAbvGr vs SalePrice")
plt.tight_layout()
```

```python
# Visualising Relationship between KitchenAbvGr and SalePrice
sns.set_style(style='white')
plt.figure(figsize=(10,4))
sns.scatterplot(x=df_fixed_pre10['KitchenAbvGr'],y=df_fixed_pre10['SalePrice'])
plt.title("KitchenAbvGr vs SalePrice")
plt.tight_layout()
```

```python
# Let's drop KitchenAbvGr 

df_fixed_pre10.drop(columns='KitchenAbvGr',inplace=True)
```

This feature we will keep because it will be usefule with the 'Has_Frontage' Feature

```python

```

##### GarageYrBlt

```python
# Visualising Relationship between GarageYrBlt and SalePrice
sns.set_style(style='white')
plt.figure(figsize=(10,4))
sns.scatterplot(x=df_fixed_pre10['GarageYrBlt'],y=df_fixed_pre10['SalePrice'])
plt.title("GarageYrBlt vs SalePrice")
plt.tight_layout()
```

```python
# Visualising Relationship between GarageYrBlt and SalePrice
sns.set_style(style='white')
plt.figure(figsize=(10,4))
sns.scatterplot(x=df_fixed_pre10[df_fixed_pre10['GarageYrBlt']>0]['GarageYrBlt'],y=df_fixed_pre10['SalePrice'])
plt.title("GarageYrBlt vs SalePrice")
plt.tight_layout()
```

```python
df_fixed_pre10['GarageYrBlt'].value_counts()
```

```python
# Remove all houses with no garage, model will be cleaner
no_garage = df_fixed_pre10[df_fixed_pre10['GarageYrBlt']==0].index
#     Get index of all no garage houses houses

df_fixed_pre10 = df_fixed_pre10.drop(index=no_garage,inplace=False)

```

<!-- #region toc-hr-collapsed=true toc-nb-collapsed=true -->
#### EXPLORING COLLINEARITY OF GARAGE FEATURES
<!-- #endregion -->

**Remove the Garage Area Feature. It's strongly correlates with a similar feature (Garage Cars), and houses advertise how many cars can fit in their garage.**

```python
df_fixed_pre10.drop(labels='GarageArea',axis=1,inplace=True)
```

<!-- #region toc-hr-collapsed=true toc-nb-collapsed=true -->
#### FINAL FEATURE SELECTION FROM BOTTOM UP EDA
<!-- #endregion -->

```python
df_fixed_pre10.drop('MSSubClass',axis=1,inplace=True)
#     Drop MSSubClass, because MSSubClass_1 exist
```

```python
for col in df_fixed_pre10.select_dtypes('number').columns:
    sns.set_style(style='dark')
    plt.figure(figsize=(10,4))
    sns.scatterplot(x=df_fixed_pre10[f'{col}'],y=df_fixed_pre10['SalePrice'])
    plt.title(f"{col} vs SalePrice")
    plt.tight_layout()
#     This code prints a scatter plot for each numerical feature
#     to SalePrice for Quicker Analysis
```

```python
# Removing Outliers in Lot Area >50000
area_outliers = df_fixed_pre10[df_fixed_pre10['LotArea']>50000].index
#     Grab index of houses with lot area >50 000

df_fixed_pre10.drop(labels=area_outliers,inplace=True)
#     Remove theses houses
```

```python
for col in df_fixed_pre10.select_dtypes('object').columns:
    sns.set_style(style='dark')
    plt.figure(figsize=(10,4))
    sns.boxplot(x=df_fixed_pre10[f'{col}'],y=df_fixed_pre10['SalePrice'])
    plt.title(f"{col} vs SalePrice")
    plt.tight_layout()
#     This code prints a box plot for each object(cat) feature
#     to SalePrice for Quicker Analysis 
```

```python
# Drop Alley, Utilities, and Street since there is little usefulness
df_fixed_pre10.drop(labels=['Alley','Utilities','Street'],axis=1,inplace=True)
#     Remove Alley and Utilities
```

## COMPARING LINEAR REGRESSION MODEL USING DIFFERENT TRAIN FEATURES
**WE COMPARED**:
- A LinearRegression Model
- A RidgeCV Model
- A LassoCV Model

**WITH**:
- Only Numerical Features
- Numerical Features with Enigneered and Category Features 



### Number Only Features

```python
numerical_features = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', '1stFlrSF',
       '2ndFlrSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageCars']
#     List of Numerical Features

X = df_fixed_pre10[numerical_features]
y = df_fixed_pre10['SalePrice']
#     Creating X and y 

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
#     Imports

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
#     scale X for Ridge and Lasso

lr = LinearRegression()
lr_ridge = RidgeCV(alphas=(0.1,1,5,10,50))
lr_lasso = LassoCV(alphas=(0.1,1,5,10,50))
#     Creating model objects

print(cross_val_score(lr,X_scaled,y=y,cv=5).mean(),cross_val_score(lr,X_scaled,y=y,cv=5).std())
print(cross_val_score(lr_ridge,X_scaled,y=y,cv=5).mean(),cross_val_score(lr_ridge,X_scaled,y=y,cv=5).std())
print(cross_val_score(lr_lasso,X_scaled,y=y,cv=5).mean(),cross_val_score(lr_ridge,X_scaled,y=y,cv=5).std())
#     Listing scores of models accuracy
```

### Number, Category, and Enigneered Features

```python
to_dummy = ['MSSubClass_1','MSZoning','LotShape','LandContour','LotConfig','LandSlope','Neighborhood',
            'Condition1','Condition2','BldgType','HouseStyle','Foundation','BsmtHeight','GarageType']
#     List of features to dummy
engineered_features = ['Has_Shed', 'has_LotFrontage', 'Has_Alley',
                       'Has_1_S', 'Has_OLDER', 'Has_2ndFlr', 'BsmtBath', 'Has_BsmtBath']
#     List of features are were engineered
df_fixed_pre10dummies = pd.get_dummies(df_fixed_pre10[to_dummy],drop_first=True)
#     Create dummies out of to_dummy features
X = pd.concat([df_fixed_pre10[numerical_features],df_fixed_pre10[engineered_features],df_fixed_pre10dummies],
             axis=1) 
#     X is a combines the numerical, engineered, dummied features
y = df_fixed_pre10['SalePrice']


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
#     Scale X for Ridge and Lasso

lr = LinearRegression()
lr_ridge = RidgeCV(alphas=(0.1,1,5,10,50))
lr_lasso = LassoCV(alphas=(0.1,1,5,10,50),max_iter=6000)
#     Create new model objects

print(cross_val_score(lr,X_scaled,y=y,cv=5).mean(),cross_val_score(lr,X_scaled,y=y,cv=5).std())
print(cross_val_score(lr_ridge,X_scaled,y=y,cv=5).mean(),cross_val_score(lr_ridge,X_scaled,y=y,cv=5).std())
print(cross_val_score(lr_lasso,X_scaled,y=y,cv=5).mean(),cross_val_score(lr_ridge,X_scaled,y=y,cv=5).std())
#     Print model scores
```

## RIDGECV WITH NUMERICAL, ENGINEERED AND CATEGORICAL FEATURES IS THE BEST MODEL


### **CREATE X_TRAIN, Y_TRAIN, Y_TEST AND Y_TRAIN DATA**  
ALL data munging and processing steps are recreated on a new dataset
Because when the dataset was split by YrSold = 2010, creating dummies afterwards caused a minimatch in the amount of features beween train and test data

**This code is a repeat of previous code** Except split by year is done at the end, after dummy creation

```python
data = pd.read_csv('./housing.csv')
```

```python
data = data[fixed_cols]
#     Creating fixed characteristics dataframe

data = data[(data['MSZoning']=='FV') | (data['MSZoning']=='RH') 
              | (data['MSZoning']=='RL') | (data['MSZoning']=='RP') | (data['MSZoning']=='RM')]
#     Keeping all houses belonging only to residental zones
```

```python
# Encode Has_Shed
data['Has_Shed'] = data['MiscFeature'].apply(lambda x : 1 if x=='Shed' else 0)

# Drop MiscFeature
data.drop(labels='MiscFeature',axis=1,inplace=True)

# Fill Null in Garagetype to NoG
data['GarageType'] = data['GarageType'].fillna(value='NoG')

# Fill Null in GarageYrBlt to 0
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(value=0)

# Renaming BsmtQual
data.rename(columns={'BsmtQual':'BsmtHeight'},inplace=True)
# Filling BsmtHeight Nulls
data['BsmtHeight'] = data['BsmtHeight'].fillna(value='NO')

# Replace null to 0
data['LotFrontage'] = data['LotFrontage'].fillna(value=0)

# Create encoding feature
data['has_LotFrontage'] = data['LotFrontage'].apply(lambda x: 0 if x==0 else 1)

# Replace nulls with 'No Alley'
data['Alley'] = data['Alley'].fillna(value='No Alley')

# encode "Has_Alley"
data['Has_Alley'] = data['Alley'].apply(lambda x: 0 if x=='No Alley' else 1)

# Rewriting MSSubClass_1
data['MSSubClass_1'] = data['MSSubClass'].apply(lambda x : "1S 46_NEWER" if x == 20
                                                       else "1_S 45_OLDER" if x == 30
                                                       else "1_S WITH_ATTIC" if x == 40
                                                       else "1.5_S NOT_FIN" if x == 45
                                                       else "1.5_S FIN" if x == 50
                                                       else "2_S 46_NEWER" if x == 60
                                                       else "2_S 45_OLDER" if x == 70
                                                       else "2.5_S" if x == 75
                                                       else "M_S or_split" if x == 80
                                                       else "SPLIT" if x == 85
                                                       else "DUPLEX" if x == 90
                                                       else "PUD 1_S 46_NEWER" if x == 120
                                                       else "PUD 1.5_S" if x == 150
                                                       else "PUD 2_S" if x == 160
                                                       else "PUD M_S" if x == 180
                                                       else "FAM_CON_2" if x == 190
                                                       else np.nan)
```

```python
data['Has_1_S'] = data['MSSubClass_1'].apply(lambda x: 1 if x in one_story_cat
                                                                else 0)
    # Create a column that encodes whether the house is a one story house
data['Has_OLDER'] = data['MSSubClass_1'].apply(lambda x: 1 if x in old_house_cat
                                                                else 0)
     # Create a column that encodes whether the house is older than 1946

# Removing LotFrontage Outliers 
outliers = data[data['LotFrontage']>150].index
    # Find index of these houses
data = data.drop(index=outliers,inplace=False)
    # Drop them

data['Has_2ndFlr'] = data['2ndFlrSF'].apply(lambda x: 1 if x==0 else 0)

data['BsmtBath'] = data['BsmtFullBath'] + data['BsmtHalfBath']
#     Add up basement bath's for each property

data['Has_BsmtBath'] = data['BsmtBath'].apply(lambda x: 0 if x==0 else 1)

data.drop(columns='KitchenAbvGr',inplace=True)

# Remove all houses with no garage, model will be cleaner
no_garage = data[data['GarageYrBlt']==0].index
#     Get index of all no garage houses houses

data = data.drop(index=no_garage,inplace=False)

data.drop(labels='GarageArea',axis=1,inplace=True)

data.drop('MSSubClass',axis=1,inplace=True)
#     Drop MSSubClass, because MSSubClass_1 exist

# Removing Outliers in Lot Area >50000
area_outliers = data[data['LotArea']>50000].index
#     Grab index of houses with lot area >50 000

data.drop(labels=area_outliers,inplace=True)
#     Remove theses houses

# Drop Alley, Utilities, and Street since there is little usefulness
data.drop(labels=['Alley','Utilities','Street'],axis=1,inplace=True)
#     Remove Alley and Utilities
```

```python
to_dummy = ['MSSubClass_1','MSZoning','LotShape','LandContour','LotConfig','LandSlope','Neighborhood',
            'Condition1','Condition2','BldgType','HouseStyle','Foundation','BsmtHeight','GarageType']

engineered_features = ['Has_Shed', 'has_LotFrontage', 'Has_Alley',
                       'Has_1_S', 'Has_OLDER', 'Has_2ndFlr', 'BsmtBath', 'Has_BsmtBath']

numerical_features = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', '1stFlrSF',
       '2ndFlrSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageCars','YrSold','SalePrice']

data_dummies = pd.get_dummies(data[to_dummy],drop_first=True)

data = pd.concat([data[numerical_features],data[engineered_features],data_dummies],axis=1)
```

```python
# Getting 2010 houses
data_pre10 = data[data['YrSold']<2010]
     # Grab only houses sold in 2010
data_10 = data[data['YrSold']==2010]
```

### **Creating RIDGECV LINEAR MODEL**

```python
X_train = data_pre10.drop('SalePrice',axis=1) 
y_train = data_pre10['SalePrice']


X_test = data_10.drop('SalePrice',axis=1)
y_test = data_10['SalePrice']
```

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

lr_ridge = RidgeCV(alphas=(np.logspace(0.1, 100, 1000)))

lr_ridge.fit(X_train_scaled,y_train)
```

### **Model Assessment**

```python
print(cross_val_score(lr_ridge,X_train_scaled,y_train,cv=10).mean())
print(cross_val_score(lr_ridge,X_train_scaled,y_train,cv=10).std())
```

```python
lr_ridge.score(X_test_scaled,y_test)
```

### **Model Visualisations**

```python
coef_values_ridgecv = pd.DataFrame(data={'Feature': X_train.columns,
                  'Coef_': lr_ridge.coef_})
#     Places all the coef_ to the side of their feature
plt.figure(figsize=(10,50))
sns.barplot(x=coef_values_ridgecv['Coef_'],y=coef_values_ridgecv['Feature'])
plt.title('Value of Feature on SalePrice',fontdict={'fontsize':17})
plt.xlabel(xlabel = 'Coef_ Value', fontdict={'fontsize':12})
plt.ylabel(ylabel = 'Feature', fontdict={'fontsize':15})
#     Create a plot to show it all
```

```python
top_bottom_10 = pd.concat([coef_values_ridgecv.sort_values('Coef_',ascending=False).head(10),
                          coef_values_ridgecv.sort_values('Coef_',ascending=False).tail(10)])
#     Grab's only the bottom 10 and 10 most impactful coef_
plt.figure(figsize=(10,13))
sns.barplot(x=top_bottom_10['Coef_'],y=top_bottom_10['Feature'])
plt.title('10 Most and Least Valuable Features on SalePrice',fontdict={'fontsize':17})
plt.xlabel(xlabel = 'Coef_ Value', fontdict={'fontsize':12})
plt.ylabel(ylabel = 'Feature', fontdict={'fontsize':15})
```

## Q1 Summary, Answers and Findings
- We performed data munging, feature selection and engineering on the dataset
- We **compared LinearRegression, RidgeCV, and LassoCV models** using either **data containing only numerical features or all selected numerical, categorical and engineered features.**
- Comparision was done using cross validation and we determined **the best model was to use all selected features on a RidgeCV model** (Using all features increased our cross validation scores by around 5%)

- We reprocessed the dataset to make it suitable to model train and testing. Data was manipulated to far during EDA and feature engineering to be useable. We repeated all munging and processing steps on the new dataset.
- We **trained a RIDGECV model with a wide range of alpha scores** on train data (pre2010 sales data)
- We **tested the fitted model on test data** (2010 sales data) with an **R2 score of 0.85.** (85% of varaince in data can be explained by our fitted model)

- We plotted the values of the models coefficents visually as they tell use the **impact of a feature on SalePrice.** We also plotted top 10 most impactful and most determental features to the SalePrice
- **What does a Coefficent Tell Us?**: It tells us that a **ONE UNIT Increase** of the Feature **changes the SalePrice by the Coefficent**. Although for Categorical Features, the change on the SalePrice is relative to the reference category. 

```python
top_bottom_10
```

# _


<img src="http://imgur.com/l5NasQj.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 2. Determine any value of *changeable* property characteristics unexplained by the *fixed* ones.

---

Now that you have a model that estimates the price of a house based on its static characteristics, we can move forward with part 2 and 3 of the plan: what are the costs/benefits of quality, condition, and renovations?

There are two specific requirements for these estimates:
1. The estimates of effects must be in terms of dollars added or subtracted from the house value. 
2. The effects must be on the variance in price remaining from the first model.

The residuals from the first model (training and testing) represent the variance in price unexplained by the fixed characteristics. Of that variance in price remaining, how much of it can be explained by the easy-to-change aspects of the property?

---

**Your goals:**
1. Evaluate the effect in dollars of the renovate-able features. 
- How would your company use this second model and its coefficients to determine whether they should buy a property or not? Explain how the company can use the two models you have built to determine if they can make money. 
- Investigate how much of the variance in price remaining is explained by these features.
- Do you trust your model? Should it be used to evaluate which properties to buy and fix up?

<!-- #region jupyter={"outputs_hidden": true} -->
## Plotting Residuals
<!-- #endregion -->

```python
residues = pd.DataFrame({'predictions':lr_ridge.predict(X_train_scaled),
                        'values':y_train})

residues['Property UnderSold By'] = residues['predictions'] - residues['values']
residues.head()
```

- **If the prediction are larger than the values .... the house is under valued**
- **If the predictions are lower than the values .... The house is over valued**

```python
sns.scatterplot(x=residues['values'],y=residues['predictions'])
# Quickly observation the quality of our model. 
# Looks Decent
```

## Data Munging and Cleaning

```python
fixable_data = pd.read_csv("./housing.csv")
```

```python
fixable_characteristics = ['OverallQual','OverallCond','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual',
              'ExterCond','BsmtCond', 'Heating','HeatingQC','CentralAir','KitchenQual',
               'GarageFinish','GarageQual','GarageCond',
                           'PavedDrive','SalePrice','YrSold']
#     List of fixable characeteristics and other data
fixable_data = fixable_data[fixable_characteristics] 
#     As shown

fixable_data = fixable_data.iloc[X_train.index]
#     Grab only the house's we have predicted the house price on.

fixable_data = pd.concat([fixable_data,residues],axis=1)
```

```python
fixable_data.info()
```

```python
# Fill MasVnrType and BsmtCond
fixable_data['MasVnrType'] = fixable_data['MasVnrType'].fillna(value='No')
fixable_data['BsmtCond'] = fixable_data['BsmtCond'].fillna(value='No')
```

## CREATING DUMMIES

```python
reference_category = fixable_data.select_dtypes('object').iloc[0]
reference_category
# For business use
```

```python
fixable_dummies = pd.get_dummies(data=fixable_data.select_dtypes('object'),drop_first=True)
#     Dummy categorical features
```

```python
final_fixable_data = pd.concat([fixable_data[['OverallQual','OverallCond']],fixable_dummies],
                              axis=1)
#     Create final data containing dummies and all predicitive features
```

## Creating Second Ridge CV Model
Ridge CV chosen cause of time

```python
X = final_fixable_data
y = fixable_data['Property UnderSold By']
# Creating X and y

lr_2_ridge = RidgeCV(alphas=(np.logspace(0.1, 100, 1000)))
#     Create second ridge CV object

lr_2_ridge.fit(X, y)
#     Fit the model
```

## Model Results

```python
print(cross_val_score(lr_2_ridge,X,y,cv=10).mean())
print(cross_val_score(lr_2_ridge,X,y,cv=10).std())
#     View scores
```

## Model Visualisations

```python
coef_values_ridgecv_2 = pd.DataFrame(data={'Feature': X.columns,
                  'Coef_': lr_2_ridge.coef_})
#     Pair the coef_ to it's feature in a dataframe

plt.figure(figsize=(10,50))
sns.barplot(x=coef_values_ridgecv_2['Coef_'],y=coef_values_ridgecv_2['Feature'])
plt.title('Value of Fixable Feature on SalePrice',fontdict={'fontsize':17})
plt.xlabel(xlabel = 'Coef_ Value', fontdict={'fontsize':12})
plt.ylabel(ylabel = 'Fixable Feature', fontdict={'fontsize':15})
#     Visualise the dataframe
```

```python
top_bottom_10_fixable = pd.concat([coef_values_ridgecv_2.sort_values('Coef_',ascending=False).head(10),
                          coef_values_ridgecv_2.sort_values('Coef_',ascending=False).tail(10)])
#     Grab the bottom and top 10 most impactful features
plt.figure(figsize=(10,13))
sns.barplot(x=top_bottom_10_fixable['Coef_'],y=top_bottom_10_fixable['Feature'])
plt.title('Top 20 Most Impactful Fixable Features on SalePrice Relative to Reference Feature',
          fontdict={'fontsize':17})
plt.xlabel(xlabel = 'Coef_ Value', fontdict={'fontsize':12})
plt.ylabel(ylabel = 'Feature', fontdict={'fontsize':15})
#     Visualise them
```

## Q2 Model Summary, Answers and Findings


- We constructed another Ridge CV Model using only data also only used in the first linear model. Since we are only looking to assess the coefficents of fixable features and not creating a stong predictive model

- Except for OverQual and OverCond, the Feature's Coefficents tell us **if that house had that feature, the SalePrice would change by the coefficent**, **RELATIVE** to the reference category

**Evaluate the effect in dollars of the renovate-able features** 

**How would your company use this second model and its coefficients to determine whether they should buy a property or not? Explain how the company can use the two models you have built to determine if they can make money.**   
- Our Company can use the coefficients to evaluate the impact of a renovation change on the SalePrice
- Just make sure you properly interpret the findings with the refernce category table
- **How to Use the Two Models:** The First Model allows use to idenfity houses that are undervalued. In conjuction with our understanding of the impact of coefficents, we can maximise our profits by *FLIPPING UNDERVALUED HOUSES THAT CAN BE RENOVATED WITH HIGH IMPACT FEATURES*

**Investigate how much of the variance in price remaining is explained by these features.**
- We only tuned the model to evaluate coefficents, so the only metric we have of the model is the cross_val_score. The cross_val_score mean = **0.11768220277065147**, std = **0.135670772693818**
- The fixable features can only explain VERY unreliably 11% of the SalePrice variance
- The remaining variance maybe variance that hasn't been captured by improving our model or variance that can't be explained by any model created by our existing data.

**Do you trust your model?**
- From a business prespective, I find the first model to be a useful tool to investigate houses that are undersold or oversold. If I had a well rounded domain knowledge in that property market, then I wouldn't be caught out by the a misprediction of the model, since I can corroborate/discuss the model's findings with my domain knowledge and expertise. 
- However, the second model's poor R2 score means it's risky to follow this model blindly, (maybe never use this model). The less riskiest way to test the accuracy of the coefficent's findings is to only apply the model's findings on properties will extreme flipping potential. (ie. MASSIVLY UNDERVALUED WITH STRONG RENOVATION POTENTIAL).
- In future, I would recreate the 2nd model using the SalePrice rather than the residual. 
- In future, I would also investigate the interesting relationships found by the PPSCORE MATRIX  The power of the PPSCORE Matrix is that you can find interesting observations in the data. With more time, we can investigate whether to use these findings to improve our model or improve our business decision making
- **A FEW INTERESTING OBSERVATIONS**: (1) A lot of features can predict the amount of full baths. (2) Neighborhood can predict well a few features. (3) The GarageType strongly predicts the Year the Garage was built. (4) YearBlt and GarageYrBlt predicts well the Foundation of the House. 

```python
# Created a Heatmap to easier understand the df
plt.figure(figsize=(13,9))
sns.heatmap(data=df_fixed_pre10_pp,cmap='viridis')
```

# _


<img src="http://imgur.com/GCAf1UX.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 3. What property characteristics predict an "abnormal" sale?

---

The `SaleCondition` feature indicates the circumstances of the house sale. From the data file, we can see that the possibilities are:

       Normal	Normal Sale
       Abnorml	Abnormal Sale -  trade, foreclosure, short sale
       AdjLand	Adjoining Land Purchase
       Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
       Family	Sale between family members
       Partial	Home was not completed when last assessed (associated with New Homes)
       
One of the executives at your company has an "in" with higher-ups at the major regional bank. His friends at the bank have made him a proposal: if he can reliably indicate what features, if any, predict "abnormal" sales (foreclosures, short sales, etc.), then in return the bank will give him first dibs on the pre-auction purchase of those properties (at a dirt-cheap price).

He has tasked you with determining (and adequately validating) which features of a property predict this type of sale. 

---

**Your task:**
1. Determine which features predict the `Abnorml` category in the `SaleCondition` feature.
- Justify your results.

This is a challenging task that tests your ability to perform classification analysis in the face of severe class imbalance. You may find that simply running a classifier on the full dataset to predict the category ends up useless: when there is bad class imbalance classifiers often tend to simply guess the majority class.

It is up to you to determine how you will tackle this problem. I recommend doing some research to find out how others have dealt with the problem in the past. Make sure to justify your solution. Don't worry about it being "the best" solution, but be rigorous.

Be sure to indicate which features are predictive (if any) and whether they are positive or negative predictors of abnormal sales.

```python jupyter={"outputs_hidden": true}
# A:
```
