#!/usr/bin/env python
# coding: utf-8

# # # House Prices 
# **Autor:** Valérie RUSSO
# 
# **Kaggle Competition:** House Prices - Advanced Regression Techniques
# * This notebook explores several regression models and ensemble techniques to predict house prices.
# 
# ## Approach
# - Feature engineering
# - Cross-validation
# - Hyperparameter tuning
# - Advanced models (XGBoost, LightGBM)
# - Model ensembling
# 

# # 1. Data Loading

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# In[2]:


from sklearn import set_config
set_config(transform_output="pandas")


# In[3]:


train_val = pd.read_csv(r"C:\Workspace\github_repository\FORMATIONS\Kaggle\House Prices\data\train.csv")
train_val.head(5)


# In[4]:


X_test = pd.read_csv(r"C:\Workspace\github_repository\FORMATIONS\Kaggle\House Prices\data\test.csv")
X_test.head(5)


# In[5]:


modele = pd.read_csv(r"C:\Workspace\github_repository\FORMATIONS\Kaggle\House Prices\data\sample_submission.csv")
modele.head(5)


# In[6]:


X_train_val=train_val.drop("SalePrice", axis=1)


# In[7]:


# Transformation of the Id column into index
X_train_val.set_index("Id", inplace=True)
X_test.set_index("Id", inplace=True)


# In[8]:


y_train_val = train_val.set_index("Id")["SalePrice"]
y_train_val.mean()

# Distribution of the target variable
import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(y_train_val, kde=True)
plt.title("Distribution of the target variable - SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.show()


# In[9]:


X_train_val.info()


# # 2. Data cleaning and Feature engineering

# In[10]:


# MSSubClass: The building class is a categorical variable, not numeric from data description. It needs to be converted to a categorical variable.
# Same for OverallQual and OverallCond, which are ratings from 1 to 10,
# but they represent categories of quality and condition, not continuous numeric values.
X_train_val["MSSubClass cat"] = X_train_val["MSSubClass"].astype(str)
X_test["MSSubClass cat"] = X_test["MSSubClass"].astype(str)

# Put those wrong numerical features in a list, and remove them from the list of numerical features
wrong_numerical_features = ["MSSubClass"]


# In[11]:


X_train_val.drop(columns=wrong_numerical_features, inplace=True)
X_test.drop(columns=wrong_numerical_features, inplace=True)


# In[12]:


# Checking the percentage of missing values for numerical features
missing_values = X_train_val.isnull().sum()
missing_percentage = (missing_values / len(X_train_val)) * 100
print("Percentage of missing values by column:")
print(missing_percentage[missing_percentage > 0])


# In[13]:


ordinal_features = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
                    "BsmtFinType1","BsmtFinType2", "Functional", "GarageFinish",
                    "PavedDrive", "LandSlope", "BsmtExposure", "OverallQual", "OverallCond"]


# In[14]:


# Checking the percentage of missing values for ordinal features
for col in ordinal_features:
    X_train_val[col] = X_train_val[col].fillna("NA")
    X_test[col] = X_test[col].fillna("NA")


# In[15]:


# Affiche toutes les colonnes numériques de X_train_val
numerical_features=X_train_val.select_dtypes(include=["int64", "float64"]).columns.tolist()
numerical_features


# In[16]:


# Looking into data description, transformation of ordinal data into numerical data is needed when there is an order in the categories.
# For example, for the feature "ExterQual" (Exterior material quality), the categories are "Ex", "Gd", "TA", "Fa", and "Po", which represent different levels of quality. We can map these categories to numerical values based on their order of quality, such as Ex=5, Gd=4, TA=3, Fa=2, Po=1. 
# This way, we can capture the ordinal relationship between the categories and use it in our model.
# Manual mapping for ordinal features is necessary to preserve the order of categories and ensure that the model can learn from the ordinal relationships in the data.
# Ordinal data will be int data type
quality_mapping = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
X_train_val["ExterQual num"] = X_train_val["ExterQual"].map(quality_mapping)
X_test["ExterQual num"] = X_test["ExterQual"].map(quality_mapping)
X_train_val["ExterCond num"] = X_train_val["ExterCond"].map(quality_mapping)
X_test["ExterCond num"] = X_test["ExterCond"].map(quality_mapping)
X_train_val["BsmtQual num"] = X_train_val["BsmtQual"].map(quality_mapping)
X_test["BsmtQual num"] = X_test["BsmtQual"].map(quality_mapping)
X_train_val["BsmtCond num"] = X_train_val["BsmtCond"].map(quality_mapping)
X_test["BsmtCond num"] = X_test["BsmtCond"].map(quality_mapping)
X_train_val["HeatingQC num"] = X_train_val["HeatingQC"].map(quality_mapping)
X_test["HeatingQC num"] = X_test["HeatingQC"].map(quality_mapping)
X_train_val["KitchenQual num"] = X_train_val["KitchenQual"].map(quality_mapping)
X_test["KitchenQual num"] = X_test["KitchenQual"].map(quality_mapping)
X_train_val["FireplaceQu num"] = X_train_val["FireplaceQu"].map(quality_mapping)
X_test["FireplaceQu num"] = X_test["FireplaceQu"].map(quality_mapping)
X_train_val["GarageQual num"] = X_train_val["GarageQual"].map(quality_mapping)
X_test["GarageQual num"] = X_test["GarageQual"].map(quality_mapping)
X_train_val["GarageCond num"] = X_train_val["GarageCond"].map(quality_mapping)
X_test["GarageCond num"] = X_test["GarageCond"].map(quality_mapping)
X_train_val["PoolQC num"] = X_train_val["PoolQC"].map(quality_mapping)
X_test["PoolQC num"] = X_test["PoolQC"].map(quality_mapping)
# Other mapping for features with different categories
exposure_mapping = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "NA": 0}
X_train_val["BsmtExposure num"] = X_train_val["BsmtExposure"].map(exposure_mapping)
X_test["BsmtExposure num"] = X_test["BsmtExposure"].map(exposure_mapping)
# Other mapping for features with different categories
bsmtf_mapping = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "NA": 0}
X_train_val["BsmtFinType1 num"] = X_train_val["BsmtFinType1"].map(bsmtf_mapping)
X_test["BsmtFinType1 num"] = X_test["BsmtFinType1"].map(bsmtf_mapping)
X_train_val["BsmtFinType2 num"] = X_train_val["BsmtFinType2"].map(bsmtf_mapping)
X_test["BsmtFinType2 num"] = X_test["BsmtFinType2"].map(bsmtf_mapping)
# Other mapping for Functional feature
functional_mapping = {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0}
X_train_val["Functional num"] = X_train_val["Functional"].map(functional_mapping)
X_test["Functional num"] = X_test["Functional"].map(functional_mapping)
# Other mapping for GarageFinish feature
garagefinish_mapping = {"Fin": 3, "RFn": 2, "Unf": 1, "NA": 0}
X_train_val["GarageFinish num"] = X_train_val["GarageFinish"].map(garagefinish_mapping)
X_test["GarageFinish num"] = X_test["GarageFinish"].map(garagefinish_mapping)
# Other mapping for PavedDrive feature
paveddrive_mapping = {"Y": 3, "P": 2, "N": 1}
X_train_val["PavedDrive num"] = X_train_val["PavedDrive"].map(paveddrive_mapping)
X_test["PavedDrive num"] = X_test["PavedDrive"].map(paveddrive_mapping)
# Other mapping for LandSlope feature
landslope_mapping = {"Gtl": 3, "Mod": 2, "Sev": 1}
X_train_val["LandSlope num"] = X_train_val["LandSlope"].map(landslope_mapping)
X_test["LandSlope num"] = X_test["LandSlope"].map(landslope_mapping)
#mapping for OverallQual and OverallCond features
overall_mapping = {10: 10, 9: 9, 8: 8, 7: 7, 6: 6, 5: 5, 4: 4, 3: 3, 2: 2, 1: 1, 0: 0}
X_train_val["OverallQual num"] = X_train_val["OverallQual"].map(overall_mapping)
X_test["OverallQual num"] = X_test["OverallQual"].map(overall_mapping)
X_train_val["OverallCond num"] = X_train_val["OverallCond"].map(overall_mapping)
X_test["OverallCond num"] = X_test["OverallCond"].map(overall_mapping)


# In[17]:


# deletion of the original ordinal features after transformation
X_train_val.drop(columns=ordinal_features, inplace=True)
X_test.drop(columns=ordinal_features, inplace=True)

#update of the list of numerical features after transformation
numerical_features = X_train_val.select_dtypes(include=["int64", "float64"]).columns.tolist()
#update of the list of categorical features after transformation
categorical_features = X_train_val.select_dtypes(include=["object", "string"]).columns.tolist()
#update of ordinal features list after transformation
ordinal_features = [feature + " num" for feature in ordinal_features]
ordinal_features


# In[18]:


X_train_val.info()


# In[19]:


#number of different values for each  feature
for col in X_train_val.columns:
    num_unique = X_train_val[col].nunique()
    if num_unique <=1:
        print(f"{col}: {num_unique} unique values")
    else:
        print(f"More than 1 value: {col}: {num_unique} unique values, {X_train_val[col].dtype}")


# In[20]:


# autotmatic test to check in categorical features if there are low variance features (one category is dominant) 
# or high missing features (most values are missing)
threshold_unique = 0.99
threshold_missing = 0.8
low_variance_cat_cols = []
high_missing_cols = []
for col in X_train_val.select_dtypes(include=["object", "string"]).columns:
    top_freq = X_train_val[col].value_counts(normalize=True).iloc[0]
    if top_freq > threshold_unique:
        low_variance_cat_cols.append(col)
    if X_train_val[col].isnull().mean() > threshold_missing:
        high_missing_cols.append(col)
print("Low variance categorical columns:", low_variance_cat_cols)
print("High missing categorical columns:", high_missing_cols)


# In[21]:


# deletion of the Low variance categorical columns and high missing categorical columns
X_train_val.drop(columns=low_variance_cat_cols + high_missing_cols, inplace=True)
X_test.drop(columns=low_variance_cat_cols + high_missing_cols, inplace=True)


# In[22]:


#categorical features without ordinal features :
categorical_features = X_train_val.select_dtypes(include=["object", "string"]).columns.tolist()
categorical_features


# In[23]:


# Correlation values for numerical features and ordinal features with the target variable
cols_to_include = list(numerical_features) + ["SalePrice"]
cols_to_include


# In[24]:


df_corr = (
    X_train_val
    .assign(SalePrice=y_train_val.values)[cols_to_include]
)
corr = df_corr.corr()["SalePrice"].sort_values(ascending=False)

print(corr.head(35))


# In[25]:


numerical_top_features = corr[abs(corr)>0.2].index.drop("SalePrice")
numerical_top_features


# In[26]:


# Transformation of the target variable with log to reduce the effect of extreme values
y_train_val_log = np.log1p(y_train_val)

# Distribution of the transformed target variable
sns.histplot(y_train_val_log, kde=True)
plt.title("Distribution of the transformed target variable - log(SalePrice)")
plt.xlabel("log(SalePrice)")
plt.ylabel("Frequency")
plt.show()


# In[27]:


continuous_numerical_features = [col for col in numerical_top_features if X_train_val[col].nunique() > 20]


# In[28]:


# Scatter plots for numerical features with high correlation - 3 plots per row
import math

num_features = len(continuous_numerical_features)
cols = 3

if num_features == 0:
    print("No contuinuous numerical features")
else:
    rows = math.ceil(num_features / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
    axes = np.array(axes).reshape(-1)

    for i, feature in enumerate(continuous_numerical_features):
        sns.scatterplot(
            x=X_train_val[feature],
            y=y_train_val_log,
            ax=axes[i]
        )
        axes[i].set_title(f'SalePrice vs {feature} (corr={corr[feature]:.2f})')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('SalePrice')

    # Remove unused axes when number of features is not a multiple of 3
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# In[29]:


# We have to separate binary features from continuous numerical features, 
# because they need to be treated differently in the preprocessing step. 
# We have to create binary features from continuous numerical features with a threshold, to capture the presence or absence of a feature. For example, for the feature "GarageCars", which represents the number of cars that can fit in the garage, we can create a binary feature "HasGarage" that indicates whether the house has a garage or not (1 if GarageCars > 0, 0 otherwise). This way, we can capture the effect of having a garage on the house price, regardless of the number of cars it can fit.

# Binary feature for GarageArea
X_train_val["HasGarage"] = (X_train_val["GarageArea"] > 0).astype(int)
X_test["HasGarage"] = (X_test["GarageArea"] > 0).astype(int)
# Binary feature for TotalBsmtSF
X_train_val["HasBasement"] = (X_train_val["TotalBsmtSF"] > 0).astype(int)
X_test["HasBasement"] = (X_test["TotalBsmtSF"] > 0).astype(int)
# Binary feature for YearRemodAdd
X_train_val["Remodeled"] = (X_train_val["YearRemodAdd"] != X_train_val["YearBuilt"]).astype(int)
X_test["Remodeled"] = (X_test["YearRemodAdd"] != X_test["YearBuilt"]).astype(int)
# Binary feature for MasVnrArea
X_train_val["HasMasVnr"] = (X_train_val["MasVnrArea"] > 0).astype(int)
X_test["HasMasVnr"] = (X_test["MasVnrArea"] > 0).astype(int)
# Binary feature for BsmtFinSF1
X_train_val["HasFinSF1"] = (X_train_val["BsmtFinSF1"] > 0).astype(int)
X_test["HasFinSF1"] = (X_test["BsmtFinSF1"] > 0).astype(int)
# Binary feature for LotFrontage
X_train_val["HasLotFrontage"] = (X_train_val["LotFrontage"] > 0).astype(int)
X_test["HasLotFrontage"] = (X_test["LotFrontage"] > 0).astype(int)
# Binary feature for WoodDeckSF
X_train_val["HasWoodDeck"] = (X_train_val["WoodDeckSF"] > 0).astype(int)
X_test["HasWoodDeck"] = (X_test["WoodDeckSF"] > 0).astype(int)
# Binary feature for 2ndFlrSF
X_train_val["Has2ndFlr"] = (X_train_val["2ndFlrSF"] > 0).astype(int)
X_test["Has2ndFlr"] = (X_test["2ndFlrSF"] > 0).astype(int)
# Binary feature for OpenPorchSF
X_train_val["HasOpenPorch"] = (X_train_val["OpenPorchSF"] > 0).astype(int)
X_test["HasOpenPorch"] = (X_test["OpenPorchSF"] > 0).astype(int)
# Binary feature for LotArea
X_train_val["HasLotArea"] = (X_train_val["LotArea"] > 0).astype(int)
X_test["HasLotArea"] = (X_test["LotArea"] > 0).astype(int)
#Binary feature for BsmtUnfSF
X_train_val["HasBsmtUnf"] = (X_train_val["BsmtUnfSF"] > 0).astype(int)
X_test["HasBsmtUnf"] = (X_test["BsmtUnfSF"] > 0).astype(int)


# In[30]:


# update of the list of numerical features
numerical_features = X_train_val.select_dtypes(include=["int64", "float64"]).columns.tolist()
numerical_features


# In[31]:


# Deletion of outliers for the feature "GrLivArea" (Above ground living area square footage) which has a strong correlation with the target variable and contains some extreme values that can negatively impact the model's performance. 
# By removing these outliers, we can improve the model's ability to learn from the data and make more accurate predictions.

#Supression of outliers
mask = X_train_val["GrLivArea"] < 4000
X_train_val = X_train_val[mask]
y_train_val = y_train_val[mask]
y_train_val_log = y_train_val_log[mask]

mask = X_train_val["TotalBsmtSF"] < 3000
X_train_val = X_train_val[mask]
y_train_val = y_train_val[mask]
y_train_val_log = y_train_val_log[mask]

mask = X_train_val["1stFlrSF"] < 3000
X_train_val = X_train_val[mask]
y_train_val = y_train_val[mask]
y_train_val_log = y_train_val_log[mask]

mask = X_train_val["GarageArea"] < 1200
X_train_val = X_train_val[mask]
y_train_val = y_train_val[mask]
y_train_val_log = y_train_val_log[mask]

mask = X_train_val["LotArea"] < 100000
X_train_val = X_train_val[mask]
y_train_val = y_train_val[mask]
y_train_val_log = y_train_val_log[mask]

mask = X_train_val["MasVnrArea"] < 1000
X_train_val = X_train_val[mask]
y_train_val = y_train_val[mask]
y_train_val_log = y_train_val_log[mask]

mask = X_train_val["WoodDeckSF"] < 1000
X_train_val = X_train_val[mask]
y_train_val = y_train_val[mask]
y_train_val_log = y_train_val_log[mask]

mask = X_train_val["OpenPorchSF"] < 500
X_train_val = X_train_val[mask]
y_train_val = y_train_val[mask]
y_train_val_log = y_train_val_log[mask]

mask = X_train_val["LotFrontage"] < 200
X_train_val = X_train_val[mask]
y_train_val = y_train_val[mask]
y_train_val_log = y_train_val_log[mask]

mask = X_train_val["BsmtFinSF1"] < 3000
X_train_val = X_train_val[mask]
y_train_val = y_train_val[mask]
y_train_val_log = y_train_val_log[mask]


# In[32]:


# Histogram for continuous numerical features - 3 plots per row

import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import skew

# Nombre total de colonnes numériques
num_cols = len(continuous_numerical_features)

# 3 graphiques par ligne
cols = 3
rows = math.ceil(num_cols / cols)

# Créer la grille de sous-graphiques
fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
axes = axes.flatten()

# Boucle sur les colonnes numériques
for i, col in enumerate(continuous_numerical_features):
    series = X_train_val[col].dropna()

    sns.histplot(
        series,
        bins=10,
        kde=True,                  # Ajoute la courbe de densité (lissage)
        color='skyblue',
        edgecolor='black',
        ax=axes[i]
    )

    # Affiche la skewness pour les colonnes numériques
    if pd.api.types.is_numeric_dtype(series):
        sk = skew(series)
        axes[i].set_title(f'Distribution of {col} | skewness={sk:.2f}', fontsize=11)
    else:
        axes[i].set_title(f'Distribution of {col} | skewness=N/A', fontsize=11)

    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Supprimer les axes inutilisés
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[33]:


# Box plot for all ordinal features - 3 plots per row
num_cat_features = len(ordinal_features)
cols = 3
rows = math.ceil(num_cat_features / cols)

# Create subplot grid
fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
axes = np.array(axes).reshape(-1)

# Loop through ordinal features
for i, feature in enumerate(ordinal_features):
    if feature not in X_train_val.columns:
        continue

    plot_df = (
        X_train_val[[feature]]
        .join(y_train_val.rename("SalePrice"), how="inner")
        .dropna(subset=[feature, "SalePrice"])
    )

    if plot_df.empty:
        continue

    sns.boxplot(data=plot_df, x=feature, y="SalePrice", ax=axes[i])
    axes[i].set_title(f"SalePrice vs {feature}", fontsize=11)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("SalePrice")
    axes[i].tick_params(axis="x", rotation=45)

# Remove unused axes
last_i = i if len(ordinal_features) > 0 else -1
for j in range(last_i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[34]:


#diplay KitchenAbvGr type
print(train_val.KitchenAbvGr.dtype)


# In[35]:


#Display the percentage for each values for the feature "KitchenAbvGr" to check if it is a binary feature or not
print(train_val.KitchenAbvGr.value_counts(normalize=True) * 100)

#display correlation between KitchenAbvGr and the target variable
corr_kitchenabvgr = X_train_val["KitchenAbvGr"].corr(y_train_val)
print(f"Correlation between KitchenAbvGr and SalePrice: {corr_kitchenabvgr:.2f}")


# In[36]:


# Transformation of the feature "KitchenAbvGr" into several binary features "HasKitchenAbvGr" that indicates whether
# the house has more than one kitchen above ground or zero above ground
X_train_val["HasKitchenAbvGr"] = (X_train_val["KitchenAbvGr"] > 1).astype(int)
X_test["HasKitchenAbvGr"] = (X_test["KitchenAbvGr"] > 1).astype(int)
X_train_val["HasNoKitchenAbvGr"] = (X_train_val["KitchenAbvGr"] == 0).astype(int)
X_test["HasNoKitchenAbvGr"] = (X_test["KitchenAbvGr"] == 0).astype(int)
#deletion of the original feature "KitchenAbvGr" after transformation
X_train_val.drop(columns=["KitchenAbvGr"], inplace=True)
X_test.drop(columns=["KitchenAbvGr"], inplace=True)


# In[37]:


# Numerical feature list update
numerical_features = X_train_val.select_dtypes(include=["int64", "float64"]).columns.tolist()
print("Updated numerical features:", numerical_features)


# In[38]:


# Categorical feature list update
categorical_features = X_train_val.select_dtypes(include=["object", "string"]).columns.tolist()
categorical_features


# In[39]:


# Box plot for all categorical features except ordinal features - 3 plots per row
num_cat_features = len(categorical_features) 
cols = 3
rows = math.ceil(num_cat_features / cols)

# Create subplot grid
fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
axes = axes.flatten()

# Loop through categorical features
for i, feature in enumerate(categorical_features):
    sns.boxplot(x=X_train_val[feature], y=y_train_val, ax=axes[i])
    axes[i].set_title(f'SalePrice vs {feature}', fontsize=11)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('SalePrice')
    axes[i].tick_params(axis='x', rotation=45)

# Remove unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[40]:


# Transformation of CentralAir feature into a binary feature "HasCentralAir" 
# that indicates whether the house has central air conditioning or not
X_train_val["HasCentralAir"] = (X_train_val["CentralAir"] == "Y").astype(int)
X_test["HasCentralAir"] = (X_test["CentralAir"] == "Y").astype(int)
# Deletion of the original feature "CentralAir" after transformation
X_train_val.drop(columns=["CentralAir"], inplace=True)
X_test.drop(columns=["CentralAir"], inplace=True)
#update of the list of categorical features after transformation
categorical_features = X_train_val.select_dtypes(include=["object", "string"]).columns.tolist()
#update of the list of numerical features after transformation
numerical_features = X_train_val.select_dtypes(include=["int64", "float64"]).columns.tolist()


# In[41]:


# Transormation of LotShape feature into ordonal feature with mapping based on the order of the categories in data description
lotshape_mapping = {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1}
X_train_val["LotShape num"] = X_train_val["LotShape"].map(lotshape_mapping)
X_test["LotShape num"] = X_test["LotShape"].map(lotshape_mapping)
# Deletion of the original feature "LotShape" after transformation
X_train_val.drop(columns=["LotShape"], inplace=True)
X_test.drop(columns=["LotShape"], inplace=True)
#update of the list of categorical features after transformation
categorical_features = X_train_val.select_dtypes(include=["object", "string"]).columns.tolist()
#update of the list of numerical features after transformation
numerical_features = X_train_val.select_dtypes(include=["int64", "float64"]).columns.tolist()


# In[42]:


X_train_val[numerical_features+categorical_features].head(5)


# In[43]:


feature_list = X_train_val.columns.to_list()
feature_list


# In[44]:


# Total living area (basement + floors)
X_train_val["TotalSF"] = X_train_val["TotalBsmtSF"] + X_train_val["1stFlrSF"] + X_train_val["2ndFlrSF"]
X_test["TotalSF"] = X_test["TotalBsmtSF"] + X_test["1stFlrSF"] + X_test["2ndFlrSF"]
# Age of the house
X_train_val["HouseAge"] = X_train_val["YrSold"] - X_train_val["YearBuilt"]
X_test["HouseAge"] = X_test["YrSold"] - X_test["YearBuilt"]
# Total number of bathrooms
X_train_val["TotalBath"] = X_train_val["FullBath"] + (0.5 * X_train_val["HalfBath"])+ X_train_val["BsmtFullBath"] + (0.5 * X_train_val["BsmtHalfBath"])
X_test["TotalBath"] = X_test["FullBath"] + (0.5 * X_test["HalfBath"])+ X_test["BsmtFullBath"] + (0.5 * X_test["BsmtHalfBath"])
# Lot area minus house area
X_train_val["LotAreaMinusSF"] = X_train_val["LotArea"] - X_train_val["1stFlrSF"]
X_test["LotAreaMinusSF"] = X_test["LotArea"] - X_test["1stFlrSF"]
# Overall quality weighted by house age
X_train_val["OverallQualAge"] = X_train_val["OverallQual num"] / (X_train_val["HouseAge"] + 1)
X_test["OverallQualAge"] = X_test["OverallQual num"] / (X_test["HouseAge"] + 1)
# Quality * area
X_train_val["QualSF"] = X_train_val["OverallQual num"] * X_train_val["TotalSF"]
X_test["QualSF"] = X_test["OverallQual num"] * X_test["TotalSF"]
# Total surface with garage
X_train_val["TotalSFWithGarage"] = X_train_val["TotalSF"] + X_train_val["GarageArea"]
X_test["TotalSFWithGarage"] = X_test["TotalSF"] + X_test["GarageArea"]
# Total surface with porch
X_train_val["TotalSFWithPorch"] = X_train_val["TotalSF"] + X_train_val["OpenPorchSF"] + X_train_val["EnclosedPorch"] + X_train_val["3SsnPorch"] + X_train_val["ScreenPorch"]
X_test["TotalSFWithPorch"] = X_test["TotalSF"] + X_test["OpenPorchSF"] + X_test["EnclosedPorch"] + X_test["3SsnPorch"] + X_test["ScreenPorch"]
# Total number of rooms
X_train_val["TotalRoomSF"] = X_train_val["TotRmsAbvGrd"] + X_train_val["FullBath"] + X_train_val["HalfBath"] + X_train_val["BsmtFullBath"] + X_train_val["BsmtHalfBath"]
X_test["TotalRoomSF"] = X_test["TotRmsAbvGrd"] + X_test["FullBath"] + X_test["HalfBath"] + X_test["BsmtFullBath"] + X_test["BsmtHalfBath"]
# Age after remodeling
X_train_val["AgeAfterRemod"] = X_train_val["YrSold"] - X_train_val["YearRemodAdd"]
X_test["AgeAfterRemod"] = X_test["YrSold"] - X_test["YearRemodAdd"]


# In[45]:


X_train_val = X_train_val.replace([np.inf, -np.inf], np.nan)


# In[46]:


X_test = X_test.replace([np.inf, -np.inf], np.nan)


# In[47]:


continuous_numerical_features = [col for col in numerical_top_features if X_train_val[col].nunique() > 20]


# In[48]:


from scipy.stats import skew

# Identify numerical features with high skewness (threshold = 1)
skewness = X_train_val[continuous_numerical_features].apply(lambda col: skew(col.dropna()))
skewed_cols = skewness[abs(skewness) > 1].index.tolist()

print(f"Colonnes transformées avec log1p ({len(skewed_cols)}) :")
print(skewed_cols)

# Apply log1p to skewed columns (log1p handles zeros safely)
X_train_val[skewed_cols] = np.log1p(X_train_val[skewed_cols].clip(lower=0))
X_test[skewed_cols] = np.log1p(X_test[skewed_cols].clip(lower=0))


# In[49]:


continuous_numerical_features = [col for col in numerical_top_features if X_train_val[col].nunique() > 20]


# In[50]:


# Histogram for log skewed numerical features - 3 plots per row

import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import skew

# Nombre total de colonnes numériques
num_cols = len(skewed_cols)

# 3 graphiques par ligne
cols = 3
rows = math.ceil(num_cols / cols)

# Créer la grille de sous-graphiques
fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
axes = axes.flatten()

# Boucle sur les colonnes numériques
for i, col in enumerate(skewed_cols):
    series = X_train_val[col].dropna()

    sns.histplot(
        series,
        bins=10,
        kde=True,                  # Ajoute la courbe de densité (lissage)
        color='skyblue',
        edgecolor='black',
        ax=axes[i]
    )

    # Affiche la skewness pour les colonnes numériques
    if pd.api.types.is_numeric_dtype(series):
        sk = skew(series)
        axes[i].set_title(f'Distribution of {col} | skewness={sk:.2f}', fontsize=11)
    else:
        axes[i].set_title(f'Distribution of {col} | skewness=N/A', fontsize=11)

    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Supprimer les axes inutilisés
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[51]:


# Scatter plots for log skewed numerical features with high correlation - 3 plots per row
import math
num_skewed_features = len(skewed_cols)
cols = 3

if num_skewed_features == 0:
    print("No feature skewed")
else:
    rows = math.ceil(num_skewed_features / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
    axes = np.array(axes).reshape(-1)

    for i, feature in enumerate(skewed_cols):
        sns.scatterplot(
            x=X_train_val[feature],
            y=y_train_val_log,
            ax=axes[i]
        )
        axes[i].set_title(f'SalePrice vs {feature} (corr={corr[feature]:.2f})')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('SalePrice')

    # Remove unused axes when number of features is not a multiple of 3
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# In[52]:


# Remove constant features (including columns that are entirely NaN)
constant_cols = [col for col in X_train_val[numerical_features+categorical_features].columns if X_train_val[col].nunique(dropna=False) <= 1]

print(f"Nombre de features constantes supprimées: {len(constant_cols)}")
if constant_cols:
    print(constant_cols)

X_train_val = X_train_val.drop(columns=constant_cols)
X_test = X_test.drop(columns=constant_cols, errors='ignore')

# Update of numerical and categorical features list after removing constant features
numerical_features = X_train_val.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train_val.select_dtypes(include=["object", "string"]).columns.tolist()


# In[53]:


#recalculate correlation on numeric features
cols_to_include = list(numerical_features) + ["SalePrice"]
df_corr = (
    X_train_val
    .assign(SalePrice=y_train_val.values)[cols_to_include]
)
corr = df_corr.corr()["SalePrice"].sort_values(ascending=False)
print(corr.head(35))


# # 3. Data split used for initial baseline model (no cross validation)
# Cross validation is done later for more robust evaluation.

# In[54]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X_train_val, y_train_val_log,  # garde l’alignement index
    test_size=0.2,
    random_state=42
)


# # 4. Preprocessing pipeline and train a baseline model (RandomForest)
# For initial performance evaluation

# In[55]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

cat_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy= 'constant', fill_value='None')), 
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
num_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy= 'median')), 
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer(transformers = [
    ('num', num_transformer, numerical_features),
    ('cat', cat_transformer, categorical_features)
], remainder='drop' # only keeps the num+cat columns
)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
        ))
])
model.fit(X_train, y_train_log)

y_train_log_pred = model.predict(X_train)
y_val_log_pred   = model.predict(X_val)


from sklearn.metrics import root_mean_squared_error, r2_score
print("Train log RMSE :", root_mean_squared_error(y_train_log, y_train_log_pred))
print("Train log R²:", r2_score(y_train_log, y_train_log_pred))
print("Validation log RMSE :", root_mean_squared_error(y_val_log, y_val_log_pred))
print("Validation log R²:", r2_score(y_val_log, y_val_log_pred))

y_train = np.expm1(y_train_log)          # Inverse of log1p
y_train_pred = np.expm1(y_train_log_pred) 

y_val = np.expm1(y_val_log)          # Inverse of log1p
y_val_pred = np.expm1(y_val_log_pred) 


# # 5. Random Forest evaluation using hyperparameter tuning with cross‑validation
# In order to optimize baseline model performance evaluation using RandomizedSearchCV

# In[56]:


rf = RandomForestRegressor(random_state=42)


# In[57]:


pip_rf = Pipeline([
    ('preproc', preprocessor),
    ('regressor', rf)
])


# In[58]:


rf.get_params()


# In[59]:


from scipy.stats import randint
param_rf = {
    'regressor__n_estimators': randint(low=400, high=700), 
    'regressor__max_depth': randint(low=14, high=20),      
    'regressor__min_samples_split': randint(low=4, high=8), 
    'regressor__min_samples_leaf': randint(low=1, high=4),  
    'regressor__max_features': [0.6,0.9]             
}


# In[60]:


from sklearn.model_selection import RandomizedSearchCV, KFold
cv = KFold(
    n_splits = 5,
    shuffle = True,
    random_state= 42
)


# In[61]:


grid_rf = RandomizedSearchCV(
    estimator = pip_rf,
    param_distributions = param_rf,
    cv=cv,
    scoring= 'neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=0,
    n_iter=20
)


# In[62]:


grid_rf_fitted = grid_rf.fit(X_train, y_train_log)


# In[63]:


grid_rf_fitted.best_params_


# In[64]:


-grid_rf_fitted.best_score_


# In[65]:


best_model = grid_rf_fitted.best_estimator_


# In[66]:


# Evaluation of the best model on the validation set
y_val_log_pred_best = best_model.predict(X_val)
print("Validation log RMSE with best model:", root_mean_squared_error(y_val_log, y_val_log_pred_best))
print("Validation log R² with best model:", r2_score(y_val_log, y_val_log_pred_best))


# # 6. Comparison of Scikit-learn Models, Catboost and LightGBM with Hyperparameter Tuning using RandomizedSearchCV

# In[67]:


from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import xgboost as xgb
from scipy.stats import loguniform
import lightgbm as lgb
from catboost import CatBoostRegressor


# In[68]:


pipelines = {
    "LinearRegression": Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())]),
    "RandomForest": Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(random_state=42))]),
    "ElasticNet": Pipeline(steps=[('preprocessor', preprocessor), ('regressor', ElasticNet(random_state=42))]),
    "Ridge": Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Ridge(random_state=42))]),
    "LightGBM": Pipeline(steps=[('preprocessor', preprocessor), ('regressor', lgb.LGBMRegressor(random_state=42, verbose=-1))]),
    "CatBoost": Pipeline(steps=[('preprocessor', preprocessor), ('regressor', CatBoostRegressor(random_state=42, verbose=0))])
}


# In[69]:


# Hyperparameter grids for tuning
param_grids = {
    "LinearRegression": {}, # No tuning for simple LinearRegression
    "RandomForest": {
        'regressor__n_estimators': randint(low=400, high=700), 
        'regressor__max_depth': randint(low=14, high=20),      
        'regressor__min_samples_split': randint(low=4, high=8), 
        'regressor__min_samples_leaf': randint(low=1, high=4),  
        'regressor__max_features': [0.6,0.9]  
    },
    "ElasticNet": {
        'regressor__alpha': uniform(0.001, 1.0), # uniform for continuous distribution
        'regressor__l1_ratio': uniform(0.1, 0.9) 
    },
    "Ridge": {
        "regressor__alpha": loguniform(0.1, 10),
        "regressor__fit_intercept": [True],
        "regressor__solver": ["auto", "lsqr", "sparse_cg"],
        "regressor__tol": loguniform(1e-4, 1e-3)
    },
    "LightGBM": {
        'regressor__num_leaves': randint(20, 50),
        'regressor__learning_rate': loguniform(0.01, 0.1),
        'regressor__n_estimators': randint(500, 1000),
        'regressor__max_depth': randint(10, 20),
        'regressor__min_child_samples': randint(20, 50),
        'regressor__subsample': uniform(0.6, 0.4),
        'regressor__colsample_bytree': uniform(0.6, 0.4)
    },
    "CatBoost": {
        'regressor__iterations': randint(500, 1000),
        'regressor__learning_rate': loguniform(0.01, 0.1),
        'regressor__depth': randint(6, 10),
        'regressor__l2_leaf_reg': loguniform(1e-4, 1e-2),
        'regressor__border_count': randint(32, 255)
    }
}


# In[70]:


# Cross validation strategy and model evaluation on train set for each model with hyperparameter tuning using RandomizedSearchCV and KFold cross validation.
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict

cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
best_models = {}
cv_scores = {}
best_params_dict = {}
y_train_pred_oof = {}

for name, pipeline in pipelines.items():
    print(f"\n Model : {name}")

    if name == "LinearRegression":
        pipeline.fit(X_train, y_train_log)
        best_models[name] = pipeline

        # OOF predictions for LinearRegression (CV on X_train only)
        y_train_pred_oof[name] = cross_val_predict(
            best_models[name],
            X_train,
            y_train_log,
            cv=cv_strategy,
            n_jobs=-1,
            method="predict",
        )
        cv_scores[name] = root_mean_squared_error(y_train_log, y_train_pred_oof[name])
        print(f"CV RMSE: {cv_scores[name]:.4f}")

    else:
        grid = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grids[name],
            cv=cv_strategy,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            n_iter=20,
            random_state=42,
            verbose=0
        )

        grid.fit(X_train, y_train_log)

        best_models[name] = grid.best_estimator_
        cv_scores[name] = -grid.best_score_
        best_params_dict[name] = grid.best_params_

        # OOF predictions other models (CV on train only)
        y_train_pred_oof[name] = cross_val_predict(
            best_models[name],
            X_train,
            y_train_log,
            cv=cv_strategy,
            n_jobs=-1,
            method="predict",
        )

        print(f"Best CV log RMSE: {-grid.best_score_:.4f}")
        print(f"Best params: {grid.best_params_}")



# # 7. XGBoost Hyperparameter tuning with Out-of-Fold cross-validation with Early Stopping
# 
# In practice, early stopping is typically applied outside of RandomizedSearchCV using a custom cross-validation loop, where full control over the training and validation splits is required.
# 

# In[71]:


from scipy.stats import uniform, randint

param_dist = {
    "max_depth": randint(3, 8),
    "learning_rate": uniform(0.01, 0.1),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
}


# In[72]:


from sklearn.model_selection import ParameterSampler

param_list = list(ParameterSampler(
    param_dist,
    n_iter=20,
    random_state=42
))


# In[73]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_model_name3 = "XGBoost"
best_score3 = float("inf")
best_n_estimators3 = None
y_train_log_pred3 = None

for params in param_list:

    fold_scores = []
    y_train_pred = np.zeros(len(X_train))
    fold_best_iters = []  

    for train_idx, val_idx in kf.split(X_train):

        X_tr = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_tr = y_train_log.iloc[train_idx]
        y_val_oof_fold = y_train_log.iloc[val_idx]

        # preprocessing
        X_tr_trans = preprocessor.fit_transform(X_tr)
        X_val_trans = preprocessor.transform(X_val_fold)

        xgb = XGBRegressor(
            n_estimators=3000,
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=50,
            random_state=42,
            **params
        )

        xgb.fit(
            X_tr_trans,
            y_tr,
            eval_set=[(X_val_trans, y_val_oof_fold)],
            verbose=False
        )

        y_train_pred[val_idx] = xgb.predict(X_val_trans)

        rmse_log = root_mean_squared_error(y_val_oof_fold, y_train_pred[val_idx])
        fold_scores.append(rmse_log)

        # best iteration found by early stopping
        if hasattr(xgb, "best_iteration") and xgb.best_iteration is not None:
            fold_best_iters.append(xgb.best_iteration + 1)
        else:
            fold_best_iters.append(3000)

    mean_rmse_log = np.mean(fold_scores)

    if mean_rmse_log < best_score3:
        best_score3 = mean_rmse_log
        best_n_estimators3 = int((np.mean(fold_best_iters)*1.1))  # Adding a 10% buffer to the average best iteration
        best_params_dict["XGBoost"] = {"params": params, "n_estimators": best_n_estimators3}
        y_train_pred_oof["XGBoost"] = y_train_pred.copy()  # Best model OOF predictions saved for final evaluation

# For XGBoost, we use the best hyperparameters found and the best n_estimators from early stopping
xgb_best = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_estimators=best_n_estimators3,
        **best_params_dict["XGBoost"]["params"]
    ))
])
# fit the best XGBoost model on the entire training set
xgb_best.fit(X_train, y_train_log)
best_models["XGBoost"] = xgb_best
cv_scores["XGBoost"] = best_score3
best_params_dict["XGBoost"]["n_estimators"] = best_n_estimators3

print("Best CV RMSE XGBoost:", best_score3)
print("Best params:", best_params_dict["XGBoost"])
print("Best n_estimators (average folds):", best_n_estimators3)


# In[74]:


# Best three models scored on the training set with cross validation scores
sorted_scores = sorted(cv_scores.items(), key=lambda x: x[1])
print("\nCV RMSE scores for all models:")
for i, (name, score) in enumerate(sorted_scores, 1):
    print(f"{i}. {name}: {score:.4f}")

best_model_name1, best_score1 = sorted_scores[0]
best_model_name2, best_score2 = sorted_scores[1]
best_model_name3, best_score3 = sorted_scores[2]

print(f"\n 1rst Best model: {best_model_name1}  — CV RMSE : {best_score1:.4f}")
print(f" 2nd Best model: {best_model_name2}  — CV RMSE : {best_score2:.4f}")
print(f" 3rd Best model: {best_model_name3}  — CV RMSE : {best_score3:.4f}")


# # 8. Optimization of ensemble weights using OOF predictions
# We combine models because they capture different patterns in the data.

# In[75]:


from sklearn.metrics import root_mean_squared_error
import numpy as np

# Learn blending weights on OOF predictions from X_train only
top3_names = [best_model_name1, best_model_name2, best_model_name3]

# Weight grid
grid = np.linspace(0, 1, 101)
best_score = float("inf")
best_w = None

for w1 in grid:
    for w2 in grid:
        w3 = 1 - w1 - w2
        if w3 < 0:
            continue

        y_train_log_pred_blend = (
            w1 * y_train_pred_oof[best_model_name1]
            + w2 * y_train_pred_oof[best_model_name2]
            + w3 * y_train_pred_oof[best_model_name3]
        )

        rmse_log = root_mean_squared_error(y_train_log, y_train_log_pred_blend)
        if rmse_log < best_score:
            best_score = rmse_log
            best_w = (w1, w2, w3)

print(f"Best weights (m1, m2, m3): {best_w}")
print(f"Best OOF RMSE (log): {best_score:.5f}")
# Compute blended OOF predictions with the best weights
y_train_log_pred_blend_full = (
    best_w[0] * y_train_pred_oof[best_model_name1]
    + best_w[1] * y_train_pred_oof[best_model_name2]
    + best_w[2] * y_train_pred_oof[best_model_name3]
)
cv_scores["Blended"] = best_score
best_models["Blended"] = "Blended model with OOF weights (not a single sklearn model)"
best_params_dict["Blended"] = {"weights": best_w}


# # 9. Stacking (Meta model)

# In[76]:


X_meta_train_oof = np.column_stack((y_train_pred_oof[best_model_name1], y_train_pred_oof[best_model_name2], y_train_pred_oof[best_model_name3]))


# In[77]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
meta_model={}
meta_model["Meta_rf"] = RandomForestRegressor(
    random_state=42
)
y_train_pred_log_meta_rf_oof = cross_val_predict(
    meta_model["Meta_rf"],
    X_meta_train_oof,
    y_train_log,
    cv=5,
    n_jobs=-1,
    method="predict",
)
# Fit the meta-model on the entire training set
meta_model["Meta_rf"].fit(X_meta_train_oof, y_train_log)
cv_scores["Meta_rf"] = root_mean_squared_error(y_train_log, y_train_pred_log_meta_rf_oof)
best_models["Meta_rf"] = "Meta-model (RandomForest) trained on OOF predictions"
best_params_dict["Meta_rf"] = {
    "random_state": 42
}
print(f"Meta-model (RandomForest) OOF RMSE (log): {cv_scores['Meta_rf']:.5f}")



# In[78]:


from sklearn.linear_model import Ridge
meta_model["Meta_ridge"] = Ridge(
    alpha=1.0,
    fit_intercept=True,
    solver="auto",
    tol=1e-4,
    random_state=42
)
y_train_pred_log_meta_ridge_oof = cross_val_predict(
    meta_model["Meta_ridge"],
    X_meta_train_oof,
    y_train_log,
    cv=5,
    n_jobs=-1,
    method="predict",
)
# Fit the meta-model on the entire training set
meta_model["Meta_ridge"].fit(X_meta_train_oof, y_train_log)
cv_scores["Meta_ridge"] = root_mean_squared_error(y_train_log, y_train_pred_log_meta_ridge_oof)
best_models["Meta_ridge"] = "Meta-model (Ridge) trained on OOF predictions"
best_params_dict["Meta_ridge"] = {
    "alpha": loguniform(0.1, 10),
    "fit_intercept": True,
    "solver": "auto",
    "tol": 1e-4
}
print(f"Meta-model (Ridge) OOF RMSE (log): {cv_scores['Meta_ridge']:.5f}")


# In[79]:


#Selection of the best model among the unitary, blended and meta models based on oof predictions on the training set with cross validation scores
best_model_name_final = min(cv_scores, key=lambda x: cv_scores[x])
print(f"\nBest model selected for final prediction: {best_model_name_final} ")
print(f"with OOF RMSE (log): {cv_scores[best_model_name_final]:.5f}")


# # 10. Predictions and scoring on validation set

# In[80]:


# Refit and validation score for the 3 individual best models
y_train_log_pred={}
y_val_log_pred={}
validation_scores_indiv = {}

for name in [best_model_name1, best_model_name2, best_model_name3]:
    print(f"\n{name} - Refit and validation evaluation:")
    best_models[name].fit(X_train, y_train_log)
    y_train_log_pred[name] = best_models[name].predict(X_train)
    y_val_log_pred[name] = best_models[name].predict(X_val)
    rmse_log = root_mean_squared_error(y_val_log, y_val_log_pred[name])
    r2 = r2_score(y_val_log, y_val_log_pred[name])
    validation_scores_indiv[name] = {"RMSE_log": rmse_log, "R2_log": r2}
    print(f"Validation log RMSE: {rmse_log:.4f}")
    print(f"Validation R²: {r2:.4f}")
if best_model_name_final in [best_model_name1, best_model_name2, best_model_name3]:
    validation_score_final = validation_scores_indiv[best_model_name_final]
    print(f"\nSelected model {best_model_name_final} validation log RMSE: {validation_score_final['RMSE_log']:.4f}")
    print(f"Selected model {best_model_name_final} validation R²: {validation_score_final['R2_log']:.4f}")


# In[81]:


# Anti-leakage: keep blend weights learned on OOF predictions only
if best_model_name_final == "Blended":
    print(f"Blended validation uses OOF weights (no retuning on validation): {best_w}")


# In[82]:


# evaluation on validation set
if best_model_name_final == "Blended":
    # Use OOF-optimized weights to avoid validation leakage
    y_val_log_pred_blend = (
        best_w[0] * y_val_log_pred[best_model_name1]
        + best_w[1] * y_val_log_pred[best_model_name2]
        + best_w[2] * y_val_log_pred[best_model_name3]
    )
    validation_score_final = {
        "RMSE_log": root_mean_squared_error(y_val_log, y_val_log_pred_blend),
        "R2_log": r2_score(y_val_log, y_val_log_pred_blend)
    }
    print(f"\nSelected model {best_model_name_final} validation log RMSE: {validation_score_final['RMSE_log']:.4f}")
    print(f"Selected model {best_model_name_final} validation R²: {validation_score_final['R2_log']:.4f}")

elif best_model_name_final.startswith("Meta_"):
    meta_model_final = meta_model[best_model_name_final]
    meta_model_final.fit(X_meta_train_oof, y_train_log)
    X_meta_val = np.column_stack((y_val_log_pred[best_model_name1], y_val_log_pred[best_model_name2], y_val_log_pred[best_model_name3]))
    y_val_log_pred[best_model_name_final] = meta_model_final.predict(X_meta_val)
    validation_score_final = {
        "RMSE_log": root_mean_squared_error(y_val_log, y_val_log_pred[best_model_name_final]),
        "R2_log": r2_score(y_val_log, y_val_log_pred[best_model_name_final])
    }
    print(f"\nSelected model {best_model_name_final} validation log RMSE: {validation_score_final['RMSE_log']:.4f}")


# # 11. Refit on all train+val data set

# In[83]:


# Refit the best final model on the entire training set (X_train_val) before predicting on the test set
y_train_val_log_pred = {}
for name in [best_model_name1, best_model_name2, best_model_name3]:
    best_models[name].fit(X_train_val, y_train_val_log)
    y_train_val_log_pred[name] = best_models[name].predict(X_train_val)
if best_model_name_final in [best_model_name1, best_model_name2, best_model_name3]:
    final_model = best_models[best_model_name_final]
    print(f"\nSelected model {best_model_name_final} refitted on the entire training set for final prediction.")
elif best_model_name_final == "Blended":
    # Blended model does not require refitting as it is a weighted average of OOF predictions
    print(f"\nSelected blended model {best_model_name_final} for final prediction.")
elif best_model_name_final.startswith("Meta_"):
    X_meta_train_val = np.column_stack((y_train_val_log_pred[best_model_name1], y_train_val_log_pred[best_model_name2], y_train_val_log_pred[best_model_name3]))
    meta_model_final.fit(X_meta_train_val, y_train_val_log)
    final_model = meta_model_final
    print(f"\nSelected meta-model {best_model_name_final} refitted on the entire training set for final prediction.")


# # 12. Final predictions on Test Data and submission

# In[84]:


# Final prediction on the test set with the selected best model
y_test_log_pred = {}
for name in [best_model_name1, best_model_name2, best_model_name3]:
    y_test_log_pred[name] = best_models[name].predict(X_test)
if best_model_name_final in [best_model_name1, best_model_name2, best_model_name3]:
        print("Final model prediction on the test set with the selected best model are ready.")
elif best_model_name_final == "Blended":
    # Predictions with the blended model using the same weights found before
    y_test_log_pred[best_model_name_final] = (
        best_w[0] * y_test_log_pred[best_model_name1] +
        best_w[1] * y_test_log_pred[best_model_name2] +
        best_w[2] * y_test_log_pred[best_model_name3]
    )
    print(f"Final blended model {best_model_name_final} predictions on the test set are ready.")
elif best_model_name_final.startswith("Meta_"):
    #predictions with the meta-model using the predictions of the base models as input
    X_meta_test = np.column_stack((y_test_log_pred[best_model_name1], y_test_log_pred[best_model_name2], y_test_log_pred[best_model_name3]))
    y_test_log_pred[best_model_name_final] = meta_model_final.predict(X_meta_test)
    print(f"Final meta-model {best_model_name_final} predictions on the test set are ready.")


# In[85]:


# Final predictions on the test set with the best model
y_test_pred_final = np.expm1(y_test_log_pred[best_model_name_final])  # Inverse of log1p


# In[86]:


# Recording predictions in a CSV file for submission
submission = pd.DataFrame({
    "Id": X_test.index,
    "SalePrice": y_test_pred_final
})
submission.to_csv("submission.csv", index=False)


# In[87]:


modele.head(5)


# In[88]:


submission.head(5)


# In[89]:


modele.SalePrice.describe()


# In[90]:


submission.SalePrice.describe()


# In[ ]:


get_ipython().system('jupyter nbconvert --to script "House_Prices.ipynb"')

