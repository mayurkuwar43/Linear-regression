# Loading libraries
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            from scipy import  stats
            import statsmodels.api as sm
            from sklearn.linear_model import LinearRegression
            from sklearn import metrics
            from sklearn.metrics import mean_squared_error
            from sklearn.model_selection import train_test_split

#Importing the dataset
            df = pd.read_csv("Downloads/Loss_Given_Default.csv")
            df
            df.shape
            df.dtypes
            # Getting the uniQue Values
            for val in df:
                print(val, " ", df[val].unique().shape)

# dealing with UniQue, or same value columns.
            df.drop("Ac_No", axis=1, inplace=True)
            df.shape

# Checking the single value value domination
            quasi_constant_feat = []
            for feature in df.columns:
                dominant = (df[feature].value_counts() / np.float(len(df))).sort_values(ascending=False).values[0]
                if dominant > 0.90:
                    quasi_constant_feat.append(feature)

            print(quasi_constant_feat)

# Null Value analysis & treatment.
            df.isnull().any()
             # df.isnull()
            df.isnull().sum()

            df.columns = ['Age', 'Experience', 'Vehicles', 'Gender', 'Married',  'tar_var']

#Plotting histogram of target variable

            tar_var = df['tar_var']
            sns.distplot(tar_var);

# Histograms : by removing density plot from top of it.
            sns.distplot(tar_var, kde=False, rug=False);
# sns.distplot(x, kde=False, bins=200, rug=True);

# Kernel density estimation :
            sns.distplot(tar_var, hist=False, rug=False);
# ####
"""      EDA steps like pair plot and other plotting analysis to Check the assumption  of Linear Regression """

            sns.pairplot(df)    # pair plotting

#Checking skewness
            tar_var.skew()
#Discover outliers with mathematical function

            z = np.abs(stats.zscore(tar_var))
            print(z)
            threshold = 3
            print(np.where(z > 3))
            x= np.where(z > 3)

            len(x[0])
            x[:1]
            df.iloc[np.where(z > 3)[:1]]
            #df_num.shape
            df.drop(df.index[x[:1]], inplace=True)

# Further skewness treatment with transformation
            df["tar_var"] = df["tar_var"] ** (1 / 2)
            df.tar_var.plot(kind="hist")
            df.tar_var.skew()

# Checking for co-relation

            df_num = df.select_dtypes(include=['int64','float64' ])

# Below Code gives corrplot
            corrmat= df_num.corr()
            fig,ax=plt.subplots()
            fig.set_size_inches(11,11)
            sns.heatmap(corrmat)

# Which are the columns that should removed
            def checkcorrelation(dataset, threshold):
                col_corr = set()
                cor_matrix = dataset.corr()
                for i in range(len(cor_matrix.columns)):
                    for j in range(i):
                        if abs(cor_matrix.iloc[i, j]) > threshold:
                            colname = cor_matrix.columns[i]
                            col_corr.add(colname)
                return col_corr

            checkcorrelation(df_num, 0.8)

# drop the highly co-related columns
            df.drop("Experience", axis=1, inplace=True)

## The Chi-Squared Test for Independence - Using Scipy

            df.columns
            #Constructing the Contingency Table
            ct = pd.crosstab(df.Gender, df.Married)
            #ct_m = pd.crosstab(df.Gender, df.Married,margins = True)
            f_obs = np.array([ct.iloc[0].values, ct.iloc[1].values])
            x = stats.chi2_contingency(f_obs)[0:3]
            print("Chi-Squared Test Analysis \n X-Squared Value :{0} "
                  "\n P-value :{1}\n degree of freedom :{2}".format(x[0],x[1],x[2]))

## The One-Way Anova test
            df.columns
            df2 = df.copy()
            data = df[['Married', 'tar_var']].dropna()
            x = data.Married.unique()
            dict = {}
            for k in x:
                dict[k] = data[data['Married'] == k]['tar_var']

            dict
            stats.f_oneway(dict['Married'], dict['Single'])

######## Multiway anova..

            from statsmodels.formula.api import ols
            #df2.columns = ['Age', 'experience', 'Vehicles', 'Gender', 'Married', 'tar_var']
            data = df[['Married', 'Gender', 'tar_var']].dropna()
            results = ols('tar_var ~ C(Married)*C(Gender)', data=data).fit()
            results.summary()


# Training & Test split of data
            #df = df2
            df.columns
            tar_var = df['tar_var']
            df.drop("tar_var", axis=1, inplace=True)

# Converting in dummies
            df_num = df.select_dtypes(include=['int64', 'float64'])
            df = pd.get_dummies(df, drop_first=False)
# Converting categorical variable into factor.
            lst = df_num.columns
            for val in df:
                if(val not in lst):
                    df[val] = df[val].astype("object")

#Spliting now.
            x_train,x_test,y_train,y_test = train_test_split(df, tar_var, random_state = 10, test_size = 0.3)


# Implementing model
            from sklearn import linear_model as lm
            model = lm.LinearRegression()
            result = model.fit(x_train, y_train)

            #Printing the coefficient
            print(model.intercept_)
            print(model.coef_)

# predicting the values.
            predictions = model.predict(x_test)

# Getting the model eveluation from predicted values
            from sklearn.metrics import mean_squared_error, r2_score
# model evaluation
            mse = mean_squared_error(y_test, predictions)
            rmse = mse**(1/2)
            r2 = r2_score(y_test, predictions)



#--------------------------Another way to get the summary from stats model-------------
            df = pd.read_csv("C:/Training/Sat Class/Loss_Given_Default.csv")
            df
            df.drop("Ac_No", axis=1, inplace=True)
            df.columns = ['Age', 'Experience', 'Vehicles', 'Gender', 'Married',  'tar_var']
            df.columns
            tar_var = df['tar_var']
            df.drop("tar_var", axis=1, inplace=True)
            df = pd.get_dummies(df, drop_first=False)
#Spliting now.
            x_train,x_test,y_train,y_test = train_test_split(df, tar_var, random_state = 10, test_size = 0.3)

# From Stats modeling
             model2 = sm.OLS(y_train, x_train).fit()
             model2.summary()

# predicting the values.
             predictions = model2.predict(x_test)

# model evaluation on predicted values.
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

# printing values
        print('Slope:' ,model.coef_)
        print('Intercept:', model.intercept_)
        print('Root mean squared error: ', rmse)
        print('R2 score: ', r2)

# Converting the y_pred as actual values ( due to transformation)

        Y_pred = predictions**2


#------------------------------------------Linear -----------------------------