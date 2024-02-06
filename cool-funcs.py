# %% [code]
# %% [code]
# %% [code] {"execution":{"iopub.status.busy":"2024-02-01T23:32:42.831458Z","iopub.execute_input":"2024-02-01T23:32:42.832411Z","iopub.status.idle":"2024-02-01T23:32:45.848144Z","shell.execute_reply.started":"2024-02-01T23:32:42.832367Z","shell.execute_reply":"2024-02-01T23:32:45.846751Z"}}
# Analayze columns 
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

def check_df(dataframe, head=5):
    """
    checks dataframe for shape, dtypes, head, tail, nuul value count, quantiles
    --------------
    params:
    dataframe : pandas dataframe
    head : int
    usage : 
    check_df(df)
    """
    print('******************************************************')
    print('shape \n' ,dataframe.shape)
    print('******************************************************')
    print('dtypes \n',dataframe.dtypes)
    print('******************************************************')    
    print('tail \n',dataframe.tail(head))
    print('******************************************************')
    print('head \n', dataframe.head(head))
    print('******************************************************')
    print('NA-sum \n', dataframe.isnull().sum())
    print('******************************************************')
    # print('quantiles \n', dataframe.quantile([0,0.05,0.50,0.95,0.99,1]).T)

"""***********************************************************************************"""
def target_summary_with_num(dataframe, target, numerical_col):
    """
    prints mean groupby target with numerical_col 
    ---------
    Params: 
    dataframe : pandas dataframe 
    target : target column
    numerical_col: columns, numerical 
    ----------------
    usage :
    for col in num_cols:
        target_summary_with_num(df, "Churn", col)
    """
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")
    print('*********************************************************')



"""***********************************************************************************"""
def target_summary_with_cat(dataframe, target, categorical_col):
    """
    prints the summirize groupy.mean with target column and the given categorical column.
    ---------
    params:
    dataframe: pandas dataframe.
    target: string
            target columns name.
    categorical_col: string
                     categorical column's name.
    ---------
    usage:
    for col in cat_cols:
        target_summary_with_cat(df, "Churn", col)
        print('***********************************')
    """
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

"""***********************************************************************************"""

"""***********************************************************************************"""
def outlier_thresholds(dataframe, col_name, q1=0.25,q3=0.75):
    """
    returns low and up limit, calculations of iqr, finds specific cols iqr, up and low limits.
    ----
    Parameters:
    dataframe : dataframe
    col_name : desired ıqr's column's names
    --------------------------
    usage:
    outlier_thresholds(df,'Age')
    ...............................
    
    """
    
    
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    
    return low_limit , up_limit
    
"""***********************************************************************************"""
def check_outlier(dataframe, col_name):

    """
    checks if there's any outlier in the specific column in the dataframe and Returns Boolean
    ---
    Parametrs:
    dataframe: dataframe
    col_name : 'col_name'
    ---------------------------------
    usage:
    check_outlier(df,'column_name')
    ..........................
    checklist = []
    for col in num_cols:
    checklist.append(check_outlier(dff,col)) 
    """
    low_limit,up_limit = outlier_thresholds(dataframe,col_name)
    if dataframe[(dataframe[col_name]< low_limit) |(dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False
        
"""***********************************************************************************"""
def grab_col_names(dataframe,cat_th=10,car_th=20):
    """
    Who is really who?
    veri setinde ki kategorik, nümerik ve kategorik fakat 'kardinal' ! değişkenlerin isimlerini return eder
    NOT: KAtegorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.
    ----
    Paramters : 
    
    dataframe: dataframe, the dataframe that col names want to be extracted.
    
    cat_th : int, optional
             a limit value for the col's that categoric but with numeric values.
    
    car_th: int, optional
            a limit value for the col's that cardinal.
    ------
    Returns:
    cat_cols : list
                Categoric col's list
    
    num_cols : list
               Numeric col's list.
               
    cat_but_car: list
                categoric but cardinals
                
    ------------------------            
    usage: 
    cat_cols,num_cols,cat_but_car = grab_col_names(dataframe=df)
    """
    #cat_cols, cat_but_car
    
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique()< cat_th and
                   dataframe[col].dtypes !='O']
    
    cat_but_car= [col for col in dataframe.columns if dataframe[col].nunique()> car_th and
                  dataframe[col].dtypes=='O']
    
    
    cat_cols= cat_cols+ num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
                   
    ## num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_cols = [col for col in num_cols if col not in num_but_cat]
    
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")    
    print(f"cat_cols: {len(cat_cols)}")    
    print(f"cat_but_car: {len(cat_but_car)}") 
    print(f"num_cols: {len(num_cols)}")     
    print(f"num_but_cat: {len(num_but_cat)}")
    
    return cat_cols,num_cols,cat_but_car

"""**********************************************************************************"""
def grab_outliers(dataframe, col_name, index=False):
    """
    grabing outliers and printing the outliers, if index is True, returns outlier_index.
    ---
    params:
    dataframe : dataframe
    
    colname: colname

    index : bool/optional
            default = False
    ----
    Returns: outlier_index, if index=True
    also prints the outliers
    ----
    usage: 
    age_index = grab_outliers(df,'Age',index=True)    
    ...............
    for col in num_cols:
        temp_index = cool_funcs.grab_outliers(df, col , index = True)
        print(f'nan includes index list for {col} : ' , temp_index , 'len: ',len(temp_index))
        print('********************', end='\n\n')
    """
    low,up = outlier_thresholds(dataframe,col_name)
    
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
        
    else:
        print((dataframe[((dataframe[col_name]<low) |(dataframe[col_name] > up))]))
        
    if index:
        outlier_index = dataframe[((dataframe[col_name]<low) | (dataframe[col_name]> up))].index
        return outlier_index
"""***********************************************************************************"""
## Outliers handle
def remove_outlier(dataframe,col_name):


    """
    removes the outliers from the dataframe
    ----
    Params:
    dataframe : dataframe
    col_name: column name
    -----
    Returns df_without_outliers
    ---------------
    usage : 
    df_without_outliers = remove_outlier(dataframe=df,col_name='Fare')
    """
    low_limit, up_limit = outlier_thresholds(dataframe,col_name)
    
    df_without_outliers = dataframe[~((dataframe[col_name]< low_limit) | (dataframe[col_name]>up_limit))]
    return df_without_outliers
    
"""***********************************************************************************"""
def replace_with_thresholds(dataframe,variable):
    """
    reassigns the outliers with low and up limits. If smaller than low limit reassignets to low_limit, 
    if bigger than up_limit reassignts to up_limit.
    ------
    params: 
    dataframe: dataframe
    variable : colname
    ------------------------------------
    usage : 
    replace_with_thresholds(df,'Age')
    ...........................
    for col in num_cols:
        replace_with_thresholds(df,col)
    """
    low_limit,up_limit = outlier_thresholds(dataframe, variable)
    
    dataframe.loc[(dataframe[variable]< low_limit), variable]= low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable]= up_limit
    
"""***********************************************************************************"""
#Missing Values
def missing_values_tabels(dataframe, na_name=False):
    """
    finds the columns contains na and returns it if na_name is set to True.
    ------------------------------------------------
    Params:
    dataframe: Pandas DataFrame
    na_name= Bool , default=False
    -------------------------
    example usage : 
    na_cols = missing_values_tabels(df,na_name=True)
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum()>0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0]*100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio,2)], axis=1, keys =['n_miss','ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns
        
"""***********************************************************************************"""
def missing_vs_target(dataframe,target,na_columns):
    """
    cheking the intercourse of the missing values and the target col. Prints the groupby, Mean report.
    ----------
    Params:
    dataframe: dataframe
    target : string
            target column
    na_columns : list
                columns name contains missing values, 
    -----------------------
    usage : 
    missing_vs_target(df,'Survived',na_cols)
    """
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col+ '_NA_Flag'] = np.where(temp_df[col].isnull(),1,0)
    na_flags= temp_df.loc[:, temp_df.columns.str.contains('_NA_')].columns
    
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN" : temp_df.groupby(col)[target].mean(),
                           'Count' : temp_df.groupby(col)[target].count()}), end=' \n\n\n')

"""***********************************************************************************"""

# Encode
def one_hot_encoder(dataframe,categorical_cols,drop_first=True):
    """
    returns a dataframe that one encoded given categorical cols.
    ----
    Params: 
    Dataframe : panads Dataframe
    
    categorical_cols :  list/string
                        categorical cols list or only one
    drop_first : bool
                 to drop first encoded column(benefits to solve dummies problem)
    -------------------------------------
    
    usage : 
    
    ohe_cols = [col for col in df.columns if 10>= df[col].nunique() > 2]
    new_df =one_hot_encoder(dataframe = df, categorical_cols= ohe_cols,drop_first=True)
    """
    
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

"""***********************************************************************************"""
def label_encoder(dataframe, binary_col):
    """
    returns encoded dataframe by LabelEncoder(). Columns to be encoded are binary_cols.
    NOTE : ONLY THIS ENCODER IS ONLY FOR BINARY COLUMNS ALSO IS NOT A TARGET COLUMN
    -------------------------------------
    Params: 
    dataframe : pandas dataframe
    binary_cols : columns to be encoded
    ------------------------------------
    usage :
    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique()==2]
    
    for col in binary_cols:
        label_encoder(df,col)
    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


"""***********************************************************************************"""
def cat_summary(dataframe, col_name, plot=False):
    """
    prints the columns value counts and gives the ratio/frequency for each class/values.
    if plot is set True, plotting countplots for certian columns.
    -----------
    Paramas:
    dataframe: Pandas DataFrame
    
    col_name: String
              column name
              
    plot: Bool
          if False no graphs, True, plots countplot for the specific columns.
    ------------
    usage :
    for col in cat_cols:
        cat_summary(dataframe=df,col_name=col)
        print('"""""""""""""""""""""""""""""""')
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                       'Ratio' : 100 * dataframe[col_name].value_counts()/len(dataframe)}))
    if plot:
        sns.countplot(x= dataframe[col_name], data = dataframe)
        plt.show()

"""***********************************************************************************"""
def rare_analyser(dataframe, target, cat_cols):
    """
    prints the mean of groupby among target column and cat_cols also gives the counts of the columns and Ratio (column count)/len(dataframe)
    -------------------------------
    Params:
    dataframe : pandas dataframe
    target : target column
    cat_cols : categoric columns
    ---------------
    usage :
    
    rare_analyser(dataframe=df, target='TARGET', cat_cols=cat_cols)
    """
    for col in cat_cols:
        print(col, ':', len(dataframe[col].value_counts()))
        print(pd.DataFrame({'COUNT' : dataframe[col].value_counts(),
                           'RATIO' : dataframe[col].value_counts()/len(dataframe),
                           'TARGET_MEAN': dataframe.groupby(col)[target].mean()}), end='\n\n\n')
        print('***************************************************')

"""***********************************************************************************"""
def rare_encoder(dataframe, rare_perc):
    """
    gathers rare columns lower than given rare_perc then creates rare columns and fills rare columns with rare expressions.
    -------
    Paramters: 
    dataframe: pandas dataframe
    
    rare_perc: float 
               rare ratio that given by user
    -----------
    returns temp_df   (pandas dataframe) 
    
    --------------------
    usage : 
    temp_df =rare_encoder(df, 0.01)
    
    """
    temp_df =  dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                   and (temp_df[col].value_counts() / len(temp_df)  < rare_perc).any(axis=None)]
    
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    
    return temp_df 

"""***********************************************************************************"""
## also analyse
def num_summary(dataframe, numerical_col, plot=False):
    """
    Sumarizes the numerical columns with it's quartiles, like df.describe()
    If plot is set to be True; plots histogram for each numerical_col
    -----------
    Params: 
    dataframe : pandas DataFrame
    numerical_col: numerical columns
    plot: Bool
    ----------------
    usage: 
    for col in num_cols:
        cool_funcs.num_summary(dataframe=df,numerical_col=col, plot=False)
    """
    quantiles = [0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot: 
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()
    else:
        print('****************************')

"""************************************************************************************"""      
"""************************************************************************************"""      
   
def plot_importance(model, features, len_X, save=False):
    """
    Plots a barplots for each feature/column in a figure, the importance level.
    If save is set to be True, saves the plotted image.
    ---------
    params:
    model : model
    features : columns
    num: length of dataframe
    
    save : bool
    -----------
    usage :
    plot_importance(rf_model,features=X, num=len(X),save=True)

    """
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:len_X])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
        
"""************************************************************************************"""