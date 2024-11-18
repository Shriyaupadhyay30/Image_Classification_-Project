#Function for data cleaning
def data_cleaning(df, name='df'):
    #Checking for null values
    num_null_vals = sum(df.isnull().sum().values)

    #When there is no null values
    if not num_null_vals:
        print(f"The {name} has no null values")

    #When there i snull values
    else:
        print(f"The {name} has {num_null_vals} null values")
        print('Total null values in each column:\n')
        print(df.isnull().sum())
        
        #Removes rows with null values
        df = df.dropna()
        print(f"\nRows with null values have been removed. The dataset now has {df.shape[0]} rows.")
    
    #Checking for duplicates
    num_duplicates = df.duplicated().sum()

    #When there is no duplication in dataset
    if num_duplicates == 0:
        print(f"\nThe {name} has no duplicate values.")
    
    #When there is duplication in dataset
    else:
        print(f"\nThe {name} has {num_duplicates} duplicate rows.")
        df = df.drop_duplicates()
        print(f"Duplicate rows have been removed. The dataset now has {df.shape[0]} rows.")

    return df

#Assiging new cleaned dataframe to the df
df = data_cleaning(df, dataset_name)
