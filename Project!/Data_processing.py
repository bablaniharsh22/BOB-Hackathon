#step:1
df=pd.read_csv("#data should be inserted here")

#step:2
df.head(no. of coloumns u want to see)

#step:3
df.coloumns = df.coloumns.str.strip()

#step:4
df.loc[:,'label'].unique()

#step:4
plt.figure(1,figsize=( 10,4))
plt.hist( df.isna().sum())

plt.xticks([0, 1], labels=['Not Null=0', 'Null=1'])
plt.title('Columns with Null Values')
plt.xlabel('Feature')
plt.ylabel('The number of features')

plt.show()

#step:5
def plotMissingValues(dataframe):
    missing_values = dataframe.isnull().sum()  
    fig = plt.figure(figsize=(16, 5))
    missing_values.plot(kind='bar')
    plt.xlabel("Features")
    plt.ylabel("Missing values")
    plt.title("Total number of Missing values in each feature")
    plt.show()

plotMissingValues(df)

#step:6
#This code is used for removing the null values
data_f=df.dropna()

#step:6
plt.figure(1,figsize=( 10,4))
plt.hist( data_f.isna().sum())

plt.title('Data aftter removing the Null Values')
plt.xlabel('null values')
plt.ylabel('Number of coloumns')

plt.show()

#step:7
pd.set_option('use_inf', True)
null_values=data_f.isnull().sum()

#step:8
(data_f.dtypes=='object')

#step:9
plt.hist(data_f['Label'], bins=[0, 0.3,0.7,1], edgecolor='black') 
plt.xticks([0, 1], labels=['benign=0', 'data should be there=1'])
plt.xlabel("Classes")
plt.ylabel("Count")
plt.show()


