#step:1
#this step is used to describe the data into different rows and coloumns
df.describe()

#step:2
plt.figure(6)
for col in data_f.columns:
    plt.hist(data_f[col])
    plt.title(col)
    plt.show()

#step:3
#converting array
X1=np.array(data_f).astype(np.float64)
y1=np.array(data_f['label'])

#step:4
X = data_f.drop('Label', axis=1)
y = data_f['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#step:5
print("The train dataset size = ",X_train.shape)
print("The test dataset size = ",X_test.shape)