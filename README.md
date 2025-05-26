# data-cleaning-and-preprocessing

'''python
from google.colab import files
files.upload()

import os
import zipfile

os.makedirs("/root/.kaggle", exist_ok=True)
!mv kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json

!kaggle competitions download -c titanic

# Unzip all downloaded files
with zipfile.ZipFile("titanic.zip", "r") as zip_ref:
    zip_ref.extractall("Titanic-Data")

os.listdir("Titanic-Data")

import pandas as pd
df = pd.read_csv("titanic_data/train.csv")  
df.head()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("/content/Titanic-Data/train.csv") 
print("First 10 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Fill missing 'Age' values with median (numeric)
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Embarked' values with mode (categorical)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' due to high number of missing values
df.drop(columns='Cabin', inplace=True)

# Double-check missing values again
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Convert 'Sex' and 'Embarked' into numeric using one-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Check the updated dataframe
print("\nColumns after encoding:")
print(df.columns)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Normalize 'Age' and 'Fare' only
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Check normalized columns
print("\nNormalized 'Age' and 'Fare':")
print(df[['Age', 'Fare']].head())

print("Mean after scaling:", df[['Age', 'Fare']].mean())
print("Std deviation after scaling:", df[['Age', 'Fare']].std())

# Visualize outliers using boxplots
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.boxplot(x=df['Age'])
plt.title("Boxplot - Age")

plt.subplot(1, 2, 2)
sns.boxplot(x=df['Fare'])
plt.title("Boxplot - Fare")

plt.tight_layout()
plt.show()

# Remove outliers in 'Fare' using IQR method
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Apply filtering
df = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]

print("\nShape after removing outliers in 'Fare':", df.shape)

print("Final Dataset Info:")
print(df.info())
'''

**output**

'''
kaggle.json(application/json) - 67 bytes, last modified: 20/02/2025 - 100% done
Saving kaggle.json to kaggle.json
{'kaggle.json': b'{"username":"hetvidpatel","key":"777939720da65628668357169aa73925"}'}

!kaggle competitions download -c titanic

Downloading titanic.zip to /content
  0% 0.00/34.1k [00:00<?, ?B/s]
100% 34.1k/34.1k [00:00<00:00, 124MB/s]

['gender_submission.csv', 'test.csv', 'train.csv']


PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	1	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	A/5 21171	7.2500	NaN	S
1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	PC 17599	71.2833	C85	C
2	3	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S
3	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	113803	53.1000	C123	S
4	5	0	3	Allen, Mr. William Henry	male	35.0	0	0	373450	8.0500	NaN	S


First 10 rows of the dataset:
   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.05000   NaN        S 

First 10 rows of the dataset:
   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S  

Summary statistics:
       PassengerId    Survived      Pclass         Age       SibSp  \
count   891.000000  891.000000  891.000000  714.000000  891.000000   
mean    446.000000    0.383838    2.308642   29.699118    0.523008   
std     257.353842    0.486592    0.836071   14.526497    1.102743   
min       1.000000    0.000000    1.000000    0.420000    0.000000   
25%     223.500000    0.000000    2.000000   20.125000    0.000000   
50%     446.000000    0.000000    3.000000   28.000000    0.000000   
75%     668.500000    1.000000    3.000000   38.000000    1.000000   
max     891.000000    1.000000    3.000000   80.000000    8.000000   

            Parch        Fare  
count  891.000000  891.000000  
mean     0.381594   32.204208  
std      0.806057   49.693429  
min      0.000000    0.000000  
25%      0.000000    7.910400  
50%      0.000000   14.454200  
75%      0.000000   31.000000  
max      6.000000  512.329200  


Missing values in each column:
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64

Missing values after cleaning:
PassengerId    0
Survived       0
Pclass         0
Name           0
Sex            0
Age            0
SibSp          0
Parch          0
Ticket         0
Fare           0
Embarked       0
dtype: int64
<ipython-input-40-cb49c2f61314>:2

Columns after encoding:
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Age', 'SibSp', 'Parch',
       'Ticket', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S'],
      dtype='object')

Normalized 'Age' and 'Fare':
        Age      Fare
0 -0.499795 -0.813749
2 -0.186156 -0.748316
4  0.519531 -0.736198
5 -0.029337 -0.696618
7 -2.067990  0.526425

Mean after scaling: Age    -4.846813e-18
Fare   -2.423406e-17
dtype: float64
Std deviation after scaling: Age     1.000683
Fare    1.000683
dtype: float64


Shape after removing outliers in 'Fare': (727, 12)

Final Dataset Info:
<class 'pandas.core.frame.DataFrame'>
Index: 727 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  727 non-null    int64  
 1   Survived     727 non-null    int64  
 2   Pclass       727 non-null    int64  
 3   Name         727 non-null    object 
 4   Age          727 non-null    float64
 5   SibSp        727 non-null    int64  
 6   Parch        727 non-null    int64  
 7   Ticket       727 non-null    object 
 8   Fare         727 non-null    float64
 9   Sex_male     727 non-null    bool   
 10  Embarked_Q   727 non-null    bool   
 11  Embarked_S   727 non-null    bool   
dtypes: bool(3), float64(2), int64(5), object(2)
memory usage: 58.9+ KB
None
'''

