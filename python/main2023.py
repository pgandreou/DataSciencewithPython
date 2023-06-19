import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 30)


def currencyCleaner(d, c):
    d[c] = d[c].str.replace(',', '')
    d[c] = d[c].str.replace('$', '')
    d[c] = d[c].apply('float64')

'''
loc
float or array_like of floats
Mean (“centre”) of the distribution.

scale
float or array_like of floats
Standard deviation (spread or “width”) of the distribution. Must be non-negative.

size
int or tuple of ints, optional
Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if loc and scale are both scalars. Otherwise, np.broadcast(loc, scale).size samples are drawn.
'''
'''
np.random.seed(10)
dist1 = np.random.normal(loc=100, scale=10, size=10000)
dist2 = np.random.normal(loc=80, scale=30, size=10000)
dist3 = np.random.normal(loc=90, scale=20, size=10000)
dist4 = np.random.normal(loc=70, scale=25, size=10000)
dist5 = np.random.poisson(lam=5, size=10000)
dist5 = [i * 12 for i in dist5]

## combine these different collections into a list
#data = [dist1, dist2, dist3, dist4, dist5]

data = pd.DataFrame({'dist1':dist1,
                    'dist2':dist2,
                    'dist3':dist3,
                    'dist4':dist4,
                    'dist5':dist5
                    })
print(data.describe())

fig, ax = plt.subplots()

bp = ax.boxplot(data,
                positions=[2, 4, 6, 8, 10],
                widths=1,
                patch_artist=True,
                showmeans=True,
                showfliers=True,
                medianprops={"color": "white", "linewidth": 0.5},
                boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.5},
                whiskerprops={"color": "C0", "linewidth": 1.5},
                capprops={"color": "C0", "linewidth": 1.5})

ax.set(#xlim=(0, 12),
       #xticks=np.arange(1, 12),
       ylim=(0, 200),
       yticks=np.arange(-10, 200, 20))

plt.tick_params(
    axis='x',           # changes apply to the x-axis
    which='both',       # both major and minor ticks are affected
    bottom=True,        # ticks along the bottom edge are off
    top=True,           # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off
plt.show()



fig, ax = plt.subplots()
ax.set(ylim=(0, 200))
vp = ax.violinplot(data)
plt.tick_params(
    axis='x',           # changes apply to the x-axis
    which='both',       # both major and minor ticks are affected
    bottom=True,        # ticks along the bottom edge are off
    top=True,           # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off
plt.show()
'''


'''
#SAMPLING
url = 'https://raw.githubusercontent.com/pgandreou/DataSciencewithPython/main/datasets/pva.csv'
df = pd.read_csv(url)
#print(df.columns)

# using filters
print(df['TARGET_B'].value_counts())
df0 = df[df['TARGET_B'] == 0]
df0 = df0.sample(n=4843)
df1 = df[df['TARGET_B'] == 1]
dfs = df0.append(df1)
print(dfs['TARGET_B'].value_counts())

# using group by
dfs = df.groupby('TARGET_B', group_keys=False).apply(lambda x: x.sample(min(len(x), 4843)))
#print(dfs['TARGET_B'].value_counts())

# using group by
dfs = df.groupby('TARGET_B', group_keys=False).apply(lambda x: x.sample(min(len(x), 4843)))
#print(dfs['TARGET_B'].value_counts())

# using group by
tCol = 'TARGET_B'
# get number of rows df.index, df.shape[0]
minVC = min(len(df.index), df[tCol].value_counts().min())
dfs = df.groupby(tCol, group_keys=False).apply(lambda x: x.sample(min(len(x), minVC)))
#print(dfs['TARGET_B'].value_counts())


# using sample and lambda
dfs = df.groupby('TARGET_B').apply(lambda x: x.sample(n=4843))
print(dfs['TARGET_B'].value_counts())
'''


url = 'https://raw.githubusercontent.com/pgandreou/DataSciencewithPython/main/datasets/pvasample.csv'
df = pd.read_csv(url)
# df['TARGET_B'].value_counts().plot(kind='bar')
# plt.show()

'''
#print(df['DemMedIncome'].head())
#print(df.dtypes)
df['DemMedIncome'] = df['DemMedIncome'].str.replace(',', '')
df['DemMedIncome'] = df['DemMedIncome'].str.replace('$', '')
#print(df['DemMedIncome'].head())
df['DemMedIncome'] = df['DemMedIncome'].apply('float64')

#print(df.dtypes)
#plt.bar(df['DemMedIncome'], df['DemMedIncome'].max())
#plt.show()

#plt.hist(df['DemMedIncome'], bins=100, density=1, alpha=0.75)
#plt.show()

# print(df['DemMedIncome'].head(100))
# df.loc[df['DemMedIncome'] < 1.0, 'DemMedIncome'] = np.nan
## df['DemMedIncome'] = np.where(df['DemMedIncome'] < 1.0, np.nan, df['DemMedIncome'])
## df['DemMedIncome'].mask(df['DemMedIncome'] < 1.0 ,np.nan, inplace=True)
#print(df['DemMedIncome'].head(100))
#plt.hist(df['DemMedIncome'], bins=100, density=1, alpha=0.75)
#plt.show()
'''
df['DemMedIncome'] = df['DemMedIncome'].str.replace(',', '')
df['DemMedIncome'] = df['DemMedIncome'].str.replace('$', '')
df['DemMedIncome'] = df['DemMedIncome'].apply('float64')
df.loc[df['DemMedIncome'] < 1.0, 'DemMedIncome'] = np.nan

for col in df:
    if col.startswith('GiftAvg'):
        currencyCleaner(df, col)

'''
print(df.columns)
df['StatusCat96NK'].value_counts().plot(kind='bar')
plt.show()
df['StatusCat96NK'] = df['StatusCat96NK'].str.replace('S', 'A')
df['StatusCat96NK'].value_counts().plot(kind='bar')
plt.show()
df.loc[df['StatusCat96NK']=='F', 'StatusCat96NK'] = 'N'
df['StatusCat96NK'].value_counts().plot(kind='bar')
plt.show()
df['StatusCat96NK'].mask(df['StatusCat96NK'] == 'E' ,'L', inplace=True)
df['StatusCat96NK'].value_counts().plot(kind='bar')
plt.show()
'''
df['StatusCat96NK'] = df['StatusCat96NK'].str.replace('S', 'A')
df.loc[df['StatusCat96NK'] == 'F', 'StatusCat96NK'] = 'N'
df['StatusCat96NK'].mask(df['StatusCat96NK'] == 'E' ,'L', inplace=True)


'''
# print(df.isnull().any())
# dfe = pd.DataFrame(df.count())
# print(dfe[dfe[0]/9686 < 0.8])
'''


#plt.hist(df['DemAge'], bins=100, density=0, alpha=0.75)
#plt.show()

#print(df['DemAge'].mean())
df['DemAge'].fillna((df['DemAge'].mean()), inplace=True)
df['DemMedIncome'].fillna((df['DemMedIncome'].mean()), inplace=True)

#plt.hist(df['DemAge'], bins=100, density=0, alpha=0.75)
#plt.show()


'''
import scipy.stats as stats
df['DemAgeZ'] = stats.zscore(df['DemAge'])
plt.hist(df['DemAgeZ'], bins=100, density=1, alpha=0.75)
plt.show()
'''

#print(df.columns)
idx = 1
for col in df:
    if col.startswith('GiftAvg') or col.startswith('GiftCnt'):
        plt.subplot(3, 3, idx).set_title(col)
        plt.hist(df[col], bins=20, density=0, alpha=0.75)
        idx = idx + 1
'''        
plt.subplot(3, 3, 1).set_title('GiftAvg36')
plt.hist(df['GiftAvg36'], bins=20, density=0, alpha=0.75)
plt.subplot(3, 3, 2)
plt.hist(df['GiftAvgAll'], bins=20, density=0, alpha=0.75)
plt.subplot(3, 3, 3)
plt.hist(df['GiftAvgCard36'], bins=20, density=0, alpha=0.75)
plt.subplot(3, 3, 4)
plt.hist(df['GiftAvgLast'], bins=20, density=0, alpha=0.75)
plt.subplot(3, 3, 5)
plt.hist(df['GiftCnt36'], bins=20, density=0, alpha=0.75)
plt.subplot(3, 3, 6)
plt.hist(df['GiftCntAll'], bins=20, density=0, alpha=0.75)
plt.subplot(3, 3, 7)
plt.hist(df['GiftCntCard36'], bins=20, density=0, alpha=0.75)
plt.subplot(3, 3, 8)
plt.hist(df['GiftTimeFirst'], bins=20, density=0, alpha=0.75)
plt.subplot(3, 3, 9)
plt.hist(df['GiftTimeLast'], bins=20, density=0, alpha=0.75)
'''
plt.show()

idx = 1
for col in df:
    if col.startswith('GiftAvg') or col.startswith('GiftCnt'):
        df[col] = df[col].apply(lambda x: np.log10(x + 1))
        plt.subplot(3, 3, idx).set_title(col)
        plt.hist(df[col], bins=20, density=0, alpha=0.75)
        idx = idx + 1
#plt.show()

df['GiftAvgCard36'].fillna((df['GiftAvgCard36'].mean()), inplace=True)

df.to_csv('data/pva_prepared.csv')
