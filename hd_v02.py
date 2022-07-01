# %% [markdown]
# # Heart Disease data base and simple analysis
# 
# - Import libraries needed;

# %%
import math
import seaborn as sns
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# - Import the heart disease data from Cleveland;
# - Check the head of the data base;

# %%
heart_disease = pd.read_csv("cleveland.csv")            # read the data

heart_disease.head()                                    # check the data base

# %% [markdown]
# - Complete attribute documentation:
#     - age: age in years
#     - sex: sex (1 = male; 0 = female)
#     - cp: chest pain type
#         - Value 1: typical angina
#         - Value 2: atypical angina
#         - Value 3: non-anginal pain
#         - Value 4: asymptomatic
#     - blood_ps: resting blood pressure (in mmHg on admission to the hospital)
#     - chol: serum cholestoral in mg/dl
#     - fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
#     - restecg: resting electrocardiographic results
#         - Value 0: normal
#         - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
#         - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
#     - thalach: maximum heart rate achieved
#     - exang: exercise induced angina (1 = yes; 0 = no)
#      - oldpeak = ST depression induced by exercise relative to rest
#     - slope: the slope of the peak exercise ST segment
#         - Value 1: upsloping
#         - Value 2: flat
#         - Value 3: downsloping
#     - ca: number of major vessels (0-3) colored by flourosopy
#     - thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
#     - num: diagnosis of heart disease (angiographic disease status)
#         - Value 0: < 50% diameter narrowing
#         - Value 1: > 50% diameter narrowing
#         (in any major vessel: attributes 59 through 68 are vessels)

# %% [markdown]
# - Now create a new column to calculate the age_group and divide the individuals by age;

# %%
#heart_disease['age'].value_counts()
heart_disease['age_group'] = heart_disease['age']           # create a new column copying the age column
heart_disease.head()                                        # check the data base

# %% [markdown]
# - Let's rearrange the columns from our data frame;

# %%
cols = ['age', 'age_group', 'sex', 'chest_pain', 'blood_ps', 'chol', 'fblood_sugar', 'restecg', 'max_ha', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
heart_disease = heart_disease[cols]
heart_disease.head()

# %% [markdown]
# - Calculate the number of classes and class amplitude;

# %%
age_k_count = heart_disease['age_group'].value_counts() # variable to receive the data counted and collated
age_k_count = age_k_count.size                          # variable receive how many elements the data have


if age_k_count > 50:                                    # compare whether the class number it's higher than 50
    age_k = 1 + (3.322 * np.log(age_k_count))           # if higher than 50 execute this function
else:                                                   # if lower or equal than 50 execute just the square root
    age_k = np.sqrt(age_k_count)

if age_k > round(age_k):                                # operation to round up the class quantity
    age_k = round(age_k) + 1                            #
else:                                                   #   
    age_k = round(age_k)                                #

max_age = heart_disease['age'].max()
min_age = heart_disease['age'].min()

class_amp_h = (max_age - min_age) / age_k               # calculate the class amplitude // maior_valor-menor_valor / num_class

if class_amp_h > round(class_amp_h):                     # operation to round up the class amplitude
    class_amp_h = round(class_amp_h) + 1                 #           
else:                                                    #
    class_amp_h = round(class_amp_h)                     #

age_group_boun = np.zeros(shape=age_k)                          # Create an aux array with the classes age limits

for i in range(age_k):                                     # Arrange and calculate the ages to each class
    if i == 0:                                             # when i equals 0, take the class amplitude plus the minimal age
        age_group_boun[0] = min_age + class_amp_h               
    else:
        age_group_boun[i] = age_group_boun[i-1] + class_amp_h        # when i not 0, take the previous age from age_group array plus the class amplitude

print(age_group_boun)                                           # check the classes boundaries


# %% [markdown]
# - Now let's organize separate the age group of each individual, for this case:
#     - 29 |- 36
#     - 36 |- 43
#     - 43 |- 50
#     - 50 |- 57
#     - 57 |- 64
#     - 64 |- 71
#     - 71 |- 78

# %%
for i in range(age_k):                               # i variable will check each boundary on the age_group class
    # IFs just for the disease data base
    if i == 0:
        aux = str(min_age) + " |- " + str(int(age_group_boun[i]))
    else:
        aux = str(int(age_group_boun[i-1])) + " |- " + str(int(age_group_boun[i]))

    for j in range(len(heart_disease['age_group'])):                        # j variable will go through all the database
        if type(heart_disease.loc[j,'age_group']) != str:                   # check if the age_group was updated to the group str or if it's still the age number
            if i == 0:                                                      # when i=0 we don't have a lower boundary, so just check whether the age it's less than the first upper boundary
                if heart_disease.loc[j,'age_group'] < age_group_boun[i]:    # check if the current db age it's lower than the first boundary
                    heart_disease.loc[j,'age_group'] = aux                  # if yes, update the position with the age_group str
            else:                                                           # if i != 0, need to check lower and upper boundaries to match the age_group
                if (heart_disease.loc[j,'age_group'] < age_group_boun[i]) and (heart_disease.loc[j,'age_group'] >= age_group_boun[i-1]): 
                    heart_disease.loc[j,'age_group'] = aux                  # if yes, update the position with the age_group str

aux = pd.DataFrame(np.zeros(shape=(age_k,2)))  # use aux variable to create a dataframe to hold some transitional information
aux = heart_disease.groupby(['age_group'])['age'].count()  # receive the count for each age_group grouped

# %% [markdown]
# - Now let's create our frequency table part by part:

# %%
freq_table= pd.DataFrame(0, index=np.arange(0, age_k), columns=['group','Xi','fi','fri','fri%','fia','fria','fria%'])  # create a dataset to organize the frequency table
freq_table['group'] = aux.index                         # from auxiliary variable get the index, which is the age_group

# calculate the class medium point
for i in range(age_k):          # for to go through all the rows from the freq_table     
    if i == 0:                  # when i=0 we don't have a lower boundary, so just check whether the age it's less than the first upper boundary
        freq_table.loc[i,'Xi'] = round((age_group_boun[i]+min_age) / 2)
    else:                       # if i != 0, need to calculate between lower and upper boundaries from the age_group
        freq_table.loc[i,'Xi'] = round((age_group_boun[i]+age_group_boun[i-1]) / 2)

# get the counted values from auxiliary variable
for i in range(age_k):          # for to go through all the rows from the freq_table
    freq_table.loc[i,'fi'] = aux.iloc[i]    # get from auxiliary variable the values counted from each age_group

# calculate the relative frequency to each age group
for i in range(age_k):          # for to go through all the rows from the freq_table
    freq_table.loc[i,'fri'] = round((freq_table.loc[i,'fi']/heart_disease['age'].size),4)      # calculate the relative frequency (fi/n)

# calculate the percentage relative frequency to each age group
for i in range(age_k):          # for to go through all the rows from the freq_table
    freq_table.loc[i,'fri%'] = round(100*freq_table.loc[i,'fri'],4)                           # calculate the percentage relative frequency (fi/n)*100

# calculate the acumulated frequency to each age group
for i in range(age_k):          # for to go through all the rows from the freq_table
    if i == 0:                  # when i = 0, it's the first row, so it's just the first number itself
        freq_table.loc[i,'fia'] = aux.iloc[i]
    else:                       # when i != 0, gets the previous value plus the one for that group
        freq_table.loc[i,'fia'] = aux.iloc[i] + freq_table.loc[i-1,'fia']

# calculate the relative acumulated frequency to each age group
for i in range(age_k):          # for to go through all the rows from the freq_table
    freq_table.loc[i,'fria'] = round((freq_table.loc[i,'fia']/heart_disease['age'].size),4)  # calculate the acumulated relative frequency (fria/n)

# calculate the percentage relative acumulated frequency to each age group
for i in range(age_k):          # for to go through all the rows from the freq_table
    freq_table.loc[i,'fria%'] = round(100*freq_table.loc[i,'fria'],4)                       # calculate the acumulated relative frequency (fria/n)*100

freq_table  # present the frequency table

# %% [markdown]
# - Some descritives measurements from the data that we have:

# %%
round(heart_disease.describe(),2)    # function that gets some descritives measurements

# %% [markdown]
# - The median and the mode values

# %%
print("Median value: " + str(heart_disease['age'].median()))       # function that gets only the median value, showed above as 50%
mode = stats.mode(heart_disease['age'])    # function that gets the mode value
print("Mode: " + str(mode[0]))


# %% [markdown]
# - Now let's find the quartiles from the age data:

# %%
agehd_sorted = heart_disease['age'].sort_values()          # sorting values in crescent order

quartile1_hd = agehd_sorted.iloc[round((heart_disease['age'].size)/4)]   # get the 1st quartile value
quartile2_hd = agehd_sorted.iloc[round((heart_disease['age'].size)/2)]   # get the 2nd quartile value, same as median value
quartile3_hd = agehd_sorted.iloc[round((heart_disease['age'].size)*3/4)] # get the 3rd quartile value

print("Q1 = " + str(quartile1_hd) + "\n" + "Q2 = " + str(quartile2_hd) + "\n" + "Q3 = " + str(quartile3_hd)) # shows the quartiles number

# %% [markdown]
# - We should also find the interquartile amplitude, which is the subtraction between the number found on Q3 and Q1. It's used as an alternative dispersion measure for standard deviation and it represents the 50% data observerd.

# %%
IQA = quartile3_hd - quartile1_hd  # Q3 - Q1 to find InterQuartile Amplitude
print(IQA)

# %% [markdown]
# - Age histogram and boxplot:

# %%
sns.histplot(heart_disease['age'], bins=7, element='poly', color='red')     # Age Polygon histogram in red
sns.histplot(heart_disease['age'], bins=7)                                  # Age Histogram in blue

# %%
print("Max Age: " + str(max_age) +"\n"+"Min Age: " + str(min_age))
sns.boxplot(data=(heart_disease.age))       #  Age boxplot

# %% [markdown]
# - Blood preasure histogram and boxplot:

# %%
sns.histplot(heart_disease['blood_ps'], bins=9, element='poly', color='red')
sns.histplot(heart_disease['blood_ps'], bins=9)

# %%
sns.boxplot(data=(heart_disease.blood_ps))

# %% [markdown]
# - 2 charts 

# %%
heart_disease.plot(kind='scatter', x='age', y='blood_ps')
heart_disease.plot(kind='scatter', x='age', y='chol', color='red')

# %%
heart_disease.plot(kind='scatter', y='chol', x='blood_ps')

# %% [markdown]
# # Analise Bidimensional

# %%
age_cols = ['Sex_Age']      # age_cols are the columns for the new dataset that will be used to do the two-way analysis
for i in range(freq_table['group'].size):                   # for to build the new data set columns
    age_cols = np.append(age_cols, freq_table.iloc[i,0])    # this will add the columns name same as the age groups form the frequency table
age_cols = np.append(age_cols,['Total'])                    # Add a column at end to sum the row total
age_cols

# %%
analise_bi = pd.DataFrame(columns = age_cols)               # Create the dataset to organize the dat.
analise_bi.loc[len(analise_bi.index)] = 0                   
analise_bi.loc[len(analise_bi.index)] = 0                   # Put zeros in each row/column
analise_bi.loc[len(analise_bi.index)] = 0
analise_bi.loc[0,'Sex_Age'] = 'Female'                      
analise_bi.loc[1,'Sex_Age'] = 'Male'                        # Add the rows index
analise_bi.loc[2,'Sex_Age'] = 'Total'


# Now let's build our two-way analysis table separated by age groups on the columns and the rows by sex
for i in range(1,len(analise_bi.columns)-1):                    # For to run throught the new dataset columns
    for j in range(heart_disease['age_group'].size):            # For to run throught the original dataset to check the individual age group and the sex
        if heart_disease.loc[j,'age_group'] == age_cols[i]:     # Check whether the individual belong to that age_group
            analise_bi.iloc[2,i] += 1                               # If yes, sum plus 1 to the number of total individuals from that age_group
            if heart_disease.loc[j,'sex'] == 0:                 # Check if the individual it's male or female and add to the row/column
                analise_bi.iloc[0,i] += 1                           # 0 means female
            else:
                analise_bi.iloc[1,i] += 1                           # 1 means male

for i in range(1,len(analise_bi.columns)-1):                    # this for it's to calculate the total individual numbers separated by row (Male/Female)
    analise_bi.loc[0,'Total'] += analise_bi.iloc[0,i]
    analise_bi.loc[1,'Total'] += analise_bi.iloc[1,i]
    analise_bi.loc[2,'Total'] += analise_bi.iloc[2,i]

# %%
analise_bi

# %%
porc_analise_bi = analise_bi.copy()  # copy the dataset to do the percentage two-way analysis

for i in range(1,len(porc_analise_bi.columns)):     # for to calculate the percentage values to each row/column
    # The formula it's done by row "(male-female_age_group_qty / male-female_total) * 100"
    porc_analise_bi.iloc[0,i] = round(((porc_analise_bi.iloc[0,i] / porc_analise_bi.loc[0,'Total']) * 100),2)
    porc_analise_bi.iloc[1,i] = round(((porc_analise_bi.iloc[1,i] / porc_analise_bi.loc[1,'Total']) * 100),2)
    porc_analise_bi.iloc[2,i] = round(((porc_analise_bi.iloc[2,i] / porc_analise_bi.loc[2,'Total']) * 100),2)


# %%
porc_analise_bi

# %% [markdown]
# # Associacao entre variaveis

# %%
val_esperado = analise_bi.copy()    # Copy the dataset to seek the 

for i in range(1,len(val_esperado.columns)-1):
    val_esperado.iloc[0,i] = val_esperado.loc[0,'Total'] * ((porc_analise_bi.iloc[2,i]) / 100)
    val_esperado.iloc[1,i] = val_esperado.loc[1,'Total'] * ((porc_analise_bi.iloc[2,i]) / 100)

# %%
desvio_ob_esp = val_esperado.copy()
desvio_ob_esp = desvio_ob_esp.drop(2)
del desvio_ob_esp['Total']
val_esperado

# %%
for i in range(1,len(desvio_ob_esp.columns)):
    for j in range(len(desvio_ob_esp.index)):
        desvio_ob_esp.iloc[j,i] = analise_bi.iloc[j,i] - val_esperado.iloc[j,i]

# %%
desvio_ob_esp

# %%
residuo_relativo = desvio_ob_esp.copy()

for i in range(1,len(residuo_relativo.columns)):
    residuo_relativo.iloc[0,i] = pow(residuo_relativo.iloc[0,i],2) / val_esperado.iloc[0,i]
    residuo_relativo.iloc[1,i] = pow(residuo_relativo.iloc[1,i],2) / val_esperado.iloc[1,i]

# %%
residuo_relativo

# %%
chi_quadrado = 0
for i in range(len(residuo_relativo.index)):    
    for j in range(1,len(residuo_relativo.columns)):
        chi_quadrado += residuo_relativo.iloc[i,j]

chi_quadrado

# %%
coef_cont_C = math.sqrt(chi_quadrado / (chi_quadrado+analise_bi.loc[2,'Total']))
coef_cont_C

# %%
coef_cont_T = math.sqrt((chi_quadrado/analise_bi.loc[2,'Total'])/((len(residuo_relativo.index)-1)*(len(residuo_relativo.columns)-1)))
coef_cont_T

# %% [markdown]
# ## Correlacao

# %%
corr_table = pd.DataFrame(columns=['age','blood_ps', 'x-x|','y-y|','zx','zy','zx*zy'])
corr_table['age']=heart_disease['age']
corr_table['blood_ps']=heart_disease['blood_ps']

# %%
age_mean = round(corr_table['age'].mean(),3)
blood_ps_mean = round(corr_table['blood_ps'].mean(),3)
age_std = round(corr_table['age'].std(),3)
blood_ps_std = round(corr_table['blood_ps'].std(),3)

# %%
for i in range (len(corr_table)):                                       ## Found the x - x_mean to each subject (age)
    corr_table.loc[i,'x-x|'] = corr_table.loc[i,'age'] - age_mean
    corr_table.loc[i,'zx'] = corr_table.loc[i,'x-x|'] / age_std

for i in range (len(corr_table)):                                       ## Found the y - y_mean to each subject (blood_ps)
    corr_table.loc[i,'y-y|'] = corr_table.loc[i,'blood_ps'] - blood_ps_mean
    corr_table.loc[i,'zy'] = corr_table.loc[i,'y-y|'] / blood_ps_std

for i in range (len(corr_table)):
    corr_table.loc[i,'zx*zy'] = corr_table.loc[i,'zx'] * corr_table.loc[i,'zy']


# %%
print("O coeficiente de correlação tem o valor de: " + str(round(corr_table['zx*zy'].sum()/len(corr_table),3)))

# %%
corr_table

# %%
corr_table.plot(kind='scatter',x='x-x|',y='y-y|')
corr_table.plot(kind='scatter',x='zx',y='zy')

# %% [markdown]
# ## Distribuicao de probabilidade

# %%
distr_prob = pd.DataFrame(columns=['blood_ps','fblood_sugar'])
distr_prob['blood_ps'] = heart_disease['blood_ps']
distr_prob['fblood_sugar'] = heart_disease['fblood_sugar']

# %%
distr_prob

# %%
for i in range (len(distr_prob)):
    if distr_prob.loc[i,'blood_ps'] > 129:
        distr_prob.loc[i,'blood_ps'] = 1
    else:
        distr_prob.loc[i,'blood_ps'] = 0

# %%
distr_prob.groupby(['blood_ps','fblood_sugar']).size()

# %%
distr_prob.groupby(['blood_ps']).size()

# %%
data = [[121,14,135],[137,31,168],[258,45,303]]
bi_distr_prob = pd.DataFrame(data, index=['n_blood_ps','h_blood_ps', 'total'],columns=['non_diab','diab','total'])
bi_distr_prob

# %%



