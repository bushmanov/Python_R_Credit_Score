
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# ## Read data

# In[2]:

names = pd.read_csv('./data/names.csv', sep = ',', header = 0).values.flatten()
names


# In[3]:

data = pd.read_csv('./data/german.data', sep= ' ', names = names)


# In[4]:

data.head(3)


# In[5]:

data.count()


# ## Prepare features

# In[6]:

y = data.status
y.value_counts()


# In[7]:

y.mean()


# #### 1. Type of checking account
# 
# A11 : ... < 0 DM   
# A12 : 0 <= ... < 200 DM   
# A13 : ... >= 200 DM / salary assignments for at least 1 year   
# A14 : no checking account   

# In[8]:

data.groupby('chk_account_').size()


# In[9]:

data.groupby('chk_account_')['status'].agg('mean')


# In[10]:

chk_account = pd.get_dummies(data.chk_account_, 'Chk_Account')
chk_account_= chk_account.drop('Chk_Account_A13', 1)
chk_account_.head(3)


# In[11]:

X = chk_account_


# #### 2. Duration of credit line

# In[12]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# In[13]:

plt.subplot(1,2,1)
data.num_duration_A2.hist() # might need pre/post -processing as data is skewed ???
plt.subplot(1,2,2)
plt.scatter( x=data.num_duration_A2, y = y) # cut data into 2-3-4 bins?
plt.show()


# In[14]:

dur = data.num_duration_A2


# In[15]:

X = pd.concat([X,dur], axis=1)


# #### 3. Credit history
# 
# A30 : no credits taken/ all credits paid back duly   
# A31 : all credits at this bank paid back duly   
# A32 : existing credits paid back duly till now   
# A33 : delay in paying off in the past   
# A34 : critical account/ other credits existing (not at this bank)   

# In[16]:

data.history_.value_counts()


# In[17]:

data.groupby('history_')['status'].agg('mean')


# In[18]:

hist = pd.get_dummies(data['history_'], 'History')
hist_ = hist.drop('History_A33', 1)
hist_.head(3)


# In[19]:

X = pd.concat([X, hist_], axis=1)


# #### 4. Purpose

# A40 : car (new)   
# A41 : car (used)   
# A42 : furniture/equipment   
# A43 : radio/television   
# A44 : domestic appliances   
# A45 : repairs   
# A46 : education   
# A47 : (vacation - does not exist?)   
# A48 : retraining   
# A49 : business   
# A410 : others   

# In[20]:

data.purpose_.value_counts()


# In[21]:

data.groupby('purpose_')['status'].agg('mean')


# In[22]:

purp = pd.get_dummies(data.purpose_, 'Purpose')
purp_ = purp.drop('Purpose_A44', 1)
X = pd.concat([X, purp_],1)


# #### 5. Credit amount

# In[23]:

amnt = data.num_amount_A5


# In[24]:

plt.scatter(x=amnt, y=data.status);


# In[25]:

data[['num_amount_A5', 'purpose_']].boxplot(by='purpose_');


# In[26]:

X = pd.concat([X, amnt], 1)


# #### 6. Savings account/bonds 
# 
# A61 : ... < 100 DM   
# A62 : 100 <= ... < 500 DM   
# A63 : 500 <= ... < 1000 DM   
# A64 : .. >= 1000 DM   
# A65 : unknown/ no savings account   

# In[27]:

data.svn_account_.value_counts()


# In[28]:

data.groupby('svn_account_')['status'].agg('mean')


# In[29]:

svn = pd.get_dummies(data.svn_account_, 'Svn_account')
svn.head(3)


# In[30]:

svn_ = svn.drop('Svn_account_A62', axis=1)


# In[31]:

X = pd.concat([X,svn_],axis=1)


# #### 7. Present employment status
# 
# A71 : unemployed   
# A72 : ... < 1 year   
# A73 : 1 <= ... < 4 years   
# A74 : 4 <= ... < 7 years   
# A75 : .. >= 7 years   

# In[32]:

data.empl_status_.value_counts()


# In[33]:

data.groupby('empl_status_')['status'].agg('mean')


# In[34]:

empl = pd.get_dummies(data.empl_status_, "Emp_status")
empl.head(3)


# In[35]:

empl_ = empl.drop('Emp_status_A73',axis=1)


# In[36]:

X = pd.concat([X,empl], axis=1)


# #### 8. Installment rate in percentage of disposable income 

# In[37]:

rt = data.num_income_share_A8


# In[38]:

rt.describe()


# In[39]:

rt.value_counts()


# In[40]:

data.groupby('num_income_share_A8')['status'].agg('mean')


# In[41]:

X = pd.concat([X,rt], axis=1)


# #### 9. Personal status and sex 
# 
# A91 : male : divorced/separated   
# A92 : female : divorced/separated/married   
# A93 : male : single   
# A94 : male : married/widowed   
# A95 : female : single    

# In[42]:

data.marital_status_.value_counts()


# In[43]:

data.groupby('marital_status_')['status'].agg('mean')


# In[44]:

mar = pd.get_dummies(data.marital_status_, 'Marital_status')


# In[45]:

mar_ = mar.drop('Marital_status_A94', axis=1)


# In[46]:

X = pd.concat([X, mar_], axis=1)


# #### 10. Other debtors / guarantors 
# 
# A101 : none  
# A102 : co-applicant  
# A103 : guarantor  

# In[47]:

data.guarantor_.value_counts()


# In[48]:

data.groupby('guarantor_')['status'].agg('mean')


# In[49]:

guar = pd.get_dummies(data.guarantor_, 'Guarantor')
guar.head(3)


# In[50]:

guar_ = guar.drop('Guarantor_A101', axis=1)


# In[51]:

X = pd.concat([X,guar_], axis=1)


# #### 11. Present residence since 

# In[52]:

res = data.num_residence_A11


# In[53]:

res.describe()


# In[54]:

data.groupby('num_residence_A11')['status'].agg('mean')


# In[55]:

X = pd.concat([X,res], axis=1)


# #### 12. Property 
# 
# A121 : real estate   
# A122 : if not A121 : building society savings agreement/ life insurance   
# A123 : if not A121/A122 : car or other, not in attribute 6   
# A124 : unknown / no property 

# In[56]:

data.property_.value_counts()


# In[57]:

data.groupby('property_')['status'].agg('mean')


# In[58]:

prop = pd.get_dummies(data.property_, 'Property')
prop.head(3)


# In[59]:

prop_ = prop.drop('Property_A122', axis=1)


# In[60]:

X = pd.concat([X,prop_], axis=1)


# #### 13. Age in years 

# In[61]:

age = data.num_age_A13


# In[62]:

age.describe()


# In[63]:

plt.subplot(1,2,1)
plt.hist(age)
plt.subplot(1,2,2)
plt.scatter(x=age, y=y);


# In[64]:

import re
ind = np.array([any(re.findall('^num',x)) for x in names])
names[ind]


# #### 14. Other installment plans 
# 
# A141 : bank  
# A142 : stores  
# A143 : none  

# In[65]:

data.other_loans_.value_counts()


# In[66]:

data.groupby('other_loans_')['status'].agg('mean')


# In[67]:

inst = pd.get_dummies(data.other_loans_, 'Installments')
inst.head(3)


# In[68]:

inst_ = inst.drop('Installments_A143', axis=1)


# In[69]:

X = pd.concat([X,inst_], axis=1)


# #### 15. Housing 
# 
# A151 : rent  
# A152 : own  
# A153 : for free  

# In[70]:

data.housing_.value_counts()


# In[71]:

data.groupby('housing_')['status'].agg('mean')


# In[72]:

hous = pd.get_dummies(data.housing_, 'Housing')
hous.head(3)


# In[73]:

hous_ = hous.drop('Housing_A152', axis=1)


# In[74]:

X = pd.concat([X, hous_], axis=1)


# #### 16. Number of existing credits at this bank 

# In[75]:

numc = data.num_credits_A16


# In[76]:

numc.describe()


# In[77]:

data.num_credits_A16.value_counts()


# In[78]:

data.groupby('num_credits_A16')['status'].agg('mean')


# In[79]:

numc = pd.get_dummies(data.num_credits_A16, 'Num_credits')
numc.head(3)


# In[80]:

numc_ = numc.drop('Num_credits_1', axis=1)


# In[81]:

X = pd.concat([X,numc_], axis=1)


# #### 17. Job 
# 
# A171 : unemployed/ unskilled - non-resident  
# A172 : unskilled - resident  
# A173 : skilled employee / official  
# A174 : management/ self-employed/ highly qualified employee/ officer  

# In[82]:

data.job_.value_counts()


# In[83]:

data.groupby('job_')['status'].agg('mean')


# In[84]:

jb = pd.get_dummies(data.job_, 'Job')
jb.head(3)


# In[85]:

jb_ = jb.drop('Job_A173', axis=1)


# In[86]:

x = pd.concat([X, jb_], axis=1)


# #### 18. Number of people being liable to provide maintenance for 

# In[87]:

dep = data.num_dependants_A18


# In[88]:

data.num_dependants_A18.value_counts()


# In[89]:

data.groupby('num_dependants_A18')['status'].agg('mean')


# In[90]:

dpn = pd.get_dummies(data.num_dependants_A18, 'Dependants')
dpn.head(3)


# In[91]:

X = pd.concat([X, dpn['Dependants_2']], axis=1)


# #### 19. Telephone 
# 
# A191 : none  
# A192 : yes, registered under the customers name  

# In[92]:

data.telephone_.value_counts()


# In[93]:

data.groupby('telephone_')['status'].agg('mean')


# In[94]:

tel = pd.get_dummies(data.telephone_, 'Telephone')
tel.head(3)


# In[95]:

X = pd.concat([X, tel['Telephone_A192']], axis=1)


# #### 20. Foreign worker 
# 
# A201 : yes  
# A202 : no   

# In[96]:

data.foreign_.value_counts()


# In[97]:

data.groupby('foreign_')['status'].agg('mean')


# In[98]:

fr = pd.get_dummies(data.foreign_, 'Foreign')
fr.head(3)


# In[99]:

X = pd.concat([X, fr['Foreign_A202']], axis=1)


# ## Train Test Split

# In[102]:

import sklearn.cross_validation as cv


# In[108]:

get_ipython().magic('pinfo cv.train_test_split')


# In[110]:

X_train, X_test, y_train, y_test = cv.train_test_split(X,y, test_size=.2, random_state=1)


# In[111]:

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ## Logistic model

# In[125]:

import sklearn.linear_model as lm
import sklearn.grid_search as gs


# In[120]:

lm_mod = lm.LogisticRegression(class_weight = 'auto', random_state=1, C=1e-5)
lm_mod


# In[174]:

folds = cv.StratifiedShuffleSplit(y_train, n_iter=20, test_size=.1, random_state=1)
lm_grid = {'C': np.logspace(-5,5,10),
           'penalty': ['l1','l2']}


lm_gsearch = gs.GridSearchCV(lm_mod, param_grid = lm_grid, n_jobs=-1, cv=folds)
lm_gsearch


# In[175]:

lm_gsearch.fit(X_train, y_train)


# In[176]:

lm_gsearch.best_score_


# In[177]:

(lm_gsearch.best_estimator_.predict(X_test) == y_test).mean()


# ## Random Forest

# In[157]:

import sklearn.ensemble as ens


# In[160]:

rf_mod = ens.ExtraTreesClassifier(class_weight='auto', n_jobs=-1, n_estimators=1000)
rf_mod


# In[178]:

rf_grid = {'max_depth': np.linspace(1,10, 10)}

rf_gridSearch = gs.GridSearchCV(rf_mod, param_grid=rf_grid, n_jobs=-1, cv=folds)
rf_gridSearch


# In[179]:

rf_gridSearch.fit(X_train, y_train)


# In[180]:

rf_gridSearch.best_score_


# In[181]:

(rf_gridSearch.best_estimator_.predict(X_test) == y_test).mean()


# ## Gradient Boosting Classifier

# In[210]:

mod_gbm = ens.GradientBoostingClassifier(random_state=1, n_estimators=1000)
mod_gbm


# In[211]:

import scipy.stats as st

params = {'max_leaf_nodes'  : st.randint(3,11),             # add 10 n_iter for every line
          'min_samples_leaf': st.randint(3,11),             # add 10 n_iter for every line
          'max_features'    : st.uniform(loc=.1,scale= .5), # add 10 n_iter for every line
          'learning_rate'   : np.logspace(-3,-1,3)
         }

rgrid_search = gs.RandomizedSearchCV(mod_gbm, param_distributions = params,
                                      n_iter=40,            # those add up to 40
                                      random_state=1, cv=folds, n_jobs=-1)
rgrid_search


# In[212]:

rgrid_search.fit(X_train, y_train)


# In[213]:

rgrid_search.best_score_


# In[214]:

rgrid_search.best_params_


# In[215]:

(rgrid_search.best_estimator_.predict(X_test) == y_test).mean()


# ## K-nearest Neghbours

# In[217]:

import sklearn.neighbors as nb


# In[218]:

mod_knn = nb.KNeighborsClassifier()
mod_knn


# In[219]:

knn_grid = {'n_neighbors': st.randint(3,11),
            'weights'    : ['uniform', 'distance'],
            'leaf_size'  : st.randint(10,31),
            'p'          : [1,2]}


knn_gridSearch = gs.RandomizedSearchCV(mod_knn, param_distributions = knn_grid,
                                      n_iter=40,            # those add up to 40
                                      random_state=1, cv=folds, n_jobs=-1)

knn_gridSearch


# In[220]:

knn_gridSearch.fit(X_train, y_train)


# In[224]:

knn_gridSearch.best_score_

