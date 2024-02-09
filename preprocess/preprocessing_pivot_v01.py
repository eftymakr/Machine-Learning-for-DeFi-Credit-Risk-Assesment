import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np 
import warnings
import pickle
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import fbeta_score
import pickle
from sklearn.model_selection import StratifiedKFold

from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn import svm

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import warnings
import pickle
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import fbeta_score
import pickle
from sklearn.model_selection import StratifiedKFold

from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn import svm

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


import pandas as pd
import matplotlib.pyplot as plt

nb_steps = 42

# dfr = pd.read_csv('ML_input_v03.csv', nrows=100*nb_steps)
dfr = pd.read_csv('../ML_input_v03.csv')
print('file read')

columns_list = dfr.columns.to_list()[2:]

dfn = pd.DataFrame()

for i in range(nb_steps):
    dfm = dfr.iloc[i::nb_steps, :]
    dfm = dfm.reset_index()
    dfm
    if i == 0:
        dfm = dfm.drop(columns=['index', 'YearMonth'])
    else:
        dfm = dfm.drop(columns=['index', 'YearMonth', 'address'])

    if i == nb_steps - 1:
        i = 999
        #dfm['liquidations'] = dfm.apply(lambda x: 1 if x.liquidation_total_count > 0 else 0, axis=1)
        dfm['liquidations'] = dfm.apply(lambda x: 1 if (x.liquidation_total_count != 0) else 2 if (x.borrow_total_count != 0) else 3 if (x.deposit_total_count != 0) else 4 if (x.flashloan_total_count != 0) else 5 if (x.repay_total_count != 0) else 0, axis=1)

        dfn = pd.concat([dfn, dfm.liquidations], axis=1)
    else:
        for c in columns_list:
            dfm = dfm.rename(columns={c: c + "_" + str(i)})

        dfn = pd.concat([dfn, dfm], axis=1)


dfn.to_csv('../Stat_Learning_set_v02.csv', index=False)


print('done')
