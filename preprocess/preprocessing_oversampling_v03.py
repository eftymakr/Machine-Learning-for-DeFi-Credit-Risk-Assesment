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


df = pd.read_csv('../Stat_Learning_set_v02.csv', nrows=400000)

y = df.liquidations.fillna(0)
X = df.drop(['address', 'liquidations'], axis=1).fillna(0)
addresses = df[['address']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
print('Percentage liquidations', y.mean(), '\nTotal number liquidations', sum(y), '\nTotal records', len(y))

Xn = X
yn = np.array(list(le.fit_transform(y)))


oversample = SMOTE()

chunk_size = 50000
num_chunks = int(np.ceil(len(X_train) / chunk_size))


oversample_in_chunks = True
print(oversample_in_chunks)


if oversample_in_chunks:
    X_train_resampled_chunks = []
    y_train_resampled_chunks = []
    print('mpike')

    output_dir = '../oversampled_chunks'
    print("Output directory:", output_dir)
    if not os.path.exists(output_dir):
       print("Directory does not exist. Creating it.")
       os.makedirs(output_dir)

    for i in range(num_chunks):
        print('Oversampling chunk {}/{}'.format(i, num_chunks-1))
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size

        X_chunk = X_train[start_idx:end_idx]
        y_chunk = y_train[start_idx:end_idx]

        X_chunk_resampled, y_chunk_resampled = oversample.fit_resample(X_chunk, y_chunk)

#        X_train_resampled_chunks.append(X_chunk_resampled)
#        y_train_resampled_chunks.append(y_chunk_resampled)

        np.save(os.path.join(output_dir, f'X_train_chunk_{i}.npy'), X_chunk_resampled)
        np.save(os.path.join(output_dir, f'y_train_chunk_{i}.npy'), y_chunk_resampled)


else:
    X_train, y_train = oversample.fit_resample(X_train, y_train)

print('Oversampling done')

# Saving resampled data

with open('../data_test.pkl', 'wb') as f:
    pickle.dump((X_test, y_test), f)

with open('../full_data.pkl', 'wb') as f:
    pickle.dump((X, y, addresses), f)

print('done')
