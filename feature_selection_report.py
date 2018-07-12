import numpy as np
import h5py
from kdigo_funcs import feature_selection
from sklearn.feature_selection import chi2
import os

# --------------------------------------------------- PARAMETERS ----------------------------------------------------- #
data_file = '../RESULTS/icu/7days_070118/kdigo_dm.h5'
features = ['apache_norm', 'all_clinical', 'all_trajectory_clusters', 'everything_clusters']
# features = ['all_trajectory_clusters', 'everything_clusters']
lbl_list = ['died_inp', ]
basepath = '../RESULTS/icu/7days_070118/feature_selection/'

dm_tag = 'norm_a1'

# selection_models = [['univariate', [[2, chi2], ]],
#                     ['variance', [[0.03, ],  [0.01, ]]],
#                     ['linear', [[0.1, ], [0.25, ]]],
#                     ['tree', [[], ]]]

selection_models = [['linear', [[0.1, ], [0.25, ]]],
                    ['tree', [[], ]]]

feature_file_path = '../DATA/icu/7days_070118/'

# -------------------------------------------------------------------------------------------------------------------- #


with open(feature_file_path + 'descriptive_features.csv', 'r') as f:
    desc_names = np.array(f.readline().rstrip().split(',')[1:], dtype=str)
with open(feature_file_path + 'template_features.csv', 'r') as f:
    temp_names = np.array(f.readline().rstrip().split(',')[1:], dtype=str)
with open(feature_file_path + 'slope_features.csv', 'r') as f:
    slope_names = np.array(f.readline().rstrip().split(',')[1:], dtype=str)

feature_names = {}
trajectory_names = np.concatenate((desc_names, temp_names, slope_names))
sofa_names = np.array(['sofa_%d' % x for x in range(6)], dtype=str)
apache_names = np.array(['apache_%d' % x for x in range(13)], dtype=str)
all_clinical_names = np.concatenate((sofa_names, apache_names))
everything_names = np.concatenate((all_clinical_names, trajectory_names))

feature_names['apache_norm'] = apache_names
feature_names['sofa_norm'] = sofa_names
feature_names['all_clinical'] = all_clinical_names
feature_names['all_trajectory_clusters'] = trajectory_names
feature_names['everything_clusters'] = everything_names


f = h5py.File(data_file, 'r')
fg = f['features']
lbls = f['meta'][lbl_list[0]]

if not os.path.exists(basepath):
    os.makedirs(basepath)

for feature in features:
    data = fg[feature][:]
    feature_path = basepath + feature + '/'
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    feature_log_name = feature_path + 'support_log.csv'

    feature_log = open(feature_log_name, 'w')
    feature_log.write('model')
    tnames = feature_names[feature]
    for i in range(len(tnames)):
        feature_log.write(',%s' % tnames[i])
    feature_log.write('\n')
    for model in selection_models:
        for i in range(len(model[1])):
            params = model[1][i]
            model_sel, sel = feature_selection(data, lbls, method=model[0], params=params)
            support = model_sel.get_support()
            if len(params) > 0:
                np.savetxt(feature_path + model[0] + '_%.2E_transformed.txt' % params[0], sel, fmt='%.4f')
                feature_log.write(model[0] + '_%.2E' % params[0])
            else:
                np.savetxt(feature_path + model[0] + '_transformed.txt', sel, fmt='%.4f')
                feature_log.write(model[0])
            for j in range(len(support)):
                feature_log.write(',%d' % support[j])
            feature_log.write('\n')
    feature_log.close()

