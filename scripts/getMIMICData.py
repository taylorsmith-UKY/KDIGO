import os
import numpy as np
import json
import pandas as pd
from utility_funcs import get_date
from kdigo_funcs import baseline_est_gfr_mdrd, calc_gfr
import datetime

fp = open('../kdigo_conf.json', 'r')
conf = json.load(fp)
fp.close()
basePath = conf['basePath']         # Path containing the DATA and RESULTS directories
cohortName = conf['cohortName']     # Will be used for folder name
t_lim = conf['analysisDays']        # How long to consider in the analysis
tRes = conf['timeResolutionHrs']    # Resolution after imputation in hours
analyze = conf['analyze']           # Time period to analyze (hospital/ICU/all)

dataPath = os.path.join(basePath, 'DATA', 'mimic-iii-clinical-database-1.4')
patients = pd.read_csv(os.path.join(dataPath, 'PATIENTS.csv'))
admissions = pd.read_csv(os.path.join(dataPath, 'ADMISSIONS.csv'))
events = pd.read_csv(os.path.join(dataPath, 'DATETIMEEVENTS.csv'))
charts = pd.read_csv(os.path.join(dataPath, 'CHARTEVENTS.csv'))
icuStays = pd.read_csv(os.path.join(dataPath, 'ICUSTAYS.csv'))

lab_key = pd.read_csv(os.path.join(dataPath, 'D_LABITEMS.csv'))
i_key = pd.read_csv(os.path.join(dataPath, 'D_ITEMS.csv'))

idx = np.where(lab_key['LABEL'].values == "Creatinine, Serum")[0][0]
labid = lab_key['ITEMID'][idx]

# hd_keys = []
# rrt_keys = []
# for i, x in enumerate(i_key['LABEL'].values.astype(str)):
#     if 'hemodia' in x.lower() and 'off' not in x.lower() and 'removal' not in x.lower():
#         hd_keys.append(['ITEMID'][i])

rrt_keys = [i_key['ITEMID'][x] for x in range(len(i_key)) if 'dialysis -' in str(i_key['LABEL'][x]).lower()]

labs = pd.read_csv(os.path.join(dataPath, 'LABEVENTS.csv'))
data = {}
for i in range(len(patients)):
    # Get patient ID, date of death, and DOB
    pid = patients['SUBJECT_ID'][i]
    eth = 2
    age = 0
    sex = 0
    dob = 'nan'
    dod = 'nan'
    if "white" in patients['ETHNICITY'][i].lower():
        eth = 0
    elif "black" in patients['ETHNICITY'][i].lower():
        eth = 1

    dod = get_date(patients['DOD'][i])
    dob = get_date(patients['DOB'][i])

    if patients['GENDER'][i] == 'M':
        sex = 1

    # Get hospital admissions and ethnicity
    idx = np.where(admissions['SUBJECT_ID'] == pid)[0]
    hadmits = [get_date(admissions['ADMITTIME'][idx[0]])]
    hdischs = [get_date(admissions['DISCHTIME'][idx[0]])]
    for tidx in idx[1:]:
        tadmit = get_date(admissions['ADMITTIME'][tidx])
        tdisch = get_date(admissions['DISCHTIME'][tidx])
        hadmits.append(tadmit)
        hdischs.append(tdisch)

    data[pid] = {}
    data[pid]['allscr'] = []
    data[pid]['alldates'] = []
    data[pid]['icuscr'] = []
    data[pid]['icudates'] = []
    data[pid]['kdigo'] = []

    # Get ICU admission
    idx = np.where(icuStays['SUBJECT_ID'] == pid)[0]
    admit = get_date(admissions['INTIME'][idx[0]])
    disch = get_date(admissions['OUTTIME'][idx[0]])
    for tidx in idx[1:]:
        tadmit = get_date(admissions['ADMITTIME'][tidx])
        tdisch = get_date(admissions['DISCHTIME'][tidx])
        if tadmit - disch < datetime.timedelta(2):
            disch = tdisch

    # Select the correct hospital admission if multiple
    hadmit = None
    hdisch = None
    for j in range(len(hadmits)):
        if hadmits[j] <= admit <= hdischs[j]:
            hadmit = hadmits[j]
            hdisch = hdischs[j]

    assert hadmit is not None

    # Calculate age
    age = (admit - dob).total_seconds() / (60 * 60 * 24 * 365)

    # get baselines
    prows = np.where(labs['SUBJECT_ID'].values == pid)[0]
    chartRows = np.where(charts['SUBJECT_ID'].values == pid)[0]
    rows = prows[np.where(labs['ITEMID'].values[prows] == labid)[0]]
    bsln = None
    btype = None
    for row in rows:
        tdate = get_date(labs['CHARTTIME'][row])
        if datetime.timedelta(365) < admit - tdate:
            if tdate >= hadmit and admit - tdate > datetime.timedelta(1):
                bsln = labs['VALUE'][row]
                btype = 'inpatient'
            elif tdate < hadmit and admit - tdate > datetime.timedelta(2):
                bsln = labs['VALUE'][row]
                btype = 'outpatient'
        elif tdate > admit:
            break

    # get RRT

    if bsln is None:
        btype = 'imputed'
        bsln = baseline_est_gfr_mdrd(75, sex, eth, age)
        bgfr = 75
    else:
        bgfr = calc_gfr(bsln, sex, eth, age)

    data[pid]['baseline_scr'] = bsln
    data[pid]['baseline_gfr'] = bgfr

    for row in rows:
        tdate = get_date(labs['CHARTTIME'][row])
        tval = labs['VALUENUM'][row]
        data[pid]['alldates'].append(tdate)
        data[pid]['allscr'].append(tval)
        if admit <= tdate < admit + datetime.timedelta(14):
            data[pid]['icudates'].append(tdate)
            data[pid]['icuscr'].append(tval)

        backIdx = row
        mval = tval
        while get_date(labs['CHARTTIME'][backIdx]) > tdate - datetime.timedelta(2):
            mval = min(mval, labs['VALUENUM'][backIdx])

        myChartItemIds = charts['ITEMID'][chartRows]

        kval = 0
        rrtrows = np.array([], dtype=int)
        for rrtid in rrt_keys:
            rrtrows = np.union1d(rrtrows, chartRows[np.where(myChartItemIds == rrtid)[0]])
        for rrtrow in rrtrows:
            start = None
            stop = None
            if start <= tdate < stop:
                kval = 4
                break

        if not kval:
            if kval < 1.5 * bsln:
                if kval > mval + 0.3:
                    kval = 1
            elif kval < 1.5 * bsln:
                kval = 1
            elif kval < 2 * bsln:
                kval = 2
            elif kval < 3 * bsln:
                kval = 3

        data[pid]['kdigo'].append(kval)
