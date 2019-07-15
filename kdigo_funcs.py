from __future__ import division
import datetime
import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import braycurtis, squareform, euclidean, cityblock
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, interp1d
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, RFECV
from sklearn.model_selection import StratifiedKFold
import multiprocessing as mp
import re
import itertools
import h5py
from tqdm import tqdm
import os


# %%
def get_dialysis_mask(dataPath='', scr_fname='SCR_ALL_VALUES.csv', rrt_fname='RENAL_REPLACE_THERAPY.csv',
                      scr_dcol='SCR_ENTERED', id_col='STUDY_PATIENT_ID', rrtSep='col', v=True):
    '''
    Returns a mask for all records in scr_m indicating whether the corresponding patient was on dialysis at the time
    that the record was taken
    :return mask: vector with 0 indicating no dialysis, 1 = HD, 2 = CRRT, 3 = PD
    '''
    # mask is same length as number of SCr records
    scr_m = pd.read_csv(os.path.join(dataPath, scr_fname))
    rrt_m = pd.read_csv(os.path.join(dataPath, rrt_fname))
    mask = np.zeros(len(scr_m))
    # Get column indices corresponding to different types of RRT
    if rrtSep == 'col':
        for k in list(rrt_m):
            kv = k.lower()
            if 'start' in kv and 'hd' in kv:
                hd_start = k
            if 'stop' in kv and 'hd' in kv:
                hd_stop = k
            if 'start' in kv and 'crrt' in kv:
                crrt_start = k
            if 'stop' in kv and 'crrt' in kv:
                crrt_stop = k
            if 'start' in kv and 'pd' in kv:
                pd_start = k
            if 'stop' in kv and 'pd' in kv:
                pd_stop = k
    # Get column indices for RRT start/stop
    elif rrtSep == 'row':
        for k in list(rrt_m):
            kv = k.lower()
            if 'start' in kv:
                start_col = k
            elif 'stop' in kv:
                stop_col = k
            elif 'type' in kv:
                type_col = k
    for i in range(len(mask)):
        # ID of the current patient
        this_id = scr_m[id_col][i]
        this_date = get_date(str(scr_m[scr_dcol][i]).lower())
        if this_date == 'nan' or this_date == 'nat':
            continue
        rows = np.where(rrt_m[id_col].values == this_id)[0]
        # check dialysis dates for this patient and assign mask value if current value was recorded during dialysis
        for row in rows:
            if rrtSep == 'col':
                left = get_date(str(rrt_m[hd_start][row]).split('.')[0])
                right = get_date(str(rrt_m[hd_stop][row]).split('.')[0])
                if left != 'nan' and right != 'nan':
                    if left < this_date < right + datetime.timedelta(2):
                        mask[i] = 1
                left = get_date(str(rrt_m[crrt_start][row]).split('.')[0])
                right = get_date(str(rrt_m[crrt_stop][row]).split('.')[0])
                if left != 'nan' and right != 'nan':
                    if left < this_date < right + datetime.timedelta(2):
                        mask[i] = 2
                left = get_date(str(rrt_m[pd_start][row]).split('.')[0])
                right = get_date(str(rrt_m[pd_stop][row]).split('.')[0])
                if left != 'nan' and right != 'nan':
                    if left < this_date < right + datetime.timedelta(2):
                        mask[i] = 3
            elif rrtSep == 'row':
                left = get_date(str(rrt_m[start_col][row]).split('.')[0])
                right = get_date(str(rrt_m[stop_col][row]).split('.')[0])
                if left != 'nan' and right != 'nan':
                    if left < this_date < right + datetime.timedelta(2):
                        if 'hd' in rrt_m[type_col][row].lower():
                            mask[i] = 1
                        elif 'crrt' in rrt_m[type_col][row].lower():
                            mask[i] = 2
                        elif 'pd' in rrt_m[type_col][row].lower():
                            mask[i] = 3
    return mask


# %%
def extract_scr_data(icu_windows, hosp_windows, dataPath='', scr_fname='SCR_ALL_VALUES.csv', rrt_fname='RENAL_REPLACE_THERAPY.csv',
                      scr_dcol='SCR_ENTERED', scr_vcol='SCR_VALUE', id_col='STUDY_PATIENT_ID', rrtSep='col'):
    scr_m = pd.read_csv(os.path.join(dataPath, 'SCR_ALL_VALUES.csv'))
    scr_m.sort_values(by=['STUDY_PATIENT_ID', 'SCR_ENTERED'], inplace=True)
    rrt_m = pd.read_csv(os.path.join(dataPath, 'RENAL_REPLACE_THERAPY.csv'))
    scrs = []
    dates = []
    crrt_masks = []
    hd_masks = []
    pd_masks = []
    rrt_masks = []
    tmasks = []
    ids = list(hosp_windows)
    if rrtSep == 'col':
        for k in list(rrt_m):
            kv = k.lower()
            if 'start' in kv and 'hd' in kv:
                hd_start_col = k
            if 'stop' in kv and 'hd' in kv:
                hd_stop_col = k
            if 'start' in kv and 'crrt' in kv:
                crrt_start_col = k
            if 'stop' in kv and 'crrt' in kv:
                crrt_stop_col = k
            if 'start' in kv and 'pd' in kv:
                pd_start_col = k
            if 'stop' in kv and 'pd' in kv:
                pd_stop_col = k
    # Get column indices for RRT start/stop
    elif rrtSep == 'row':
        for k in list(rrt_m):
            kv = k.lower()
            if 'start' in kv:
                start_col = k
            elif 'stop' in kv:
                stop_col = k
            elif 'type' in kv:
                type_col = k
    for i in range(len(ids)):
        tid = ids[i]
        sidx = np.where(scr_m[id_col].values == tid)[0]
        didx = np.where(rrt_m[id_col].values == tid)[0]
        hosp_admit, hosp_disch = hosp_windows[tid]
        icu_admit, icu_disch = icu_windows[tid]
        crrt_start = crrt_stop = hd_start = hd_stop = pd_start = pd_stop = None
        if didx.size > 0:
            if rrtSep == 'col':
                idx = didx[0]
                tstart = get_date(str(rrt_m[crrt_start_col][idx]))
                if tstart != 'nan':
                    crrt_start = tstart
                    crrt_stop = get_date(str(rrt_m[crrt_stop_col][idx]))
                tstart = get_date(str(rrt_m[hd_start_col][idx]))
                if tstart != 'nan':
                    hd_start = tstart
                    hd_stop = get_date(str(rrt_m[hd_stop_col][idx]))
                tstart = get_date(str(rrt_m[pd_start_col][idx]))
                if tstart != 'nan':
                    pd_start = tstart
                    pd_stop = get_date(str(rrt_m[pd_stop_col][idx]))
            elif rrtSep == 'row':
                for idx in didx:
                    tstart = get_date(str(rrt_m[start_col][idx]))
                    if tstart != 'nan':
                        tstop = get_date(str(rrt_m[stop_col][idx]))
                        tp = rrt_m[type_col][idx].lower()
                        if 'hd' in tp:
                            hd_start = tstart
                            hd_stop = tstop
                        if 'crrt' in tp:
                            crrt_start = tstart
                            crrt_stop = tstop
                        if 'pd' in tp:
                            pd_start = tstart
                            pd_stop = tstop
        scr = []
        datel = []
        tmask = []
        crrt_mask = []
        hd_mask = []
        pd_mask = []
        rrt_mask = []
        for idx in sidx:
            tval = scr_m[scr_vcol][idx]
            tdate = get_date(scr_m[scr_dcol][idx])
            if tdate == 'nan' or str(tval) == 'nan':
                continue
            scr.append(tval)
            datel.append(tdate)
            if icu_admit <= tdate <= icu_disch:
                tmask.append(2)
            elif hosp_admit <= tdate <= hosp_disch:
                tmask.append(1)
            else:
                tmask.append(0)
            dval = 0
            if hd_start is not None and hd_start <= tdate <= hd_stop:
                hd_mask.append(1)
                dval = 1
            else:
                hd_mask.append(0)
            if crrt_start is not None and crrt_start <= tdate <= crrt_stop:
                crrt_mask.append(1)
                dval = 1
            else:
                crrt_mask.append(0)
            if pd_start is not None and pd_start <= tdate <= pd_stop:
                pd_mask.append(1)
                dval = 1
            else:
                pd_mask.append(0)
            rrt_mask.append(dval)
        scrs.append(np.array(scr))
        dates.append(np.array(datel, dtype=str))
        tmasks.append(np.array(tmask, dtype=int))
        hd_masks.append(np.array(hd_mask, dtype=int))
        crrt_masks.append(np.array(crrt_mask, dtype=int))
        pd_masks.append(np.array(pd_mask, dtype=int))
        rrt_masks.append(np.array(rrt_mask, dtype=int))
    return scrs, dates, tmasks, hd_masks, crrt_masks, pd_masks, rrt_masks


def extract_masked_data(data_list, masks, sel=1):
    out = []
    for i in range(len(data_list)):
        out_d = []
        data = data_list[i]
        assert len(data) == len(masks)
        for j in range(len(data)):
            mask = masks[j]
            vec = np.array(data[j])
            idx = np.where(mask == sel)[0]
            if idx.size > 0:
                out_d.append(vec[idx])
            else:
                out_d.append([])
        out.append(out_d)
    return out

# %%
def get_t_mask(dataPath='', scr_fname='SCR_ALL_VALUES.csv', date_fname = 'ADMISSION_INDX.csv', scr_dcol='SCR_ENTERED', id_col='STUDY_PATIENT_ID'):
    '''
    Returns a mask indicating whether each SCr value was recorded prior to admission, in the hospital, or in the ICU
    :return:
    '''
    scr_m = pd.read_csv(os.path.join(dataPath, scr_fname))
    date_m = pd.read_csv(os.path.join(dataPath, date_fname))
    for k in list(date_m):
        kv = k.lower()
        if 'hosp' in kv or 'hsp' in kv:
            if 'admit' in kv or 'admsn' in kv:
                hosp_start = k
            elif 'disch' in kv:
                hosp_stop = k
        if 'icu' in kv:
            if 'admit' in kv or 'admsn' in kv:
                icu_start = k
            elif 'disch' in kv:
                icu_stop = k
    mask = np.zeros(len(scr_m))
    current_id = 0
    hosp_windows = {}
    icu_windows = {}
    cur_hosp_window = None
    cur_icu_window = None
    no_admit_info = []
    for i in range(len(mask)):
        this_id = scr_m[id_col][i]
        # if new ID, previous is finished.
        if this_id != current_id:
            if cur_hosp_window is not None:
                hosp_windows[current_id] = cur_hosp_window
            if cur_icu_window is not None:
                icu_windows[current_id] = cur_icu_window
            current_id = this_id
            date_idx = np.where(date_m[id_col].values == current_id)[0]
            if len(date_idx) == 0:
                continue
            cur_hosp_window = None
            cur_icu_window = None
            for j in range(len(date_idx)):
                idx = date_idx[j]
                cur_hosp_start = get_date(date_m[hosp_start][idx])
                cur_hosp_stop = get_date(date_m[hosp_stop][idx])
                if cur_hosp_start == 'nan' or cur_hosp_stop == 'nan':
                    continue

                if cur_hosp_window is None:
                    cur_hosp_window = [cur_hosp_start, cur_hosp_stop]
                else:
                    if cur_hosp_start > cur_hosp_window[0]:
                        if cur_hosp_stop > cur_hosp_window[1]:
                            cur_hosp_window[1] = cur_hosp_stop
                    elif cur_hosp_start < cur_hosp_window[0]:
                        if cur_hosp_stop > cur_hosp_window[0]:
                            cur_hosp_window[0] = cur_hosp_start
                    elif cur_hosp_stop < cur_hosp_window[0]:
                        if (cur_hosp_window[0] - cur_hosp_stop).days >= 2:
                            cur_hosp_window = [cur_hosp_start, cur_hosp_stop]
                        else:
                            cur_hosp_window[0] = cur_hosp_start
                    elif cur_hosp_start > cur_hosp_window[1]:
                        if (cur_hosp_start - cur_hosp_window[1]).days >= 2:
                            continue
                        else:
                            cur_hosp_window[1] = cur_hosp_stop

                cur_icu_start = get_date(date_m[icu_start][idx])
                cur_icu_stop = get_date(date_m[icu_stop][idx])
                if cur_icu_start == 'nan' or cur_icu_stop == 'nan':
                    continue

                if cur_icu_window is None:
                    cur_icu_window = [cur_icu_start, cur_icu_stop]
                else:
                    if cur_icu_start > cur_icu_window[0]:
                        if cur_icu_stop > cur_icu_window[1]:
                            cur_icu_window[1] = cur_icu_stop
                    elif cur_icu_start < cur_icu_window[0]:
                        if cur_icu_stop > cur_icu_window[0]:
                            cur_icu_window[0] = cur_icu_start
                    elif cur_icu_stop < cur_icu_window[0]:
                        if (cur_icu_window[0] - cur_icu_stop).days >= 2:
                            cur_icu_window = [cur_icu_start, cur_icu_stop]
                        else:
                            cur_icu_window[0] = cur_icu_start
                    elif cur_icu_start > cur_icu_window[1]:
                        if (cur_icu_start - cur_icu_window[1]).days >= 2:
                            continue
                        else:
                            cur_icu_window[1] = cur_icu_stop
                
        this_date = get_date(scr_m[scr_dcol][i])
        if this_date == 'nan':
            continue
        if cur_hosp_window is not None:
            try:
                if cur_hosp_window[0] <= this_date <= cur_hosp_window[1]:
                    mask[i] = 1
            except TypeError:
                pass
        else:
            no_admit_info.append(this_id)
        if cur_icu_window is not None:
            try:
                if cur_icu_window[0] <= this_date <= cur_icu_window[1]:
                    mask[i] = 2
            except TypeError:
                pass
        else:
            no_admit_info.append(this_id)
    hosp_windows[current_id] = cur_hosp_window
    icu_windows[current_id] = cur_icu_window
    return mask, hosp_windows, icu_windows


def get_exclusion_criteria(ids, mask, icu_windows, genders, races, ages, dataPath='', id_col='STUDY_PATIENT_ID'):
    scr_all_m = pd.read_csv(os.path.join(dataPath, 'SCR_ALL_VALUES.csv'))
    diag_m = pd.read_csv(os.path.join(dataPath, 'DIAGNOSIS.csv'))
    scm_esrd_m = pd.read_csv(os.path.join(dataPath, 'ESRD_STATUS.csv'))
    usrds_esrd_m = pd.read_csv(os.path.join(dataPath, 'USRDS_ESRD.csv'))
    esrd_man_rev = pd.read_csv(os.path.join(dataPath, 'ESRD_MANUAL_REVISION.csv'))
    bsln_m = pd.read_csv(os.path.join(dataPath, 'all_baseline_info.csv'))
    date_m = pd.read_csv(os.path.join(dataPath, 'ADMISSION_INDX.csv'))
    surg_m = pd.read_csv(os.path.join(dataPath, 'SURGERY_INDX.csv'))
    mort_m = pd.read_csv(os.path.join(dataPath, 'OUTCOMES_COMBINED.csv'))

    hdr = 'patientID,No values in first 7 days, Only 1 value, Last SCr - First SCr < 6hrs, Missing Demographics, Missing DOB, Missing Admit Info, Baseline SCr > 4, Age < 18, Died < 48 hrs after ICU admission, ESRD, Baseline eGFR < 15, Kidney Transplant'
    # NOT_ENOUGH = 0
    NO_VALS_ICU = 0
    ONLY_1VAL = 1
    TOO_SHORT = 2
    MISSING_DEM = 3
    NO_DOB = 4
    NO_ADMIT = 5
    SCR_GT4 = 6
    AGE_LT18 = 7
    DIED_LT48 = 8
    ESRD = 9
    GFR_LT15 = 10
    KID_TPLT = 11
    exclusions = np.zeros((len(ids), 12))
    for i in tqdm(range(len(ids)), desc='Getting Exclusion Criteria'):
        tid = ids[i]
        exc = np.zeros(12, dtype=int)
        exc_name = ''
        all_rows = np.where(scr_all_m[id_col].values == tid)[0]
        sel = np.where(mask[all_rows] == 2)[0]
        keep = all_rows[sel]
        if len(sel) > 1:
            first_date = get_date(scr_all_m['SCR_ENTERED'][keep[0]])
            last_date = get_date(scr_all_m['SCR_ENTERED'][keep[-1]])
            dif = (last_date - first_date).total_seconds() / (60 * 60)
        # Ensure this patient has values in time period of interest
        if len(sel) == 0:
            exclusions[i][NO_VALS_ICU] = 1
        elif len(sel) == 1:
            exclusions[i][ONLY_1VAL] = 1
        elif dif < 6:
            exclusions[i][TOO_SHORT] = 1

        sex = genders[i]
        race = races[i]
        # get demographics and remove if any required are missing
        if np.isnan(sex) or np.isnan(race):
            sex = 'nan'
            race = 'nan'
            exclusions[i][MISSING_DEM] = 1

        age = ages[i]
        if np.isnan(age):
            exclusions[i][NO_DOB] = 1
            dob = 'nan'

        # Get Baseline or remove if no admit dates provided
        bsln_idx = np.where(bsln_m[id_col].values == tid)[0]
        bsln = bsln_m['bsln_val'][bsln_idx[0]]
        btype = bsln_m['bsln_type'][bsln_idx[0]]
        if btype == 'mdrd':
            btype = 'imputed'
        else:
            btype = 'measured'
        if np.isnan(bsln):
            exclusions[i][NO_ADMIT] = 1
        if bsln >= 4.0:
            exclusions[i][SCR_GT4] = 1

        try:
            icu_admit, icu_disch = icu_windows[tid]
            if type(icu_admit) != datetime.datetime:
                icu_admit = get_date(icu_admit)
                icu_disch = get_date(icu_disch)
        except KeyError:
            exclusions[i][NO_ADMIT] = 1
            icu_admit = 'nan'
            icu_disch = 'nan'

        # get age and remove if less than 18
        if age != 'nan':
            if age < 18:
                exclusions[i][AGE_LT18] = 1
        else:
            age = 'nan'

        # get mortality date if available
        mort_idx = np.where(mort_m['STUDY_PATIENT_ID'].values == tid)[0]
        mort_date = 'nan'
        if mort_idx.size > 0:
            for j in range(len(mort_idx)):
                midx = mort_idx[j]
                mort_date = get_date(str(mort_m['DECEASED_DATE'][midx]).split('.')[0])
                if mort_date != 'nan':
                    break
        if mort_date != 'nan' and icu_admit != 'nan':
            death_dur = (mort_date - icu_admit).total_seconds() / (60 * 60 * 24)
            if death_dur < 2:
                exclusions[i][DIED_LT48] = 1
        else:
            death_dur = np.nan

        # remove if ESRD status
        # Check manual revision
        esrd_idx = np.where(esrd_man_rev['STUDY_PATIENT_ID'].values == tid)[0]
        if len(esrd_idx) > 0:
            esrd_idx = esrd_idx[0]
            if esrd_man_rev['before'][esrd_idx]:
                exclusions[i][ESRD] = 1
        else:
            # Check scm flags
            esrd_idx = np.where(scm_esrd_m['STUDY_PATIENT_ID'].values == tid)[0]
            if len(esrd_idx) > 0:
                esrd_idx = esrd_idx[0]
                if scm_esrd_m['BEFORE_INDEXED_INDICATOR'][esrd_idx] == 'Y' or scm_esrd_m['AT_INDEXED_INDICATOR'][esrd_idx] == 'Y':
                    exclusions[i][ESRD] = 1
            # Check USRDS
            esrd_idx = np.where(usrds_esrd_m['STUDY_PATIENT_ID'].values == tid)[0]
            if len(esrd_idx) > 0:
                esrd_idx = esrd_idx[0]
                tdate = get_date(str(usrds_esrd_m['ESRD_DATE'][esrd_idx]))
                if tdate != 'nan':
                    if tdate < icu_admit:  # Before admit
                        exclusions[i][ESRD] = 1
                    elif icu_admit - tdate < 1 and icu_admit.day == tdate.day:  # Same day as admit
                        exclusions[i][ESRD] = 1

        # remove patients with baseline GFR < 15
        if bsln != 'nan' and age != 'nan':
            gfr = calc_gfr(bsln, sex, race, age)
            if gfr < 15:
                exclusions[i][GFR_LT15] = 1

        # remove patients with kidney transplant
        diagnosis_rows = np.where(diag_m['STUDY_PATIENT_ID'].values == tid)[0]
        for row in diagnosis_rows:
            str_des = str(diag_m['DIAGNOSIS_DESC'][row]).upper()
            icd_type = str(diag_m['ICD_TYPECODE'][row]).upper()
            if icd_type == 'nan':  # No ICD code provided
                if str_des == 'KIDNEY/PANCREAS FROM BATAVIA  ETA 1530':
                    exclusions[i][KID_TPLT] = 1
                    break
                if 'TRANS' in str_des:
                    if 'KID' in str_des or 'RENA' in str_des:
                        exclusions[i][KID_TPLT] = 1
                        break
            elif icd_type == 'ICD9':
                icd_code = str(diag_m['ICD_CODE'][row]).upper()
                if icd_code in ['V42.0', '996.81']:
                    exclusions[i][KID_TPLT] = 1
                    break
            elif icd_type == 'ICD0':
                icd_code = str(diag_m['ICD_CODE'][row]).upper()
                if icd_code in ['Z94.0', 'T86.10', 'T86.11', 'T86.12', 'T86.13', 'T86.19']:
                    exclusions[i][KID_TPLT] = 1
                    break
        transplant_rows = np.where(surg_m['STUDY_PATIENT_ID'].values == tid)  # rows in surgery sheet
        for row in transplant_rows:
            str_des = str(surg_m['SURGERY_DESCRIPTION'][row]).upper()
            if 'TRANS' in str_des:
                if 'KID' in str_des or 'RENA' in str_des:
                    exclusions[i][KID_TPLT] = 1
                    break

    del scr_all_m
    del bsln_m
    del diag_m
    del date_m
    del surg_m
    return exclusions, hdr


def get_patients(scr_all_m, scr_val_loc, scr_date_loc, d_disp_loc,
                     mask, rrt_mask, icu_windows, hosp_windows,
                     diag_m, diag_loc, icd_type_loc, icd_code_loc,
                     scm_esrd_m, scm_esrd_before, scm_esrd_at,
                     esrd_man_rev, man_rev_bef,
                     usrds_esrd_m, usrds_esrd_date_loc,
                     bsln_m, bsln_scr_loc, bsln_type_loc,
                     date_m,
                     surg_m, surg_desc_loc,
                     dem_m, sex_loc, eth_loc,
                     dob_m, birth_loc,
                     mort_m, mdate_loc,
                     log, exc_log, exc_list='exclusions.csv', v=True, pre_ids=None):
    # Lists to store records for each patient
    scr = []
    tmasks = []  # time/date
    dmasks = []  # dialysis
    dates = []
    d_disp = []
    ids_out = []
    bslns = []
    bsln_gfr = []
    btypes = []
    t_range = []
    ages = []
    days = []

    # Counters for total number of patients and how many removed for each exclusion criterium
    if pre_ids is None:
        ids = np.unique(scr_all_m[:, 0]).astype(int)
        ids.sort()
    else:
        ids = pre_ids
    if type(log) == str:
        log = open(log, 'w')
    if type(exc_log) == str:
        exc_log = open(exc_log, 'w')
    if type(exc_list) == str:
        exc_list = open(exc_list, 'w')

    log.write('Patient_ID,Admit_Date,Discharge_Date,Baseline_SCr,Mort_Date,Days_To_Death\n')
    exc_log.write(
        'patientID,No values in first 7 days, Only 1 value, Last SCr - First SCr < 6hrs, Missing Demographics, Missing DOB, Missing Admit Info, Baseline SCr > 4, Age < 18, Died < 48 hrs after ICU admission, ESRD, Baseline eGFR < 15, Kidney Transplant\n')
    exc_list.write('Patient_ID,First_Exclusion\n')
    # NOT_ENOUGH = 0
    NO_VALS_ICU = 0
    ONLY_1VAL = 1
    TOO_SHORT = 2
    MISSING_DEM = 3
    NO_DOB = 4
    NO_ADMIT = 5
    SCR_GT4 = 6
    AGE_LT18 = 7
    DIED_LT48 = 8
    ESRD = 9
    GFR_LT15 = 10
    KID_TPLT = 11
    for tid in tqdm(ids, desc='Getting Patient Data'):
        exc = np.zeros(12, dtype=int)
        exc_name = ''
        all_rows = np.where(scr_all_m[:, 0] == tid)[0]
        sel = np.where(mask[all_rows])[0]
        keep = all_rows[sel]
        if len(sel) > 1:
            first_date = get_date(scr_all_m[keep[0], scr_date_loc])
            last_date = get_date(scr_all_m[keep[-1], scr_date_loc])
            dif = (last_date - first_date).total_seconds() / (60 * 60)
        # Ensure this patient has values in time period of interest
        if len(sel) == 0:
            if not np.any(exc):
                exc_name = "No ICU Values"
            exc[NO_VALS_ICU] = 1
        elif len(sel) == 1:
            if not np.any(exc):
                exc_name = "Only 1 Value in ICU"
            exc[ONLY_1VAL] = 1
        elif dif < 6:
            if not np.any(exc):
                exc_name = "Last SCr - First SCr <6 hours"
            exc[TOO_SHORT] = 1

        # get demographics and remove if any required are missing
        if tid not in dem_m[:, 0]:
            sex = 'nan'
            race = 'nan'
            if not np.any(exc):
                exc_name = "Missing Demographics"
            exc[MISSING_DEM] = 1
        else:
            dem_idx = np.where(dem_m[:, 0] == tid)[0]
            if len(dem_idx) > 1:
                dem_idx = dem_idx[0]
            sex = str(dem_m[dem_idx, sex_loc])
            race = str(dem_m[dem_idx, eth_loc])
            if sex == 'nan' or race == 'nan':
                if not np.any(exc):
                    exc_name = "Missing Demographics"
                exc[MISSING_DEM] = 1
            else:
                if sex == 'M':
                    sex = 1
                elif sex == 'F':
                    sex = 0

                if "BLACK" in race:
                    race = 1
                else:
                    race = 0

        # get dob, sex, and race
        if tid not in dob_m[:, 0]:
            if not np.any(exc):
                exc_name = "Missing DOB"
            exc[NO_DOB] = 1
            dob = 'nan'
        else:
            birth_idx = np.where(dob_m[:, 0] == tid)[0][0]
            dob = str(dob_m[birth_idx, birth_loc]).split('.')[0]
            dob = get_date(dob)
            if dob == 'nan':
                if not np.any(exc):
                    exc_name = "Missing DOB"
                exc[NO_DOB] = 1

        # Get Baseline or remove if no admit dates provided
        bsln_idx = np.where(bsln_m[:, 0] == tid)[0]
        if bsln_idx.size == 0 or bsln_m[bsln_idx, bsln_scr_loc] == 'None' or bsln_m[bsln_idx, bsln_scr_loc] is None:
            if not np.any(exc):
                exc_name = "Missing Admit Info"
            exc[NO_ADMIT] = 1
        else:
            bsln = float(bsln_m[bsln_idx, bsln_scr_loc])
            btype = bsln_m[bsln_idx, bsln_type_loc]
            if btype == 'mdrd':
                btype = 'imputed'
            else:
                btype = 'measured'
            if np.isnan(bsln):
                if not np.any(exc):
                    exc_name = "Missing Admit Info"
                exc[NO_ADMIT] = 1
            if bsln >= 4.0:
                if not np.any(exc):
                    exc_name = "Baseline SCr > 4.0"
                exc[SCR_GT4] = 1
        try:
            icu_admit, icu_disch = icu_windows[tid]
            if type(icu_admit) != datetime.datetime:
                icu_admit = get_date(icu_admit)
                icu_disch = get_date(icu_disch)
        except KeyError:
            if not np.any(exc):
                exc_name = "Missing Admit Info"
            exc[NO_ADMIT] = 1
            icu_admit = 'nan'
            icu_disch = 'nan'

        # get age and remove if less than 18
        if dob != 'nan' and icu_admit != 'nan':
            age = icu_admit - dob
            age = age.total_seconds() / (60 * 60 * 24 * 365)
            if age < 18:
                if not np.any(exc):
                    exc_name = "Age < 18"
                exc[AGE_LT18] = 1
        else:
            age = 'nan'

        # get mortality date if available
        mort_idx = np.where(mort_m[:, 0] == tid)[0]
        mort_date = 'nan'
        if mort_idx.size > 0:
            for i in range(len(mort_idx)):
                midx = mort_idx[i]
                mdate = str(mort_m[midx, mdate_loc]).split('.')[0]
                mort_date = get_date(mdate)
        if mort_date != 'nan' and icu_admit != 'nan':
            death_dur = (mort_date - icu_admit).total_seconds() / (60 * 60 * 24)
            if death_dur < 2:
                if not np.any(exc):
                    exc_name = "Died < 48 Hrs from ICU Admission"
                exc[DIED_LT48] = 1
        else:
            death_dur = np.nan

        # remove if ESRD status
        # Check manual revision
        esrd_idx = np.where(esrd_man_rev[:, 0] == tid)[0]
        if len(esrd_idx) > 0:
            esrd_idx = esrd_idx[0]
            if esrd_man_rev[esrd_idx, man_rev_bef]:
                if not np.any(exc):
                    exc_name = "ESRD"
                exc[ESRD] = 1
        else:
            # Check scm flags
            esrd_idx = np.where(scm_esrd_m[:, 0] == tid)[0]
            if len(esrd_idx) > 0:
                esrd_idx = esrd_idx[0]
                if scm_esrd_m[esrd_idx, scm_esrd_before] == 'Y' or scm_esrd_m[esrd_idx, scm_esrd_at] == 'Y':
                    if not np.any(exc):
                        exc_name = "ESRD"
                    exc[ESRD] = 1
            # Check USRDS
            esrd_idx = np.where(usrds_esrd_m[:, 0] == tid)[0]
            if len(esrd_idx) > 0:
                esrd_idx = esrd_idx[0]
                tdate = get_date(str(usrds_esrd_m[esrd_idx, usrds_esrd_date_loc]))
                if tdate != 'nan':
                    if tdate < icu_admit:  # Before admit
                        if not np.any(exc):
                            exc_name = "ESRD"
                        exc[ESRD] = 1
                    elif icu_admit - tdate < 1 and icu_admit.day == tdate.day:  # Same day as admit
                        if not np.any(exc):
                            exc_name = "ESRD"
                        exc[ESRD] = 1

        # remove patients with baseline GFR < 15
        if bsln != 'nan' and age != 'nan':
            gfr = calc_gfr(bsln, sex, race, age)
            if gfr < 15:
                if not np.any(exc):
                    exc_name = "GFR < 15"
                exc[GFR_LT15] = 1

        # remove patients with kidney transplant
        diagnosis_rows = np.where(diag_m[:, 0] == tid)[0]
        for row in diagnosis_rows:
            str_des = str(diag_m[row, diag_loc]).upper()
            icd_type = str(diag_m[row, icd_type_loc]).upper()
            if icd_type == 'nan':  # No ICD code provided
                if str_des == 'KIDNEY/PANCREAS FROM BATAVIA  ETA 1530':
                    if not np.any(exc):
                        exc_name = "Kidney Transplant"
                    exc[KID_TPLT] = 1
                    break
                if 'TRANS' in str_des:
                    if 'KID' in str_des or 'RENA' in str_des:
                        if not np.any(exc):
                            exc_name = "Kidney Transplant"
                        exc[KID_TPLT] = 1
                        break
            elif icd_type == 'ICD9':
                icd_code = str(diag_m[row, icd_code_loc]).upper()
                if icd_code in ['V42.0', '996.81']:
                    if not np.any(exc):
                        exc_name = "Kidney Transplant"
                    exc[KID_TPLT] = 1
                    break
            elif icd_type == 'ICD0':
                icd_code = str(diag_m[row, icd_code_loc]).upper()
                if icd_code in ['Z94.0', 'T86.10', 'T86.11', 'T86.12', 'T86.13', 'T86.19']:
                    if not np.any(exc):
                        exc_name = "Kidney Transplant"
                    exc[KID_TPLT] = 1
                    break
        transplant_rows = np.where(surg_m[:, 0] == tid)  # rows in surgery sheet
        for row in transplant_rows:
            str_des = str(surg_m[row, surg_desc_loc]).upper()
            if 'TRANS' in str_des:
                if 'KID' in str_des or 'RENA' in str_des:
                    if not np.any(exc):
                        exc_name = "Kidney Transplant"
                    exc[KID_TPLT] = 1
                    break

        if np.any(exc):
            exc = ','.join(exc.astype(str))
            exc_log.write('%d,%s\n' % (tid, exc))
            exc_list.write('%d,%s\n' % (tid, exc_name))
            if pre_ids is None:
                continue

        # get discharge disposition
        all_drows = np.where(date_m[:, 0] == tid)[0]
        disch_disp = date_m[all_drows[0], d_disp_loc]
        if type(disch_disp) == np.ndarray:
            disch_disp = disch_disp[0]
        disch_disp = str(disch_disp).upper()
        if 'DIED' in disch_disp or 'EXPIRED' in disch_disp:
            disch_disp = 'died'
        else:
            disch_disp = 'alive'
        # disch_disp = 'alive'
        if mort_date != 'nan' and 'died' not in disch_disp:
            hosp_disch = get_date(hosp_windows[tid][1])
            if mort_date < hosp_disch:
                disch_disp = 'died'
            elif (abs(mort_date - hosp_disch) < datetime.timedelta(1)) and mort_date.day == hosp_disch.day:
                disch_disp = 'died'

        # calculate duration vector
        tdate_strs = scr_all_m[keep, scr_date_loc]
        tdates = []
        for i in range(len(tdate_strs)):
            date_str = tdate_strs[i].split('.')[0]
            tdate = get_date(date_str)
            tdates.append(tdate)
        duration = [tdates[x] - icu_admit for x in range(len(tdates))]
        duration = np.array(duration)

        tdays = []
        for i in range(len(tdates)):
            tdays.append((tdates[i] - tdates[0]).days)

        log.write('%d,%s,%s,%.3f,%s,%.3f\n' % (tid, icu_admit, icu_disch, bsln, mort_date, death_dur))
        d_disp.append(disch_disp)
        bslns.append(bsln)
        bsln_gfr.append(gfr)

        tmask = mask[keep]
        tmasks.append(tmask)
        dmask = rrt_mask[keep]
        dmasks.append(dmask)
        scr.append(scr_all_m[keep, scr_val_loc])
        dates.append(tdates)
        ages.append(age)
        days.append(tdays)
        if len(duration) > 0:
            tmin = duration[0].total_seconds() / (60 * 60)
            tmax = duration[-1].total_seconds() / (60 * 60)
        else:
            tmin = tmax = 0
        ids_out.append(tid)
        t_range.append([tmin, tmax])
        btypes.append(btype)
    bslns = np.array(bslns)
    del scr_all_m
    del bsln_m
    del diag_m
    del date_m
    del surg_m
    del dem_m
    del dob_m
    return ids_out, scr, dates, days, tmasks, dmasks, bslns, bsln_gfr, btypes, d_disp, t_range, ages


# %%
def get_patients(scr_all_m, scr_val_loc, scr_date_loc, d_disp_loc,
                 mask, rrt_mask, icu_windows, hosp_windows,
                 diag_m, diag_loc, icd_type_loc, icd_code_loc,
                 scm_esrd_m, scm_esrd_before, scm_esrd_at,
                 esrd_man_rev, man_rev_bef,
                 usrds_esrd_m, usrds_esrd_date_loc,
                 bsln_m, bsln_scr_loc, bsln_type_loc,
                 date_m,
                 surg_m, surg_desc_loc,
                 dem_m, sex_loc, eth_loc,
                 dob_m, birth_loc,
                 mort_m, mdate_loc,
                 log, exc_log, exc_list='exclusions.csv', v=True, pre_ids=None):
    # Lists to store records for each patient
    scr = []
    tmasks = []  # time/date
    dmasks = []  # dialysis
    dates = []
    d_disp = []
    ids_out = []
    bslns = []
    bsln_gfr = []
    btypes = []
    t_range = []
    ages = []
    days = []

    # Counters for total number of patients and how many removed for each exclusion criterium
    if pre_ids is None:
        ids = np.unique(scr_all_m[:, 0]).astype(int)
        ids.sort()
    else:
        ids = pre_ids
    if type(log) == str:
        log = open(log, 'w')
    if type(exc_log) == str:
        exc_log = open(exc_log, 'w')
    if type(exc_list) == str:
        exc_list = open(exc_list, 'w')

    log.write('Patient_ID,Admit_Date,Discharge_Date,Baseline_SCr,Mort_Date,Days_To_Death\n')
    exc_log.write(
        'patientID,No values in first 7 days, Only 1 value, Last SCr - First SCr < 6hrs, Missing Demographics, Missing DOB, Missing Admit Info, Baseline SCr > 4, Age < 18, Died < 48 hrs after ICU admission, ESRD, Baseline eGFR < 15, Kidney Transplant\n')
    exc_list.write('Patient_ID,First_Exclusion\n')
    # NOT_ENOUGH = 0
    NO_VALS_ICU = 0
    ONLY_1VAL = 1
    TOO_SHORT = 2
    MISSING_DEM = 3
    NO_DOB = 4
    NO_ADMIT = 5
    SCR_GT4 = 6
    AGE_LT18 = 7
    DIED_LT48 = 8
    ESRD = 9
    GFR_LT15 = 10
    KID_TPLT = 11
    for tid in tqdm(ids, desc='Getting Patient Data'):
        exc = np.zeros(12, dtype=int)
        exc_name = ''
        all_rows = np.where(scr_all_m[:, 0] == tid)[0]
        sel = np.where(mask[all_rows])[0]
        keep = all_rows[sel]
        if len(sel) > 1:
            first_date = get_date(scr_all_m[keep[0], scr_date_loc])
            last_date = get_date(scr_all_m[keep[-1], scr_date_loc])
            dif = (last_date - first_date).total_seconds() / (60 * 60)
        # Ensure this patient has values in time period of interest
        if len(sel) == 0:
            if not np.any(exc):
                exc_name = "No ICU Values"
            exc[NO_VALS_ICU] = 1
        elif len(sel) == 1:
            if not np.any(exc):
                exc_name = "Only 1 Value in ICU"
            exc[ONLY_1VAL] = 1
        elif dif < 6:
            if not np.any(exc):
                exc_name = "Last SCr - First SCr <6 hours"
            exc[TOO_SHORT] = 1

        # get demographics and remove if any required are missing
        if tid not in dem_m[:, 0]:
            sex = 'nan'
            race = 'nan'
            if not np.any(exc):
                exc_name = "Missing Demographics"
            exc[MISSING_DEM] = 1
        else:
            dem_idx = np.where(dem_m[:, 0] == tid)[0]
            if len(dem_idx) > 1:
                dem_idx = dem_idx[0]
            sex = str(dem_m[dem_idx, sex_loc])
            race = str(dem_m[dem_idx, eth_loc])
            if sex == 'nan' or race == 'nan':
                if not np.any(exc):
                    exc_name = "Missing Demographics"
                exc[MISSING_DEM] = 1
            else:
                if sex == 'M':
                    sex = 1
                elif sex == 'F':
                    sex = 0

                if "BLACK" in race:
                    race = 1
                else:
                    race = 0

        # get dob, sex, and race
        if tid not in dob_m[:, 0]:
            if not np.any(exc):
                exc_name = "Missing DOB"
            exc[NO_DOB] = 1
            dob = 'nan'
        else:
            birth_idx = np.where(dob_m[:, 0] == tid)[0][0]
            dob = str(dob_m[birth_idx, birth_loc]).split('.')[0]
            dob = get_date(dob)
            if dob == 'nan':
                if not np.any(exc):
                    exc_name = "Missing DOB"
                exc[NO_DOB] = 1

        # Get Baseline or remove if no admit dates provided
        bsln_idx = np.where(bsln_m[:, 0] == tid)[0]
        if bsln_idx.size == 0 or bsln_m[bsln_idx, bsln_scr_loc] == 'None' or bsln_m[bsln_idx, bsln_scr_loc] is None:
            if not np.any(exc):
                exc_name = "Missing Admit Info"
            exc[NO_ADMIT] = 1
        else:
            bsln = float(bsln_m[bsln_idx, bsln_scr_loc])
            btype = bsln_m[bsln_idx, bsln_type_loc]
            if btype == 'mdrd':
                btype = 'imputed'
            else:
                btype = 'measured'
            if np.isnan(bsln):
                if not np.any(exc):
                    exc_name = "Missing Admit Info"
                exc[NO_ADMIT] = 1
            if bsln >= 4.0:
                if not np.any(exc):
                    exc_name = "Baseline SCr > 4.0"
                exc[SCR_GT4] = 1
        try:
            icu_admit, icu_disch = icu_windows[tid]
            if type(icu_admit) != datetime.datetime:
                icu_admit = get_date(icu_admit)
                icu_disch = get_date(icu_disch)
        except KeyError:
            if not np.any(exc):
                exc_name = "Missing Admit Info"
            exc[NO_ADMIT] = 1
            icu_admit = 'nan'
            icu_disch = 'nan'

        # get age and remove if less than 18
        if dob != 'nan' and icu_admit != 'nan':
            age = icu_admit - dob
            age = age.total_seconds() / (60 * 60 * 24 * 365)
            if age < 18:
                if not np.any(exc):
                    exc_name = "Age < 18"
                exc[AGE_LT18] = 1
        else:
            age = 'nan'

        # get mortality date if available
        mort_idx = np.where(mort_m[:, 0] == tid)[0]
        mort_date = 'nan'
        if mort_idx.size > 0:
            for i in range(len(mort_idx)):
                midx = mort_idx[i]
                mdate = str(mort_m[midx, mdate_loc]).split('.')[0]
                mort_date = get_date(mdate)
        if mort_date != 'nan' and icu_admit != 'nan':
            death_dur = (mort_date - icu_admit).total_seconds() / (60 * 60 * 24)
            if death_dur < 2:
                if not np.any(exc):
                    exc_name = "Died < 48 Hrs from ICU Admission"
                exc[DIED_LT48] = 1
        else:
            death_dur = np.nan

        # remove if ESRD status
        # Check manual revision
        esrd_idx = np.where(esrd_man_rev[:, 0] == tid)[0]
        if len(esrd_idx) > 0:
            esrd_idx = esrd_idx[0]
            if esrd_man_rev[esrd_idx, man_rev_bef]:
                if not np.any(exc):
                    exc_name = "ESRD"
                exc[ESRD] = 1
        else:
            # Check scm flags
            esrd_idx = np.where(scm_esrd_m[:, 0] == tid)[0]
            if len(esrd_idx) > 0:
                esrd_idx = esrd_idx[0]
                if scm_esrd_m[esrd_idx, scm_esrd_before] == 'Y' or scm_esrd_m[esrd_idx, scm_esrd_at] == 'Y':
                    if not np.any(exc):
                        exc_name = "ESRD"
                    exc[ESRD] = 1
            # Check USRDS
            esrd_idx = np.where(usrds_esrd_m[:, 0] == tid)[0]
            if len(esrd_idx) > 0:
                esrd_idx = esrd_idx[0]
                tdate = get_date(str(usrds_esrd_m[esrd_idx, usrds_esrd_date_loc]))
                if tdate != 'nan':
                    if tdate < icu_admit:  # Before admit
                        if not np.any(exc):
                            exc_name = "ESRD"
                        exc[ESRD] = 1
                    elif icu_admit - tdate < 1 and icu_admit.day == tdate.day:  # Same day as admit
                        if not np.any(exc):
                            exc_name = "ESRD"
                        exc[ESRD] = 1

        # remove patients with baseline GFR < 15
        if bsln != 'nan' and age != 'nan':
            gfr = calc_gfr(bsln, sex, race, age)
            if gfr < 15:
                if not np.any(exc):
                    exc_name = "GFR < 15"
                exc[GFR_LT15] = 1

        # remove patients with kidney transplant
        diagnosis_rows = np.where(diag_m[:, 0] == tid)[0]
        for row in diagnosis_rows:
            str_des = str(diag_m[row, diag_loc]).upper()
            icd_type = str(diag_m[row, icd_type_loc]).upper()
            if icd_type == 'nan':  # No ICD code provided
                if str_des == 'KIDNEY/PANCREAS FROM BATAVIA  ETA 1530':
                    if not np.any(exc):
                        exc_name = "Kidney Transplant"
                    exc[KID_TPLT] = 1
                    break
                if 'TRANS' in str_des:
                    if 'KID' in str_des or 'RENA' in str_des:
                        if not np.any(exc):
                            exc_name = "Kidney Transplant"
                        exc[KID_TPLT] = 1
                        break
            elif icd_type == 'ICD9':
                icd_code = str(diag_m[row, icd_code_loc]).upper()
                if icd_code in ['V42.0', '996.81']:
                    if not np.any(exc):
                        exc_name = "Kidney Transplant"
                    exc[KID_TPLT] = 1
                    break
            elif icd_type == 'ICD0':
                icd_code = str(diag_m[row, icd_code_loc]).upper()
                if icd_code in ['Z94.0', 'T86.10', 'T86.11', 'T86.12', 'T86.13', 'T86.19']:
                    if not np.any(exc):
                        exc_name = "Kidney Transplant"
                    exc[KID_TPLT] = 1
                    break
        transplant_rows = np.where(surg_m[:, 0] == tid)  # rows in surgery sheet
        for row in transplant_rows:
            str_des = str(surg_m[row, surg_desc_loc]).upper()
            if 'TRANS' in str_des:
                if 'KID' in str_des or 'RENA' in str_des:
                    if not np.any(exc):
                        exc_name = "Kidney Transplant"
                    exc[KID_TPLT] = 1
                    break

        if np.any(exc):
            exc = ','.join(exc.astype(str))
            exc_log.write('%d,%s\n' % (tid, exc))
            exc_list.write('%d,%s\n' % (tid, exc_name))
            if pre_ids is None:
                continue

        # get discharge disposition
        all_drows = np.where(date_m[:, 0] == tid)[0]
        disch_disp = date_m[all_drows[0], d_disp_loc]
        if type(disch_disp) == np.ndarray:
            disch_disp = disch_disp[0]
        disch_disp = str(disch_disp).upper()
        if 'DIED' in disch_disp or 'EXPIRED' in disch_disp:
            disch_disp = 'died'
        else:
            disch_disp = 'alive'
        # disch_disp = 'alive'
        if mort_date != 'nan' and 'died' not in disch_disp:
            hosp_disch = get_date(hosp_windows[tid][1])
            if mort_date < hosp_disch:
                disch_disp = 'died'
            elif (abs(mort_date - hosp_disch) < datetime.timedelta(1)) and mort_date.day == hosp_disch.day:
                disch_disp = 'died'

        # calculate duration vector
        tdate_strs = scr_all_m[keep, scr_date_loc]
        tdates = []
        for i in range(len(tdate_strs)):
            date_str = tdate_strs[i].split('.')[0]
            tdate = get_date(date_str)
            tdates.append(tdate)
        duration = [tdates[x] - icu_admit for x in range(len(tdates))]
        duration = np.array(duration)

        tdays = []
        for i in range(len(tdates)):
            tdays.append((tdates[i] - tdates[0]).days)

        log.write('%d,%s,%s,%.3f,%s,%.3f\n' % (tid, icu_admit, icu_disch, bsln, mort_date, death_dur))
        d_disp.append(disch_disp)
        bslns.append(bsln)
        bsln_gfr.append(gfr)

        tmask = mask[keep]
        tmasks.append(tmask)
        dmask = rrt_mask[keep]
        dmasks.append(dmask)
        scr.append(scr_all_m[keep, scr_val_loc])
        dates.append(tdates)
        ages.append(age)
        days.append(tdays)
        if len(duration) > 0:
            tmin = duration[0].total_seconds() / (60 * 60)
            tmax = duration[-1].total_seconds() / (60 * 60)
        else:
            tmin = tmax = 0
        ids_out.append(tid)
        t_range.append([tmin, tmax])
        btypes.append(btype)
    bslns = np.array(bslns)
    del scr_all_m
    del bsln_m
    del diag_m
    del date_m
    del surg_m
    del dem_m
    del dob_m
    return ids_out, scr, dates, days, tmasks, dmasks, bslns, bsln_gfr, btypes, d_disp, t_range, ages


# %%
def get_patients_dallas(scr_all_m, scr_val_loc, scr_date_loc,
                        mask, rrt_mask, icu_windows, hosp_windows,
                        diag_m, icd9_code_loc, icd10_code_loc, diag_desc_loc,
                        esrd_m, esrd_before,
                        bsln_m, bsln_scr_loc, bsln_type_loc,
                        date_m,
                        dem_m, sex_loc, eth_loc, dob_loc, dod_locs,
                        usrds_m, usrds_mort_loc, usrds_esrd_loc,
                        log, exc_log, exc_list='exclusions.csv', v=True):
    # Lists to store records for each patient
    scr = []
    tmasks = []  # time/date
    dmasks = []  # dialysis
    dates = []
    d_disp = []
    ids_out = []
    bslns = []
    bsln_gfr = []
    btypes = []
    t_range = []
    ages = []
    days = []

    # Counters for total number of patients and how many removed for each exclusion criterium
    ids = np.unique(scr_all_m[:, 0]).astype(int)
    ids.sort()
    if type(log) == str:
        log = open(log, 'w')
    if type(exc_log) == str:
        exc_log = open(exc_log, 'w')
    if type(exc_list) == str:
        exc_list = open(exc_list, 'w')

    log.write('Patient_ID,Admit_Date,Discharge_Date,Baseline_SCr,Mort_Date,Days_To_Death\n')
    exc_log.write(
        'PATIENT_NUM,No values in first 7 days, Only 1 value, Last SCr - First SCr < 6hrs, Missing Demographics, Missing DOB, Missing Admit Info, Baseline SCr > 4, Age < 18, Died < 48 hrs after ICU admission, ESRD, Baseline eGFR < 15, Kidney Transplant\n')
    exc_list.write('Patient_ID,First_Exclusion\n')
    # NOT_ENOUGH = 0
    NO_VALS_ICU = 0
    ONLY_1VAL = 1
    TOO_SHORT = 2
    MISSING_DEM = 3
    NO_DOB = 4
    NO_ADMIT = 5
    SCR_GT4 = 6
    DIED_LT48 = 7
    AGE_LT18 = 8
    ESRD = 9
    GFR_LT15 = 10
    KID_TPLT = 11
    for tid in tqdm(ids, desc='Getting Patient Data'):
        exc = np.zeros(12, dtype=int)
        exc_name = ''
        all_rows = np.where(scr_all_m[:, 0] == tid)[0]
        sel = np.where(mask[all_rows])[0]
        keep = all_rows[sel]
        if len(sel) > 1:
            first_date = get_date(scr_all_m[keep[0], scr_date_loc])
            last_date = get_date(scr_all_m[keep[-1], scr_date_loc])
            dif = (last_date - first_date).total_seconds() / (60 * 60)
        # Ensure this patient has values in time period of interest
        if len(sel) == 0:
            if not np.any(exc):
                exc_name = "No ICU Values"
            exc[NO_VALS_ICU] = 1
        elif len(sel) == 1:
            if not np.any(exc):
                exc_name = "Only 1 Value in ICU"
            exc[ONLY_1VAL] = 1
        elif dif < 6:
            if not np.any(exc):
                exc_name = "Last SCr - First SCr <6 hours"
            exc[TOO_SHORT] = 1
        # if len(sel) < 2 or dif < 6:
        #     if not np.any(exc):
        #         exc_name = "Not Enough Values"
        #     exc[NOT_ENOUGH] = 1

        # get demographics and remove if any required are missing
        if tid not in dem_m[:, 0]:
            male = 'nan'
            black = 'nan'
            if not np.any(exc):
                exc_name = "Missing Demographics"
            exc[MISSING_DEM] = 1
        else:
            dem_idx = np.where(dem_m[:, 0] == tid)[0]
            if len(dem_idx) > 1:
                dem_idx = dem_idx[0]
            male = dem_m[dem_idx, sex_loc]
            black = dem_m[dem_idx, eth_loc]

        # get dob
        if tid not in dem_m[:, 0]:
            if not np.any(exc):
                exc_name = "Missing DOB"
            exc[NO_DOB] = 1
            dob = 'nan'
        else:
            birth_idx = np.where(dem_m[:, 0] == tid)[0][0]
            dob = str(dem_m[birth_idx, dob_loc]).split('.')[0]
            dob = get_date(dob)
            if dob == 'nan':
                if not np.any(exc):
                    exc_name = "Missing DOB"
                exc[NO_DOB] = 1

        # Get Baseline or remove if no admit dates provided
        bsln_idx = np.where(bsln_m[:, 0] == tid)[0]
        if bsln_idx.size == 0 or bsln_m[bsln_idx, bsln_scr_loc] == 'None' or bsln_m[bsln_idx, bsln_scr_loc] is None:
            if not np.any(exc):
                exc_name = "Missing Admit Info"
            exc[NO_ADMIT] = 1
        else:
            bsln = float(bsln_m[bsln_idx, bsln_scr_loc])
            btype = bsln_m[bsln_idx, bsln_type_loc]
            if btype == 'mdrd':
                btype = 'imputed'
            else:
                btype = 'measured'
            if np.isnan(bsln):
                if not np.any(exc):
                    exc_name = "Missing Admit Info"
                exc[NO_ADMIT] = 1
            if bsln >= 4.0:
                if not np.any(exc):
                    exc_name = "Baseline SCr > 4.0"
                exc[SCR_GT4] = 1
        try:
            icu_admit, icu_disch = icu_windows[tid]
            if type(icu_admit) != datetime.datetime:
                icu_admit = get_date(icu_admit)
                icu_disch = get_date(icu_disch)
        except KeyError:
            if not np.any(exc):
                exc_name = "Missing Admit Info"
            exc[NO_ADMIT] = 1
            icu_admit = 'nan'
            icu_disch = 'nan'

        # get age and remove if less than 18
        if dob != 'nan' and icu_admit != 'nan':
            age = icu_admit - dob
            age = age.total_seconds() / (60 * 60 * 24 * 365)
            if age < 18:
                if not np.any(exc):
                    exc_name = "Age < 18"
                exc[AGE_LT18] = 1
        else:
            age = 'nan'

        # get mortality date if available
        mort_idx = np.where(dem_m[:, 0] == tid)[0]
        mort_date = 'nan'
        if mort_idx.size > 0:
            midx = mort_idx[0]
            for dod_loc in dod_locs:
                mdate = str(dem_m[midx, dod_loc]).split('.')[0]
                mort_date = get_date(mdate)
                if mort_date != 'nan':
                    break
        if mort_date != 'nan' and icu_admit != 'nan':
            death_dur = (mort_date - icu_admit).total_seconds() / (60 * 60 * 24)
            if death_dur < 2:
                if not np.any(exc):
                    exc_name = "Died < 48 Hrs from ICU Admission"
                exc[DIED_LT48] = 1
        else:
            death_dur = np.nan

        if mort_date == 'nan':
            mort_idx = np.where(usrds_m[:, 0] == tid)[0]
            if mort_idx.size > 0:
                midx = mort_idx[0]
                mdate = str(usrds_m[midx, usrds_mort_loc]).split('.')[0]
                mort_date = get_date(mdate)
            if mort_date != 'nan' and icu_admit != 'nan':
                death_dur = (mort_date - icu_admit).total_seconds() / (60 * 60 * 24)
                if death_dur < 2:
                    if not np.any(exc):
                        exc_name = "Died < 48 Hrs from ICU Admission"
                    exc[DIED_LT48] = 1
            else:
                death_dur = np.nan

        # remove if ESRD status
        esrd_idx = np.where(esrd_m[:, 0] == tid)[0]
        if len(esrd_idx) > 0:
            esrd_idx = esrd_idx[0]
            if esrd_m[esrd_idx, esrd_before] == 'BEFORE_INDEXED_ADT':
                if not np.any(exc):
                    exc_name = "ESRD"
                exc[ESRD] = 1
        # Check USRDS
        esrd_idx = np.where(usrds_m[:, 0] == tid)[0]
        if len(esrd_idx) > 0:
            esrd_idx = esrd_idx[0]
            tdate = get_date(str(usrds_m[esrd_idx, usrds_esrd_loc]))
            if tdate != 'nan':
                if tdate < icu_admit:  # Before admit
                    if not np.any(exc):
                        exc_name = "ESRD"
                    exc[ESRD] = 1
                elif icu_admit - tdate < datetime.timedelta(1) and icu_admit.day == tdate.day:  # Same day as admit
                    if not np.any(exc):
                        exc_name = "ESRD"
                    exc[ESRD] = 1

        # remove patients with baseline GFR < 15
        if bsln != 'nan' and age != 'nan':
            gfr = calc_gfr(bsln, male, black, age)
            if gfr < 15:
                if not np.any(exc):
                    exc_name = "GFR < 15"
                exc[GFR_LT15] = 1

        # remove patients with kidney transplant
        diagnosis_rows = np.where(diag_m[:, 0] == tid)[0]
        for row in diagnosis_rows:
            icd9_code = str(diag_m[row, icd9_code_loc]).upper()
            icd10_code = str(diag_m[row, icd10_code_loc]).upper()
            if icd9_code in ['V42.0', '996.81']:
                if not np.any(exc):
                    exc_name = "Kidney Transplant"
                exc[KID_TPLT] = 1
                break
            elif icd10_code in ['Z94.0', 'T86.10', 'T86.11', 'T86.12', 'T86.13', 'T86.19']:
                if not np.any(exc):
                    exc_name = "Kidney Transplant"
                exc[KID_TPLT] = 1
                break
        if not exc[KID_TPLT]:  # rows in surgery sheet
            for row in diagnosis_rows:
                str_des = str(diag_m[row, diag_desc_loc]).upper()
                if 'TRANS' in str_des:
                    if 'KID' in str_des or 'RENA' in str_des:
                        if not np.any(exc):
                            exc_name = "Kidney Transplant"
                        exc[KID_TPLT] = 1
                        break

        if np.any(exc):
            exc = ','.join(exc.astype(str))
            exc_log.write('%d,%s\n' % (tid, exc))
            exc_list.write('%d,%s\n' % (tid, exc_name))
            continue

        # get discharge disposition
        # all_drows = np.where(date_m[:, 0] == tid)[0]
        # disch_disp = date_m[all_drows[0], d_disp_loc]
        # if type(disch_disp) == np.ndarray:
        #     disch_disp = disch_disp[0]
        # disch_disp = str(disch_disp).upper()
        disch_disp = 'alive'
        if mort_date != 'nan':
            hosp_disch = get_date(hosp_windows[tid][1])
            if mort_date < hosp_disch:
                disch_disp = 'died'
            elif (abs(mort_date - hosp_disch) < datetime.timedelta(1)) and mort_date.day == hosp_disch.day:
                disch_disp = 'died'

        # calculate duration vector
        tdate_strs = scr_all_m[keep, scr_date_loc]
        tdates = []
        for i in range(len(tdate_strs)):
            date_str = tdate_strs[i].split('.')[0]
            tdate = get_date(date_str)
            tdates.append(tdate)
        duration = [tdates[x] - icu_admit for x in range(len(tdates))]
        duration = np.array(duration)

        tdays = []
        for i in range(len(tdates)):
            tdays.append((tdates[i] - tdates[0]).days)

        log.write('%d,%s,%s,%.3f,%s,%.3f\n' % (tid, icu_admit, icu_disch, bsln, mort_date, death_dur))
        d_disp.append(disch_disp)
        bslns.append(bsln)
        bsln_gfr.append(gfr)

        tmask = mask[keep]
        tmasks.append(tmask)
        dmask = rrt_mask[keep]
        dmasks.append(dmask)
        scr.append(scr_all_m[keep, scr_val_loc])
        dates.append(tdates)
        ages.append(age)
        days.append(tdays)
        tmin = duration[0].total_seconds() / (60 * 60)
        tmax = duration[-1].total_seconds() / (60 * 60)
        ids_out.append(tid)
        t_range.append([tmin, tmax])
        btypes.append(btype)
    bslns = np.array(bslns)
    del scr_all_m
    del bsln_m
    del date_m
    del diag_m
    del dem_m
    return ids_out, scr, dates, days, tmasks, dmasks, bslns, bsln_gfr, btypes, d_disp, t_range, ages


# %%
# def get_patients_dallas(scr_all_m, scr_val_loc, scr_date_loc,
#                         icu_mask, rrt_mask, hosp_windows,
#                         esrd_m, esrd_bef_loc,
#                         bsln_m, bsln_scr_loc,
#                         date_m,
#                         dem_m, sex_loc, eth_loc, dob_loc, dod_loc,
#                         diag_m, diag_loc,
#                         log, exc_log, v=True):
#
#     # Lists to store records for each patient
#     scr = []
#     tmasks = []  # time/date
#     dmasks = []  # dialysis
#     dates = []
#     d_disp = []
#     ids_out = []
#     bslns = []
#     bsln_gfr = []
#     t_range = []
#     ages = []
#     days = []
#
#     # Counters for total number of patients and how many removed for each exclusion criterium
#     count = 0
#     gfr_count = 0
#     high_bsln_count = 0
#     no_recs_count = 0
#     no_bsln_count = 0
#     kid_xplt_count = 0
#     esrd_count = 0
#     dem_count = 0
#     age_count = 0
#     ids = np.unique(dem_m[:, 0]).astype(int)
#     ids.sort()
#     if v:
#         print('Getting patient vectors')
#         print('Patient_ID\tAdmit_Date\tDischarge_Date\tBaseline_SCr\tMort_Date\tDays_To_Death')
#         log.write('Patient_ID,Admit_Date,Discharge_Date,Baseline_SCr,Mort_Date,Days_To_Death\n')
#         exc_log.write(
#             'Patient_ID,Not Enough Values,No Admit Info,SCr > 4.0,Missing DOB, Missing Demographics, Age < 18, ESRD at/before admission,GFR < 15,Kidney Transplant\n')
#     for tid in ids:
#         skip = False  # required to skip patients where exclusion is found in interior loop
#         all_rows = np.where(scr_all_m[:, 0] == tid)[0]
#         sel = np.where(icu_mask[all_rows])[0]
#         keep = all_rows[sel]
#         exc = np.zeros(9, dtype=int)
#         # Ensure this patient has values in time period of interest
#         if len(sel) < 2:
#             no_recs_count += 1
#             exc[0] = 1
#             if v:
#                 print(str(tid) + ', removed due to not enough values in the time period of interest')
#                 # exc_log.write(str(tid) + ', removed due to not enough values in the time period of interest\n')
#             skip = True
#
#         # Get Baseline or remove if no admit dates provided
#         bsln_idx = np.where(bsln_m[:, 0] == tid)[0]
#         if bsln_idx.size == 0:
#             no_bsln_count += 1
#             if v:
#                 print(str(tid) + ', removed due to no valid baseline')
#                 # exc_log.write(str(tid) + ', removed due to no valid baseline\n')
#             skip = True
#         else:
#             bsln_idx = bsln_idx[0]
#         bsln = bsln_m[bsln_idx, bsln_scr_loc]
#         if str(bsln).lower() == 'nan' or str(bsln).lower() == 'none' or str(bsln).lower() == 'nat':
#             no_bsln_count += 1
#             if v:
#                 print(str(tid) + ', removed due to missing baseline')
#                 # exc_log.write(str(tid) + ', removed due to missing baseline\n')
#             skip = True
#         bsln = float(bsln)
#         if bsln >= 4.0:
#             high_bsln_count += 1
#             if v:
#                 print(str(tid) + ', removed due to baseline SCr > 4.0')
#                 # exc_log.write(str(tid) + ', removed due to baseline SCr > 4.0\n')
#             skip = True
#             exc[2] = 1
#
#         try:
#             admit, discharge = hosp_windows[tid]
#         except KeyError:
#             admit = discharge = 'nan'
#             skip = True
#             exc[1] = 1
#
#         # get demographics and remove if any required are missing
#         if tid not in dem_m[:, 0]:
#             # ids = np.delete(ids, count)
#             dem_count += 1
#             if v:
#                 print(str(tid) + ', removed due to missing demographics')
#                 # exc_log.write(str(tid) + ', removed due to missing demographics\n')
#             skip = True
#             exc[4] = 1
#         dem_idx = np.where(dem_m[:, 0] == tid)[0]
#         if len(dem_idx) >= 1:
#             dem_idx = dem_idx[0]
#         sex = dem_m[dem_idx, sex_loc]
#         race = dem_m[dem_idx, eth_loc]
#         if str(sex) == 'nan' or str(race) == 'nan':
#             # ids = np.delete(ids, count)
#             dem_count += 1
#             if v:
#                 print(str(tid) + ', removed due to missing demographics')
#                 # exc_log.write(str(tid) + ', removed due to missing demographics\n')
#             skip = True
#             exc[4] = 1
#
#         # get mortality date if available
#         # mort_date = 'nan'
#         # for dod_loc in dod_locs:
#         #     mort_date = get_date(dem_m[dem_idx, dod_loc])
#         #     if mort_date != 'nan':
#         #         death_dur = (mort_date - admit).total_seconds() / (60 * 60 * 24)
#         #         disch_disp = 'died'
#         #         break
#         # if mort_date == 'nan':
#         #     disch_disp = 'alive'
#         #     death_dur = np.nan
#         # mort_date = get_date(dem_m[dem_idx, dod_locs])
#         # dodl = dem_m[dem_idx, dod_locs]
#         # mort_date = 'nan'
#         # for x in dodl:
#         #     dodDate = get_date(str(x))
#         #     if dodDate != 'nan':
#         #         mort_date = dodDate
#         mort_date = dem_m[dem_idx, dod_loc]
#         if type(mort_date) == str:
#             mort_date = get_date(mort_date)
#
#         # mort_date = get_date(dem_m[dem_idx, dod_locs[0]])
#         # if mort_date == 'nan':
#         #     for j in range(1, len(dod_locs)):
#         #         tdod = get_date(dem_m[dem_idx, dod_locs[j]])
#         #         if tdod != 'nan':
#         #             mort_date = tdod
#
#         if mort_date != 'nan' and admit != 'nan':
#             death_dur = (mort_date - admit).total_seconds() / (60 * 60 * 24)
#         else:
#             death_dur = np.nan
#
#         # dob = get_date(dem_m[dem_idx, dob_loc])
#         # if dob == 'nan':
#         #     dob_count += 1
#         #     if v:
#         #         print(str(idx) + ', removed due to missing DOB')
#         #         exc_log.write(str(idx) + ', removed due to missing DOB\n')
#         #     continue
#         dob = dem_m[dem_idx, dob_loc]
#         if type(dob) == str:
#             dob = get_date(dob)
#
#         # get age and remove if less than 18
#         if admit != 'nan':
#             age = admit - dob
#             age = age.total_seconds() / (60 * 60 * 24 * 365)
#             if age < 18:
#                 age_count += 1
#                 if v:
#                     print(str(tid) + ', removed due to age < 18')
#                     # exc_log.write(str(tid) + ', removed due to age < 18\n')
#                 skip = True
#                 exc[5] = 1
#
#         # remove if ESRD status
#         esrd_idx = np.where(esrd_m[:, 0] == tid)[0]
#         if esrd_idx.size > 0:
#             esrd_idx = esrd_idx[0]
#             if esrd_m[esrd_idx, esrd_bef_loc] == 'BEFORE_INDEXED_ADT':
#                 skip = True
#                 # ids = np.delete(ids, count)
#                 esrd_count += 1
#                 exc[6] = 1
#                 if v:
#                     print(str(tid) + ', removed due to ESRD status')
#                     # exc_log.write(str(tid) + ', removed due to ESRD status\n')
#                 break
#
#         # remove patients with baseline GFR < 15
#         gfr = calc_gfr(bsln, sex, race, age)
#         if gfr < 15:
#             # ids = np.delete(ids, count)
#             gfr_count += 1
#             if v:
#                 print(str(tid) + ', removed due to initial GFR too low')
#                 # exc_log.write(str(tid) + ', removed due to initial GFR too low\n')
#             skip = True
#             exc[7] = 1
#
#         # remove patients with kidney transplant
#         transplant_rows = np.where(diag_m[:, 0] == tid)  # rows in surgery sheet
#         for row in transplant_rows:
#             str_des = str(diag_m[row, diag_loc]).upper()
#             if 'KID' in str_des and 'TRANS' in str_des:
#                 skip = True
#                 exc[8] = 1
#                 kid_xplt_count += 1
#                 # ids = np.delete(ids, count)
#                 if v:
#                     print(str(tid) + ', removed due to kidney transplant')
#                     # exc_log.write(str(tid) + ', removed due to kidney transplant\n')
#                 break
#
#         if skip:
#             exc = ','.join(exc.astype(str))
#             exc_log.write('%d,%s\n' % (tid, exc))
#             continue
#
#         # calculate duration vector
#         tdate_strs = scr_all_m[keep, scr_date_loc]
#         tdates = []
#         for i in range(len(tdate_strs)):
#             date_str = tdate_strs[i].split('.')[0]
#             tdate = get_date(date_str)
#             tdates.append(tdate)
#         duration = [tdates[x] - admit for x in range(len(tdates))]
#         duration = np.array(duration)
#
#         if v:
#             print('%d\t%s\t%s\t%.3f\t%s\t%.3f' % (tid, admit, discharge, bsln, mort_date, death_dur))
#             log.write('%d,%s,%s,%.3f,%s,%.3f\n' % (tid, admit, discharge, bsln, mort_date, death_dur))
#
#         disch_disp = 'alive'
#         if mort_date != 'nan':
#             if mort_date < discharge:
#                 disch_disp = 'died'
#
#         d_disp.append(disch_disp)
#         bslns.append(bsln)
#         bsln_gfr.append(gfr)
#
#         tmask = icu_mask[keep]
#         tmasks.append(tmask)
#         dmask = rrt_mask[keep]
#         dmasks.append(dmask)
#         scr.append(scr_all_m[keep, scr_val_loc])
#         dates.append(tdates)
#         ages.append(age)
#         tmin = duration[0].total_seconds() / (60 * 60)
#         tmax = duration[-1].total_seconds() / (60 * 60)
#         ids_out.append(tid)
#         t_range.append([tmin, tmax])
#         count += 1
#     bslns = np.array(bslns)
#     # if v:
#         # exc_log.write('# Patients Kept: ' + str(count) + '\n')
#         # exc_log.write('# Patients removed for ESRD: ' + str(esrd_count) + '\n')
#         # exc_log.write('# Patients w/ GFR < 15: ' + str(gfr_count) + '\n')
#         # exc_log.write('# Patients w/ no admit info: ' + str(no_bsln_count) + '\n')
#         # exc_log.write('# Patients w/ missing demographics: ' + str(dem_count) + '\n')
#         # exc_log.write('# Patients w/ < 2 ICU records: ' + str(no_recs_count) + '\n')
#         # exc_log.write('# Patients w/ no valid baseline: ' + str(no_bsln_count) + '\n')
#         # exc_log.write('# Patients w/ kidney transplant: ' + str(kid_xplt_count) + '\n')
#     del scr_all_m
#     del bsln_m
#     del diag_m
#     del date_m
#     del dem_m
#     return ids_out, scr, dates, tmasks, dmasks, bslns, bsln_gfr, d_disp, t_range, ages
    # return ids_out, scr, dates, tmasks, dmasks, bslns, bsln_gfr, d_disp, t_range, ages, admits, discharges


def linear_interpo(scrs, ids, dates, dmasks, scale):
    '''
    For each provided list of SCr values with corresponding dates and dialysis masks, do the following
        1) Fit each record to a fixed grid with resolution defined by scale
            a) If there are multiple records in the same bin, select the maximum
        2) Using linear interpolation, impute any missing values


    :param scrs: list of arrays containing raw SCr values
    :param ids: list of corresponding patient IDs
    :param dates: list of arrays of dates corresponding to SCr values
    :param dmasks: list of arrays indicating which points are during dialysis
    :param scale: in hours... e.g. scale=6 means 4 points per day
    :return interp_scr_all: list of arrays corresponding to interpolated SCr
    :return interp_dmasks_all: list of arrays indicated which points correspond to dialysis
    :return interp_days_all: list of arrays indicating to which day each point belongs
    :return interp_masks_all: indicates which points are real values vs. interpolated
    '''
    interp_scr_all = []           # stores all SCr after interpolation
    interp_rrt_all = []          # stores which interpolated points correspond to dialysis
    interp_days_all = []            # stores which the day to which each interpolated point corresponds
    interp_masks_all = []           # indicates which points are original vs. interpolated
    out_ids = []
    too_few_records = 0
    for i in range(len(scrs)):
        tscrs = scrs[i]             # raw SCr values
        tid = ids[i]                # patient ID
        tdates = dates[i]           # raw SCr dates
        tdmask = dmasks[i]          # indicates dialysis for raw values
        if len(tscrs) < 2:
            interp_scr = tscrs
            interp_dia = tdmask
            if len(tscrs) == 1:
                interp_days = [0]
            interp_mask = np.ones(len(interp_scr))

        else:
            interp_scr, interp_days, interp_mask, interp_dia = fill_grid(tscrs, tdates, tdmask, scale)
            # It is possible a patient has more than 1 record, but they all lie in the same bin. If so, remove the patient
            if len(interp_scr) < 2:
                too_few_records += 1
                # continue
            else:
                interp_scr = interpolate_scr(interp_scr, interp_mask)

        interp_scr_all.append(interp_scr)
        interp_days_all.append(interp_days)
        interp_rrt_all.append(interp_dia)
        interp_masks_all.append(interp_mask)
        out_ids.append(tid)
    #
    # print('Total # patients provided: %d' % len(ids))
    # print('Number patients kept: %d' % len(interp_scr_all))
    # print('Number removed due to not enough records: %d' % too_few_records)
    return interp_scr_all, interp_rrt_all, interp_days_all, interp_masks_all


def fill_grid(scrs, dates, dmasks, scale):
    '''
    Fits SCr values and dialysis masks to a uniform grid defined by scale, where scale is the number of hours in each
    bin. If multiple values provided for the same bin, select the maximum.

    :param scrs: List of all raw SCr values
    :param dates: Corresponding dates (as datetime.datetime)
    :param dmasks: Binary mask indicating whether or not patient was on dialysis for each record
    :param scale: Defines grid resolution, so that each bin represents 'scale' hours
    :return: All values fit to a uniform grid, with missing points
    '''
    bins_per_day = int(np.floor(24 / scale))

    # Determine the final length of the interpolated records
    start_date = get_date(dates[0])
    start_bin = int(np.floor(start_date.hour / scale))

    end_date = get_date(dates[-1])
    end_bin = int(np.floor(end_date.hour / scale))

    # If final hour is later than first hour, subract 1
    ndays_all = int((end_date - start_date).days)
    if start_date.time() <= end_date.time():
        ndays_all -= 1

    # Number of bins for day 0
    day0_l = int(bins_per_day - start_bin)

    # Final length
    ilen = day0_l + ndays_all * bins_per_day + (end_bin + 1)

    if ilen < 2:
        return [0, ], [0, ], [0, ], [1, ]

    iscr = np.zeros(ilen)
    idays = np.zeros(ilen, dtype=int)
    idia = np.zeros(ilen, dtype=bool)
    imask = np.zeros(ilen, dtype=bool)

    # Assign each individual record to the appropriate bin
    for i in range(len(scrs)):
        current_date = get_date(dates[i])
        current_bin = int(np.floor(current_date.hour / scale))
        ndays = int((current_date - start_date).days)
        if start_date.time() <= current_date.time():
            ndays -= 1
        idx = day0_l + (ndays * bins_per_day) + (current_bin)

        # If this is not the first record assigned to this bin, select the maximum
        iscr[idx] = max(scrs[i], iscr[idx])

        # Indicate that this bin contains an real-world record and is not interpolated
        imask[idx] = 1

        # If patient was on dialysis for this record, indicate dialysis for the subsequent 48 hours (or rest of record,
        # whichever is shorter.
        bins_left = ilen - idx
        if dmasks[i]:
            if bins_left <= 2 * bins_per_day:
                idia[idx:] = np.repeat(1, bins_left)
            else:
                idia[idx:idx + (2 * bins_per_day)] = np.repeat(1, 2 * bins_per_day)

    # Build the corresponding vector indicating to which day each point belongs (including for those yet to be imputed)
    cur = 1
    for i in range(ndays_all):
        for j in range(bins_per_day):
            idays[day0_l + (i * 4) + j] = cur
        cur += 1
    try:
        idays[-(end_bin + 1):] = np.repeat(cur, end_bin + 1)
    except ValueError:
        l = len(idays[-(end_bin + 1):])
        if start_bin <= end_bin:
            idays[-l:] = np.repeat(0, l)
        else:
            idays[-l:] = np.repeat(1, l)

    return iscr, idays, imask, idia


def interpolate_scr(iscr, imask):
    real_vals = np.where(imask)[0]
    current_val = iscr[0]
    current_idx = 0
    for next_idx in real_vals[1:]:
        next_val = iscr[next_idx]
        step = (next_val - current_val) / (next_idx - current_idx)

        nsteps = (next_idx - current_idx)
        for i in range(1, nsteps):
            iscr[current_idx + i] = iscr[current_idx + i - 1] + step
        current_idx = next_idx
        current_val = iscr[current_idx]
    return iscr



# def interpolate_scrs(start_scr, end_scr, start_date, end_date, start_dia, end_dia, scale):
#     '''
#     :param start_scr:
#     :param end_scr:
#     :param start_date:
#     :param end_date:
#     :param start_dia:
#     :param end_dia:
#     :param scale:
#
#     >>> import numpy as np
#     >>> import datetime
#     >>> start_scr = 1.2
#     >>> end_scr = 2.7
#     >>> start_date = datetime.datetime.strptime('2015-02-16 03:25:00', '%Y-%m-%d %H:%M:%S')
#     >>> end_date = datetime.datetime.strptime('2015-02-17 10:15:00', '%Y-%m-%d %H:%M:%S')
#     >>> start_dia = 0
#     >>> end_dia = 0
#     >>> scale = 6
#     >>> iscr, idia, imask = interpolate_scrs(start_scr, end_scr, start_date, end_date, start_dia, end_dia, scale)
#     >>> iscr
#     [1.2, 1.5, 1.8, 2.1, 2.4, 2.7]
#     >>> idia
#     [0, 0, 0, 0, 0, 0]
#     >>> imask
#     [1, 0, 0, 0, 0, 1]
#
#     >>> start_dia = 1
#     >>> iscr, idia, imask = interpolate_scrs(start_scr, end_scr, start_date, end_date, start_dia, end_dia, scale)
#     >>> idia
#     [1, 1, 1, 1, 1, 1]
#     '''
#     start_hour = start_date.hour
#
#     end_hour = end_date.hour
#
#     bins_per_day = int(24 / scale)
#     start_bin = np.floor(start_hour / scale)
#     end_bin = np.floor(end_hour / scale)
#
#     ndays = (end_date - start_date).days
#
#     if ndays == 0:
#         nbins = int(end_bin - start_bin)
#     elif ndays == 1:
#         nbins = int((bins_per_day - start_bin) + end_bin)
#     else:
#         nbins = int((bins_per_day - start_bin) + ndays * bins_per_day + end_bin)
#
#     if nbins == 0:
#         interp_scr = [start_scr, end_scr]
#         interp_dia = [start_dia, end_dia]
#         interp_mask = [1, 1]
#     else:
#         step = (end_scr - start_scr) / nbins
#         interp_scr = [start_scr, ]
#         interp_mask = [1, ]
#         for i in range(nbins):
#             interp_scr.append(interp_scr[-1] + step)
#             interp_mask.append(0)
#         interp_mask[-1] = 1
#
#         if start_dia:
#             interp_dia = [1, ]
#             for i in range(nbins):
#                 if end_dia or i < bins_per_day * 2:
#                     interp_dia.append(1)
#                 else:
#                     interp_dia.append(0)
#         else:
#             interp_dia = [0, ]
#             for i in range(nbins):
#                 if end_dia and i == nbins - 1:
#                     interp_dia.append(1)
#                 else:
#                     interp_dia.append(0)
#
#     return interp_scr, interp_dia, interp_mask


# # %%
# def linear_interpo(scr, ids, dates, masks, dmasks, scale, log, v=True):
#     post_interpo = []
#     dmasks_interp = []
#     days_interps = []
#     interp_masks = []
#     count = 0
#     if v:
#         log.write('Raw SCr\n')
#         log.write('Stretched SCr\n')
#         log.write('Interpolated\n')
#         log.write('Original Dialysis\n')
#         log.write('Interpolated Dialysis\n')
#     print('Interpolating missing values')
#     for i in range(len(scr)):
#         print('Patient #' + str(ids[i]))
#         mask = masks[i]
#         dmask = dmasks[i]
#         print(mask)
#         tmin = datetime.datetime.strptime(str(dates[i][0]).split('.')[0], '%Y-%m-%d %H:%M:%S')
#         tmax = datetime.datetime.strptime(str(dates[i][-1]).split('.')[0], '%Y-%m-%d %H:%M:%S')
#         tstart = tmin.hour
#         n = nbins(tmin, tmax, scale)
#         thisp = np.repeat(-1., n)
#         interp_mask = np.zeros(n, dtype=int)  # indicates which points are raw vs. interpolated
#         this_start = datetime.datetime.strptime(str(dates[i][0]).split('.')[0], '%Y-%m-%d %H:%M:%S')
#         thisp[0] = scr[i][0]
#         dmask_i = np.repeat(-1, len(thisp))
#         dmask_i[0] = dmask[0]
#         for j in range(1, len(scr[i])):
#             tdate = datetime.datetime.strptime(str(dates[i][j]).split('.')[0], '%Y-%m-%d %H:%M:%S')
#             dt = (tdate - this_start).total_seconds()
#             idx = int(math.floor(dt / (60 * 60 * scale)))
#             if mask[j] != -1:
#                 thisp[idx] = max(thisp[idx], scr[i][j])
#                 interp_mask[idx] = 1
#             dmask_i[idx] = dmask[j]
#         for j in range(len(dmask_i)):
#             if dmask_i[j] != -1:
#                 k = j + 1
#                 while k < len(dmask_i) and dmask_i[k] == -1:
#                     dmask_i[k] = dmask_i[j]
#                     k += 1
#         print(str(thisp))
#         if v:
#             log.write('%d\n' % (ids[i]))
#             log.write(arr2str(scr[i]) + '\n')
#             log.write(arr2str(thisp) + '\n')
#         print(dmask_i)
#         dmasks_interp.append(dmask_i)
#         interp_masks.append(interp_mask)
#         j = 0
#         while j < len(thisp):
#             if thisp[j] == -1:
#                 pre_id = j - 1
#                 pre_val = thisp[pre_id]
#                 while thisp[j] == -1 and j < len(thisp) - 1:
#                     j += 1
#                 post_id = j
#                 post_val = thisp[post_id]
#                 if post_val == -1:
#                     post_val = pre_val
#                 step = (post_val - pre_val) / (post_id - pre_id)
#                 for k in range(pre_id + 1, post_id + 1):
#                     thisp[k] = thisp[k - 1] + step
#             j += 1
#         if v:
#             log.write(arr2str(thisp) + '\n')
#             log.write(arr2str(dmask) + '\n')
#             log.write(arr2str(dmask_i) + '\n')
#             log.write('\n')
#         print(str(thisp))
#         post_interpo.append(thisp)
#         interp_len = len(thisp)
#
#         if tstart >= 18:
#             n_zeros = 1
#         elif tstart >= 12:
#             n_zeros = 2
#         elif tstart >= 6:
#             n_zeros = 3
#         else:
#             n_zeros = 4
#
#         days_interp = np.zeros(interp_len, dtype=int)
#         tday = 1
#         ct = 0
#         for i in range(n_zeros, interp_len):
#             days_interp[i] = tday
#             if ct == 3:
#                 ct = 0
#                 tday += 1
#             else:
#                 ct += 1
#         days_interps.append(days_interp)
#
#         count += 1
#     return post_interpo, dmasks_interp, days_interps, interp_masks
#
#
# # %%
# def nbins(start, stop, scale):
#     dt = (stop - start).total_seconds()
#     div = scale * 60 * 60  # hrs * minutes * seconds
#     bins, _ = divmod(dt, div)
#     return int(bins + 1)


# # %%
# def get_baselines(date_m, hosp_locs, scr_all_m, scr_val_loc, scr_date_loc, scr_desc_loc,
#                   dem_m, sex_loc, eth_loc, dob_m, dob_loc, fname, outp_rng=(1, 365), inp_rng=(7, 365)):
#     log = open(fname, 'w')
#     log.write('ID,bsln_val,bsln_type,bsln_date,admit_date,time_delta\n')
#     cur_id = None
#
#     out_lim = (datetime.timedelta(outp_rng[0]), datetime.timedelta(outp_rng[1]))
#     inp_lim = (datetime.timedelta(inp_rng[0]), datetime.timedelta(inp_rng[1]))
#
#     for i in range(len(date_m)):
#         idx = date_m[i, 0]
#         if cur_id != idx:
#             cur_id = idx
#             log.write(str(idx) + ',')
#             # determine earliest admission date
#             admit = datetime.datetime.now()
#             didx = np.where(date_m[:, 0] == idx)[0]
#             for did in didx:
#                 tdate = get_date(date_m[did, hosp_locs[0]])
#                 if tdate < admit:
#                     admit = tdate
#             if type(admit) == np.ndarray:
#                 admit = admit[0]
#
#             # find indices of all SCr values for this patient
#             all_rows = np.where(scr_all_m[:, 0] == idx)[0]
#
#             # extract record types
#             # i.e. indexed admission, before indexed, after indexed
#             scr_desc = scr_all_m[all_rows, scr_desc_loc]
#             for j in range(len(scr_desc)):
#                 scr_desc[j] = scr_desc[j].split()[0].upper()
#
#             scr_tp = scr_all_m[all_rows, scr_desc_loc]
#             for j in range(len(scr_tp)):
#                 scr_tp[j] = scr_tp[j].split()[-1].upper()
#
#             # find indexed admission rows for this patient
#             idiag_rows = np.where(scr_desc == 'INDEXED')[0]
#             idiag_rows = all_rows[idiag_rows]
#
#             pre_admit_rows = np.where(scr_desc == 'BEFORE')[0]
#             inp_rows = np.where(scr_tp == 'INPATIENT')[0]
#             out_rows = np.where(scr_tp == 'OUTPATIENT')[0]
#
#             pre_inp_rows = np.intersect1d(pre_admit_rows, inp_rows)
#             pre_inp_rows = list(all_rows[pre_inp_rows])
#
#             pre_out_rows = np.intersect1d(pre_admit_rows, out_rows)
#             pre_out_rows = list(all_rows[pre_out_rows])
#
#             # default values
#             bsln_val = None
#             bsln_date = None
#             bsln_type = None
#             bsln_delta = None
#
#             # find the baseline
#             if len(idiag_rows) == 0:  # no indexed tpts, so disregard entirely
#                 bsln_type = 'No_indexed_values'
#             elif np.all([np.isnan(scr_all_m[x, scr_val_loc]) for x in idiag_rows]):
#                 bsln_type = 'No_indexed_values'
#             # BASELINE CRITERIUM A
#             # first look for valid outpatient values before admission
#             if len(pre_out_rows) != 0 and bsln_type is None:
#                 row = pre_out_rows.pop(-1)
#                 this_date = get_date(scr_all_m[row, scr_date_loc])
#                 delta = admit - this_date
#                 # find latest point that is > 24 hrs prior to admission
#                 while delta < out_lim[0] and len(pre_out_rows) > 0:
#                     row = pre_out_rows.pop(-1)
#                     this_date = get_date(scr_all_m[row, scr_date_loc])
#                     delta = admit - this_date
#                 # if valid point found save it
#                 if out_lim[0] < delta < out_lim[1]:
#                     bsln_date = get_date(str(this_date).split('.')[0])
#                     bsln_val = scr_all_m[row, scr_val_loc]
#                     bsln_type = 'OUTPATIENT'
#                     bsln_delta = delta.total_seconds() / (60 * 60 * 24)
#             # BASLINE CRITERIUM B
#             # no valid outpatient, so look for valid inpatient
#             if len(pre_inp_rows) != 0 and bsln_type is None:
#                 row = pre_inp_rows.pop(-1)
#                 this_date = get_date(scr_all_m[row, scr_date_loc])
#                 delta = admit - this_date
#                 # find latest point that is > 24 hrs prior to admission
#                 while delta < inp_lim[0] and len(pre_inp_rows) > 0:
#                     row = pre_inp_rows.pop(-1)
#                     this_date = get_date(scr_all_m[row, scr_date_loc])
#                     delta = admit - this_date
#                 # if valid point found save it
#                 if inp_lim[0] < delta < inp_lim[1]:
#                     bsln_date = get_date(str(this_date).split('.')[0])
#                     bsln_val = scr_all_m[row, scr_val_loc]
#                     bsln_type = 'INPATIENT'
#                     bsln_delta = delta.total_seconds() / (60 * 60 * 24)
#             # BASELINE CRITERIUM C
#             # no values prior to admission, calculate MDRD derived
#             if bsln_type is None:
#                 dem_idx = np.where(dem_m[:, 0] == cur_id)[0][0]
#                 dob_idx = np.where(dob_m[:, 0] == cur_id)[0]
#                 if dob_idx.size == 0:
#                     bsln_type = 'no_dob'
#                 else:
#                     dob_idx = dob_idx[0]
#                     sex = dem_m[dem_idx, sex_loc]
#                     eth = dem_m[dem_idx, eth_loc]
#                     dob = get_date(dob_m[dob_idx, dob_loc])
#                     age = float((admit - dob).total_seconds()) / (60 * 60 * 24 * 365)
#                     if age > 0:
#                         bsln_val = baseline_est_gfr_mdrd(75, sex, eth, age)
#                         bsln_date = get_date(str(admit).split('.')[0])
#                         bsln_type = 'mdrd'
#                         bsln_delta = 'na'
#             admit = str(admit).split('.')[0]
#             log.write(str(bsln_val) + ',' + str(bsln_type) + ',' + str(bsln_date) + ',' +
#                       str(admit) + ',' + str(bsln_delta) + '\n')
#     log.close()


# %%
def get_baselines(dataPath, hosp_windows, genders, races, ages, scr_fname='SCR_ALL_VALUES.csv',
                  scr_dcol='SCR_ENTERED', scr_vcol='SCR_VALUE', scr_typecol='SCR_ENCOUNTER_TYPE',
                  id_col='STUDY_PATIENT_ID', outp_rng=(1, 365), inp_rng=(7, 365)):

    log = open(os.path.join(dataPath, 'all_baseline_info.csv'), 'w')
    log.write('STUDY_PATIENT_ID,bsln_val,bsln_type,bsln_date,admit_date,time_delta,black,male,age\n')

    out_lim = (datetime.timedelta(outp_rng[0]), datetime.timedelta(outp_rng[1]))
    inp_lim = (datetime.timedelta(inp_rng[0]), datetime.timedelta(inp_rng[1]))

    scr_all_m = pd.read_csv(os.path.join(dataPath, scr_fname))
    ids = list(hosp_windows)

    for i in range(len(ids)):
        tid = ids[i]
        log.write(str(tid) + ',')
        # determine earliest admission date
        admit = get_date(hosp_windows[tid][0])

        # find indices of all SCr values for this patient
        all_rows = np.where(scr_all_m[id_col].values == tid)[0]
        dates = []
        inpatients = []
        for row in all_rows:
            t = get_date(scr_all_m[scr_dcol][row])
            if t == 'nan':
                dates.append(datetime.datetime.now())
            else:
                dates.append(t)
            inpatients.append('inpatient' in scr_all_m[scr_typecol][row].lower())
        dates = np.array(dates)
        inpatients = np.array(inpatients)

        pre_admit_rows = all_rows[np.where([x < admit for x in dates])[0]]
        inp_rows = all_rows[np.where(inpatients)[0]]
        out_rows = all_rows[np.where(inpatients == 0)[0]]

        pre_inp_rows = list(np.intersect1d(pre_admit_rows, inp_rows))

        pre_out_rows = list(np.intersect1d(pre_admit_rows, out_rows))

        # default values
        bsln_val = None
        bsln_date = None
        bsln_type = None
        bsln_delta = None

        # find the baseline

        # BASELINE CRITERIUM A
        # first look for valid outpatient values before admission
        if len(pre_out_rows) != 0 and bsln_type is None:
            row = pre_out_rows.pop(-1)
            this_date = get_date(scr_all_m[scr_dcol][row])
            delta = admit - this_date
            # find latest point that is > 24 hrs prior to admission
            while delta < out_lim[0] and len(pre_out_rows) > 0:
                row = pre_out_rows.pop(-1)
                this_date = get_date(scr_all_m[scr_dcol][row])
                delta = admit - this_date
            # if valid point found save it
            if out_lim[0] < delta < out_lim[1]:
                bsln_date = get_date(str(this_date).split('.')[0])
                bsln_val = scr_all_m[scr_vcol][row]
                bsln_type = 'OUTPATIENT'
                bsln_delta = delta.total_seconds() / (60 * 60 * 24)
        # BASLINE CRITERIUM B
        # no valid outpatient, so look for valid inpatient
        if len(pre_inp_rows) != 0 and bsln_type is None:
            row = pre_inp_rows.pop(-1)
            this_date = get_date(scr_all_m[scr_dcol][row])
            delta = admit - this_date
            # find latest point that is > 24 hrs prior to admission
            while delta < inp_lim[0] and len(pre_inp_rows) > 0:
                row = pre_inp_rows.pop(-1)
                this_date = get_date(scr_all_m[scr_dcol][row])
                delta = admit - this_date
            # if valid point found save it
            if inp_lim[0] < delta < inp_lim[1]:
                bsln_date = get_date(str(this_date).split('.')[0])
                bsln_val = scr_all_m[scr_vcol][row]
                bsln_type = 'INPATIENT'
                bsln_delta = delta.total_seconds() / (60 * 60 * 24)
        # BASELINE CRITERIUM C
        # no values prior to admission, calculate MDRD derived
        sex = genders[i]
        eth = races[i]
        age = ages[i]
        if bsln_type is None:
            bsln_val = baseline_est_gfr_mdrd(75, sex, eth, age)
            bsln_date = get_date(str(admit).split('.')[0])
            bsln_type = 'mdrd'
            bsln_delta = 'na'
        admit = str(admit).split('.')[0]
        log.write(str(bsln_val) + ',' + str(bsln_type) + ',' + str(bsln_date) + ',' +
                  str(admit) + ',' + str(bsln_delta) + ',' + str(eth) + ',' +
                  str(sex) + ',' + str(age) + '\n')
    log.close()
    return


# %%
def get_baselines_dallas(ids, hosp_windows, scr_all_m, scr_val_loc, scr_date_loc, scr_ip_loc,
                         genders, races, ages, fname, outp_rng=(1, 365), inp_rng=(7, 365)):
    log = open(fname, 'w')
    log.write('ID,bsln_val,bsln_type,bsln_date,admit_date,time_delta\n')
    cur_id = None

    out_lim = (datetime.timedelta(outp_rng[0]), datetime.timedelta(outp_rng[1]))
    inp_lim = (datetime.timedelta(inp_rng[0]), datetime.timedelta(inp_rng[1]))

    for i in range(len(ids)):
        tid = ids[i]
        if tid not in ids:
            continue
        if cur_id != tid:
            cur_id = tid
            log.write(str(tid) + ',')
            # determine earliest admission date
            admit = hosp_windows[tid][0]
            if type(admit) != datetime.datetime:
                admit = get_date(admit)

            # find indices of all SCr values for this patient
            all_rows = np.where(scr_all_m[:, 0] == tid)[0]
            all_dates = np.array([get_date(x) for x in scr_all_m[all_rows, scr_date_loc]])
            all_ips = np.array([x for x in scr_all_m[all_rows, scr_ip_loc]])

            sel = np.array([x != 'nan' for x in all_dates])

            if sel.size > 0:
                all_rows = all_rows[sel]
                all_dates = all_dates[sel]
                all_ips = all_ips[sel]

            pre_rows = all_rows[[x < admit for x in all_dates]]
            inp_rows = all_rows[[x == 'inpatient' for x in all_ips]]
            outp_rows = all_rows[[x == 'outpatient' for x in all_ips]]

            pre_out_rows = list(np.intersect1d(pre_rows, outp_rows))
            pre_inp_rows = list(np.intersect1d(pre_rows, inp_rows))

            # default values
            bsln_val = None
            bsln_date = None
            bsln_type = None
            bsln_delta = None

            # BASELINE CRITERIUM A
            # first look for valid outpatient values before admission
            if len(pre_out_rows) != 0 and bsln_type is None:
                row = pre_out_rows.pop(-1)
                this_date = get_date(scr_all_m[row, scr_date_loc])
                delta = admit - this_date
                # find latest point that is > 24 hrs prior to admission
                while delta < out_lim[0] and len(pre_out_rows) > 0:
                    row = pre_out_rows.pop(-1)
                    this_date = get_date(scr_all_m[row, scr_date_loc])
                    delta = admit - this_date
                # if valid point found save it
                if out_lim[0] < delta < out_lim[1]:
                    bsln_date = get_date(str(this_date).split('.')[0])
                    bsln_val = scr_all_m[row, scr_val_loc]
                    bsln_type = 'OUTPATIENT'
                    bsln_delta = delta.total_seconds() / (60 * 60 * 24)
            # BASLINE CRITERIUM B
            # no valid outpatient, so look for valid inpatient
            if len(pre_inp_rows) != 0 and bsln_type is None:
                row = pre_inp_rows.pop(-1)
                this_date = get_date(scr_all_m[row, scr_date_loc])
                delta = admit - this_date
                # find latest point that is > 24 hrs prior to admission
                while delta < inp_lim[0] and len(pre_inp_rows) > 0:
                    row = pre_inp_rows.pop(-1)
                    this_date = get_date(scr_all_m[row, scr_date_loc])
                    delta = admit - this_date
                # if valid point found save it
                if inp_lim[0] < delta < inp_lim[1]:
                    bsln_date = get_date(str(this_date).split('.')[0])
                    bsln_val = scr_all_m[row, scr_val_loc]
                    bsln_type = 'INPATIENT'
                    bsln_delta = delta.total_seconds() / (60 * 60 * 24)
            # BASELINE CRITERIUM C
            # no values prior to admission, calculate MDRD derived
            if bsln_type is None:
                sel = np.where(ids == cur_id)[0][0]
                sex = genders[sel]
                eth = races[sel]
                age = ages[sel]
                bsln_val = baseline_est_gfr_mdrd(75, sex, eth, age)
                bsln_date = get_date(str(admit).split('.')[0])
                bsln_type = 'mdrd'
                bsln_delta = 'na'
            admit = str(admit).split('.')[0]
            log.write(str(bsln_val) + ',' + str(bsln_type) + ',' + str(bsln_date) + ',' +
                      str(admit) + ',' + str(bsln_delta) + '\n')
    log.close()



# # %%
# def get_baselines_dallas(date_m, icu_locs, bsln_loc, bsln_date_loc, bsln_typ_loc, fname):
#     log = open(fname, 'w')
#     log.write('ID,bsln_val,bsln_type,bsln_date,admit_date,disch_date,time_delta\n')
#
#     for i in range(len(date_m)):
#         idx = date_m[i, 0]
#         log.write(str(idx) + ',')
#         # determine earliest admission date
#         tstr = str(date_m[i, icu_locs[0]]).lower()
#         if tstr != 'nat' and tstr != 'nan':
#             admit = datetime.datetime.strptime(tstr, "%Y-%m-%d %H:%M:%S")
#         else:
#             admit = 'nat'
#         tstr = str(date_m[i, icu_locs[1]]).lower()
#         if tstr != 'nat' and tstr != 'nan':
#             disch = datetime.datetime.strptime(tstr, "%Y-%m-%d %H:%M:%S")
#         else:
#             disch = 'nat'
#         bsln_val = date_m[i, bsln_loc]
#         bsln_type = date_m[i, bsln_typ_loc]
#         tstr = str(date_m[i, bsln_date_loc]).lower()
#         if tstr != 'nat' and tstr != 'nan':
#             bsln_date = datetime.datetime.strptime(tstr, "%Y-%m-%d %H:%M:%S")
#         else:
#             bsln_date = 'nat'
#         if admit != 'nat' and bsln_date != 'nat':
#             bsln_delta = (admit - bsln_date).total_seconds() / (60 * 60 * 24)
#         else:
#             bsln_delta = 'nan'
#
#         log.write(str(bsln_val) + ',' + str(bsln_type) + ',' + str(bsln_date) + ',' +
#                   str(admit) + ',' + str(disch) + ',' + str(bsln_delta) + '\n')
#     log.close()


# %%
# def calc_gfr(bsln, date_base, sex, race, dob):
#     if type(date_base) == str or type(date_base) == unicode:
#         date_base = datetime.datetime.strptime(date_base.split('.')[0], '%Y-%m-%d %H:%M:%S')
#     if type(dob) == str or type(dob) == unicode:
#         dob = datetime.datetime.strptime(dob.split('.')[0], '%Y-%m-%d %H:%M:%S')
#     time_diff = date_base - dob
#     year_val = time_diff.years
#     mon_val = time_diff.months
#     age = year_val+mon_val/12
#     min_value = 1
#     max_value = 1
#     race_value = 1
#     if sex == 'M':
#         k_value = 0.9
#         a_value = -0.411
#         f_value = 1
#     else:  # female
#         k_value = 0.7
#         a_value = -0.329
#         f_value = 1.018
#     if race == "BLACK/AFR AMERI":
#         race_value = 1.159
#     if bsln/k_value < 1:
#         min_value = bsln/k_value
#     else:
#         max_value = bsln/k_value
#     min_power = math.pow(min_value, a_value)
#     max_power = math.pow(max_value, -1.209)
#     age_power = math.pow(0.993, age)
#     gfr = 141 * min_power * max_power * age_power * f_value * race_value
#     return gfr
def calc_gfr(scr, sex, race, age):
    if sex == 'M' or sex == 1:
        sex_val = 1.0
    else:  # female
        sex_val = 0.742
    if race == "BLACK/AFR AMERI" or race == "B" or race == 1:
        race_val = 1.212
    else:
        race_val = 1.0
    gfr = 174 * math.pow(scr, -1.154) * math.pow(age, -0.203) * sex_val * race_val
    return gfr


# %%
def baseline_est_gfr_epi(gfr, sex, race, age):
    race_value = 1
    sel = 0
    if sex:
        k_value = 0.9
        a_value = -0.411
        f_value = 1
    else:  # female
        k_value = 0.7
        a_value = -0.329
        f_value = 1.018
    if sex and not race and age >= 109:
        sel = 1
    if sex and race and age >= 88:
        sel = 1
    if not sex and not race and age >= 90:
        sel = 1
    if not sex and race and age >= 111:
        sel = 1

    if race:
        race_value = 1.159
    if sel == 0:
        a_value = -1.209
    numerator = gfr * math.pow(k_value, a_value)
    denominator = 141 * math.pow(0.993, age) * f_value * race_value
    scr = math.pow((numerator / denominator), 1. / a_value)

    return scr


# %%
def baseline_est_gfr_mdrd(gfr, sex, race, age):
    race_value = 1
    if sex == 1 or sex == "M":
        f_value = 1
    else:  # female
        f_value = 0.742
    if race == 1 or race == "BLACK/AFR AMERI" or race == "B":
        race_value = 1.212
    numerator = gfr
    denominator = 175 * age ** (-0.203) * f_value * race_value
    scr = (numerator / denominator) ** (1. / -1.154)
    return scr


# %%
def scr2kdigo(scr, base, masks, days, valid, useAbs=True):
    kdigos = []
    for i in range(len(scr)):
        kdigo = np.zeros(len(scr[i]), dtype=int)
        for j in range(len(scr[i])):
            if masks[i][j] > 0:
                kdigo[j] = 4
                continue
            elif scr[i][j] <= (1.5 * base[i]):
                if j > 7 and useAbs:
                    window = np.where(days[i] >= days[i][j] - 2)[0]
                    window = window[np.where(window < j)[0]]
                    window = np.intersect1d(window, np.where(valid[i])[0])
                    if window.size > 0:
                        if scr[i][j] >= np.min(scr[i][window]) + 0.3:
                            kdigo[j] = 1
                        else:
                            kdigo[j] = 0
                    else:
                        kdigo[j] = 0
                else:
                    kdigo[j] = 0
            elif scr[i][j] < (2 * base[i]):
                kdigo[j] = 1
            elif scr[i][j] < (3 * base[i]):
                kdigo[j] = 2
            elif (scr[i][j] >= (3 * base[i])) or (scr[i][j] >= 4.0):
                kdigo[j] = 3
            elif scr[i][j] >= 4.0:
                kdigo[j] = 3
        kdigos.append(kdigo)
    return kdigos


def pairwise_feat_dist(features, ids, dm_fname, dist=lambda x, xx: np.abs(x - xx),
                       desc='Feature Based Distance Calculation'):
    df = open(dm_fname, 'w')
    dis = []
    for i in tqdm(range(len(features)), desc=desc):
        for j in range(i + 1, len(features)):
            d = dist(features[i], features[j])
            df.write('%d,%d,%f' % (ids[i], ids[j], d))
            dis.append(d)
    dis = np.array(dis)
    df.close()
    return dis


# %%
def pairwise_dtw_dist(patients, days, ids, dm_fname, dtw_name, v=True,
                      mismatch=lambda y, yy: abs(y-yy),
                      extension=lambda y: 0,
                      dist=braycurtis,
                      alpha=1.0, t_lim=7):
    '''
    For each pair of arrays in patients, this function first applies dynamic time warping to
    align the arrays to the same length and then computes the distance between the aligned arrays.

    :param patients: List containing N individual arrays of variable length to be aligned
    :param days: List containing N arrays, each indicating the day of each individual point
    :param ids: List of N patient IDs
    :param dm_fname: Filename to save distances
    :param dtw_name: Filename to save DTW alignment
    :param v: verbose... if True prints extra information
    :param mismatch: function handle... determines cost of mismatched values
    :param extension: function handle... if present, introduces extension penalty coresponding to value
    :param dist: function handle... calculates the total distance between the aligned curves
    :param alpha: float value... specifies weight of extension penalty vs. mismatch penalty
    :param t_lim: float/integer... only include data where day <= t_lim
    :return: condensed pair-wise distance matrix
    '''
    df = open(dm_fname, 'w')
    dis = []
    if not os.path.exists(dtw_name):
        if v and dtw_name is not None:
            log = open(dtw_name, 'w')
        for i in tqdm(range(len(patients)), desc='DTW and Distance Calculation'):
            sel = np.where(days[i] <= t_lim)[0]
            patient1 = np.array(patients[i])[sel]
            dlist = []
            for j in range(i + 1, len(patients)):
                df.write('%d,%d,' % (ids[i], ids[j]))
                sel = np.where(days[j] < t_lim)[0]
                patient2 = np.array(patients[j])[sel]
                if np.all(patient1 == patient2):
                    df.write('%f\n' % 0)
                    dis.append(0)
                    dlist.append(0)
                else:
                    if len(patient1) > 1 and len(patient2) > 1:
                        d, _, _, path, xext, yext = dtw_p(patient1, patient2, mismatch=mismatch, extension=extension, alpha=alpha)
                        p1_path = path[0]
                        p2_path = path[1]
                        p1 = np.array([patient1[p1_path[x]] for x in range(len(p1_path))])
                        p2 = np.array([patient2[p2_path[x]] for x in range(len(p2_path))])
                    elif len(patient1) == 1:
                        p1 = np.repeat(patient1[0], len(patient2))
                        p2 = patient2
                    elif len(patient2) == 1:
                        p1 = patient1
                        p2 = np.repeat(patient2[0], len(patient1))
                    if np.all(p1 == p2):
                        df.write('%f\n' % 0)
                        dis.append(0)
                        dlist.append(0)
                    else:
                        d = dist(p1, p2)
                        df.write('%f\n' % d)
                        dis.append(d)
                        dlist.append(d)
                if v and dtw_name is not None:
                    log.write(arr2str(p1, fmt='%d') + '\n')
                    log.write(arr2str(p2, fmt='%d') + '\n\n')
        if v and dtw_name is not None:
            log.close()
    else:
        dtw = open(dtw_name, 'r')
        for i in tqdm(range(len(ids)), desc='Distance Calculation using Previous DTW'):
            for j in range(i+1, len(ids)):
                p1 = np.array(dtw.readline().rstrip().split(','), dtype=int)
                p2 = np.array(dtw.readline().rstrip().split(','), dtype=int)
                d = dist(p1, p2)
                dis.append(d)
                df.write('%d,%d,%f\n' % (ids[i], ids[j], d))
                _ = dtw.readline()
        dtw.close()
    df.close()
    return dis

#
# def mismatch(x, y):
#     tcosts = [1.00,  # [0 - 1]
#               2.73,  # [1 - 2]
#               4.36,  # [2 - 3]
#               6.74]  # [3 - 4]
#     cost_dic = {}
#     for i in range(len(tcosts)):
#         cost_dic[tuple(set((i, i+1)))] = tcosts[i]
#     for i in range(len(tcosts)):
#         for j in range(i + 2, len(tcosts) + 1):
#             cost_dic[tuple(set((i, j)))] = cost_dic[tuple(set((i, j-1)))] + cost_dic[tuple(set((j-1, j)))]
#     if x == y:
#         return 0
#     elif x < y:
#         return cost_dic[tuple(set((x, y)))]
#     else:
#         return cost_dic[tuple(set((y, x)))]
#
# def extension(x):
#     tcosts = [1.00,  # [0 - 1]
#               2.73,  # [1 - 2]
#               4.36,  # [2 - 3]
#               6.74]  # [3 - 4]
#     costs = np.zeros(len(tcosts) + 1)
#     for i in range(len(tcosts)):
#         costs[i + 1] = tcosts[i]
#
#     return costs[x]
#
#
# def dist(x, y):
#     coordinates = [0, 1, 2, 3, 4]
#     return braycurtis([coordinates[xi] for xi in x], [coordinates[yi] for yi in y])
#
#
# def parallel_pairwise_dtw_dist(ids, kdigos, dm_fname, dtw_name, alpha=1.0):
#     df = open(dm_fname, 'w')
#     log_dtw = False
#     if not os.path.exists(dtw_name):
#         log = open(dtw_name, 'w')
#         log_dtw = True
#     pool = mp.Pool(processes=mp.cpu_count())
#     d = []
#     for i in tqdm(range(len(kdigos)), desc='Parallel DTW and Distance Calculation'):
#         results = []
#         for j in range(i + 1, len(kdigos)):
#             results.append(individual_dtw_dist(ids[i], ids[j], kdigos[i], kdigos[j], mismatch, extension, dist, alpha))
#         # results = [pool.apply(individual_dtw_dist, args=(ids[i], ids[j], kdigos[i], kdigos[j], mismatch, extension, dist, alpha)) for j in range(i + 1, len(kdigos))]
#         for r in results:
#             if log_dtw:
#                 log.write('%d,' % r[0])
#                 log.write(arr2str(r[2], fmt='%d') + '\n')
#                 log.write('%d,' % r[1])
#                 log.write(arr2str(r[3], fmt='%d') + '\n\n')
#             df.write('%d,%d,%f\n' % (r[0], r[1], r[4]))
#             d.append(r[4])
#     if log_dtw:
#         log.close()
#     df.close()
#     return d
#
#
# def individual_dtw_dist(id1, id2, X, Y,
#                         mismatch=lambda y, yy: abs(y-yy), extension=lambda y: 0, dist=braycurtis, alpha=1.0):
#     if np.all(X == Y):
#         d = 0
#     else:
#         _, _, _, path = dtw_p(X, Y, mismatch=mismatch, extension=extension, alpha=alpha)
#         X_path = path[0]
#         Y_path = path[1]
#         X = np.array([X[X_path[x]] for x in range(len(X_path))])
#         Y = np.array([Y[Y_path[x]] for x in range(len(Y_path))])
#         if np.all(X == Y):
#             d = 0
#         else:
#             d = dist(X, Y)
#     return (id1, id2, X, Y, d)


# %%
def dtw_p(x, y, mismatch=lambda y, yy: abs(y-yy),
                extension=lambda y: 0,
                alpha=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences with weighted penalty exponentiation.
    Designed for sequences of distinct integer values in the set [0, 1, 2, 3, 4]

    :param array x: N1*M array
    :param array y: N2*M array
    :param func mismatch: distance used as cost measure
    :param func extension: extension penalty applied when repeating index
    :param alpha: float value indicating relative weight of extension penalty

    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the warp path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))  # distance matrix
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = mismatch(x[i], y[j])
    C = D1.copy()
    ext_y = np.zeros(c)
    ext_x = np.zeros(r)
    for i in range(r):
        for j in range(c):
            diag = D0[i, j]
            sel = np.argmin((D0[i, j], D0[i, j + 1], D0[i + 1, j]))
            if sel == 1:
                ext_y[j] += alpha * extension(y[j])
                D1[i, j] += D0[i, j + 1] + ext_y[j]
            elif sel == 2:
                ext_x[i] += alpha * extension(x[i])
                D1[i, j] += D0[i + 1, j] + ext_x[i]
            else:
                D1[i, j] += diag
    if len(x) == 1:
        path = np.zeros(len(y))
    elif len(y) == 1:
        path = np.zeros(len(x))
    else:
        path = _traceback(D0, x, y)
    xext = 0
    for i in range(len(x)):
        idx = np.where(path[0] == i)[0]
        for j in range(len(idx)):
            xext += extension(x[i]) * j
    yext = 0
    for i in range(len(y)):
        idx = np.where(path[1] == i)[0]
        for j in range(len(idx)):
            yext += extension(y[i]) * j
    return D1[-1, -1] / sum(D1.shape), C, D1, path, xext, yext


def _traceback(D, x, y):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while i > 0 or j > 0:
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1

        else:  # (tb == 2):
            j -= 1

        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


def pairwise_zeropad_dist(patients, days, ids, dm_fname, dist=braycurtis, t_lim=7):
    df = open(dm_fname, 'w')
    dis = []
    for i in tqdm(range(len(patients)), desc='Zero-Padded Distance Calculation'):
        sel = np.where(days[i] <= t_lim)[0]
        patient1 = np.array(patients[i])[sel]
        dlist = []
        for j in range(i + 1, len(patients)):
            df.write('%d,%d,' % (ids[i], ids[j]))
            sel = np.where(days[j] < t_lim)[0]
            patient2 = np.array(patients[j])[sel]
            if np.all(patient1 == patient2):
                df.write('%f\n' % 0)
                dis.append(0)
                dlist.append(0)
            else:
                if len(patient1) > 1 and len(patient2) > 1:
                    l = max(len(patient1), len(patient2))
                    p1 = np.zeros(l, dtype=int)
                    p2 = np.zeros(l, dtype=int)
                    p1[:len(patient1)] = patient1
                    p2[:len(patient2)] = patient2
                elif len(patient1) == 1:
                    p1 = np.repeat(patient1[0], len(patient2))
                    p2 = patient2
                elif len(patient2) == 1:
                    p1 = patient1
                    p2 = np.repeat(patient2[0], len(patient1))
                if np.all(p1 == p2):
                    df.write('%f\n' % 0)
                    dis.append(0)
                    dlist.append(0)
                else:
                    d = dist(p1, p2)
                    df.write('%f\n' % d)
                    dis.append(d)
                    dlist.append(d)
    df.close()
    return dis


# %%
def get_uky_dynamic_features(ids, dataPath):
    '''
    Extracts dynamic features from csv files for the UKY KDIGO data
    Dynamic Features:
        BUN         LABS_SET1 -> BUN_DX_HIGH_VALUE
        Diastolic BP (low)
        FiO2
        GCS
        HR
        PO2
        Potassium
        Sodium
        Systolic BP (high)
        Temperature
        Urine Out
        WBC
    :param ids:
    :param dataPath:
    :return: dynamic features
    '''

def get_uky_static_features(ids, dataPath):
    '''
    Extracts dynamic features from csv files for the UKY KDIGO data
    Dynamic Features:
        Bicarbonate LABS_SET1 -> CO2_D1_LOW_VALUE
        Bilirubin   LABS_SET1 -> BILIRUBIN_D1_HIGH_VALUE
        BUN
        Diastolic BP (low)
        FiO2
        GCS
        HR
        PO2
        Potassium
        Sodium
        Systolic BP (high)
        Temperature
        Urine Out
        WBC
    :param ids:
    :param dataPath:
    :return: static features
    '''


# %%
def rel_scr(ids, scrs, bslns, outPath):
    rel_chg = []
    abs_chg = []
    assert len(ids) == len(scrs) == len(bslns)
    for i in range(len(ids)):
        scrl = scrs[i]
        bsln = bslns[i]
        tabs = [scrl[0] - bsln]
        trel = [scrl[0] / bsln]
        for j in range(1, len(scrl)):
            tabs.append(scrl[j] - bsln)
            trel.append(scrl[j] / bsln)

        abs_chg.append(tabs)
        rel_chg.append(trel)

    arr2csv(os.path.join(outPath, 'scr_rel_chg.csv'), rel_chg, ids)
    arr2csv(os.path.join(outPath, 'scr_abs_chg.csv'), abs_chg, ids)


# %%
def dict2csv(fname, inds, fmt='%f', header=False):
    ids = sorted(list(inds))
    outFile = open(fname, 'w')
    if header:
        outFile.write(header)
        outFile.write('\n')
    for i in range(len(inds)):
        tid = ids[i]
        outFile.write(str(tid))
        if np.size(inds[tid]) > 1:
            for j in range(len(inds[tid])):
                outFile.write(',' + fmt % (inds[tid][j]))
        elif np.size(inds[i]) == 1:
            try:
                outFile.write(',' + fmt % (inds[tid]))
            except TypeError:
                outFile.write(',' + fmt % (inds[tid][0]))
        else:
            outFile.write(',')
        outFile.write('\n')
    outFile.close()


# %%
def arr2csv(fname, inds, ids=None, fmt='%f', header=False):
    outFile = open(fname, 'w')
    if ids is None:
        ids = np.arange(len(inds))
    if header:
        outFile.write(header)
        outFile.write('\n')
    for i in range(len(inds)):
        outFile.write(str(ids[i]))
        if np.size(inds[i]) > 1:
            for j in range(len(inds[i])):
                outFile.write(',' + fmt % (inds[i][j]))
        elif np.size(inds[i]) == 1:
            try:
                outFile.write(',' + fmt % (inds[i]))
            except TypeError:
                outFile.write(',' + fmt % (inds[i][0]))
        else:
            outFile.write(',')
        outFile.write('\n')
    outFile.close()


# %%
def arr2str(arr, fmt='%f'):
    s = fmt % (arr[0])
    for i in range(1, len(arr)):
        s = s + ',' + fmt % (arr[i])
    return s


# %%
def load_csv(fname, ids=None, dt=float, skip_header=False, idxs=None, targets=None, struct='list', id_dtype=int):
    if struct == 'list':
        res = []
    elif struct == 'dict':
        res = {}
    rid = []
    f = open(fname, 'r')
    hdr = None
    if skip_header is not False or targets is not None:
        hdr = f.readline().rstrip().split(',')[1:]
        hdr = np.array(hdr)
    if targets is not None:
        assert hdr is not None
        idxs = []
        try:
            for target in targets:
                tidx = np.where(hdr == target)[0]
                if tidx.size == 0:
                    raise ValueError('%s was not found in file %s' % (target, fname.split('/')[-1]))
                elif tidx.size > 1:
                    raise ValueError('%s is not unique in file %s' % (target, fname.split('/')[-1]))
                idxs.append(tidx)
        except TypeError:
            tidx = np.where(hdr == targets)[0]
            if tidx.size == 0:
                raise ValueError('%s was not found in file %s' % (target, fname.split('/')[-1]))
            elif tidx.size > 1:
                raise ValueError('%s is not unique in file %s' % (target, fname.split('/')[-1]))
            idxs.append(tidx)
    for line in f:
        l = np.array(line.rstrip().split(','))
        tid = id_dtype(l[0])
        if ids is None or tid in ids:
            if idxs is None:
                if len(l) > 1 and l[1] != '':
                    if type(dt) == str and dt == 'date':
                        d = [get_date(x) for x in l[1:]]
                        if struct == 'list':
                            res.append(d)
                        elif struct == 'dict':
                            res[tid] = d
                    else:
                        if struct == 'list':
                            res.append(np.array(l[1:], dtype=dt))
                        elif struct == 'dict':
                            res[tid] = np.array(l[1:], dtype=dt)
                else:
                    if struct == 'list':
                        res.append(())
                    elif struct == 'dict':
                        res[tid] = []
            else:
                if type(dt) == str and dt == 'date':
                    d = [get_date(l[idx]) for idx in idxs]
                    if struct == 'list':
                        res.append(d)
                    elif struct == 'dict':
                        res[tid] = d
                else:
                    if struct == 'list':
                        res.append(np.array([l[idx] for idx in idxs], dtype=dt))
                    elif struct == 'dict':
                        res[tid] = np.array([l[idx] for idx in idxs], dtype=dt)
            if ids is not None:
                rid.append(type(ids[0])(l[0]))
            else:
                rid.append(l[0])
    if struct == 'list':
        try:
            if np.all([len(res[x]) == len(res[0]) for x in range(len(res))]):
                res = np.array(res)
                if res.ndim > 1:
                    if res.shape[1] == 1:
                        res = np.squeeze(res)
        except (ValueError, TypeError):
            res = res
    if ids is not None:
        if len(rid) != len(ids):
            print('Missing ids in file: ' + fname)
        if skip_header == 'keep':
            return res, hdr
        else:
            return res
    else:
        rid = np.array(rid)
        if skip_header == 'keep':
            return hdr, res, rid
        else:
            return res, rid


# %%
def descriptive_trajectory_features(kdigos, ids, days=None, t_lim=None, filename='descriptive_features.csv'):
    npts = len(kdigos)
    features = np.zeros((npts, 26))
    header = 'id,no_AKI,peak_at_KDIGO1,peak_at_KDIGO2,peak_at_KDIGO3,peak_at_KDIGO3D,KDIGO1_at_admit,KDIGO2_at_admit,' + \
             'KDIGO3_at_admit,KDIGO3D_at_admit,KDIGO1_at_disch,KDIGO2_at_disch,KDIGO3_at_disch,KDIGO3D_at_disch,' + \
             'onset_lt_3days,onset_gte_3days,complete_recovery_lt_3days,multiple_hits,KDIGO1_gt_24hrs,KDIGO2_gt_24hrs,' + \
             'KDIGO3_gt_24hrs,KDIGO4_gt_24hrs,flat,strictly_increase,strictly_decrease,slope_posTOneg,slope_negTOpos'
    for i in range(len(kdigos)):
        kdigo = kdigos[i]
        if days is not None:
            tdays = days[i]
            sel = np.where(tdays <= t_lim)[0]
            kdigo = kdigo[sel]
        kdigo1 = np.where(kdigo == 1)[0]
        kdigo2 = np.where(kdigo == 2)[0]
        kdigo3 = np.where(kdigo == 3)[0]
        kdigo4 = np.where(kdigo == 4)[0]
        # No AKI
        if np.all(kdigo == 0):
            features[i, 0] = 1
        # KDIGO 1 Peak
        temp = 0  # previous score
        direction = 0  # 1: Increased     -1: Decreased
        for j in range(len(kdigo)):
            if kdigo[j] < temp:
                if direction == 1:
                    features[i, temp] = 1
                direction = -1

            elif kdigo[j] > temp:
                direction = 1
            temp = kdigo[j]

        if direction == 1:
            features[i, temp] = 1
        '''
        if kdigo[0] > 0:
            temp = kdigo[0]
            i = 1
            while i < len(kdigo):
                if kdigo[i] > temp:
                    break
                elif kdigo[i] < temp:
                    if temp == 1:
                        features[i, 1] = 1
                    elif temp == 2:
                        features[i, 2] = 1
                    elif temp == 3:
                        features[i, 3] = 1
                    elif temp == 4:
                        features[i, 4] = 1
                    break
                i += 1
        '''
        # KDIGO @ admit
        if kdigo[0] == 1:
            features[i, 5] = 1
        elif kdigo[0] == 2:
            features[i, 6] = 1
        elif kdigo[0] == 3:
            features[i, 7] = 1
        elif kdigo[0] == 4:
            features[i, 8] = 1

        # KDIGO @ discharge
        if kdigo[-1] == 1:
            features[i, 9] = 1
        elif kdigo[-1] == 2:
            features[i, 10] = 1
        elif kdigo[-1] == 3:
            features[i, 11] = 1
        elif kdigo[-1] == 4:
            features[i, 12] = 1

        # AKI onset <3 days after admit
        if kdigo[0] == 0 and np.any(kdigo[:12] > 0):
            features[i, 13] = 1

        # AKI onset >=3 days after admit
        if np.all(kdigo[:12] == 0) and np.any(kdigo[12:] > 0):
            features[i, 14] = 1

        # Complete recovery within 3 days
        if len(kdigo) >= 12:
            if np.any(kdigo > 0) and np.all(kdigo[12:] == 0):
                features[i, 15] = 1
        else:
            if kdigo[-1] == 0:
                features[i, 15] = 1

        # Multiple hits separated by >= 24 hrs
        temp = 0
        count = 0
        for j in range(len(kdigo)):
            if kdigo[j] > 0:
                if temp == 0:
                    if count >= 4:
                        features[i, 16] = 1
                temp = 1
                count = 0
            else:
                if temp == 1:
                    count = 1
                temp = 0
                if count > 0:
                    count += 1

        # >=24 hrs at KDIGO 1
        if len(kdigo1) >= 4:
            features[i, 17] = 1
        # >=24 hrs at KDIGO 2
        if len(kdigo2) >= 4:
            features[i, 18] = 1
        # >=24 hrs at KDIGO 3
        if len(kdigo3) >= 4:
            features[i, 19] = 1
        # >=24 hrs at KDIGO 3D
        if len(kdigo4) >= 4:
            features[i, 20] = 1
        # Flat trajectory
        if np.all(kdigo == kdigo[0]):
            features[i, 21] = 1

        # KDIGO strictly increases
        diff = kdigo[1:] - kdigo[:-1]
        if np.any(diff > 0):
            if np.all(diff >= 0):
                features[i, 22] = 1
        '''
        j = 1
        while j < len(kdigo):
            if kdigo[j] < kdigo[j-1]:
                break
            if j == len(kdigo) - 1:
                if kdigo[j] >= kdigo[0]:
                    features[i, 22] = 1
            j += 1
        '''

        # KDIGO strictly decreases
        if np.any(diff < 0):
            if np.all(diff <= 0):
                features[i, 23] = 1
        '''
        j = 1
        while j < len(kdigo):
            if kdigo[j] > kdigo[j - 1]:
                break
            if j == len(kdigo) - 1:
                if kdigo[j] <= kdigo[0]:
                    features[i, 23] = 1
            j += 1
        '''

        # Slope changes sign
        direction = 0
        temp = kdigo[0]
        for j in range(len(kdigo)):
            if kdigo[j] < temp:
                # Pos to neg
                if direction == 1:
                    features[i, 24] = 1
                direction = -1
            elif kdigo[j] > temp:
                # Neg to pos
                if direction == -1:
                    features[i, 25] = 1
                direction = 1
            temp = kdigo[j]
    arr2csv(filename, features, ids, fmt='%d', header=header)
    return features, header


# %%
def new_descriptive_trajectory_features(kdigos, ids, days=None, t_lim=None, filename='new_descriptive_features.csv'):
    npts = len(kdigos)
    features = np.zeros((npts, 15))
    header = 'id,peak_KDIGO,KDIGO_at_admit,KDIGO_at_7d,AKI_first_3days,AKI_after_3days,' + \
             'multiple_hits,KDIGO1+_gt_24hrs,KDIGO2+_gt_24hrs,KDIGO3+_gt_24hrs,KDIGO4_gt_24hrs,' + \
             'flat,strictly_increase,strictly_decrease,slope_posTOneg,slope_negTOpos'
    PEAK_KDIGO = 0
    KDIGO_ADMIT = 1
    KDIGO_7D = 2
    AKI_FIRST_3D = 3
    AKI_AFTER_3D = 4
    MULTI_HITS = 5
    KDIGO1_GT1D = 6
    KDIGO2_GT1D = 7
    KDIGO3_GT1D = 8
    KDIGO4_GT1D = 9
    FLAT = 10
    ONLY_INC = 11
    ONLY_DEC = 12
    POStoNEG = 13
    NEGtoPOS = 14

    for i in range(len(kdigos)):
        kdigo = kdigos[i]
        if days is not None:
            tdays = days[i]
            sel = np.where(tdays <= t_lim)[0]
            kdigo = kdigo[sel]
            tdays = tdays[sel]
        kdigo1 = np.where(kdigo == 1)[0]
        kdigo2 = np.where(kdigo == 2)[0]
        kdigo3 = np.where(kdigo == 3)[0]
        kdigo4 = np.where(kdigo == 4)[0]
        # No AKI
        features[i, PEAK_KDIGO] = max(kdigo)
        features[i, KDIGO_ADMIT] = kdigo[0]
        features[i, KDIGO_7D] = kdigo[-1]
        features[i, AKI_FIRST_3D] = max(kdigo[np.where(tdays <= 3)])
        if max(tdays) > 3:
            features[i, AKI_AFTER_3D] = max(kdigo[np.where(tdays > 3)])

        # Multiple hits separated by >= 24 hrs
        temp = 0
        count = 0
        eps = 0
        for j in range(len(kdigo)):
            if kdigo[j] > 0:
                if temp == 0:
                    if count >= 4:
                        features[i, MULTI_HITS] = 1
                temp = 1
                count = 0
            else:
                if temp == 1:
                    count = 1
                temp = 0
                if count > 0:
                    count += 1

        # >=24 hrs at KDIGO 1
        if len(np.where(kdigo >= 1)[0]) >= 4:
            features[i, KDIGO1_GT1D] = 1
        # >=24 hrs at KDIGO 2
        if len(np.where(kdigo >= 2)[0]) >= 4:
            features[i, KDIGO2_GT1D] = 1
        # >=24 hrs at KDIGO 3
        if len(np.where(kdigo >= 3)[0]) >= 4:
            features[i, KDIGO3_GT1D] = 1
        # >=24 hrs at KDIGO 3D
        if len(np.where(kdigo >= 4)[0]) >= 4:
            features[i, KDIGO4_GT1D] = 1
        # Flat trajectory
        if np.all(kdigo == kdigo[0]):
            features[i, FLAT] = 1

        # KDIGO strictly increases
        diff = kdigo[1:] - kdigo[:-1]
        if np.any(diff > 0):
            if np.all(diff >= 0):
                features[i, ONLY_INC] = 1

        # KDIGO strictly decreases
        if np.any(diff < 0):
            if np.all(diff <= 0):
                features[i, ONLY_DEC] = 1

        # Slope changes sign
        direction = 0
        temp = kdigo[0]
        for j in range(len(kdigo)):
            if kdigo[j] < temp:
                # Pos to neg
                if direction == 1:
                    features[i, POStoNEG] = 1
                direction = -1
            elif kdigo[j] > temp:
                # Neg to pos
                if direction == -1:
                    features[i, NEGtoPOS] = 1
                direction = 1
            temp = kdigo[j]
    arr2csv(filename, features, ids, fmt='%d', header=header)
    return features, header


def descriptive_trajectory_features_param(kdigos, ids, days=None, t_lim=None, lenDur=2, filename='descriptive_features_param.csv'):
    npts = len(kdigos)
    features = np.zeros((npts, 18))
    header = 'id,peak_KDIGO,KDIGO_at_admit,KDIGO_at_7d,HIGH_KDIGO_first_3days,HIGH_KDIGO_after_3days,' + \
             'multiple_hits,KDIGO1+_dur,KDIGO2+_dur,KDIGO3+_dur,KDIGO4_dur,' + \
             'flat,strictly_increase,strictly_decrease,slope_posTOneg,slope_negTOpos,KDIGO0_dur,LOW_KDIGO_first_3days,LOW_KDIGO_after_3days'
    PEAK_KDIGO = 0
    KDIGO_ADMIT = 1
    KDIGO_7D = 2
    HIGH_AKI_FIRST_3D = 3
    HIGH_AKI_AFTER_3D = 4
    MULTI_HITS = 5
    KDIGO1_DUR = 6
    KDIGO2_DUR = 7
    KDIGO3_DUR = 8
    KDIGO3D_DUR = 9
    FLAT = 10
    ONLY_INC = 11
    ONLY_DEC = 12
    POStoNEG = 13
    NEGtoPOS = 14
    KDIGO0_DUR = 15
    LOW_AKI_FIRST_3D = 16
    LOW_AKI_AFTER_3D = 17

    for i in range(len(kdigos)):
        kdigo = kdigos[i]
        if days is not None:
            tdays = days[i]
            sel = np.where(tdays <= t_lim)[0]
            kdigo = kdigo[sel]
            tdays = tdays[sel]
        kdigo1 = np.where(kdigo == 1)[0]
        kdigo2 = np.where(kdigo == 2)[0]
        kdigo3 = np.where(kdigo == 3)[0]
        kdigo4 = np.where(kdigo == 4)[0]
        # No AKI
        features[i, PEAK_KDIGO] = max(kdigo)
        features[i, KDIGO_ADMIT] = kdigo[0]
        features[i, KDIGO_7D] = kdigo[-1]
        features[i, HIGH_AKI_FIRST_3D] = max(kdigo[np.where(tdays <= 3)])
        features[i, LOW_AKI_FIRST_3D] = min(kdigo[np.where(tdays <= 3)])
        if max(tdays) > 3:
            features[i, HIGH_AKI_AFTER_3D] = max(kdigo[np.where(tdays > 3)])
            features[i, LOW_AKI_AFTER_3D] = min(kdigo[np.where(tdays > 3)])
        else:
            features[i, HIGH_AKI_AFTER_3D] = max(kdigo[np.where(tdays <= 3)])
            features[i, LOW_AKI_AFTER_3D] = min(kdigo[np.where(tdays <= 3)])

        # Multiple hits separated by >= 24 hrs
        temp = 0
        count = 0
        for j in range(len(kdigo)):
            if kdigo[j] > 0:
                if temp == 0:
                    if count >= 4:
                        features[i, MULTI_HITS] = 1
                temp = 1
                count = 0
            else:
                if temp == 1:
                    count = 1
                temp = 0
                if count > 0:
                    count += 1

        if lenDur is not None:
            # >=24 hrs at KDIGO 1
            if len(np.where(kdigo >= 1)[0]) >= 4 * lenDur:
                features[i, KDIGO1_DUR] = 1
            # >=24 hrs at KDIGO 2
            if len(np.where(kdigo >= 2)[0]) >= 4 * lenDur:
                features[i, KDIGO2_DUR] = 1
            # >=24 hrs at KDIGO 3
            if len(np.where(kdigo >= 3)[0]) >= 4 * lenDur:
                features[i, KDIGO3_DUR] = 1
            # >=24 hrs at KDIGO 3D
            if len(np.where(kdigo >= 4)[0]) >= 4 * lenDur:
                features[i, KDIGO3D_DUR] = 1
        else:
            features[i, KDIGO1_DUR] = len(np.where(kdigo >= 1)[0])
            features[i, KDIGO2_DUR] = len(np.where(kdigo >= 2)[0])
            features[i, KDIGO3_DUR] = len(np.where(kdigo >= 3)[0])
            features[i, KDIGO3D_DUR] = len(np.where(kdigo >= 4)[0])
        features[i, KDIGO0_DUR] = len(np.where(kdigo == 0)[0])
        # Flat trajectory
        if np.all(kdigo == kdigo[0]):
            features[i, FLAT] = 1

        # KDIGO strictly increases
        diff = kdigo[1:] - kdigo[:-1]
        if np.any(diff > 0):
            if np.all(diff >= 0):
                features[i, ONLY_INC] = 1

        # KDIGO strictly decreases
        if np.any(diff < 0):
            if np.all(diff <= 0):
                features[i, ONLY_DEC] = 1

        # Slope changes sign
        direction = 0
        temp = kdigo[0]
        for j in range(len(kdigo)):
            if kdigo[j] < temp:
                # Pos to neg
                if direction == 1:
                    features[i, POStoNEG] = 1
                direction = -1
            elif kdigo[j] > temp:
                # Neg to pos
                if direction == -1:
                    features[i, NEGtoPOS] = 1
                direction = 1
            temp = kdigo[j]
    arr2csv(filename, features, ids, fmt='%d', header=header)
    return features, header


def template_trajectory_features(kdigos, ids, days=None, t_lim=None, filename='template_trajectory_features.csv',
                                 scores=np.array([0, 1, 2, 3, 4], dtype=int), npoints=3, ratios=False, gap=0, stride=1):
    combination = scores
    for i in range(npoints - 1):
        combination = np.vstack((combination, scores))
    npts = len(kdigos)
    templates = cartesian(combination)
    header = 'id'
    for i in range(len(templates)):
        header += ',' + str(templates[i])
    features = np.zeros((npts, len(templates)))
    for i in range(npts):
        kdigo = kdigos[i]
        if len(kdigo) < npoints:
            continue
        if days is not None:
            tdays = days[i]
            sel = np.where(tdays <= t_lim)[0]
            kdigo = kdigo[sel]
        nwin = np.floor((len(kdigo) - npoints + 1) / stride).astype(int)
        for j in range(nwin):
            start = j * stride
            # tk = kdigo[start:start + npoints]
            tk = [kdigo[x] for x in range(start, start + npoints + (gap * (npoints - 1)), gap + 1)]
            sel = np.where(templates[:, 0] == tk[0])[0]
            loc = [x for x in sel if np.all(templates[x, :] == tk)]
            features[i, loc] += 1
        if ratios:
            features[i, :] = features[i, :] / np.sum(features[i, :])
    if ratios:
        arr2csv(filename, features, ids, fmt='%f', header=header)
    else:
        arr2csv(filename, features, ids, fmt='%d', header=header)
    return features, header


def slope_trajectory_features(kdigos, ids, days=None,t_lim=None, scores=np.array([0, 1, 2, 3, 4]),
                              filename='slope_features.csv', ratios=False, gap=0, stride=1):
    slopes = []
    header = 'ids'
    for i in range(len(scores)):
        slopes.append(scores[i] - scores[0])
        header += ',%d' % (scores[i] - scores[0])
    for i in range(len(slopes)):
        if slopes[i] > 0:
            slopes.append(-slopes[i])
            header += ',%d' % (-slopes[i])
    slopes = np.array(slopes)
    npts = len(kdigos)
    features = np.zeros((npts, len(slopes)))
    for i in range(npts):
        kdigo = kdigos[i]
        if days is not None:
            tdays = days[i]
            sel = np.where(tdays <= t_lim)[0]
            kdigo = kdigo[sel]
        nwin = np.floor((len(kdigo) - 1) / stride).astype(int)
        for j in range(nwin):
            start = stride * j
            ts = kdigo[start + gap + 1] - kdigo[start]
            loc = np.where(slopes == ts)[0][0]
            features[i, loc] += 1
        if ratios:
            features[i, :] = features[i, :] / np.sum(features[i, :])
    if ratios:
        arr2csv(filename, features, ids, fmt='%f', header=header)
    else:
        arr2csv(filename, features, ids, fmt='%d', header=header)
    return features, header


def feature_voting(ds):
    features = np.zeros(ds.shape[1])
    flags = np.zeros(ds.shape[1])
    for i in range(ds.shape[1]):
        avg = np.mean(ds[:, i], axis=0)
        std = np.std(ds[:, i], axis=0)
        if avg > 0.5:
            features[i] = 1
        if avg < 2 * std:
            flags[i] = 1
    print('Features with High Variability:')
    sel = np.where(flags)[0]
    for idx in sel:
        print('Feature #' + str(idx))
    return features


def perf_measure(y_actual, y_hat):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            tp += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            fp += 1
        if y_actual[i] == y_hat[i] == 0:
            tn += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            fn += 1
    prec = precision_score(y_actual, y_hat)
    rec = recall_score(y_actual, y_hat)
    f1 = f1_score(y_actual, y_hat)
    return np.array((prec, rec, f1, tp, fp, tn, fn))


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def feature_selection(data, lbls, method='univariate', params=[]):
    '''

    :param data: Matrix of the form n_samples x n_features
    :param lbls: n_samples binary features
    :param method: {'variance', 'univariate', 'recursive', 'FromModel'}

    :param params: variance     - [float pct]
                   univariate   - [int n_feats, function scoring]
                   recursive    - [obj estimator, int cv]
                   FromModel
                        linear  - [float c]
                        tree    - []

    :return:
    '''

    if method == 'variance':
        assert len(params) == 1
        pct = params[0]
        sel = VarianceThreshold(threshold=(pct * (1 - pct)))
        sf = sel.fit_transform(data)
        return sel, sf

    elif method == 'univariate':
        assert len(params) == 2
        n_feats = params[0]
        scoring = params[1]
        sel = SelectKBest(scoring, k=n_feats)
        sf = sel.fit_transform(data, lbls)
        return sel, sf

    elif method == 'recursive':
        assert len(params) == 2
        estimator = params[0]
        cv = params[1]
        rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(cv),
                      scoring='accuracy')
        rfecv.fit(data, lbls)
        sel = rfecv.support_
        sf = data[:, sel]
        return rfecv, sf

    elif method == 'linear':
        assert len(params) == 1
        c = params[0]
        lsvc = LinearSVC(C=c, penalty='l1', dual=False).fit(data, lbls)
        model = SelectFromModel(lsvc, prefit=True)
        sf = model.transform(data)
        return model, sf

    elif method == 'tree':
        assert params == []
        clf = RandomForestClassifier()
        clf.fit(data, lbls)
        model = SelectFromModel(clf, prefit=True)
        sf = model.transform(data)
        return model, sf

    else:
        print("Please select one of the following methods:")
        print("[variance, univariate, recursive, linear, tree")


# %%
def daily_max_kdigo(scr, days, bsln, dmask, tlim=7):
    mk = []
    temp = days[0]
    tmax = 0
    for i in range(len(scr)):
        if days[i] != temp:
            mk.append(tmax)
            temp = days[i]
            tmax = 0
        start = np.where(days >= temp - 2)[0][0]
        if start == i:
            tmin = scr[i]
        else:
            tmin = np.min(scr[start:i])
        if np.any(dmask[start:i]):
            tmax = 4
        elif scr[i] <= (1.5 * bsln):
            # absolute change
            if scr[i] >= tmin + 0.3:
                tmax = max(tmax, 1)
        elif scr[i] < (2.0 * bsln):
            tmax = max(tmax, 1)
        elif scr[i] < (3.0 * bsln):
            tmax = max(tmax, 2)
        elif scr[i] >= (3.0 * bsln) or scr[i] >= 4.0:
            tmax = max(tmax, 3)
    mk.append(tmax)
    return np.array(mk, dtype=int)


def daily_max_kdigo_interp(kdigos, days, tlim=7):
    mk = []
    for i in range(tlim + 1):
        sel = np.where(days == i)[0]
        if sel.size == 0:
            tmax = np.nan
        else:
            tmax = np.max(kdigos[sel])
        mk.append(tmax)
    return np.array(mk)


def daily_max_kdigo_aligned(kdigo, pts_per_day=4):
    mk = []
    for start in range(0, len(kdigo), pts_per_day):
        if start + pts_per_day < len(kdigo):
            tmax = np.max(kdigo[start:start+pts_per_day])
        else:
            tmax = np.max(kdigo[start:])
        mk.append(tmax)
    return np.array(mk)

# def daily_max_kdigo(scr, dates, bsln, admit_date, dmask, tlim=7):
#     mk = []
#     temp = dates[0].day
#     tmax = 0
#     # Prior 2 days minimum (IRRESPECTIVE OF TIME!!!)
#     #   i.e. any record on day 2 is compared to the minimum value during days 0-1, even if the minimum is at the
#     #        beginning of day 0 and the current point is at the end of day 2
#     for i in range(len(scr)):
#         date = dates[i]
#         day = date.day
#         if day != temp:
#             mk.append(tmax)
#             tmax = 0
#             temp = day
#         if dmask[i] > 0:
#             tmax = 4
#         elif scr[i] <= (1.5 * bsln):
#             if (date - admit_date).day >= datetime.timedelta(2):
#                 minim = scr[i]
#                 for j in range(i)[::-1]:
#                     delta = (date - dates).day
#                     if delta < datetime.timedelta(2):
#                         break
#                     if scr[i] >= minim + 0.3:
#                         if tmax < 1:
#                             tmax = 1
#                         break
#                     else:
#                         if scr[i] < minim:
#                             minim = scr[i]
#         elif scr[i] < (2 * bsln):
#             if tmax < 1:
#                 tmax = 1
#         elif scr[i] < (3 * bsln):
#             if tmax < 2:
#                 tmax = 2
#         elif (scr[i] >= (3 * bsln)) or (scr[i] >= 4.0):
#             if tmax < 3:
#                 tmax = 3
#     mk.append(tmax)
#     return mk


def get_cluster_features(individual, lbls, dm, op='mean'):
    clbls = np.unique(lbls)
    n_clust = len(clbls)
    n_feats = individual.shape[1]
    cluster_feats = np.zeros((n_clust, n_feats))
    cluster_feats_ind = np.zeros((len(lbls), n_feats))

    if dm.ndim == 1:
        dm = squareform(dm)
    for i in range(n_clust):
        tlbl = clbls[i]
        sel = np.where(lbls == tlbl)[0]
        cluster_ind = individual[sel, :]

        sqsel = np.ix_(sel, sel)
        cdm = dm[sqsel]
        cdm = np.sum(cdm, axis=0)
        cidx = np.argsort(cdm)[0]

        if type(op) == str:
            if op == 'mean':
                cluster_ind = np.mean(cluster_ind, axis=0)
            elif op == 'mean_bin':
                cluster_ind = np.mean(cluster_ind, axis=0)
                cluster_ind[np.where(cluster_ind >= 0.5)] = 1
                cluster_ind[np.where(cluster_ind < 1)] = 0
            elif op == 'center':
                cluster_ind = cluster_ind[cidx, :]
        else:
            cluster_ind = op(cluster_ind, axis=0)

        cluster_feats[i, :] = cluster_ind
        cluster_feats_ind[sel, :] = cluster_ind

    return cluster_feats_ind, cluster_feats


def cluster_feature_vectors(desc, temp, slope, lbls, dm):
    clbls = np.unique(lbls)
    n_clust = len(clbls)
    n_desc = desc.shape[1]
    n_temp = temp.shape[1]
    n_slope = slope.shape[1]
    desc_c = np.zeros((n_clust, n_desc))
    desc_c_bin = np.zeros((n_clust, n_desc))
    temp_c = np.zeros((n_clust, n_temp))
    slope_c = np.zeros((n_clust, n_slope))
    desc_c_center = np.zeros((n_clust, n_desc))
    temp_c_center = np.zeros((n_clust, n_temp))
    slope_c_center = np.zeros((n_clust, n_slope))
    if dm.ndim == 1:
        dm = squareform(dm)
    for i in range(n_clust):
        tlbl = clbls[i]
        sel = np.where(lbls == tlbl)[0]
        tdesc = desc[sel, :]
        ttemp = temp[sel, :]
        tslope = slope[sel, :]
        
        sqsel = np.ix_(sel, sel)
        cdm = dm[sqsel]
        cdm = np.sum(cdm, axis=0)
        cidx = np.argsort(cdm)[0]

        tdesc = np.mean(tdesc, axis=0)
        tdesc_bin = np.array(tdesc, copy=True)
        tdesc_bin[np.where(tdesc >= 0.5)] = 1
        tdesc_bin[np.where(tdesc < 1)] = 0
        ttemp = np.mean(ttemp, axis=0)
        tslope = np.mean(tslope, axis=0)

        desc_c[i, :] = tdesc
        desc_c_bin[i, :] = tdesc_bin
        temp_c[i, :] = ttemp
        slope_c[i, :] = tslope
        desc_c_center[i, :] = desc[sel[cidx], :]
        temp_c_center[i, :] = temp[sel[cidx], :]
        slope_c_center[i, :] = slope[sel[cidx], :]
    return desc_c_bin, desc_c, temp_c, slope_c, desc_c_center, temp_c_center, slope_c_center


def mismatch_penalty_func(*tcosts):
    '''
    Returns distance function for the mismatch penalty between any two KDIGO
     scores in the range [0, len(tcosts) - 1]
    :param tcosts: List of float values corresponding to transitions between
                   consecutive KDIGO scores. E.g.
                    tcosts[i] = cost(i, i + 1)
    :return:
    '''
    cost_dic = {}
    for i in range(len(tcosts)):
        cost_dic[tuple(set((i, i+1)))] = tcosts[i]
    for i in range(len(tcosts)):
        for j in range(i + 2, len(tcosts) + 1):
            cost_dic[tuple(set((i, j)))] = cost_dic[tuple(set((i, j-1)))] + cost_dic[tuple(set((j-1, j)))]

    def penalty(x, y):
        if x == y:
            return 0
        elif x < y:
            return cost_dic[tuple(set((x, y)))]
        else:
            return cost_dic[tuple(set((y, x)))]

    penalty.nvals = len(tcosts) + 1

    return penalty


def continuous_mismatch(discrete):
    nvals = discrete.nvals
    cm = np.zeros((nvals, nvals))
    for i in range(nvals):
        for j in range(nvals):
            cm[i, j] = discrete(i, j)

    x = np.arange(nvals)
    y = np.arange(nvals)

    penalty = interp2d(x, y, cm)

    return penalty


def extension_penalty_func(*tcosts):
    costs = {0: 0}
    for i in range(len(tcosts)):
        costs[i + 1] = tcosts[i]

    def penalty(x):
        return costs[x]

    penalty.nvals = len(tcosts) + 1

    return penalty


def continuous_extension(discrete):
    nvals = discrete.nvals
    cm = np.zeros(nvals)
    for i in range(1, nvals):
        cm[i] = discrete(i)
    x = np.arange(nvals)
    penalty = interp1d(x, cm)

    return penalty


# def get_custom_braycurtis(*tcosts, **kwargs):
#     coordinates = np.zeros(len(tcosts) + 1)
#     shift = 0
#     if 'shift' in list(kwargs):
#         shift = kwargs['shift']
#
#     coordinates[0] = np.sum(tcosts) + shift
#     for i in range(len(tcosts)):
#         coordinates[i + 1] = coordinates[i] - tcosts[i]
#
#     def dist(x, y):
#         return braycurtis(coordinates[x], coordinates[y])
#
#     return dist


def get_custom_distance_discrete(coordinates, dfunc='braycurtis'):
    if dfunc == 'braycurtis':
        dist = get_custom_braycurtis(coordinates)
    elif dfunc == 'euclidean':
        def dist(x, y):
            return man_euclidean_norm(coordinates[x], coordinates[y],
                                      minval=min(coordinates), maxval=max(coordinates))
    elif dfunc == 'cityblock':
        def dist(x, y):
            return man_cityblock_norm(coordinates[x], coordinates[y],
                                      minval=min(coordinates), maxval=max(coordinates))
    return dist


def get_custom_braycurtis(coordinates):
    def dist(x, y):
        return braycurtis(coordinates[x], coordinates[y])

    return dist


def get_continuous_laplacian_braycurtis(coordinates, lf=0, lf_type='aggregated'):
    def dist(x, y):
        out = 0
        if hasattr(x, '__len__'):
            # If vector input, accumulate numerator and denominator separately
            num = 0
            denom = 0
            for i in range(len(x)):
                # Use linear interpolation to determine new coordinate given a float value
                # within the range of the original discrete set
                # I.e. given a float value 0 <= x <= 4 indicating a KDIGO severity, the new
                # coordinate is computed using linear interpolation between floor(x) and ceil(x),
                xv = x[i]
                start = coordinates[int(np.floor(xv))]
                stop = coordinates[int(np.ceil(xv))]
                step = xv - np.floor(xv)
                xSlope = stop - start
                xCoord = start + step * xSlope
                yv = y[i]
                start = coordinates[int(np.floor(yv))]
                stop = coordinates[int(np.ceil(yv))]
                step = yv - np.floor(yv)
                ySlope = stop - start
                yCoord = start + step * ySlope
                # Bray-Curtis is then computed like normal, with the addition of the Laplacian factor
                # in the denominator
                num += np.abs(xCoord - yCoord)
                if lf_type == 'aggregated':
                    denom += np.abs(xCoord + yCoord + lf)
                else:
                    denom += np.abs(xCoord + yCoord)
            if lf_type == 'individual':
                out = num / (denom + lf)
            else:
                out = num / denom
            return out
        else:
            # Get new coordinates from float values
            xstart = coordinates[int(np.floor(x))]
            xstop = coordinates[int(np.ceil(x))]
            ystart = coordinates[int(np.floor(y))]
            ystop = coordinates[int(np.ceil(y))]

            xSlope = xstop - xstart
            xCoord = xstart + (x - np.floor(x)) * xSlope

            ySlope = ystop - ystart
            yCoord = ystart + (y - np.floor(y)) * ySlope

            out += (np.abs(xCoord - yCoord) / np.abs(xCoord + yCoord + lf))
        return out

    return dist


def get_custom_cityblock_continuous(coordinates):
    minval = min(coordinates)
    maxval = max(coordinates)

    def dist(x, y):
        newXCoords = np.zeros(len(x))
        newYCoords = np.zeros(len(y))
        for i in range(len(x)):
            xv = x[i]
            start = coordinates[int(np.floor(xv))]
            stop = coordinates[int(np.ceil(xv))]
            xSlope = stop - start
            newXCoords[i] = start + (xv - start) * xSlope
            yv = y[i]
            start = coordinates[int(np.floor(yv))]
            stop = coordinates[int(np.ceil(yv))]
            ySlope = stop - start
            newYCoords[i] = start + (yv - start) * ySlope
        n = cityblock(newXCoords, newYCoords)
        d = cityblock(np.zeros(len(x)) + minval, np.zeros(len(x)) + maxval)
        return n / d

    return dist


def man_euclidean_norm(x, y, minval=0, maxval=1):
    n = euclidean(x, y)
    d = euclidean(np.zeros(len(x)) + minval, np.zeros(len(x)) + maxval)
    return n / d


def man_cityblock_norm(x, y, minval=0, maxval=1):
    n = cityblock(x, y)
    d = cityblock(np.zeros(len(x)) + minval, np.zeros(len(x)) + maxval)
    return n / d


def get_euclidean_norm(coordinates):
    memo = {}

    def dist(x, y):
        d = euclidean(coordinates[x], coordinates[y])
        try:
            d /= memo[len(x)]
        except KeyError:
            denom = euclidean(np.zeros(len(x)) + max(coordinates), np.zeros(len(x)) + min(coordinates))
            memo[len(x)] = denom
            d /= denom
        return d

    return dist


def get_cityblock_norm(coordinates):
    memo = {}

    def dist(x, y):
        d = cityblock(coordinates[x], coordinates[y])
        try:
            d /= memo[len(x)]
        except KeyError:
            denom = cityblock(np.zeros(len(x)) + max(coordinates), np.zeros(len(x)) + min(coordinates))
            memo[len(x)] = denom
            d /= denom
        return d
    return dist


def get_weighted_braycurtis(coordinates, min_weight=0.5):
    def dist(x, y):
        n = len(x)
        w = np.linspace(min_weight, 1, n)
        num = 0
        den = 0
        for i in range(n):
            num += (w[i] * abs(coordinates[x[i]] - coordinates[y[i]]))
            den += (w[i] * abs(coordinates[x[i]] + coordinates[y[i]]))
        distance = (num / den)
        return distance

    return dist


def get_laplacian_braycurtis(coordinates, lf=0.2):
    def dist(x, y):
        if hasattr(x, '__len__'):
            x = np.array([coordinates[xi] for xi in x])
            y = np.array([coordinates[yi] for yi in y])
            d = float(np.sum(np.abs(x - y))) / (np.sum(np.abs(x + y)) + lf*len(x))
        else:
            d = (np.abs(x - y)) / (np.abs(x + y) + lf)
        return d
    return dist


def get_pop_dist(*tcosts):
    coordinates = np.zeros(len(tcosts) + 1)
    for i in range(len(tcosts)):
        coordinates[i + 1] = coordinates[i] + tcosts[i]

    def dist(x, y):
        return np.sum(np.abs(coordinates[x] - coordinates[y]))

    return dist


def binary_dist(x, y):
    x = np.array(x, dtype=bool)
    y = np.array(y, dtype=bool)
    o = np.logical_or(x, y)
    return np.sum(o)


def count_transitions(kdigos):
    k_lbls = range(5)
    transitions = cartesian((k_lbls, k_lbls))
    t_counts = np.zeros((len(kdigos), len(transitions)))
    for i in range(len(kdigos)):
        for j in range(len(kdigos[i]) - 1):
            idx = np.intersect1d(np.where(transitions[:, 0] == kdigos[i][j])[0],
                                 np.where(transitions[:, 1] == kdigos[i][j + 1])[0])
            t_counts[i, idx] += 1

    return t_counts


def get_transition_weights(kdigos):
    t_counts = count_transitions(kdigos)
    t_counts = np.mean(t_counts, axis=0)
    k_lbls = range(5)
    transitions = cartesian((k_lbls, k_lbls))
    # KDIGO 0-1
    weights = {}
    tot = 0
    for k1 in range(4):
        k2 = k1 + 1
        idx1 = np.intersect1d(np.where(transitions[:, 0] == k1)[0], np.where(transitions[:, 1] == k2)[0])[0]
        idx2 = np.intersect1d(np.where(transitions[:, 0] == k2)[0], np.where(transitions[:, 1] == k1)[0])[0]
        weights[(k1, k2)] = t_counts[idx1] + t_counts[idx2]
        tot += t_counts[idx1] + t_counts[idx2]
    minval = 100
    for k in list(weights):
        weights[k] = np.log10(tot / weights[k])
        minval = min(minval, weights[k])

    out = []
    for k in list(weights):
        weights[k] /= minval
        out.append(weights[k])

    return out




def get_usrds_dod(death_fname, id_fname, out_name='dod.csv', delimiter='^'):
    f = open(id_fname, 'r')
    hdr = f.readline().rstrip().split(delimiter)
    pid_loc = np.where([x == 'PTID' for x in hdr])[0][0]
    usrds_id_loc = np.where([x == 'USRDS_ID' for x in hdr])[0][0]
    id_ref = {}
    for line in f:
        l = line.rstrip().split(delimiter)
        usrdsid = int(l[usrds_id_loc])
        if usrdsid not in list(id_ref):
            id_ref[usrdsid] = int(l[pid_loc])
    f.close()

    f = open(death_fname)
    hdr = f.readline().rstrip().split(delimiter)
    id_loc = np.where([x == 'USRDS_ID' for x in hdr])[0][0]
    dod_loc = np.where([x == 'DOD' for x in hdr])[0][0]

    out = open(out_name, 'w')
    for line in f:
        l = line.rstrip().split(delimiter)
        usrdsid = int(l[id_loc])
        try:
            pid = id_ref[usrdsid]
        except KeyError:
            print('Patient w/ USRDS ID \"%d\" does not have a corresponding PID' % usrdsid)
            continue
        try:
            dod = str(datetime.datetime.strptime(l[dod_loc], '%m/%d/%Y'))
        except ValueError:
            dod = ''
        out.write('%d,%s\n' % (pid, dod))
    out.close()


def add_usrds_dod(usrds_f, orig_f, out_f):
    ids = np.loadtxt(usrds_f, delimiter=',', usecols=0, dtype=int)
    dod = load_csv(usrds_f, ids, str)

    with open(orig_f, 'r') as f:
        hdr = f.readline().rstrip()

    ids2 = np.loadtxt(orig_f, delimiter=',', skiprows=1, usecols=0, dtype=int)
    orig = np.loadtxt(orig_f, delimiter=',', skiprows=1, dtype=str)

    ids = list(ids)
    ids2 = list(ids2)

    dod = list(dod)

    out = open(out_f, 'w')
    out.write(hdr + ',source\n')

    of = open(orig_f, 'r')
    _ = of.readline()
    while len(ids) > 0 or len(ids2) > 0:
        if len(ids2) == 0 or ids[0] < ids2[0]:
            tid = ids.pop(0)
            tdod = dod.pop(0)
            out.write('%d,,%s,usrds\n' % (tid, tdod))
        elif len(ids) == 0 or ids2[0] < ids[0]:
            _ = ids2.pop(0)
            l = of.readline().rstrip()
            out.write(l + ',uk\n')
        else:
            tid = ids.pop(0)
            tdod = dod.pop(0)
            out.write('%d,,%s,both\n' % (tid, tdod))
            _ = ids2.pop(0)
            _ = of.readline()
    out.close()
    of.close()
    return


def get_usrds_esrd(esrd_fname, id_fname, out_name='esrd.csv', delimiter='^'):
    f = open(id_fname, 'r')
    hdr = f.readline().rstrip().split(delimiter)
    pid_loc = np.where([x == 'PTID' for x in hdr])[0][0]
    usrds_id_loc = np.where([x == 'USRDS_ID' for x in hdr])[0][0]
    id_ref = {}
    for line in f:
        l = line.rstrip().split(delimiter)
        usrdsid = int(l[usrds_id_loc])
        if usrdsid not in list(id_ref):
            id_ref[usrdsid] = int(l[pid_loc])
    f.close()

    f = open(esrd_fname)
    hdr = f.readline().rstrip().split(delimiter)
    id_loc = np.where([x == 'USRDS_ID' for x in hdr])[0][0]
    esrd_loc = np.where([x == 'FIRST_SE' for x in hdr])[0][0]

    out = open(out_name, 'w')
    no_ref_count = 0
    pids = []
    for line in f:
        l = line.rstrip().split(delimiter)
        usrdsid = int(l[id_loc])
        try:
            pid = id_ref[usrdsid]
            pids.append(pid)
        except KeyError:
            no_ref_count += 1
            print('Patient w/ USRDS ID \"%d\" does not have a corresponding PID' % usrdsid)
            continue
        try:
            esrd = str(datetime.datetime.strptime(l[esrd_loc], '%m/%d/%Y'))
        except ValueError:
            esrd = ''
        out.write('%d,%s\n' % (pid, esrd))
    out.close()
    print('Total number of patients with no matching PID: %d' % no_ref_count)
    return pids


# %%
def get_icd9_codes(filename):
    icd9 = {}
    f = open(filename, 'r')
    for line in f:
        l = line.rstrip().split(', ')
        if len(l[0]) == 0:
            continue
        key, l[0] = l[0].split(': ')
        out = []
        for i in range(len(l)):
            if '-' in l[i]:
                v1, v2 = l[i].split(' - ')
                v1_pt1, v1_pt2 = v1.split('.')
                v2_pt1, v2_pt2 = v2.split('.')
                if v1_pt1 == v2_pt1:
                    start = int(v1_pt2)
                    stop = int(v2_pt2)
                    tl = range(start, stop + 1)
                    for j in range(len(tl)):
                        out.append('%s.%d' % (v1_pt1, tl[j]))
                else:
                    start = int(v1_pt1)
                    stop = int(v2_pt1)
                    tl = range(start, stop + 1)
                    for j in range(len(tl)):
                        out.append('%d.%s' % (tl[j], v1_pt2))
            else:
                out.append(l[i])
        # for i in range(len(out)):
        #     out[i] = out[i].replace("x", "")
        icd9[key] = out
    return icd9



# %%
def get_comorbidities(ids, diag_m, icd_loc, icd_ref):
    icd9 = get_icd9_codes(icd_ref)
    comorbidities = np.zeros((len(ids), len(list(icd9))))
    hdr = 'ids,' + ','.join(list(icd9))
    conditions = list(icd9)
    for i in range(len(ids)):
        tid = ids[i]
        rrt_idx = np.where(diag_m[:, 0] == tid)[0]
        for idx in rrt_idx:
            for j in range(len(conditions)):
                condition = conditions[j]
                for code in icd9[condition]:
                    if 'x' in code:
                        ref = code.split('.')[0] + '.'
                    else:
                        ref = code
                    cur = str(diag_m[idx, icd_loc])
                    if cur != 'nan' and ref in diag_m[idx, icd_loc]:
                        comorbidities[i, j] = 1
    return comorbidities


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True, label_names=None, show=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    sens = tp / (tp + fn)

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if show:
        plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=16)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=16)


    plt.tight_layout()
    if label_names is None:
        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}\nsensitivity={:0.4f}; specificity={:0.4f}'.format(accuracy, misclass, sens, spec), fontsize=18)
    else:
        plt.ylabel(label_names[0], fontsize=18)
        plt.xlabel(
            label_names[1] + '\naccuracy={:0.4f}; misclass={:0.4f}\nsensitivity={:0.4f}; specificity={:0.4f}'.format(
                accuracy, misclass, sens, spec), fontsize=18)
    if show:
        plt.show()


def get_rel_scr(ids, dataPath):
    scr_interp = load_csv(os.path.join(dataPath, "scr_interp.csv"), ids)
    bslns = load_csv(os.path.join(dataPath, "baselines.csv"), ids)
    out = open(os.path.join(dataPath, 'rel_scr_interp.csv'), 'w')
    rel = []
    for i in range(len(ids)):
        out.write('%d' % ids[i])
        trel = np.zeros(len(scr_interp[i]))
        for j in range(len(scr_interp[i])):
            out.write(',%4f' % (scr_interp[i][j] / bslns[i]))
            trel[j] = scr_interp[i][j] / bslns[i]
        out.write('\n')
        rel.append(trel)
    return rel


def getESRDFlags(f, dataPath):
    udf = pd.read_csv(os.path.join(dataPath, 'USRDS_ESRD.csv'))
    rdf = pd.read_csv(os.path.join(dataPath, 'RENAL_REPLACE_THERAPY.csv'))
    ids = f['meta']['ids'][:]
    deltas = np.full((len(ids), 2), np.nan)
    flags = np.zeros((len(ids), 7))
    BEF_ADMIT = 0
    BEF_DISCH = 1
    BEF_DISCH_120 = 2
    BEF_DISCH_365 = 3
    RRT_LT24 = 4
    RRT_LT48 = 5
    RRT_LT72 = 6
    for i in range(len(ids)):
        tid = ids[i]
        idx = np.where(udf['STUDY_PATIENT_ID'][:] == tid)[0]
        if idx.size > 0:
            idx = idx[0]
            esrd = get_date(udf['ESRD_DATE'][idx])
            if esrd != 'nan':
                icu = get_array_dates(f['meta']['icu_dates'][i].astype(str))
                admit = get_date(icu[0])
                disch = get_date(icu[1])
                aDelta = (esrd - admit).total_seconds() / (60 * 60 * 24)
                dDelta = (esrd - disch).total_seconds() / (60 * 60 * 24)
                deltas[i, 0] = aDelta
                deltas[i, 1] = dDelta
                if aDelta < 0:
                    flags[i, BEF_ADMIT] = 1
                if dDelta < 0:
                    flags[i, BEF_DISCH] = 1
                if dDelta < 120:
                    flags[i, BEF_DISCH_120] = 1
                if dDelta < 365:
                    flags[i, BEF_DISCH_365] = 1
        idx = np.where(rdf['STUDY_PATIENT_ID'].values == tid)[0]
        if idx.size > 0:
            idx = idx[0]
            stop = get_date(rdf['HD_STOP_DATE'][idx])
            disch = get_date(f['meta']['icu_dates'][i, 0].astype(str))
            if stop != 'nan':
                if (disch - stop) < datetime.timedelta(1):
                    flags[i, RRT_LT24] = 1
                if (disch - stop) < datetime.timedelta(2):
                    flags[i, RRT_LT48] = 1
                if (disch - stop) < datetime.timedelta(3):
                    flags[i, RRT_LT72] = 1
            stop = get_date(rdf['CRRT_STOP_DATE'][idx])
            if stop != 'nan':
                if (disch - stop) < datetime.timedelta(1):
                    flags[i, RRT_LT24] = 1
                if (disch - stop) < datetime.timedelta(2):
                    flags[i, RRT_LT48] = 1
                if (disch - stop) < datetime.timedelta(3):
                    flags[i, RRT_LT72] = 1
    return deltas, flags


def build_class_vectors(ids, vecs, starts, dods, targLen=8*4, pad='zero', fillDir='back'):
    out = np.zeros((len(ids), targLen))
    for i in range(len(vecs)):
        if dods[i] != 'nan':
            dodBin = int((get_date(dods[i]) - get_date(starts[i])).total_seconds() / (60 * 60 * 6))
            dodBin = min(len(vecs[i]), dodBin)
        else:
            dodBin = len(vecs[i])
        stop = min(dodBin, targLen)
        if fillDir == 'back':
            out[i, :stop] = vecs[i][:stop]
            if pad == 'fill':
                out[i, stop:] = vecs[i][stop-1]
        else:
            out[i, -stop:] = vecs[i][:stop]
            if pad == 'fill':
                out[i, :(targLen - stop)] = vecs[i][0]
    return out



def load_all_csv(datapath, sort_id='STUDY_PATIENT_ID'):
    print('Loading encounter info...')
    # Get IDs and find indices of all used metrics
    date_m = pd.read_csv(datapath + 'all_sheets/ADMISSION_INDX.csv')
    date_m.sort_values(by=[sort_id, 'HOSP_ADMIT_DATE'], inplace=True)
    hosp_locs = [date_m.columns.get_loc("HOSP_ADMIT_DATE"), date_m.columns.get_loc("HOSP_DISCHARGE_DATE")]
    icu_locs = [date_m.columns.get_loc("ICU_ADMIT_DATE"), date_m.columns.get_loc("ICU_DISCHARGE_DATE")]
    adisp_loc = date_m.columns.get_loc('DISCHARGE_DISPOSITION')
    date_m = date_m.values

    ### GET SURGERY SHEET AND LOCATION
    print('Loading surgery information...')
    surg_m = pd.read_csv(datapath + 'all_sheets/SURGERY_INDX.csv')
    surg_m.sort_values(by=sort_id, inplace=True)
    surg_des_loc = surg_m.columns.get_loc("SURGERY_DESCRIPTION")
    surg_m = surg_m.values

    ### GET DIAGNOSIS SHEET AND LOCATION
    print('Loading diagnosis information...')
    diag_m = pd.read_csv(datapath + 'all_sheets/DIAGNOSIS.csv')
    diag_m.sort_values(by=sort_id, inplace=True)
    diag_loc = diag_m.columns.get_loc("DIAGNOSIS_DESC")
    diag_nb_loc = diag_m.columns.get_loc("DIAGNOSIS_SEQ_NB")
    icd_type_loc = diag_m.columns.get_loc("ICD_TYPECODE")
    icd_code_loc = diag_m.columns.get_loc("ICD_CODE")
    diag_m = diag_m.values

    print('Loading ESRD status...')
    # ESRD status
    scm_esrd_m = pd.read_csv(datapath + 'all_sheets/ESRD_STATUS.csv')
    scm_esrd_m.sort_values(by=sort_id, inplace=True)
    scm_esrd_before = scm_esrd_m.columns.get_loc("BEFORE_INDEXED_INDICATOR")
    scm_esrd_at = scm_esrd_m.columns.get_loc("AT_ADMISSION_INDICATOR")
    scm_esrd_during = scm_esrd_m.columns.get_loc("DURING_INDEXED_INDICATOR")
    scm_esrd_after = scm_esrd_m.columns.get_loc("AFTER_INDEXED_INDICATOR")
    scm_esrd_m = scm_esrd_m.values

    # Dialysis dates
    print('Loading dialysis dates...')
    rrt_m = pd.read_csv(datapath + 'all_sheets/RENAL_REPLACE_THERAPY.csv')
    rrt_m.sort_values(by=sort_id, inplace=True)
    crrt_locs = [rrt_m.columns.get_loc('CRRT_START_DATE'), rrt_m.columns.get_loc('CRRT_STOP_DATE')]
    hd_locs = [rrt_m.columns.get_loc('HD_START_DATE'), rrt_m.columns.get_loc('HD_STOP_DATE')]
    pd_locs = [rrt_m.columns.get_loc('PD_START_DATE'), rrt_m.columns.get_loc('PD_STOP_DATE')]
    hd_trt_loc = rrt_m.columns.get_loc('HD_TREATMENTS')
    rrt_m = rrt_m.values

    # All SCR
    print('Loading SCr values (may take a while)...')
    scr_all_m = pd.read_csv(datapath + 'all_sheets/SCR_ALL_VALUES.csv')
    scr_all_m.sort_values(by=[sort_id, 'SCR_ENTERED'], inplace=True)
    scr_date_loc = scr_all_m.columns.get_loc('SCR_ENTERED')
    scr_val_loc = scr_all_m.columns.get_loc('SCR_VALUE')
    scr_desc_loc = scr_all_m.columns.get_loc('SCR_ENCOUNTER_TYPE')
    scr_all_m = scr_all_m.values

    # Demographics
    print('Loading demographics...')
    dem_m = pd.read_csv(datapath + 'all_sheets/DEMOGRAPHICS_INDX.csv')
    dem_m.sort_values(by=sort_id, inplace=True)
    sex_loc = dem_m.columns.get_loc('GENDER')
    eth_loc = dem_m.columns.get_loc('RACE')
    dem_m = dem_m.values

    # DOB
    print('Loading birthdates...')
    dob_m = pd.read_csv(datapath + 'all_sheets/DOB.csv')
    dob_m.sort_values(by=sort_id, inplace=True)
    birth_loc = dob_m.columns.get_loc("DOB")
    dob_m = dob_m.values

    # load death data
    print('Loading dates of death...')
    mort_m = pd.read_csv(datapath + 'all_sheets/OUTCOMES_COMBINED.csv')
    mort_m.sort_values(by=sort_id, inplace=True)
    mdate_loc = mort_m.columns.get_loc("DECEASED_DATE")
    mort_m = mort_m.values

    # load fluid input/output info
    print('Loading fluid I/O totals...')
    io_m = pd.read_csv(datapath + 'all_sheets/IO_TOTALS.csv')
    io_m.sort_values(by=sort_id, inplace=True)
    io_m = io_m.values

    # load dates of birth
    print('Loading charlson scores...')
    charl_m = pd.read_csv(datapath + 'all_sheets/CHARLSON_SCORE.csv')
    charl_m.sort_values(by=sort_id, inplace=True)
    charl_loc = charl_m.columns.get_loc("CHARLSON_INDEX")
    charl_m = charl_m.values

    # load dates of birth
    print('Loading elixhauser scores...')
    elix_m = pd.read_csv(datapath + 'all_sheets/ELIXHAUSER_SCORE.csv')
    elix_m.sort_values(by=sort_id, inplace=True)
    elix_loc = elix_m.columns.get_loc("ELIXHAUSER_INDEX")
    elix_m = elix_m.values

    print('Loading blood gas labs...')
    blood_m = pd.read_csv(datapath + 'all_sheets/BLOOD_GAS.csv')
    blood_m.sort_values(by=sort_id, inplace=True)
    pao2_locs = [blood_m.columns.get_loc('PO2_D1_LOW_VALUE'),
                 blood_m.columns.get_loc('PO2_D1_HIGH_VALUE')]
    paco2_locs = [blood_m.columns.get_loc('PCO2_D1_LOW_VALUE'),
                  blood_m.columns.get_loc('PCO2_D1_HIGH_VALUE')]
    ph_locs = [blood_m.columns.get_loc('PH_D1_LOW_VALUE'),
               blood_m.columns.get_loc('PH_D1_HIGH_VALUE')]
    blood_m = blood_m.values

    print('Loading clinical others...')
    clinical_oth = pd.read_csv(datapath + 'all_sheets/CLINICAL_OTHERS.csv')
    clinical_oth.sort_values(by=sort_id, inplace=True)
    resp_locs = [clinical_oth.columns.get_loc('RESP_RATE_D1_LOW_VALUE'),
                 clinical_oth.columns.get_loc('RESP_RATE_D1_HIGH_VALUE')]
    fio2_locs = [clinical_oth.columns.get_loc('FI02_D1_LOW_VALUE'),
                 clinical_oth.columns.get_loc('FI02_D1_HIGH_VALUE')]
    gcs_loc = clinical_oth.columns.get_loc('GLASGOW_SCORE_D1_LOW_VALUE')
    weight_loc = clinical_oth.columns.get_loc('INITIAL_WEIGHT_KG')
    height_loc = clinical_oth.columns.get_loc('HEIGHT_CM_VALUE')
    clinical_oth = clinical_oth.values

    print('Loading clinical vitals...')
    clinical_vit = pd.read_csv(datapath + 'all_sheets/CLINICAL_VITALS.csv')
    clinical_vit.sort_values(by=sort_id, inplace=True)
    temp_locs = [clinical_vit.columns.get_loc('TEMPERATURE_D1_LOW_VALUE'),
                 clinical_vit.columns.get_loc('TEMPERATURE_D1_HIGH_VALUE')]
    map_locs = [clinical_vit.columns.get_loc('ART_MEAN_D1_LOW_VALUE'),
                clinical_vit.columns.get_loc('ART_MEAN_D1_HIGH_VALUE')]
    cuff_locs = [clinical_vit.columns.get_loc('CUFF_MEAN_D1_LOW_VALUE'),
                 clinical_vit.columns.get_loc('CUFF_MEAN_D1_HIGH_VALUE')]
    hr_locs = [clinical_vit.columns.get_loc('HEART_RATE_D1_LOW_VALUE'),
               clinical_vit.columns.get_loc('HEART_RATE_D1_HIGH_VALUE')]
    clinical_vit = clinical_vit.values

    print('Loading standard labs...')
    labs1_m = pd.read_csv(datapath + 'all_sheets/LABS_SET1.csv')
    labs1_m.sort_values(by=sort_id, inplace=True)
    bili_loc = labs1_m.columns.get_loc('BILIRUBIN_D1_HIGH_VALUE')
    pltlt_loc = labs1_m.columns.get_loc('PLATELETS_D1_LOW_VALUE')
    na_locs = [labs1_m.columns.get_loc('SODIUM_D1_LOW_VALUE'),
               labs1_m.columns.get_loc('SODIUM_D1_HIGH_VALUE')]
    pk_locs = [labs1_m.columns.get_loc('POTASSIUM_D1_LOW_VALUE'),
               labs1_m.columns.get_loc('POTASSIUM_D1_HIGH_VALUE')]
    hemat_locs = [labs1_m.columns.get_loc('HEMATOCRIT_D1_LOW_VALUE'),
                  labs1_m.columns.get_loc('HEMATOCRIT_D1_HIGH_VALUE')]
    hemo_locs = [labs1_m.columns.get_loc('HEMOGLOBIN_D1_LOW_VALUE'),
                 labs1_m.columns.get_loc('HEMOGLOBIN_D1_HIGH_VALUE')]
    wbc_locs = [labs1_m.columns.get_loc('WBC_D1_LOW_VALUE'),
                labs1_m.columns.get_loc('WBC_D1_HIGH_VALUE')]
    bun_locs = [labs1_m.columns.get_loc('BUN_D0_HIGH_VALUE'),
                labs1_m.columns.get_loc('BUN_D1_HIGH_VALUE'),
                labs1_m.columns.get_loc('BUN_D2_HIGH_VALUE'),
                labs1_m.columns.get_loc('BUN_D3_HIGH_VALUE')]
    bicarb_locs = [labs1_m.columns.get_loc('CO2_D1_LOW_VALUE'),
                   labs1_m.columns.get_loc('CO2_D1_HIGH_VALUE')]
    labs1_m = labs1_m.values

    labs2_m = pd.read_csv(datapath + 'all_sheets/LABS_SET2.csv')
    labs2_m.sort_values(by=sort_id, inplace=True)
    alb_loc = labs2_m.columns.get_loc('ALBUMIN_VALUE')
    lac_loc = labs2_m.columns.get_loc('LACTATE_SYRINGE_ION_VALUE')
    labs2_m = labs2_m.values

    print('Loading medications...')
    med_m = pd.read_csv(datapath + 'all_sheets/MEDICATIONS_INDX.csv')
    med_m.sort_values(by=sort_id, inplace=True)
    med_type = med_m.columns.get_loc('MEDICATION_TYPE')
    med_name = med_m.columns.get_loc('MEDICATION_NAME')
    med_date = med_m.columns.get_loc('ORDER_ENTERED_DATE')
    med_dur = med_m.columns.get_loc('DAYS_ON_MEDICATION')
    med_m = med_m.values

    print('Loading mechanical ventilation support data...')
    organ_sup_mv = pd.read_csv(datapath + 'all_sheets/ORGANSUPP_VENT.csv')
    organ_sup_mv.sort_values(by=sort_id, inplace=True)
    mech_vent_dates = [organ_sup_mv.columns.get_loc('VENT_START_DATE'), organ_sup_mv.columns.get_loc('VENT_STOP_DATE')]
    mech_vent_days = organ_sup_mv.columns.get_loc('TOTAL_DAYS')
    organ_sup_mv = organ_sup_mv.values

    print('Loading ECMO data...')
    organ_sup_ecmo = pd.read_csv(datapath + 'all_sheets/ORGANSUPP_ECMO.csv')
    organ_sup_ecmo.sort_values(by=sort_id, inplace=True)
    ecmo_dates = [organ_sup_ecmo.columns.get_loc('ECMO_START_DATE'), organ_sup_ecmo.columns.get_loc('ECMO_STOP_DATE')]
    ecmo_days = organ_sup_ecmo.columns.get_loc('TOTAL_DAYS')
    organ_sup_ecmo = organ_sup_ecmo.values

    print('Loading IABP data...')
    organ_sup_iabp = pd.read_csv(datapath + 'all_sheets/ORGANSUPP_IABP.csv')
    organ_sup_iabp.sort_values(by=sort_id, inplace=True)
    iabp_dates = [organ_sup_iabp.columns.get_loc('IABP_START_DATE'), organ_sup_iabp.columns.get_loc('IABP_STOP_DATE')]
    iabp_days = organ_sup_iabp.columns.get_loc('TOTAL_DAYS')
    organ_sup_iabp = organ_sup_iabp.values

    print('Loading VAD data...')
    organ_sup_vad = pd.read_csv(datapath + 'all_sheets/ORGANSUPP_VAD.csv')
    organ_sup_vad.sort_values(by=sort_id, inplace=True)
    vad_dates = [organ_sup_vad.columns.get_loc('VAD_START_DATE'), organ_sup_vad.columns.get_loc('VAD_STOP_DATE')]
    vad_days = organ_sup_vad.columns.get_loc('TOTAL_DAYS')
    organ_sup_vad = organ_sup_vad.values

    print('Loading urine output data...')
    urine_m = pd.read_csv(datapath + 'all_sheets/URINE_OUTPUT.csv')
    urine_m.sort_values(by=sort_id, inplace=True)
    urine_locs = [urine_m.columns.get_loc('U0_INDWELLING_URETHRAL_CATHTER_D0_VALUE'),
                  urine_m.columns.get_loc('U0_VOIDED_ML_D0_VALUE'),
                  urine_m.columns.get_loc('U0_INDWELLING_URETHRAL_CATHTER_D1_VALUE'),
                  urine_m.columns.get_loc('U0_VOIDED_ML_D1_VALUE'),
                  urine_m.columns.get_loc('U0_INDWELLING_URETHRAL_CATHTER_D2_VALUE'),
                  urine_m.columns.get_loc('U0_VOIDED_ML_D2_VALUE')]
    urine_m = urine_m.values

    print('Loading smoking data...')
    smoke_m = pd.read_csv(datapath + 'all_sheets/SMOKING_HIS.csv')
    smoke_m.sort_values(by=sort_id, inplace=True)
    former_smoke = smoke_m.columns.get_loc('SMOKING_HISTORY_STATUS')
    current_smoke = smoke_m.columns.get_loc('SMOKING_CURRENT_STATUS')
    smoke_m = smoke_m.values

    print('Loading USRDS ESRD data...')
    usrds_esrd_m = pd.read_csv(datapath + 'all_sheets/USRDS_ESRD.csv')
    usrds_esrd_m.sort_values(by=sort_id, inplace=True)
    usrds_esrd_date_loc = usrds_esrd_m.columns.get_loc('ESRD_DATE')
    usrds_esrd_m = usrds_esrd_m.values

    print('Loading ESRD manual revision data...')
    esrd_man_rev = pd.read_csv(datapath + 'all_sheets/ESRD_MANUAL_REVISION.csv')
    esrd_man_rev.sort_values(by=sort_id, inplace=True)
    man_rev_bef = esrd_man_rev.columns.get_loc('before')
    man_rev_dur = esrd_man_rev.columns.get_loc('during')
    man_rev_rrt = esrd_man_rev.columns.get_loc('rrt_dependent')
    esrd_man_rev = esrd_man_rev.values

    return ((date_m, hosp_locs, icu_locs, adisp_loc,
             surg_m, surg_des_loc,
             diag_m, diag_loc, diag_nb_loc, icd_type_loc, icd_code_loc,
             scm_esrd_m, scm_esrd_before, scm_esrd_at, scm_esrd_during, scm_esrd_after,
             rrt_m, crrt_locs, hd_locs, pd_locs, hd_trt_loc,
             scr_all_m, scr_date_loc, scr_val_loc, scr_desc_loc,
             dem_m, sex_loc, eth_loc,
             dob_m, birth_loc,
             mort_m, mdate_loc,
             io_m, charl_m, charl_loc, elix_m, elix_loc,
             blood_m, pao2_locs, paco2_locs, ph_locs,
             clinical_oth, resp_locs, fio2_locs, gcs_loc, weight_loc, height_loc,
             clinical_vit, temp_locs, map_locs, cuff_locs, hr_locs,
             labs1_m, bili_loc, pltlt_loc, na_locs, pk_locs, hemat_locs, wbc_locs, hemo_locs, bun_locs, bicarb_locs,
             labs2_m, alb_loc, lac_loc,
             med_m, med_type, med_name, med_date, med_dur,
             organ_sup_mv, mech_vent_dates, mech_vent_days,
             organ_sup_ecmo, ecmo_dates, ecmo_days,
             organ_sup_iabp, iabp_dates, iabp_days,
             organ_sup_vad, vad_dates, vad_days,
             urine_m, urine_locs,
             smoke_m, former_smoke, current_smoke,
             usrds_esrd_m, usrds_esrd_date_loc,
             esrd_man_rev, man_rev_bef, man_rev_dur, man_rev_rrt))


def load_all_csv_dallas(datapath, sort_id='PATIENT_NUM'):
    print('Loading encounter info...')
    # Get IDs and find indices of all used metrics
    date_m = pd.read_csv(datapath + 'csv/tIndexedIcuAdmission.csv')
    date_m.sort_values(by=[sort_id, 'HSP_ADMSN_TIME'], inplace=True)
    hosp_locs = [date_m.columns.get_loc("HSP_ADMSN_TIME"), date_m.columns.get_loc("HSP_DISCH_TIME")]
    icu_locs = [date_m.columns.get_loc("ICU_ADMSN_TIME"), date_m.columns.get_loc("ICU_DISCH_TIME")]

    # adisp_loc = date_m.columns.get_loc('DISCHARGE_DISPOSITION')
    date_m = date_m.values

    print('Loading ESRD status...')
    # ESRD status
    esrd_m = pd.read_csv(datapath + 'csv/tESRDSummary.csv')
    esrd_m.sort_values(by=sort_id, inplace=True)
    esrd_bef_loc = esrd_m.columns.get_loc("BEFORE_INDEXED_ADT")
    esrd_during_loc = esrd_m.columns.get_loc("DURING_INDEXED_ADT")
    esrd_after_loc = esrd_m.columns.get_loc("AFTER_INDEXED_ADT")
    esrd_date_loc = esrd_m.columns.get_loc("EFFECTIVE_TIME")
    esrd_m = esrd_m.values

    # All SCR
    print('Loading SCr values (may take a while)...')
    scr_all_m = pd.read_csv(datapath + 'csv/all_scr_data.csv')
    scr_all_m.sort_values(by=[sort_id, 'SPECIMN_TEN_TIME'], inplace=True)
    scr_date_loc = scr_all_m.columns.get_loc('SPECIMN_TEN_TIME')
    scr_val_loc = scr_all_m.columns.get_loc('ORD_VALUE')
    scr_ip_loc = scr_all_m.columns.get_loc('IP_FLAG')
    rrt_locs = [scr_all_m.columns.get_loc('CRRT_0_1'),
                scr_all_m.columns.get_loc('HD_0_1'),
                scr_all_m.columns.get_loc('PD_0_1')]
    scr_all_m = scr_all_m.values

    # Demographics
    print('Loading demographics...')
    dem_m = pd.read_csv(datapath + 'csv/tPatients.csv')
    dem_m.sort_values(by=sort_id, inplace=True)
    sex_loc = dem_m.columns.get_loc('SEX_ID')
    eth_locs = [dem_m.columns.get_loc('RACE_WHITE'),
                dem_m.columns.get_loc('RACE_BLACK')]
    dob_loc = dem_m.columns.get_loc('DOB')
    dod_locs = [dem_m.columns.get_loc('DOD_Epic'),
                dem_m.columns.get_loc('DOD_NDRI'),
                dem_m.columns.get_loc('DMF_DEATH_DATE')]
    dem_m = dem_m.values

    # load lab values
    print('Loading laboratory values...')
    lab_m = pd.read_csv(datapath + 'csv/icu_lab_data.csv')
    lab_col = lab_m.columns.get_loc('TERM_GRP_NAME')
    lab_day = lab_m.columns.get_loc('DAY_NO')
    lab_min = lab_m.columns.get_loc('D_MIN_VAL')
    lab_max = lab_m.columns.get_loc('D_MAX_VAL')
    lab_m = lab_m.values

    # load blood gas/flow values
    print('Loading laboratory values...')
    flw_m = pd.read_csv(datapath + 'csv/icu_flw_data.csv')
    flw_col = flw_m.columns.get_loc('TERM_GRP_NAME')
    flw_day = flw_m.columns.get_loc('DAY_NO')
    flw_min = flw_m.columns.get_loc('D_MIN_VAL')
    flw_max = flw_m.columns.get_loc('D_MAX_VAL')
    flw_sum = flw_m.columns.get_loc('D_SUM_VAL')
    flw_m = flw_m.values

    print('Loading medications...')
    medications = pd.read_csv(datapath + 'csv/tMedications.csv')
    medications.sort_values(by=sort_id, inplace=True)
    amino_loc = medications.columns.get_loc('AMINOGLYCOSIDES')
    nsaid_loc = medications.columns.get_loc('NSAIDS')
    acei_loc = medications.columns.get_loc('ACEI')
    arb_loc = medications.columns.get_loc('ARB')
    press_loc = medications.columns.get_loc('PRESSOR_INOTROPE')
    medications = medications.values

    print('Loading mechanical ventilation support data...')
    organ_sup = pd.read_csv(datapath + 'csv/tAOS.csv')
    organ_sup.sort_values(by=sort_id, inplace=True)
    mech_vent_dates = [organ_sup.columns.get_loc('MV_START_DATE'), organ_sup.columns.get_loc('MV_END_DATE')]
    iabp_dates = [organ_sup.columns.get_loc('IABP_START_DATE'), organ_sup.columns.get_loc('IABP_END_DATE')]
    ecmo_dates = [organ_sup.columns.get_loc('ECMO_START_DATE'), organ_sup.columns.get_loc('ECMO_END_DATE')]
    vad_dates = [organ_sup.columns.get_loc('VAD_START_DATE'), organ_sup.columns.get_loc('VAD_END_DATE')]
    organ_sup = organ_sup.values

    print('Loading diagnosis data...')
    diag_m = pd.read_csv(datapath + 'csv/tHospitalFinalDiagnoses.csv')
    diag_m.sort_values(by=sort_id, inplace=True)
    diag_desc_loc = diag_m.columns.get_loc('DX_NAME')
    icd9_code_loc = diag_m.columns.get_loc('CURRENT_ICD9_LIST')
    icd10_code_loc = diag_m.columns.get_loc('CURRENT_ICD10_LIST')
    diag_m = diag_m.values

    print('Loading dialysis data...')
    rrt_m = pd.read_csv(datapath + 'csv/tDialysis.csv')
    rrt_m.sort_values(by=sort_id, inplace=True)
    rrt_start_loc = rrt_m.columns.get_loc('START_DATE')
    rrt_stop_loc = rrt_m.columns.get_loc('END_DATE')
    rrt_type_loc = rrt_m.columns.get_loc('DIALYSIS_TYPE')
    rrt_m = rrt_m.values
    
    print('Loading all hospitalization data...')
    hosp_m = pd.read_csv(datapath + 'csv/tHospitalization.csv')
    hosp_m.sort_values(by=sort_id, inplace=True)
    hosp_icu_locs = [hosp_m.columns.get_loc('HSP_ADMSN_TIME'), hosp_m.columns.get_loc('HSP_DISCH_TIME')]
    hosp_hosp_locs = [hosp_m.columns.get_loc('ICU_ADMSN_TIME'), hosp_m.columns.get_loc('ICU_DISCH_TIME')]
    hosp_m = hosp_m.values

    print('Loading USRDS data...')
    usrds_m = pd.read_csv(datapath + 'csv/tUSRDS_CORE_Patients.csv')
    usrds_m.sort_values(by=sort_id, inplace=True)
    usrds_mort_loc = usrds_m.columns.get_loc('DIED')
    usrds_esrd_loc = usrds_m.columns.get_loc('FIRST_SE')
    usrds_m = usrds_m.values


    return ((date_m, hosp_locs, icu_locs,
             esrd_m, esrd_bef_loc, esrd_during_loc, esrd_after_loc, esrd_date_loc,
             scr_all_m, scr_date_loc, scr_val_loc, scr_ip_loc, rrt_locs,
             dem_m, sex_loc, eth_locs, dob_loc, dod_locs,
             lab_m, lab_col, lab_day, lab_min, lab_max,
             flw_m, flw_col, flw_day, flw_min, flw_max, flw_sum,
             medications, amino_loc, nsaid_loc, acei_loc, arb_loc, press_loc,
             organ_sup, mech_vent_dates, iabp_dates, ecmo_dates, vad_dates,
             diag_m, icd9_code_loc, icd10_code_loc, diag_desc_loc,
             rrt_m, rrt_start_loc, rrt_stop_loc, rrt_type_loc,
             hosp_m, hosp_icu_locs, hosp_hosp_locs,
             usrds_m, usrds_mort_loc, usrds_esrd_loc))


