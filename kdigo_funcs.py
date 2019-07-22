from __future__ import division
import datetime
import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
import os
from utility_funcs import get_date, get_array_dates, arr2csv, cartesian, load_csv


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
    scr_m.sort_values(by=['STUDY_PATIENT_ID', 'SCR_ENTERED'], inplace=True)
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
            if hasattr(sel, '__len__'):
                idx = np.where(mask == sel[0])[0]
                for tsel in sel[1:]:
                    idx = np.union1d(idx, np.where(mask == tsel)[0])
            else:
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
    scr_m.sort_values(by=['STUDY_PATIENT_ID', 'SCR_ENTERED'], inplace=True)
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

        # Check scm flags
        esrd_idx = np.where(scm_esrd_m['STUDY_PATIENT_ID'].values == tid)[0]
        scm = 0
        if len(esrd_idx) > 0:
            esrd_idx = esrd_idx[0]
            if scm_esrd_m['BEFORE_INDEXED_INDICATOR'][esrd_idx] == 'Y' or scm_esrd_m['AT_ADMISSION_INDICATOR'][esrd_idx] == 'Y':
                scm = 1
                # exclusions[i][ESRD] = 1
        # Check USRDS
        esrd_idx = np.where(usrds_esrd_m['STUDY_PATIENT_ID'].values == tid)[0]
        usrds = 0
        if len(esrd_idx) > 0:
            esrd_idx = esrd_idx[0]
            tdate = get_date(str(usrds_esrd_m['ESRD_DATE'][esrd_idx]))
            if tdate != 'nan':
                if tdate < icu_admit:  # Before admit
                    exclusions[i][ESRD] = 1
                    usrds = 1
                elif icu_admit.toordinal() == tdate.toordinal():  # Same day as admit
                    exclusions[i][ESRD] = 1
                    usrds = 1
        if scm != usrds:
            # Check manual revision
            esrd_idx = np.where(esrd_man_rev['STUDY_PATIENT_ID'].values == tid)[0]
            if len(esrd_idx) > 0:
                esrd_idx = esrd_idx[0]
                if esrd_man_rev['before'][esrd_idx]:
                    exclusions[i][ESRD] = 1
        else:
            exclusions[i][ESRD] = scm

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
            interp_scr, interp_days, interp_mask, interp_dia = interpolate_scr(tscrs, tdates, tdmask, scale)

        interp_scr_all.append(interp_scr)
        interp_days_all.append(interp_days)
        interp_rrt_all.append(interp_dia)
        interp_masks_all.append(interp_mask)
        out_ids.append(tid)
    return interp_scr_all, interp_rrt_all, interp_days_all, interp_masks_all


def interpolate_scr(scrs, dates, dmasks, scale):
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
    start_bin_time = datetime.datetime(start_date.year, start_date.month, start_date.day, hour=scale*start_bin)

    end_date = get_date(dates[-1])

    tot_hrs = (end_date - start_bin_time).total_seconds() / (60 * 60)
    nbins = int(np.floor((tot_hrs / scale)) + 1)

    trueVals = {}

    iscr = np.zeros(nbins)
    idia = np.zeros(nbins, dtype=bool)
    imask = np.zeros(nbins, dtype=bool)

    # Assign each individual record to the appropriate bin
    for i in range(len(scrs)):
        current_date = get_date(dates[i])
        hrs = (current_date - start_bin_time).total_seconds() / (60 * 60)
        idx = int(np.floor((hrs / scale)))
        # idx = (current_date - start_bin_time).total_seconds() / (60 * 60)
        # current_bin = int(np.floor(current_date.hour / scale))
        # ndays = (current_date.toordinal() - start_date.toordinal())
        # idx = (ndays * bins_per_day) + (current_bin - start_bin)
        try:
            trueVals[idx] = max(trueVals[idx], scrs[i])
        except KeyError:
            trueVals[idx] = scrs[i]

        # Indicate that this bin contains an real-world record and is not interpolated
        imask[idx] = 1

        # If patient was on dialysis for this record, indicate dialysis for the subsequent 48 hours (or rest of record,
        # whichever is shorter.
        bins_left = nbins - idx
        if dmasks[i]:
            if bins_left <= 2 * bins_per_day:
                idia[idx:] = np.repeat(1, bins_left)
            else:
                idia[idx:idx + (2 * bins_per_day)] = np.repeat(1, 2 * bins_per_day)

    if nbins > 1:
        # Build the corresponding vector indicating to which day each point belongs (including for those yet to be imputed)
        interpolator = interp1d(sorted(trueVals), [trueVals[x] for x in sorted(trueVals)])
        for i in range(nbins):
            iscr[i] = interpolator(i)

        idays = list(
            np.concatenate([[d for _ in range(bins_per_day)] for d in range(int(np.ceil(nbins / bins_per_day)) + 1)]))

        for i in range(start_bin):
            _ = idays.pop(0)
        idays = np.array(idays, dtype=int)[:nbins]
    else:
        iscr = scrs
        if len(scrs) > 1:
            iscr = [np.mean(scrs)]
            idia = [max(dmasks)]
            idays = [0]
        elif len(scrs) == 1:
            iscr = scrs
            idays = [0]
            imask = [1]
            idia = dmasks
        else:
            idays = []
            imask = []
            idia = []
    assert len(iscr) == len(idays) == len(imask) == len(idia)
    return iscr, idays, imask, idia


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


def rolling_average(data, masks=[], times=None, window=2, step=1, maxDiff=24, debug=False):
    out = []
    time_out = []
    omasks = [[] for _ in range(len(masks))]
    for i, vec in enumerate(data):
        if times is not None:
            t = times[i]
            new_time = []
            assert len(t) == len(vec)
        tmasks = [[] for _ in range(len(masks))]
        if len(vec) <= window:
            out.append(vec)
            for j in range(len(masks)):
                omasks[j].append(masks[j][i])
            if times is not None:
                time_out.append(t)
            continue
        l = int((np.ceil(len(vec) / step) - (window - step)))
        nvec = []
        for ni, oi in enumerate(range(0, len(vec), step)):
            tt = datetime.timedelta(0)
            ct = 0
            for j in range(1, min(window, len(t) - oi)):
                if (t[oi + j] - t[oi]).total_seconds() / (60 * 60) > maxDiff:
                    continue
                tt += (t[oi + j] - t[oi])
                ct += 1
            tt /= (ct + 1)
            new_time.append(t[oi] + tt)
            for j in range(len(masks)):
                tmasks[j].append(max(masks[j][i][oi:oi+ct+1]))
            try:
                nvec.append(np.nanmean(vec[oi:oi+ct+1]))
                if np.isnan(nvec[ni]) and debug:
                    print('Length of vec/t: %d/%d' % (len(vec), len(t)))
                    print('Original start index + (# points): %d + (%d) = %d' % (oi, ct, oi + ct - 1))
                    print(vec[oi:oi+ct])
                    print(nvec[ni])
                    raise ValueError("NaN encountered in average.")
            except IndexError:
                if debug:
                    print('Length of vec/t: %d/%d' % (len(vec), len(t)))
                    print('Original start index + (# points): %d + (%d) = %d' % (oi,  ct, oi + ct))
                    print('Length of new vector: %d' % len(nvec))
                    print('New index: %d' % ni)
                raise RuntimeError("Index error when performing rolling average.")
        out.append(np.array(nvec))
        for j in range(len(masks)):
            omasks[j].append(tmasks[j])
        if times is not None:
            time_out.append(new_time)
    if len(omasks) == 0:
        if times is None:
            return out
        else:
            return out, time_out
    else:
        if times is None:
            return out, omasks
        else:
            return out, omasks, time_out
