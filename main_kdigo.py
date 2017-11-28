import kdigo_funcs as kf
import numpy as np
import os

#------------------------------- PARAMETERS ----------------------------------#
inFile = "/Users/taylorsmith/Google Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/DATA/KDIGO_full.xlsx"
outPath = "result/"
sort_id = 'STUDY_PATIENT_ID'
sort_id_date = 'SCR_ENTERED'
incl_esrd = False
t_analyze = 'ICU'
timescale = 6       #in hours

#-----------------------------------------------------------------------------#
def main():
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    ############ Get Indices for All Used Values ################
    print('Loading encounter info...')
    #Get IDs and find indices of all used metrics
    date_m = kf.get_mat(inFile,'ADMISSION_INDX',[sort_id])
    id_loc=date_m.columns.get_loc("STUDY_PATIENT_ID")
    hosp_locs=[date_m.columns.get_loc("HOSP_ADMIT_DATE"),date_m.columns.get_loc("HOSP_DISCHARGE_DATE")]
    icu_locs=[date_m.columns.get_loc("ICU_ADMIT_DATE"),date_m.columns.get_loc("ICU_DISCHARGE_DATE")]
    date_m=date_m.as_matrix()

    ### GET SURGERY SHEET AND LOCATION
    print('Loading surgery information...')
    surg_m = kf.get_mat(inFile,'SURGERY_INDX',[sort_id])
    surg_loc = surg_m.columns.get_loc("SURGERY_CATEGORY")
    surg_des_loc = surg_m.columns.get_loc("SURGERY_DESCRIPTION")
    surg_m = surg_m.as_matrix()

    ### GET DIAGNOSIS SHEET AND LOCATION
    print('Loading diagnosis information...')
    dx_m = kf.get_mat(inFile,'DIAGNOSIS',[sort_id])
    dx_loc = dx_m.columns.get_loc("DIAGNOSIS_DESC")
    dx_m = dx_m.as_matrix()

    print('Loading ESRD status...')
    #ESRD status
    esrd_m = kf.get_mat(inFile,'ESRD_STATUS',[sort_id])
    esrd_locs = [esrd_m.columns.get_loc("AT_ADMISSION_INDICATOR"),esrd_m.columns.get_loc("DURING_INDEXED_INDICATOR"),esrd_m.columns.get_loc("BEFORE_INDEXED_INDICATOR")]
    esrd_m = esrd_m.as_matrix()

    #Dialysis dates
    print('Loading dialysis dates...')
    dia_m = kf.get_mat(inFile,'RENAL_REPLACE_THERAPY',[sort_id])
    crrt_locs = [dia_m.columns.get_loc('CRRT_START_DATE'),dia_m.columns.get_loc('CRRT_STOP_DATE')]
    hd_locs = [dia_m.columns.get_loc('HD_START_DATE'),dia_m.columns.get_loc('HD_STOP_DATE')]
    pd_locs = [dia_m.columns.get_loc('PD_START_DATE'),dia_m.columns.get_loc('PD_STOP_DATE')]
    dia_m = dia_m.as_matrix()

    #All SCR
    print('Loading SCr values (may take a while)...')
    scr_all_m = kf.get_mat(inFile,'SCR_ALL_VALUES',[sort_id,sort_id_date])
    scr_date_loc = scr_all_m.columns.get_loc('SCR_ENTERED')
    scr_val_loc = scr_all_m.columns.get_loc('SCR_VALUE')
    scr_all_m = scr_all_m.as_matrix()

    #Baselines
    print('Loading baselines...')
    bsln_m = kf.get_mat(inFile,'BASELINE_SCR',[sort_id])
    bsln_scr_loc = bsln_m.columns.get_loc('BASELINE_VALUE')
    bsln_date_loc = bsln_m.columns.get_loc('BASELINE_DATE')
    bsln_m = bsln_m.as_matrix()

    #Demographics
    dem_m = kf.get_mat(inFile,'DEMOGRAPHICS_INDX',[sort_id])
    sex_loc = dem_m.columns.get_loc('GENDER')
    eth_loc = dem_m.columns.get_loc('RACE')
    dem_m = dem_m.as_matrix()

    #DOB
    dob_m = kf.get_mat(inFile,'DOB',[sort_id])
    birth_loc = dob_m.columns.get_loc("DOB")
    dob_m = dob_m.as_matrix()
    ###### Get masks for ESRD, dialysis, etc.

    #Get mask inidicating which points are during dialysis
    dia_mask=kf.get_dialysis_mask(scr_all_m,scr_date_loc,dia_m,crrt_locs,hd_locs,pd_locs)

    #Get mask indicating whether each point was in hospital or ICU
    t_mask=kf.get_t_mask(scr_all_m,scr_date_loc,scr_val_loc,date_m,hosp_locs,icu_locs)

    #Get mask for all patients with ESRD
    #esrd_mask=kf.get_esrd_mask(scr_all_m,id_loc,esrd_m,esrd_locs)

    #Get mask for the desired data
    mask=np.zeros(len(scr_all_m))
    for i in range(len(scr_all_m)):
        if t_analyze == 'ICU':
            if t_mask[i] == 2:
                if dia_mask[i]:
                    mask[i]=-1
                else:
                    mask[i]=1
        elif t_analyze == 'HOSP':
            if t_mask[i] >= 1:
                if dia_mask[i]:
                    mask[i]=-1
                else:
                    mask[i]=1


    #Extract patients into separate list elements and get baselines
    ids,scr,dates,masks,dmasks,baselines,bsln_gfr = kf.get_patients(scr_all_m,scr_val_loc,scr_date_loc,\
                                                        mask,dia_mask,\
                                                        dx_m,dx_loc,\
                                                        esrd_m,esrd_locs,\
                                                        bsln_m,bsln_scr_loc,bsln_date_loc,\
                                                        date_m,id_loc,icu_locs,\
                                                        surg_m,surg_loc,surg_des_loc,\
                                                        dem_m,sex_loc,eth_loc,\
                                                        dob_m,birth_loc)
    kf.arr2csv(outPath+'scr_ICU_raw.csv',scr,ids)
    kf.str2csv(outPath+'dates_ICU.csv',dates,ids)
    kf.arr2csv(outPath+'masks_ICU.csv',masks,ids)
    kf.arr2csv(outPath+'dialysis_ICU.csv',dmasks,ids)
    kf.arr2csv(outPath+'baselines.csv',baselines,ids)
    kf.arr2csv(outPath+'baseline_gfr.csv',bsln_gfr,ids)

    #Interpolate missing values
    post_interpo,dmasks_interp=kf.linear_interpo(scr,ids,dates,masks,dmasks,timescale)

    #Get SCr Distance Matrix
    #scr_dm=pairwise_dtw_dist(post_interpo)

    #np.savetxt(outPath+'scr_dm.csv',np.hstack((row_lbls,scr_dm)),delimiter=',',header=cols,fmt='%s')

    #Convert SCr to KDIGO
    kdigo = kf.scr2kdigo(post_interpo,baselines,dmasks_interp)
    kf.arr2csv(outPath+'kdigo_ex.csv',kdigo,ids)
    #Get KDIGO Distance Matrix
    kf.pairwise_dtw_dist(kdigo,'kdigo_ex_dm.csv','kdigo_ex_dtwlog.csv')

main()