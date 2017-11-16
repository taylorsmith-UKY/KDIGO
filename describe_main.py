import pandas as pd
from kdigo_funcs import * 
import os
import h5py
from describe import *
from plotcloudy import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#------------------------------- PARAMETERS ----------------------------------#
inFile = "KDIGO_full.xlsx"
outPath = "result/"
sort_id = 'STUDY_PATIENT_ID'
sort_id_date = 'SCR_ENTERED'
incl_esrd = False
t_analyze = 'ICU'
timescale = 6       #in hours
file_cluster = 'cluster_noaki.txt'
data_file = 'kdigo_data.hdf5'
result_folder="result_cluster/"
#-----------------------------------------------------------------------------#
def main():
    outPath="result/"
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    #f = h5py.File(data_file,'w')
    ############ Get Indices for All Used Values ################
    print('Loading encounter info...')
    #Get IDs and find indices of all used metrics
    date_m = get_mat(inFile,'ADMISSION_INDX',[sort_id])
    id_loc=date_m.columns.get_loc("STUDY_PATIENT_ID")
    hosp_locs=[date_m.columns.get_loc("HOSP_ADMIT_DATE"),date_m.columns.get_loc("HOSP_DISCHARGE_DATE")]
    icu_locs=[date_m.columns.get_loc("ICU_ADMIT_DATE"),date_m.columns.get_loc("ICU_DISCHARGE_DATE")]
    date_m=date_m.as_matrix()

    print('Loading ESRD status...')
    #ESRD status
    esrd_m = get_mat(inFile,'ESRD_STATUS',[sort_id])
    esrd_locs = [esrd_m.columns.get_loc("AT_ADMISSION_INDICATOR"),esrd_m.columns.get_loc("DURING_INDEXED_INDICATOR"),esrd_m.columns.get_loc("BEFORE_INDEXED_INDICATOR")]
    esrd_m = esrd_m.as_matrix()

    #Dialysis dates
    print('Loading dialysis dates...')
    dia_m = get_mat(inFile,'RENAL_REPLACE_THERAPY',[sort_id])
    crrt_locs = [dia_m.columns.get_loc('CRRT_START_DATE'),dia_m.columns.get_loc('CRRT_STOP_DATE')]
    hd_locs = [dia_m.columns.get_loc('HD_START_DATE'),dia_m.columns.get_loc('HD_STOP_DATE')]
    pd_locs = [dia_m.columns.get_loc('PD_START_DATE'),dia_m.columns.get_loc('PD_STOP_DATE')]
    dia_m = dia_m.as_matrix()

    #All SCR
    print('Loading SCr values (may take a while)...')
    scr_all_m = get_mat(inFile,'SCR_ALL_VALUES',[sort_id,sort_id_date])
    scr_date_loc = scr_all_m.columns.get_loc('SCR_ENTERED')
    scr_val_loc = scr_all_m.columns.get_loc('SCR_VALUE')
    scr_all_m = scr_all_m.as_matrix()

    baseline_m = get_mat(inFile,'BASELINE_SCR',[sort_id])
    baseline_scr_loc = baseline_m.columns.get_loc('BASELINE_VALUE')
    baseline_m = baseline_m.as_matrix()

    ''' Will be used for post-analysis
    #Post discharge SCR
    scr_aft1yr_m = get_mat(inFile,'SCR_AFTINDX_1YR',sort_id)

    #Outcomes
    outcms_m = get_mat(inFile,'OUTCOMES',sort_id)

    #Demographics
    dem_m = get_mat(inFile,'DEMOGRAPHICS_INDX',sort_id)
    '''

    ###### Get masks for ESRD, dialysis, etc.

    #Get mask inidicating which points are during dialysis
    dia_mask=get_dialysis_mask(scr_all_m,scr_date_loc,dia_m,crrt_locs,hd_locs,pd_locs)

    #Get mask indicating whether each point was in hospital or ICU
    t_mask=get_t_mask(scr_all_m,scr_date_loc,date_m,hosp_locs,icu_locs)

    #Get mask for all patients with ESRD
    esrd_mask=get_esrd_mask(scr_all_m,id_loc,esrd_m,esrd_locs)

    #Get mask for the desired data
    mask=np.zeros(len(scr_all_m))
    for i in range(len(scr_all_m)):
        if t_analyze == 'ICU':
            if not incl_esrd:
                if esrd_mask[i]:
                    continue
            if t_mask[i] == 2:
                if dia_mask[i]:
                    mask[i]=-1
                else:
                    mask[i]=1
        elif t_analyze == 'HOSP':
            if not incl_esrd:
                if esrd_mask[i]:
                    continue
            if t_mask[i] == 1:
                if dia_mask[i]:
                    mask[i]=-1
                else:
                    mask[i]=1

    #Extract patients into separate list elements and get baselines
    ids,scr,dates,masks,dmasks,baselines = get_patients(scr_all_m,scr_val_loc,scr_date_loc,mask,dia_mask,incl_esrd,baseline_m,baseline_scr_loc,date_m,id_loc,icu_locs)

    row_lbls = np.array(ids, dtype='|S5')[:, np.newaxis]
    cols = ','+str(ids)[1:-1]

    #Interpolate missing values
    post_interpo,dmasks_interp=linear_interpo(scr,ids,dates,masks,dmasks,timescale)

    #Convert SCr to KDIGO
    kdigo = scr2kdigo(post_interpo,baselines,dmasks_interp)
    arr2csv(outPath+'kdigo.csv',kdigo,ids)
    #count how many record for each patient in each stages
    #columns name: patient id;stage 0;stage1;stage 2;stage 3;stage4;total record
    stage_record,max_kdigo = cal_days_stages(kdigo,ids)
    p=open("stage_record.txt","w")
    print_file(stage_record,p)
    #transfer the stage record into percentage eg. total # of stage0 record / total record per patient
    percent_result = make_percentage(stage_record)
    
    #read the cluster value
    cluster_id=[]
    line_cluster=[]
    p = open(file_cluster,'r')
    for line in p:
        cluster_id.append(line.rstrip().split('\t'))
    cluster_id.pop(0)
    temp_clusters = -1
    clusters=[]
    name_cluster=[]
    for i in range (len(cluster_id)):
        if cluster_id[i][1] == str(temp_clusters):
            line_cluster.append(cluster_id[i][0])
        else:
            name_cluster.append(cluster_id[i][1])
            if len(line_cluster) !=0:
                clusters.append(line_cluster)
                line_cluster=[]
            line_cluster.append(cluster_id[i][0])
            temp_clusters=cluster_id[i][1]
    if len(line_cluster) !=0:
        clusters.append(line_cluster)
    #================================================================================
    cluster_avg_std_all=[]
    cluster_id_all=[]
    cluster_kdigo_all=[]
    for i in range(len(clusters)):
        #print("clusters[i]-----------------")
        #print(clusters[i])
        cluster_days_stage=cluster_avg_std(clusters[i],percent_result,ids)#one cluster
        avg_std_mix=find_avg_std_stage(cluster_days_stage)#one cluster
        cluster_avg_std_all.append(avg_std_mix)#store all clusters with avg and std
        cluster_kdigo,cluster_kdigo_id=find_cluster_kdigo(clusters[i],ids,kdigo)
        cluster_id_all.append(cluster_kdigo_id)
        cluster_kdigo_all.append(cluster_kdigo)
    p=open(result_folder+"cluster_kdigo_all.txt","w")
    print_file(cluster_kdigo_all,p)
    p=open(result_folder+"cluster_kdigo_id.txt","w")
    print_file(cluster_kdigo_all,p)
    p=open(result_folder+"avg_std_linepair.txt","w")
    print_avg_std_linepair(cluster_avg_std_all,p,cluster_days_stage)
    p=open(result_folder+"cluster_days_stage.txt","w")
    print_file(cluster_days_stage,p)
    #======================================================================
    warped_all=[]
    valist=[]
    for i in range(len(clusters)):
        print("kdigo_all----")
        print(cluster_kdigo_all[i])
        warped=use_dtw_cluster(cluster_kdigo_all[i])
        warped_all.append(warped)
    p=open(result_folder+"warpedall.txt","w")
    print_file(warped_all,p)
    #================================================
    record_dayinorder=cluster_record_day(cluster_id_all,stage_record,ids,max_kdigo,name_cluster,ids)
    p=open(result_folder+"cluster_record_day.txt","w")
    print_cluster_record_day(record_dayinorder,p)
    #print(record_dayinorder[0])
    #================================================
    ######################work over here
    clusters_max_kdigo=[]
    valist1=[]
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            item = clusters[i][j]
            item1=int(item)
            for n in range(len(max_kdigo)):
                if item1 == max_kdigo[n][0]:
                    valist1.append(int(max_kdigo[n][1]))
                    break
        clusters_max_kdigo.append(valist1)
        valist1=[]
    avg_max_result=[]
    valist=[]
    for i in range(len(clusters_max_kdigo)):
        average_store,stand_de = avg_max(clusters_max_kdigo[i])
        valist.append(average_store)
        valist.append(stand_de)
        avg_max_result.append(valist)
        valist=[]
    f=open(result_folder+"avg_max.txt","w")
    f.write("Average"+"\t"+"standard dervation"+"\n")
    print_file(avg_max_result,f)
    #===========================================
    #print(len(warped_all))
    #for i in range (len(warped_all)):
        #plt.figure(i)        
        #draw_cloudy(warped_all[i])
        #plt.show()
    plot_picture(warped_all)
    
def get_mat(fname,page_name,sort_id):
    return pd.read_excel(fname,page_name).sort_values(sort_id)

main()
