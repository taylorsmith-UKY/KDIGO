import datetime
from datetime import date
from scipy.spatial import distance
import collections
import numpy as np
#from calculate import *
#=================================================
def cover_scr_kidney(afterinterpo,min_max_inicuid,base_id,dialysis_index):
    '''
    covert all scr value into kdigo. 
    '''
    valist=[]
    covert_kid=[]
    dialyid=[]
    index_dia=[]
    val_dia=[]
    for i in range(len(dialysis_index)):
        dialyid.append(dialysis_index[i][0])
        for j in range(1,len(dialysis_index[i])):
            val_dia.append(dialysis_index[i][j])
        index_dia.append(val_dia)
        val_dia=[]
    for i in range(len(afterinterpo)):
        valist.append(min_max_inicuid[i])
        base_result=-10
        for n in range(len(base_id)):
            if min_max_inicuid[i] == base_id[n][0]:
                base_result=base_id[n][1][0]
                break
        if min_max_inicuid[i] in dialyid:
            index_dia1=dialyid.index(min_max_inicuid[i])
            for j in range(len(afterinterpo[i])):
                if j in index_dia[index_dia1]:
                    valist.append(4)
                else:
                    result=cover_kdigo(afterinterpo[i][j],base_result)
                    valist.append(result)
            covert_kid.append(valist)
            valist=[]
        else:
            for n in range(len(afterinterpo[i])):
                result=cover_kdigo(afterinterpo[i][n],base_result)
                valist.append(result)
            covert_kid.append(valist)
            valist=[]
    return covert_kid
#=======================================================================
#=============================================
def cal_days_stages(covert_kid,min_max_inicuid):
    '''
    purpose:for each patient find out how many calculate in stage 0, 1, 2, 3, 4 
    output: patient id and detail data for different stage and the total calculation they have
    '''
    valist=[]
    days_record=[]
    max_kdigo=[]
    kdigo_temp=[]#patient id and max kdigo
    for i in range(len(covert_kid)):
        valist.append(min_max_inicuid[i])
        kdigo_temp.append(min_max_inicuid[i])
        kdigo_temp.append(max(covert_kid[i]))
        max_kdigo.append(kdigo_temp)
        kdigo_temp=[]
        counter=collections.Counter(covert_kid[i])
        valist.append(counter[0])
        valist.append(counter[1])
        valist.append(counter[2])
        valist.append(counter[3])
        valist.append(counter[4])
        valist.append(len(covert_kid[i]))
        days_record.append(valist)
        valist=[]
    return days_record,max_kdigo
#==========================================
def make_percentage(covert_kid):
    '''
    purpose: transfer all patient information in to the percentage. 
    '''
    percent_result=[]
    valist=[]
    for i in range(len(covert_kid)):
        valist.append(covert_kid[i][0])
        total=covert_kid[i][len(covert_kid[i])-1]
        for j in range(1,len(covert_kid[i])-1):
            item=covert_kid[i][j]/total
            valist.append(item)
        percent_result.append(valist)
        valist=[]
    return percent_result
#=====================================
def cluster_avg_std(clustervector,days_record,id_num):
    '''
    input: clustervector---patient id in one cluster,days_record--for our case, we use the
           percentage result; id_num: patient id in order corespoinding the percentage
           result.
    output: cluster_days_stage:all percentage data in this cluster.
    purpose: use this function to put the data as cluster for later used.
    '''
    valist=[]
    cluster_days_stage=[]
    for i in range(len(clustervector)):
        #item=clustervector[i][1:-1]
        item = clustervector[i]
        item1=int(item)
        index_num=id_num.index(item1)
        for j in range(1,len(days_record[index_num])):
            valist.append(days_record[index_num][j])
        cluster_days_stage.append(valist)
        valist=[]
    return cluster_days_stage
#==============================================
def avg_max(num_list):
    average_num=cal_avg(num_list)
    std_result=np.std(num_list)
    return average_num,std_result
#======================
def cal_avg(item):
    ret=sum(item)/(len(item))
    return ret
#================================
def find_avg_std_stage(cluster_days_stage):
    v0=[]#stage 0
    v1=[]#stage 1
    v2=[]#stage 2
    v3=[]#stage 3
    v4=[]#stage 4
    #use the for loop put each cluster data into five split group
    for i in range(len(cluster_days_stage)):
        v0.append(cluster_days_stage[i][0])
        v1.append(cluster_days_stage[i][1])
        v2.append(cluster_days_stage[i][2])
        v3.append(cluster_days_stage[i][3])
        v4.append(cluster_days_stage[i][4])
    avg=[]
    std_val=[]
    avg_result,std_result=avg_max(v0)
    avg.append(avg_result)
    std_val.append(std_result)
    avg_result,std_result=avg_max(v1)    
    avg.append(avg_result)
    std_val.append(std_result)
    avg_result,std_result=avg_max(v2)   
    avg.append(avg_result)
    std_val.append(std_result)
    avg_result,std_result=avg_max(v3)
    avg.append(avg_result)
    std_val.append(std_result)
    avg_result,std_result=avg_max(v4)   
    avg.append(avg_result)
    std_val.append(std_result)
    avg_std_mix=[]
    avg_std_mix.append(avg)
    avg_std_mix.append(std_val)
    return avg_std_mix
#=================================================
def print_cluster_avg_std(cluster_avg_std,f,name_cluster):
    for i in range(len(cluster_avg_std)):
        f.write("For cluster "+str(name_cluster[i])+"\n")
        f.write("Average: "+"\n")
        for n in range(len(cluster_avg_std[i][0])):
            f.write(str(cluster_avg_std[i][0][n])+"\t")
        f.write("\n")
        f.write("Standard division:"+"\n")
        for n in range(len(cluster_avg_std[i][0])):
            f.write(str(cluster_avg_std[i][1][n])+"\t")
        f.write("\n")
    f.close()
#=======================================================
def print_cluster_percent_avgstd(cluster_avg_std,f,name_cluster,cluster_days_stage):
    for i in range(len(cluster_avg_std)):
        f.write("For cluster "+str(name_cluster[i])+"\n")
        #f.write("Patient id"+"\t"+"stage 0"+"\t"+"stage1"+"\t"+"stage2"+"\t"+"stage3"+"\t"+"stage 4"+"\n")
        #for j in range(len(cluster_days_stage[i])):
            #f.write(str(cluster_days_stage[i][j])+"\t")
        #f.write("\n")
        f.write("Average: "+"\n")
        for n in range(len(cluster_avg_std[i][0])):
            f.write(str(cluster_avg_std[i][0][n])+"\t")
        f.write("\n")
        f.write("Standard division: "+"\n")
        for w in range(len(cluster_avg_std[i][0])):
            f.write(str(cluster_avg_std[i][1][w])+"\t")
        f.write("\n")
    f.close()
#==========================================================
def print_avg_std_linepair(cluster_avg_std,f,cluster_days_stage):
    f.write("Average:"+"\n")
    for i in range(len(cluster_avg_std)):
        for n in range(len(cluster_avg_std[i][0])):
            f.write(str(cluster_avg_std[i][0][n])+"\t")
        f.write("\n")
    f.write("std:" +"\n")
    for i in range(len(cluster_avg_std)):
        for w in range(len(cluster_avg_std[i][0])):
            f.write(str(cluster_avg_std[i][1][w])+"\t")
        f.write("\n")
    f.close()
#=================================================
def cluster_record_day(cluster_id_all,day_records,min_max_inicuid,max_scr_kdigo,name_cluster,allid):
    valist=[]
    result=[]
    cluster_all=[]
    for i in range(len(cluster_id_all)):
        for j in range(len(cluster_id_all[i])):
            index=min_max_inicuid.index(cluster_id_all[i][j])
            for n in range(len(day_records[index])):
                valist.append(day_records[index][n])
            for w in range(len(max_scr_kdigo)):
                if day_records[index][0] == max_scr_kdigo[w][0]:
                    valist.append(max_scr_kdigo[w][1])
                    break
            valist.append(name_cluster[i])
            result.append(valist)
            valist=[]
        cluster_all.append(result)
        result=[]
    return cluster_all
#============================================================
def print_cluster_record_day(record_dayinorder,f):
    f.write("Patient ID"+"\t"+"Stage 0"+"\t"+"stage 1"+"\t"+"stage2"+"\t"+"stage3"+"\t"+"stage4"+"\t"+"total day"+"\t"+"max_kdigo"+"\t"+"number_cluster"+"\n")
    for i in range(len(record_dayinorder)):
        for j in range(len(record_dayinorder[i])):
            for n in range(len(record_dayinorder[i][j])):
                f.write(str(record_dayinorder[i][j][n])+"\t")
            f.write("\n")
    f.close()

