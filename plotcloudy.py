import matplotlib.pyplot as plt
import dtw
from matplotlib.backends.backend_pdf import PdfPages
def find_cluster_scr(clustervector,min_maxid,afterinterpo):
    '''
    input: clustervector--one cluster's patient ids
     output:cluster_scr---all patient scr values
    '''
    cluster_scr=[]
    valist=[]
    cluster_scr_id=[]
    for i in range(len(clustervector)):
        item=clustervector[i][1:-1]
        item1=int(item)
        cluster_scr_id.append(item1)
        index_num=min_maxid.index(item1)
        for j in range(len(afterinterpo[index_num])):
            valist.append(afterinterpo[index_num][j])
        cluster_scr.append(valist)
        valist=[]
    return cluster_scr,cluster_scr_id
#==============================
def find_cluster_kdigo(clustervector,min_maxid,covert_kid):
    '''
    purpose: find each patient id related kdigo number
    input: clustervector--one cluster patient id number, min_maxid -- patient id. covert_kid--no aki afterinterpo data with kdigo instand of scr value
    output: cluster_kdigo--the kdigo number for each patient id group them by their cluster number. cluster_kdigo_id--the patient id corsponding to the cluster_kdigo
    '''
    cluster_kdigo=[]
    valist=[]
    cluster_kdigo_id=[]
    valist1=[]
    interpo=[]
    for i in range(len(covert_kid)):
        #print(covert_kid[i])
        for j in range(len(covert_kid[i])):
            valist1.append(covert_kid[i][j])
        interpo.append(valist1)
        valist1=[]
    for i in range(len(clustervector)):
        #item = clustervector[i][1:-1]
        item = clustervector[i]
        #print("clustervector----")
        #print(clustervector)
        #print("item----")
        #print(item)
        item1=int(item)
        cluster_kdigo_id.append(item1)
        index_num = min_maxid.index(item1)
        #print(index_num)
        #print(min_maxid[index_num])
        #print(interpo[index_num])
        for j in range(len(interpo[index_num])):
            valist.append(interpo[index_num][j])
            #print(valist)
        cluster_kdigo.append(valist)
        valist=[]
    return cluster_kdigo,cluster_kdigo_id
#==================================================================
def L1(a,b):
    return abs(a-b)
def use_dtw_cluster(clusterinterpo):
    '''
    using dtw on each cluster, and dtw them with the longest one
    '''
    max_index=-1
    max_value=0
    warped=[]
    valist=[]
    valist1=[]
    for i in range (len(clusterinterpo)):
        if len(clusterinterpo[i]) > max_value:
            max_value = len(clusterinterpo[i])
            max_index = i
    for i in range(len(clusterinterpo)):
        if i != max_index:
            a,b,c,path=dtw.dtw(clusterinterpo[i],clusterinterpo[max_index],L1)
            for idx in path[0]:
                valist.append(clusterinterpo[i][int(idx)])
            valist1=[]
            warped.append(valist)
            valist=[]
        else:
            warped.append(clusterinterpo[max_index])
    return warped

#===============================================
def draw_cloudy(clusterinterpo):
    '''
    draw the plot picture for each cluster based on their kdigo number.
    '''
    fig = plt.figure()
    x=[]
    y=[]
    sub_x=[]
    sub_y=[]
    for i in range(len(clusterinterpo)):
        day=0
        for j in range(len(clusterinterpo[i])):
            day+=1
            sub_x.append(day)
        x.append(sub_x)
        sub_x=[]
    for i in range (len(clusterinterpo)):
        for j in range(len(clusterinterpo[i])):
            sub_y.append(clusterinterpo[i][j])
        y.append(sub_y)
        sub_y=[]
    for i in range(len(clusterinterpo)):
        plt.plot(x[i],y[i])
    plt.xlabel('6hour')
    plt.ylabel('scrvalue')

#===================================================
def plot_picture(warped_all):
    pp=PdfPages('kdigo_cluster_noaki.pdf')
    for i in range (len(warped_all)):
        picture=draw_cloudy(warped_all[i])
        pp.savefig(picture)
    pp.close()
