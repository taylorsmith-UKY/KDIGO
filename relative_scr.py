import numpy as np

#-------------------------------------
//the reason I copy arr2csv function from kdigo_funcs because if can not run on my 
//computer.
def arr2csv(fname,inds,ids,fmt='%f',header=False):
    outFile=open(fname,'w')
    if header:
        outFile.write('id')
        for idx in ids:
            outFile.write(',%d' % (idx))
        outFile.write('\n')
    try:
        for i in range(len(inds)):
            outFile.write('%d' % (ids[i]))
            for j in range(len(inds[i])):
                outFile.write(','+fmt % (inds[i][j]))
            outFile.write('\n')
        outFile.close()
    except:
        for i in range(len(inds)):
            outFile.write('%d' % (ids[i]))
            outFile.write(','+fmt % (inds[i])+'\n')
        outFile.close()
#-------------------------------------
//remove one paramater and one line called "if int(l.split(',')[0] in ids
//from the load_csv in kdigo_funcs.py.
def load_csv_file(fname,dt=float):
    res=[]
    rid=[]
    f = open(fname,'r')
    for line in f:
        l = line.rstrip()
        res.append(np.array(l.split(',')[1:],dtype=dt))
        rid.append(int(l.split(',')[0]))
    return rid,res
#------------------------------------------------
#load scr data and baseline value data
scr_id,scr_vals = load_csv_file('scr_interp.csv', dt = float)
base_id,base_vals = load_csv_file('baselines.csv', dt = float)
scr_id = np.array(scr_id,dtype=int)
base_id = np.array(base_id,dtype = int)
base_id_lst=[]
for i in range(len(base_id)):
    base_id_lst.append(base_id[i])
#
perc_change=[]
abs_change=[]
temp_perc=[]
temp_abs=[]
for i in range(len(scr_vals)):
    index = base_id_lst.index(scr_id[i])
    base_value=base_vals[index]
    for j in range(len(scr_vals[i])):
        diff = scr_vals[i][j] - base_value
        temp_perc.append(diff/scr_vals[i][j])
        temp_abs.append(abs(diff))
    perc_change.append(temp_perc)
    abs_change.append(temp_abs)
    temp_perc =[]
    temp_abs = []
arr2csv('percent_change.csv',perc_change,scr_id)
arr2csv('abs_change.csv',abs_change,scr_id)
