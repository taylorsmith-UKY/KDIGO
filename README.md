# KDIGO

Source code for KDIGO study on AKI trajectory analysis

Files:
kdigo_funcs.py 		- Definitions of most functions used in the full analysis
main_kdigo.py 		- Driver program designed specifically for the KDIGO
					  dataset. Takes SCr values and additional data to provide
					  a distance matrix for use in subsequent clustering.

Dependencies:
	Python 2.7.x
	NumPy
	SciPy
	Pandas
	dtw


Description:
	The program performs the tasks below in order, where the function
definitions are primarily provided in `kdigo_funcs.py`. The primary input
is a single 1-D vector of SCr values, with associated dates, and each assigned
to a patient.
	1) Extract raw data as matrices from excel file (highly specific to dataset)
	2) Generate masks for the SCr vector corresponding to:
		a) records taken during dialysis
		b) whether each record was outpatient, in the hospital, and/or in ICU
	3) Extract, hospital, ICU, or outpatient (user defined parameter) data,
	   separated by patient, using masks from above
		a) SCr values
		b) Dates
		c) Individual dialysis masks
		e) Exclude patients based on exclusion criteria 
			(see exclusion_criteria.txt)
	4) Discretize the data into a user defined time resolution (in hours) 
	   and interpolate missing values
	   	a) Also interpolate across points recorded during dialysis and for
	   	   the 2 days following the end of dialysis
	5) Convert from raw SCr values to KDIGO score (reference to be provided)
	6) Perform pairwise dynamic time warping (DTW) pair-wise on the KDIGO
	   vectors for each patient and calculate the bray-curtis distance
	   between the curves.
	7) Results in a condensed distance matrix (see the pdist function in MATLAB
	   for reference) to be used for clustering.

Usage:
	cd to KDIGO folder
	verify all paths and parameters are correct at beginning of main_kdigo.py
	from the shell, python main_kdigo.py
	results will be in the sub-folder 'result/'