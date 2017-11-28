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
	The main program takes a single Microsoft Excel file with multiple sheets as
	input. See the example `kdigo_test.xlsx` linked below. The following is a summary
	of the information and sheets required:
		1) ADMISSION_INDX
			Hospital and ICU dates of admission/discharge
			Discharge disposition
		2) DEMOGRAPHICS_INDX
			Sex and ethnicity
		3) DOB
		4) DIAGNOSIS
			In particular, information about kidney transplants
		5) SURGERY_INDX
			Same as number 4
		6) RENAL_REPLACE_THERAPY
			Dates and types of dialysis treatment
		7) ESRD_STATUS
			Note whether patient has end stage renal disease before,
			at, during, or after the indexed admission
		8) ORGANSUPP_VENT
			Dates for mechanical ventillation support (not yet used)
		9) BASELINE_SCR
			Note: function provided for determining baselines if not provided
		10) SCR_ALL_VALUES
			All SCr values and dates for each patient, all stored in a single
			column. The program will separate by patient.
		11) SCR_AFTINDX_1YR
			SCr values up to 1 year post-discharge (for those available; not yet used)
		12) OUTCOMES
			Deceased dates (not yet used)
			
	Before running the main application, create a new base directory and update the PARAMETERS section
	of `main_kdigo.py` accordingly. Within the base folder there should be a `DATA` folder, with the Excel file inside.
	With the data in the correct location and the function parameters updated accordingly, execute the main application
	(e.g. python main_kdigo.py). This will extract ALL data for patients during the specified time period, while excluding
	any patients who meet any of the exclusion criteria. For each of the corresponding files data is extracted with
	each line representing a single patient vector.:
		BASE-FOLDER/DATA/(ICU)
			raw SCr values (e.g. in ICU)
			masks indicating missing values and dialysis periods
			corresponding dates
			baselines (incl. baseline GFR)
			discharge disposition
	
	Once the raw data has been extracted, the raw SCr values are interpolated such that there are no missing values and
	then the SCr values are converted to KDIGO scores.
		BASE-FOLDER/DATA/(ICU)
			interpolated SCr values
			final KDIGO scores
			
	Finally, the KDIGO scores are used for pair-wise dynamic time-warping and bray-curtis distance calculation. The
	distances are written in a condensed form (i.e. no redundancy). For relatively small subsets, the full, square
	distance matrix is calculated.
	
	Finally, once the distance matrix is obtained, clustering and visualization can be performed by running the R script.
	The results from clustering can then be used with 'get_stats.py' to get comparative statistics to visualize with the
	distance matrix heatmap and clustering dendrogram.

	
	https://drive.google.com/file/d/1WAiTB3zpTIzvdO4FsxpQ4sFaJY_Uu2Yw/view?usp=sharing
