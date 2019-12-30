#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sstream>
#include <limits>
#include <stdlib.h>

#define MAXLEN 512
#define v true
using namespace std;

double distanceWrapper(vector<int> X, vector<int> Y, double *coords, bool popCoords, string dfunc);
double bcDist(vector<int> X, vector<int> Y, double *coords, bool popCoords);
double normEuclidean(vector<int> X, vector<int> Y, double *coords, bool popCoords);
double normCityblock(vector<int> X, vector<int> Y, double *coords, bool popCoords);

int main(int argc, char **argv) {
    vector<vector<int> > seqs;
    vector<double> tweights;
    vector<int> ids;
    vector<double> dist;
    vector<double> xext;
    vector<double> yext;
    vector<vector<int> > alignments;
    vector<vector<int> > pairs;
    vector<int> pair(2);
    long idx1, idx2, nIterations, number;
    double fnum;
    stringstream ss;
    stringstream ss2;
    string token;
    string line;
    string pairFileName;
    char filename[80] = {"\0"};
    char folderName[256] = {"\0"};
    char basePath[256] = {"\0"};
    char distFileName[256] = {"\0"};
    char dist_tag[80] = {"\0"};
    char dtw_tag[80] = {"\0"};
    char subFolderName[80] = {"\0"};
    char tempStr[256] = {"\0"};
    char inputFilenameStem[80] = {"\0"};
    char sequenceFilename[80] = {"\0"};
    char weightFilename[80] = {"\0"};
    vector<int> seq;
    struct stat info;
    char *delim = ",";

    if (argc < 4) {
        cout << "Usage:" << endl;
        cout << argv[0] << " seq_filename weight_filename output_dir (-opts)" << endl;
        return 1;
    }

    bool popDist = false;
    bool verbose = false;
    bool getDfunc = false;
    string dfunc = "braycurtis";
    for (int i = 4; i < argc; i++) {
        ss.clear();
        if (strcmp("-popDist", argv[i]) == 0) {
            if (verbose)
                cout << "Using population bray-curtis distance." << endl;
            popDist = true;
        }
        else if (strcmp("-dfunc", argv[i]) == 0) {
            if (verbose)
                cout << "Using " << argv[i+1] << " distance." << endl;
            getDfunc = true;
        }
        else if (strcmp("-v", argv[i]) == 0) {
            verbose = true;
        }
        else if (strcmp("-delim", argv[i]) == 0) {
            delim = argv[i+1];
        }
        else {
            if (getDfunc) {
                ss << argv[i];
                ss >> dfunc;
                tempStr[0] = '\0';
                strcat(dist_tag, tempStr);
                strcat(dist_tag, dfunc.c_str());
//                            i++;
                getDfunc = false;
            }
        }
    }

    // Construct tag for Distance
    if (popDist)
        strcat(dist_tag, "_popCoords");
    else
        strcat(dist_tag, "_absCoords");

    // strcat(folderName, dtw_tag);
    // First get sequence file-name without extension and store in line
    char * tch;
    strcpy(inputFilenameStem, argv[1]);
    tch = strtok(inputFilenameStem, ".");

//    cout << "Base Path: " << argv[3] << endl;
    strcpy(basePath, argv[3]);
    strcpy(subFolderName, argv[4]);
    if (verbose)
        cout << "Working in folder: " << basePath << endl;
    // Construct folder and dtw/distance filenames
    if (strcmp("", argv[4]) == 0)
        sprintf(folderName, "%s/%s", basePath, tch);    // sequenceFile_dtwTag
    else
        sprintf(folderName, "%s/%s/%s", basePath, subFolderName, tch);
//    sprintf(dtwFileName, "%s/dtw_alignment.csv", folderName);
//    sprintf(distFileName, "%s/kdigo_dm_%s_%s.csv", folderName, dtw_tag, dist_tag);

    stat(folderName, &info);
    if( ~S_ISDIR(info.st_mode))
        mkdir(folderName, ACCESSPERMS);

    unsigned long minSize = 10000;

    sprintf(weightFilename, "%s/%s", basePath, argv[2]);
//    strcpy(weightFilename, argv[2]);
    ifstream is(weightFilename);
    if (verbose)
        cout << "Reading weights from " << argv[2] << endl;
    getline(is, line);
    while (getline(is, line)) {
        seq.clear();
        ss.clear();
        ss.str("");
        ss << line;
        getline(ss, token, *delim);
        getline(ss, token, *delim);
        ss2 << token;
        ss2 >> fnum;
        tweights.push_back(fnum);
        ss2.clear();
        ss2.str("");
    }
    is.clear();

    sprintf(sequenceFilename, "%s/%s", basePath, argv[1]);
    ifstream is2(sequenceFilename);
    if (verbose)
        cout << "Reading sequences from " << argv[1] << endl;
    ss.clear();
    string line2;
    while (getline(is2,line2)) {
        ss2.clear();
        ss.clear();
        ss2.str("");
        ss.str("");
        seq.clear();

        ss << line2;
        getline(ss, token, *delim);
        ss2 << token;
        ss2 >> fnum;

        ids.push_back((int) fnum);
        while (getline(ss, token, *delim)) {
            ss2.clear();
            ss2.str("");
            ss2 << token;
            ss2 >> fnum;
            seq.push_back(fnum);
        }
        seqs.push_back(seq);
        minSize = min(minSize, seq.size());
        line2.clear();
    }
    number = seqs.size();
    if (verbose)
        cout << "Computing for " << number << " sequences." << endl;

    double coords[5];
    double temp_coord;

    for (int i = 0; i < 5; i++) {
        temp_coord = 0;
        for (int j = 0; j < i; j++) {
            temp_coord += tweights[j];
        }
        coords[i] = temp_coord;
    }

    for (int i = 0; i < number; i++) {
        for (int j = i + 1; j < number; j++) {
            pair[0] = i;
            pair[1] = j;
            pairs.push_back(pair);
        }
    }

    nIterations = pairs.size();
    double d;

    memset(distFileName, 0, 80);
//    sprintf(distFileName, "%s/kdigo_dm_%s_%s.csv", folderName, dtw_tag, dist_tag);
    sprintf(distFileName, "%s/kdigo_dm_%s.csv", folderName, dist_tag);

    // sprintf(filename, "%s/dtw_alignment_proc%d.csv", folderName, myid);
    memset(filename, 0, sizeof(filename));
    // sprintf(filename, "%s/%s_proc%d.csv", folderName, distFileName, myid);
    ofstream distanceFile;
    distanceFile.open(distFileName, ios::out | ios::trunc);
//    distanceFile << "ID1,ID2,Distance,PT1_Extension_Penalty,PT2_Extension_Penalty" << endl;
    for (long i = 0; i < nIterations; i++) {
        if (((i % (nIterations / 10)) == 0) & verbose)
            cout << (i / (nIterations / 10)) * 10 << "% done with alignment." << endl;
        idx1 = pairs[i][0];
        idx2 = pairs[i][1];
        d = distanceWrapper(seqs[idx1], seqs[idx2], coords, popDist, dfunc);

        distanceFile << ids[pairs[i][0]] << "," << ids[pairs[i][1]] << "," << d << endl;
    }
    distanceFile.close();

    return 0;
}

double distanceWrapper(vector<int> X, vector<int> Y, double *coords, bool popCoords, string dfunc) {
    if (dfunc == "braycurtis")
        return bcDist(X, Y, coords, popCoords);
    else if (dfunc == "euclidean")
        return normEuclidean(X, Y, coords, popCoords);
    else if (dfunc == "cityblock")
        return normCityblock(X, Y, coords, popCoords);
    else
        cout << "Didn't understand distance function: " << dfunc << endl;
    return -1;
}

double bcDist(vector<int> X, vector<int> Y, double *coords, bool popCoords) {
    double num = 0;
    double den = 0;
    double xCoord, yCoord;
    for (int i = 0; i < X.size(); i++) {
        if (popCoords) {
            xCoord = coords[X[i]];
            yCoord = coords[Y[i]];
            num += fabs(xCoord - yCoord);
            den += fabs(xCoord + yCoord);
        }
        else {
            num += fabs(X[i] - Y[i]);
            den += fabs(X[i] + Y[i]);
        }
    }
    double out;
    out = num / den;
    if (isnan(out))
        cout << "Distance is NaN" << endl;
    return out;
}

double normEuclidean(vector<int> X, vector<int> Y, double *coords, bool popCoords) {
    double num = 0;
    double den = 0;
    double xCoord, yCoord;
    for (int i = 0; i < X.size(); i++) {
        if (popCoords) {
            xCoord = coords[X[i]];
            yCoord = coords[Y[i]];
            num += pow(fabs(xCoord - yCoord), 2);
            den += pow(fabs(coords[4] - coords[0]), 2);
        }
        else {
            num += pow(fabs(X[i] - Y[i]), 2);
            den += pow(4, 2);
        }
    }
    return sqrt(num) / sqrt(den);
}

double normCityblock(vector<int> X, vector<int> Y, double *coords, bool popCoords) {
    double num = 0;
    double den = 0;
    double xCoord, yCoord;
    for (int i = 0; i < X.size(); i++) {
        if (popCoords) {
            xCoord = coords[X[i]];
            yCoord = coords[Y[i]];
            num += fabs(xCoord - yCoord);
            den += fabs(coords[4] - coords[0]);
        }
        else {
            num += fabs(X[i] - Y[i]);
            den += 4;
        }
    }
    return num / den;
}
