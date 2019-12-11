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

vector<vector<double> > dtw(vector<double> X, vector<double> Y,  double mismatch[5][5], double extension[5], double *coords, bool popCoords, bool aggExt=0, bool aggLap=0, double alpha=1.0, double laplacian = 0.0);
void priorDTW(char * dtwFileName, char * distFileName, double *coords, bool popCoords, double laplacian = 0, bool aggLap = 0);

double mismatchInterp(double val1, double val2, double mismatch[5][5]);
double extensionInterp(double val, double extension[5]);
double coordInterp(double val, double coords[5]);

double myDist(vector<double> X, vector<double> Y, double *coords, bool popCoords, double laplacian = 0, bool aggLap = 0);

int main(int argc, char **argv) {
    vector<vector<double> > seqs;
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
    char dtwFileName[256] = {"\0"};
    char distFileName[256] = {"\0"};
    char dist_tag[80] = {"\0"};
    char dtw_tag[80] = {"\0"};
    char subFolderName[80] = {"\0"};
    char tempStr[256] = {"\0"};
    char inputFilenameStem[80] = {"\0"};
    char sequenceFilename[80] = {"\0"};
    char weightFilename[80] = {"\0"};
    vector<double> seq;
    struct stat info;
    char *delim = ",";

    if (argc < 4) {
        cout << "Usage:" << endl;
        cout << argv[0] << " seq_filename weight_filename output_dir (-opts)" << endl;
        return 1;
    }

    bool popDTW = false;
    bool popDist = false;
    bool aggLap = false;
    bool aggExt = false;
    bool getLap = false;
    bool getAlpha = false;
    bool verbose = false;
    double alpha = 0.0;
    double laplacian = 0.0;
    for (int i = 4; i < argc; i++) {
        ss.clear();
        if (strcmp("-popDTW", argv[i]) == 0) {

            popDTW = true;
            if (verbose)
                cout << "Using population mismatch and extension penalties." << endl;
        }
        else if (strcmp("-popDist", argv[i]) == 0) {
            if (verbose)
                cout << "Using population bray-curtis distance." << endl;
            popDist = true;
        }
        else if (strcmp("-laplacian", argv[i]) == 0) {
            if (verbose)
                cout << "Laplacian factor value " << argv[i+1] << "." << endl;
            getLap = true;
        }
        else if (strcmp("-aggLap", argv[i]) == 0) {
            strcat(dist_tag, "_aggLap");
            if (verbose)
                cout << "Aggregating Laplacian factor." << endl;
            aggLap = true;
        }
        else if (strcmp("-aggExt", argv[i]) == 0) {
            strcat(dtw_tag, "_aggExt");
            if (verbose)
                cout << "Aggregating extension penalty." << endl;
            aggExt = true;
        }
        else if (strcmp("-alpha", argv[i]) == 0) {
            getAlpha = true;
        }
        else if (strcmp("-v", argv[i]) == 0) {
            verbose = true;
        }
        else if (strcmp("-delim", argv[i]) == 0) {
            delim = argv[i+1];
        }
        else {
            if (getLap) {
                ss << argv[i];
                ss >> laplacian;
                tempStr[0] = '\0';
                sprintf(tempStr, "_lap%d", (int) laplacian);
                strcat(dist_tag, tempStr);
//                            i++;
                if (verbose)
                    cout << "Using Laplacian value of " << laplacian << "." << endl;
                getLap = false;
            }
            if (getAlpha) {
                ss << argv[i];
                ss >> alpha;
                tempStr[0] = '\0';
                if (verbose)
                    cout << "Using Extension Penalty weight of alpha = " << alpha << "." << endl;
                //            i++;
                getAlpha = false;
            }
        }
    }

    // Construct tag for DTW
    if (popDTW)
//        strcat(dtw_tag, "popDTW");
        sprintf(dtw_tag, "popDTW");
    else
//        strcat(dtw_tag, "normDTW");
        sprintf(dtw_tag, "normDTW");
//    if (alpha > 0) {
        tempStr[0] = '\0';
        if (alpha >= 1)
            sprintf(tempStr, "_a%dE+00", (int) alpha);
        else
            sprintf(tempStr, "_a%dE-04", (int) (floor((alpha*10000))));
        strcat(dtw_tag, tempStr);
//    }
    if (aggExt)
        strcat(dtw_tag, "_aggExt");

    // Construct tag for Distance
    if (popDist)
        sprintf(dist_tag, "popCoords");
    else
        sprintf(dist_tag, "absCoords");
    if (laplacian > 0) {
//        tempStr[0] = '\0';
        memset(tempStr, 0, 256);
        sprintf(tempStr, "_lap%.0f", laplacian);
        strcat(dist_tag, tempStr);
    }
    if (aggLap)
        strcat(dist_tag, "_aggLap");

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
        sprintf(folderName, "%s/%s_%s", basePath, tch, dtw_tag);    // sequenceFile_dtwTag
    else
        sprintf(folderName, "%s/%s/%s_%s", basePath, subFolderName, tch, dtw_tag);
//    sprintf(dtwFileName, "%s/dtw_alignment.csv", folderName);
//    sprintf(distFileName, "%s/kdigo_dm_%s_%s.csv", folderName, dtw_tag, dist_tag);

    stat(folderName, &info);
    if( ~S_ISDIR(info.st_mode))
        mkdir(folderName, ACCESSPERMS);

    unsigned long minSize = 10000;

//    sprintf(weightFilename, "%s/%s", basePath, argv[2]);
    strcpy(weightFilename, argv[2]);
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

    double mismatch[5][5] = {};
    double extension[5] = {0};
    if (popDTW) {
        for (int i = 0; i < 4; i++) {
            mismatch[i][i+1] = tweights[i];
        }
        for (int i = 0; i < 4; i++) {
            for (int j = i + 2; j < 5; j++) {
                mismatch[i][j] = mismatch[i][j-1] + mismatch[j-1][j];
            }
        }
        for (int i = 0; i < 4; i++)
            extension[i+1] = tweights[i];
    } else {
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                mismatch[i][j] = fabs(i - j);
                mismatch[j][i] = fabs(i - j);
            }
        }
    }

    nIterations = pairs.size();
    vector<vector<double> > ret;
    vector<vector<double> > allPaths;

    memset(dtwFileName, 0, 80);
    memset(distFileName, 0, 80);
    sprintf(dtwFileName, "%s/dtw_alignment.csv", folderName);
//    sprintf(distFileName, "%s/kdigo_dm_%s_%s.csv", folderName, dtw_tag, dist_tag);
    sprintf(distFileName, "%s/kdigo_dm.csv", folderName);

    // sprintf(filename, "%s/dtw_alignment_proc%d.csv", folderName, myid);
    ofstream alignmentFile;
    alignmentFile.open(dtwFileName, ios::out | ios::trunc);
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
        ret = dtw(seqs[idx1], seqs[idx2], mismatch, extension, coords, popDist, aggExt, aggLap, alpha, laplacian);

        alignmentFile << (int) ids[pairs[i][0]];
        for (long j = 0; j < ret[1].size(); j++)
            alignmentFile << "," << ret[1][j];
        alignmentFile << endl;
        alignmentFile << (int) ids[pairs[i][1]];
        for (long j = 0; j < ret[1].size(); j++)
            alignmentFile << "," << ret[2][j];
        alignmentFile << endl << endl;
        distanceFile << ids[pairs[i][0]] << "," << ids[pairs[i][1]] << "," << ret[0][0]
                     << "," << ret[0][1] << "," << ret[0][2] << endl;
    }
    alignmentFile.close();
    distanceFile.close();

    return 0;
}

vector<vector<double> > dtw(vector<double> X, vector<double> Y,  double mismatch[5][5], double extension[5], double *coords, bool popCoords, bool aggExt, bool aggLap, double alpha, double laplacian) {
    int r = X.size();
    int c = Y.size();
    vector<vector<double> > D0(r+1, vector<double>(c+1, 0));
    vector<vector<double> > ext_x(r+1, vector<double>(c+1, 0));
    vector<vector<double> > dup_x(r+1, vector<double>(c+1, 0));
    vector<vector<double> > ext_y(r+1, vector<double>(c+1, 0));
    vector<vector<double> > dup_y(r+1, vector<double>(c+1, 0));

    for (int i = 1; i < r+1; i++) {
        D0[i][0] = numeric_limits<double>::max();
    }

    for (int i = 1; i < c+1; i++) {
        D0[0][i] = numeric_limits<double>::max();
    }

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            if ((int)X[i] <= (int)Y[j])
//                D0[i+1][j+1] = mismatch[(int)X[i]][(int)Y[j]];
                D0[i+1][j+1] = mismatchInterp(X[i], Y[j], mismatch);
            else
                D0[i+1][j+1] = mismatchInterp(Y[j], X[i], mismatch);
        }
    }


    double C[r][c];
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            C[i][j] = D0[i+1][j+1];
        }
    }
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            double options[3];
            if (aggExt) {
                options[0] = D0[i][j] + ext_x[i][j] + ext_y[i][j];
                options[1] = D0[i][j + 1] + ext_x[i][j + 1] + ext_y[i][j + 1] +
                             alpha * (dup_y[i][j + 1] + 1) * extensionInterp(Y[j], extension);
                options[2] = D0[i + 1][j] + ext_x[i + 1][j] + ext_y[i + 1][j] +
                             alpha * (dup_x[i + 1][j] + 1) * extensionInterp(X[i], extension);
                if (options[1] <= options[0] && options[1] <= options[2]) {
                    D0[i + 1][j + 1] += D0[i][j + 1] + alpha * ((dup_y[i][j + 1] + 1) * extensionInterp(Y[j], extension));
                    dup_y[i + 1][j + 1] = dup_y[i][j + 1] + 1;
                    dup_x[i + 1][j + 1] = 0;
                } else if (options[2] <= options[0] && options[2] <= options[1]) {
                    D0[i + 1][j + 1] += D0[i + 1][j] + alpha * ((dup_x[i + 1][j] + 1) * extensionInterp(X[i], extension));
                    dup_y[i + 1][j + 1] = 0;
                    dup_x[i + 1][j + 1] = dup_x[i + 1][j] + 1;
                } else {
                    D0[i + 1][j + 1] += D0[i][j];
                    dup_y[i + 1][j + 1] = 0;
                    dup_x[i + 1][j + 1] = 0;
                }
            } else {
                options[0] = D0[i][j];
                options[1] = D0[i][j + 1] + alpha * ((dup_y[i][j + 1] + 1) * extensionInterp(Y[j], extension));
                options[2] = D0[i + 1][j] + alpha * ((dup_x[i + 1][j] + 1) * extensionInterp(X[i], extension));
                if (options[0] <= options[1] && options[0] <= options[2]) {
                    D0[i + 1][j + 1] += D0[i][j];
                    dup_x[i + 1][j + 1] = 0;
                    dup_y[i + 1][j + 1] = 0;
                } else if (options[1] <= options[0] && options[1] <= options[2]) {
                    D0[i + 1][j + 1] += D0[i][j + 1] + alpha * ((dup_y[i][j + 1] + 1) * extensionInterp(Y[j], extension));
                    dup_y[i + 1][j + 1] = dup_y[i][j + 1] + 1;
                    dup_x[i + 1][j + 1] = 0;
                } else{
                    D0[i + 1][j + 1] += D0[i + 1][j] + alpha * ((dup_x[i + 1][j] + 1) * extensionInterp(X[i], extension));
                    dup_y[i + 1][j + 1] = 0;
                    dup_x[i + 1][j + 1] = dup_x[i + 1][j] + 1;
                }
            }
        }
    }

    vector<vector<double> > paths;
    int i = r - 1;
    int j = c - 1;
    double v1, v2, v3;
    vector<double> distance(3, 0);
    int xdup = 0;
    int ydup = 0;
    vector<int> p;
    vector<int> q;
    vector<double> xout, yout, temp;
    p.push_back(i);
    q.push_back(j);
    while(i > 0 || j > 0) {

        v1 = D0[i][j];
        v2 = D0[i][j+1];
        v3 = D0[i+1][j];
        if (v1 <= v2 && v1 <= v3) {
            i -= 1;
            j -= 1;
            xdup = ydup = 0;
        }
        else if (v2 <= v1 && v2 <= v3) {
            i -= 1;
            ydup += 1;
            distance[2] += (ydup * extensionInterp(Y[j], extension));
            xdup = 0;
        }
        else {
            j -= 1;
            xdup += 1;
            distance[1] += (xdup * extensionInterp(X[i], extension));
            ydup = 0;
        }

        p.insert(p.begin(), i);
        q.insert(q.begin(), j);
    }
    for (int i = 0; i < p.size(); i++) {
        xout.push_back(X[p[i]]);
        yout.push_back(Y[q[i]]);
    }
    distance[0] = myDist(xout, yout, coords, popCoords, laplacian, aggLap);
    paths.push_back(distance);
    paths.push_back(xout);
    paths.push_back(yout);
    return paths;
}

double mismatchInterp(double val1, double val2, double mismatch[5][5]) {
    double f1, f2, baseMism, slope1, slope2, intMism;
    f1 = floor(val1);
    f2 = floor(val2);

    baseMism = mismatch[(int) f1][(int) f2];
    if (f1 < 4)
        slope1 = mismatch[(int) f1 + 1][(int) f2] - baseMism;
    else
        slope1 = 0;

    if (f2 < 4)
        slope2 = mismatch[(int) f1][(int) f2 + 1] - baseMism;
    else
        slope2 = 0;

    intMism = baseMism + ((val1 - f1) * slope1) + ((val2 - f2) * slope2);
    return intMism;
}

double extensionInterp(double val, double extension[5]) {
    double flr, baseExt, slope, intExt;
    flr = floor(val);
    baseExt = extension[(int) flr];
    if (flr < 4) {
        slope = extension[(int) flr + 1] - baseExt;
        intExt = baseExt + ((val - flr) * slope);
        return intExt;
    }
    else
        return baseExt;
}


double coordInterp(double val, double coords[5]) {
    double flr, baseCoord, slope, intCoord;
    flr = floor(val);
    baseCoord = coords[(int) flr];
    if (flr < 4) {
        slope = coords[(int) flr + 1] - baseCoord;
        intCoord = baseCoord + ((val - flr) * slope);
        return intCoord;
    }
    else
        return baseCoord;
}

void priorDTW(char * dtwFileName, char * distFileName, double *coords, bool popCoords, double laplacian, bool aggLap) {
    string line;
    vector<double> X;
    vector<double> Y;
    stringstream ss;
    vector<double> id1;
    vector<double> id2;
    double number;

    ifstream alignments(dtwFileName);
    ofstream dist_out(distFileName);

    while (getline(alignments, line)) {
        X.clear();
        Y.clear();
        ss.clear();
        ss << line;
        ss >> number;
        dist_out << number;
        while (ss >> number) {
            X.push_back(number);
        }
        getline(alignments, line);
        ss.clear();
        ss << line;
        ss >> number;
        dist_out << "," << number;
        while (ss >> number) {
            Y.push_back(number);
        }
        number = myDist(X, Y, coords, popCoords, laplacian, aggLap);
        dist_out << "," << number << endl;
        getline(alignments, line);
    }
    alignments.close();
    dist_out.close();
    return;
}

double myDist(vector<double> X, vector<double> Y, double *coords, bool popCoords, double laplacian, bool aggLap) {
    double num = 0;
    double den = 0;
    double xCoord, yCoord;
    for (int i = 0; i < X.size(); i++) {
        if (popCoords) {
            xCoord = coordInterp(X[i], coords);
            yCoord = coordInterp(Y[i], coords);
            num += fabs(xCoord - yCoord);
            den += fabs(xCoord + yCoord);
        }
        else {
            num += fabs(X[i] - Y[i]);
            den += fabs(X[i] + Y[i]);
        }
    }
    double out;
    if (aggLap)
        laplacian *= X.size();
    out = num / (den + laplacian);
    if (isnan(out))
        cout << "Distance is NaN" << endl;
    return out;
}
