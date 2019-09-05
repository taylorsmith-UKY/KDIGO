#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <mpi.h>
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
    long val, idx1, idx2, nIterations, myStart, myStop, number;
    int size, myid, nweights;
    double msg[MAXLEN];
    double fnum;
    MPI_Status status;
    stringstream ss;
    stringstream ss2;
    string token;
    string line;
    char filename[80] = {"\0"};
    char folderName[80] = {"\0"};
    char dtwFileName[80] = {"\0"};
    char distFileName[80] = {"\0"};
    char dist_tag[80] = {"\0"};
    char dtw_tag[80] = {"\0"};
    char tempStr[256] = {"\0"};
    char inputFilenameStem[80] = {"\0"};
    char mergeCommand[8*80] = {"\0"};
    vector<double> seq;
    struct stat info;
    char *delim = ",";

    if (argc < 4) {
        cout << "Usage:" << endl;
        cout << argv[0] << " seq_filename weight_filename output_dir (-opts)" << endl;
        return 1;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);


    bool popDTW = false;
    bool popDist = false;
    bool aggLap = false;
    bool aggExt = false;
    bool getLap = false;
    bool getAlpha = false;
    double alpha = 0.0;
    double laplacian = 0.0;
    for (int i = 2; i < argc; i++) {
        ss.clear();
        if (strcmp("-popDTW", argv[i]) == 0) {

            popDTW = true;
            if (myid == 0)
                cout << "Using population mismatch and extension penalties." << endl;
        }
        else if (strcmp("-popDist", argv[i]) == 0) {
            if (myid == 0)
                cout << "Using population bray-curtis distance." << endl;
            popDist = true;
        }
        else if (strcmp("-laplacian", argv[i]) == 0) {
            if (myid == 0)
                cout << "Received command for laplacian with value " << argv[i+1] << "." << endl;
            getLap = true;
        }
        else if (strcmp("-aggLap", argv[i]) == 0) {
            aggLap = true;
        }
        else if (strcmp("-aggExt", argv[i]) == 0) {
            strcat(dtw_tag, "_aggExt");
            if (myid == 0)
                cout << "Aggregating extension penalty." << endl;
            aggExt = true;
        }
        else if (strcmp("-alpha", argv[i]) == 0) {
            getAlpha = true;
        }
        else if (strcmp("-delim", argv[i]) == 0) {
            delim = argv[i+1];
        }
        else {
            if (getLap) {
                ss << argv[i];
                ss >> laplacian;
                if (myid == 0)
                    cout << "Using Laplacian value of " << laplacian << "." << endl;
                getLap = false;
            }
            if (getAlpha) {
                ss << argv[i];
                ss >> alpha;
                tempStr[0] = '\0';
                if (myid == 0)
                    cout << "Using Extension Penalty weight of alpha = " << alpha << "." << endl;
                getAlpha = false;
            }
        }
    }

    // Construct tag for DTW
    if (popDTW)
        sprintf(dtw_tag, "popDTW");
    else
        sprintf(dtw_tag, "normDTW");
    if (alpha > 0) {
        tempStr[0] = '\0';
        if (alpha >= 1)
            sprintf(tempStr, "_a%dE+00", (int) alpha);
        else
            sprintf(tempStr, "_a%dE-02", (int) (alpha*100));
        strcat(dtw_tag, tempStr);
    }
    if (aggExt)
        strcat(dtw_tag, "_aggExt");

    // Construct tag for Distance
    if (popDist)
        sprintf(dist_tag, "popCoords");
    else
        sprintf(dist_tag, "absCoords");
    if (laplacian > 0) {
        memset(tempStr, 0, 256);
        sprintf(tempStr, "_lap%.0f", laplacian);
        strcat(dist_tag, tempStr);
    }
    if (aggLap)
        strcat(dist_tag, "_aggLap");

    // First get sequence file-name without extension and store in line
    char * tch;
    strcpy(inputFilenameStem, argv[1]);
    tch = strtok(inputFilenameStem, ".");

    // Construct folder and dtw/distance filenames
    sprintf(folderName, "%s_%s", tch, dtw_tag);    // sequenceFile_dtwTag

    stat(folderName, &info);
    if( (info.st_mode & S_IFDIR) == 0 )
        mkdir(folderName, ACCESSPERMS);

    memset(dtwFileName, 0, 80);
    memset(distFileName, 0, 80);
    sprintf(dtwFileName, "%s/dtw_alignment.csv", folderName);
    sprintf(distFileName, "%s/kdigo_dm_%s_%s_proc%d.csv", folderName, dtw_tag, dist_tag, myid);

    if (myid == 0) {
            ifstream is(argv[2]);
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
            nweights = tweights.size();
    }

    ifstream is2(dtwFileName);
    memset(distFileName, 0, 80);
    sprintf(distFileName, "%s/kdigo_dm_%s_%s.csv", folderName, dtw_tag, dist_tag);
    ofstream distanceFile;
    distanceFile.open(distFileName, ios::out | ios::trunc);
    cout << "Reading alignments from " << dtwFileName << endl;
    bool IDSaved = false;
    ss.clear();
    string line2;
    vector<int> vec1, vec2;
    int kval;
    double d;
    double coords[nweights+1];
    double temp_coord;
    for (int i = 0; i < nweights+1; i++){
        temp_coord = 0;
        for (int j = 0; j < i; j++) {
            temp_coord += tweights[j];
        }
        coords[i] = temp_coord;
    }
    while (getline(is2,line2)) {
        vec1.clear();
        vec2.clear();
        ss2.clear();
        ss.clear();
        ss2.str("");
        ss.str("");
        seq.clear();

        ss << line2;
        getline(ss, token, *delim);
        ss2 << token;
        ss2 >> kval;

        distanceFile << kval;
        while (getline(ss, token, *delim)) {
            ss2.clear();
            ss2.str("");
            ss2 << token;
            ss2 >> kval;
            vec1.push_back(fnum);
        }
        lin2.clear();
        getline(is2,line2);

        ss2.clear();
        ss.clear();
        ss2.str("");
        ss.str("");
        seq.clear();

        ss << line2;
        getline(ss, token, *delim);
        ss2 << token;
        ss2 >> kval;

        distanceFile << "," << kval;
        while (getline(ss, token, *delim)) {
            ss2.clear();
            ss2.str("");
            ss2 << token;
            ss2 >> kval;
            vec2.push_back(fnum);
        }

        d = myDist(vec1, vec2, coords, popCoords, laplacian, aggLap);
        distanceFile << "," << d << endl;
        line2.clear();
        getline(is2,line2);
        line2.clear();
    }
    distanceFile.close();

    MPI_Finalize();
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
                D0[i+1][j+1] = mismatch[(int)X[i]][(int)Y[j]];
            else
                D0[i+1][j+1] = mismatch[(int)Y[j]][(int)X[i]];
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
                             alpha * (dup_y[i][j + 1] + 1) * extension[(int)Y[j]];
                options[2] = D0[i + 1][j] + ext_x[i + 1][j] + ext_y[i + 1][j] +
                             alpha * (dup_x[i + 1][j] + 1) * extension[(int)X[i]];
                if (options[1] <= options[0] && options[1] <= options[2]) {
                    ext_y[i + 1][j + 1] = ext_y[i][j + 1] + alpha * (dup_y[i][j + 1] + 1) * extension[(int)Y[j]];
                    ext_x[i + 1][j + 1] = ext_x[i][j + 1];
                    D0[i + 1][j + 1] += D0[i][j + 1];
                    dup_y[i + 1][j + 1] = dup_y[i][j + 1] + 1;
                    dup_x[i + 1][j + 1] = 0;
                } else if (options[2] <= options[0] && options[2] <= options[1]) {
                    ext_y[i + 1][j + 1] = ext_y[i + 1][j];
                    ext_x[i + 1][j + 1] = ext_x[i + 1][j] + alpha * (dup_x[i + 1][j] + 1) * extension[(int)X[i]];
                    D0[i + 1][j + 1] += D0[i + 1][j];
                    dup_y[i + 1][j + 1] = 0;
                    dup_x[i + 1][j + 1] = dup_x[i + 1][j] + 1;
                } else {
                    D0[i + 1][j + 1] += D0[i][j];
                    ext_y[i + 1][j + 1] = ext_y[i + 1][j];
                    ext_x[i + 1][j + 1] = ext_x[i + 1][j];
                    dup_y[i + 1][j + 1] = 0;
                    dup_x[i + 1][j + 1] = 0;
                }
                D0[i+1][j+1] += ext_x[i+1][j+1] + ext_y[i+1][j+1];
            } else {
                options[0] = D0[i][j];
                options[1] = D0[i][j + 1] + alpha * (dup_y[i][j + 1] + 1) * extension[(int)Y[j]];
                options[2] = D0[i + 1][j] + alpha * (dup_x[i + 1][j] + 1) * extension[(int)X[i]];
                if (options[0] <= options[1] && options[0] <= options[2]) {
                    D0[i + 1][j + 1] += D0[i][j];
                    dup_y[i + 1][j + 1] = 0;
                    dup_x[i + 1][j + 1] = 0;
                } else if (options[1] <= options[0] && options[1] <= options[2]) {
                    D0[i + 1][j + 1] += D0[i][j + 1] + alpha * (dup_y[i][j + 1] + 1) * extension[(int)Y[j]];
                    dup_y[i + 1][j + 1] = dup_y[i][j + 1] + 1;
                    dup_x[i + 1][j + 1] = 0;
                } else{
                    D0[i + 1][j + 1] += D0[i + 1][j] + alpha * (dup_x[i + 1][j] + 1) * extension[(int)X[i]];
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
            distance[2] += (ydup * extension[(int)Y[j]]);
            xdup = 0;
        }
        else {
            j -= 1;
            xdup += 1;
            distance[1] += (xdup * extension[(int)X[i]]);
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

double myDist(vector<double> X, vector<double> Y, double *coords, bool popCoords, double laplacian, bool aggLap) {
    double num = 0;
    double den = 0;
    for (int i = 0; i < X.size(); i++) {
        if (popCoords) {
            num += fabs(coords[(int)X[i]] - coords[(int)Y[i]]);
            den += fabs(coords[(int)X[i]] + coords[(int)Y[i]]);
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
    return out;
}
