#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>
#include <mpi.h>
#include <math.h>

#define MAXLEN 256

using namespace std;

vector<vector<int> > dtw(vector<int> X, vector<int> Y,  double mismatch[5][5], double extension[5], bool aggExt=0, double alpha=0.0);
vector<vector<int> > _traceback(double **D, int r, int c);
double dist(vector<double> X, vector<double> Y, double *coords, bool popCoords, int laplacian=0);

int main(int argc, char **argv) {
    vector<vector<int> > seqs;
    vector<int> ids;
    vector<double> dist;
    vector<vector<int> > alignments;
    int myid, size, val, idx1, idx2, nIterations, myStart, myStop, number;
    int msg[256];
    MPI_Status status;
    stringstream ss;
    string line;
    vector<int> seq;

    double tweights[4] = {1.0, 2.513116, 4.077875, 6.156196};
    double coords[5];
    for (int i = 0; i < 5; i++){
        for (int j = 0; j < i; j++) {
            coords[i] += tweights[j];
        }
    }
//
//
//    MPI_Init(&argc, &argv);
//
//    MPI_Comm_size(MPI_COMM_WORLD, &size);
//    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    myid = 0;
    size = 4;
    bool popDTW = 0;
    bool popDist = 0;
    int laplacian = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp("-popDTW", argv[i]) != 0)
            popDTW = 1;
        else if (strcmp("-popDist", argv[i]) != 0)
            popDist = 1;
        else if (strcmp("-laplacian", argv[i]) != 0) {
            ss << argv[i+1];
            ss >> laplacian;
        }
    }

    if (myid == 0) {
        ifstream is("kdigo.csv");
        while (getline(is, line)) {
            ss << line;
            ss >> number;
            ids.push_back(number);
            while (ss >> number) {
                seq.push_back(number);
            }
            seqs.push_back(seq);
            seq.clear();
            ss.clear();
        }
        number = seqs.size();
    }
//    MPI_Bcast(&number, 1, MPI_INT, 0, MPI_COMM_WORLD);
    vector<vector<int> > pairs;
    vector<int> pair(2);
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
            mismatch[i+1][i] = tweights[i];
        }
        for (int i = 0; i < 4; i++) {
            for (int j = i + 2; j < 5; j++) {
                mismatch[i][j] = mismatch[i][j-1] + mismatch[j-1][j];
                mismatch[j][i] = mismatch[i][j-1] + mismatch[j-1][i];
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

//
//    if (myid == 0) {
//        for (int i = 0; i < number; i++) {
//            msg[0] = ids[i];
//            msg[1] = seqs[i].size();
//            copy(seqs[i].begin(), seqs[i].end(), &msg[2]);
//            for (int j = 0; j < size; j++) {
//                MPI_Send(&msg, msg[1]+2, MPI_INT, j, i, MPI_COMM_WORLD);
//            }
//
//        }
//    } else {
//        for (int i = 0; i < number; i++) {
//            MPI_Recv(&msg, MAXLEN, MPI_INT, 0, i, MPI_COMM_WORLD, &status);
//            ids.push_back(msg[0]);
//            val = msg[1];
//            vector<int> seq;
//            for (int j = 2; j < val+2; j++) {
//                seq.push_back(msg[j]);
//            }
//            seqs.push_back(seq);
//            seq.clear();
//        }
//    }

    nIterations = ceil((float) pairs.size() / size);
    myStart = myid * nIterations;
    myStop = min(myStart + nIterations, (int) pairs.size());

    vector<vector<int> > ret;
    vector<vector<int> > allPaths;
    for (int i = myStart; i < myStop; i++) {
        idx1 = pairs[i][0];
        idx2 = pairs[i][1];
        ret = dtw(seqs[idx1], seqs[idx2], mismatch, extension);
        allPaths.push_back(ret[0]);
        allPaths.push_back(ret[1]);
        ret.clear();
    }
//    if (myid == 0) {
//        int stopIdx, startIdx;
//        for (int i = 1; i < size; i++) {
//            startIdx = i*nIterations;
//            stopIdx = min(startIdx + nIterations, (int) pairs.size());
//            for (int j = startIdx; j < stopIdx; j++) {
//                seq.clear();
//                MPI_Recv(&msg, MAXLEN, MPI_INT, i, 2*j, MPI_COMM_WORLD, &status);
//                for (int k = 1; k < msg[0]+1; k++)
//                    seq.push_back(msg[k]);
//                allPaths.push_back(seq);
//                MPI_Recv(&msg, MAXLEN, MPI_INT, i, 2*j+1, MPI_COMM_WORLD, &status);
//                seq.clear();
//                for (int k = 1; k < msg[0]+1; k++)
//                    seq.push_back(msg[k]);
//                allPaths.push_back(seq);
//            }
//        }
//        if (allPaths.size() != 2*pairs.size()) {
//            cout << "Mismatch between ID pairs and aligned sequences" << endl;
//            cout << "Number of alignments: " << allPaths.size() << endl;
//            cout << "Number of pairs of IDs: " << pairs.size() << endl;
//            return 0;
//        } else {
//            cout << "Alignments and pairs match.";
//        }
//        ofstream f("dtw_alignment.csv");
//        for (int i = 0; i < pairs.size(); i++) {
//            f << ids[pairs[i][0]];
//            for (int j = 0; j < allPaths[2*i].size(); j++) {
//                f << "," << allPaths[2*i][j];
//            }
//            f << endl;
//            f << ids[pairs[i][1]];
//            for (int j = 0; j < allPaths[2*i+1].size(); j++) {
//                f << "," << allPaths[2*i+1][j];
//            }
//            f << endl;
//            f << endl;
//        }
//    } else {
//        int stopIdx, startIdx;
//        startIdx = myid*nIterations;
//        for (int i = 0; i < (allPaths.size() / 2); i++) {
//            msg[0] = allPaths[2*i].size();
//            copy(allPaths[2*i].begin(), allPaths[2*i].end(), &msg[1]);
//            MPI_Send(&msg, msg[0]+1, MPI_INT, 0, (startIdx + i)*2, MPI_COMM_WORLD);
//            msg[0] = allPaths[2*i+1].size();
//            copy(allPaths[2*i+1].begin(), allPaths[2*i+1].end(), &msg[1]);
//            MPI_Send(&msg, msg[0]+1, MPI_INT, 0, (startIdx + i)*2+1, MPI_COMM_WORLD);
//        }
//    }
//    MPI_Finalize();
}

vector<vector<int> > dtw(vector<int> X, vector<int> Y, double mismatch[5][5], double extension[5], bool aggExt, double alpha) {
    int r = X.size();
    int c = Y.size();
    double D0[r+1][c+1];
    double ext_x[r+1][c+1];
    double dup_x[r+1][c+1];
    double ext_y[r+1][c+1];
    double dup_y[r+1][c+1];
    cout << r << "," << c << endl;

    for (int i = 1; i < r+1; i++) {
        D0[i][0] = 100000;
    }

    for (int i = 1; i < c+1; i++) {
        D0[0][i] = 100000;
    }

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            D0[i+1][j+1] = mismatch[X[i]][Y[j]];
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
                             alpha * (dup_y[i][j + 1] + 1) * extension[Y[j]];
                options[2] = D0[i + 1][j] + ext_x[i + 1][j] + ext_y[i + 1][j] +
                             alpha * (dup_x[i + 1][j] + 1) * extension[X[i]];
                if (options[1] <= options[0] && options[1] <= options[2]) {
                    ext_y[i + 1][j + 1] = ext_y[i][j + 1] + alpha * (dup_y[i][j + 1] + 1) * extension[Y[j]];
                    ext_x[i + 1][j + 1] = ext_x[i][j + 1];
                    D0[i + 1][j + 1] += D0[i][j + 1];
                    dup_y[i + 1][j + 1] = dup_y[i][j + 1] + 1;
                    dup_x[i + 1][j + 1] = 0;
                } else if (options[2] <= options[0] && options[2] <= options[1]) {
                    ext_y[i + 1][j + 1] = ext_y[i + 1][j];
                    ext_x[i + 1][j + 1] = ext_x[i + 1][j] + alpha * (dup_x[i + 1][j] + 1) * extension[X[i]];
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
                options[1] = D0[i][j + 1] + alpha * (dup_y[i][j + 1] + 1) * extension[Y[j]];
                options[2] = D0[i + 1][j] + alpha * (dup_x[i + 1][j] + 1) * extension[X[i]];
                if (options[0] <= options[1] && options[0] <= options[2]) {
                    D0[i + 1][j + 1] += D0[i][j];
                    dup_y[i + 1][j + 1] = 0;
                    dup_x[i + 1][j + 1] = 0;
                } else if (options[1] <= options[0] && options[1] <= options[2]) {
                    D0[i + 1][j + 1] += D0[i][j + 1] + alpha * (dup_y[i][j + 1] + 1) * extension[Y[j]];
                    dup_y[i + 1][j + 1] = dup_y[i][j + 1] + 1;
                    dup_x[i + 1][j + 1] = 0;
                } else{
                    D0[i + 1][j + 1] += D0[i + 1][j] + alpha * (dup_x[i + 1][j] + 1) * extension[X[i]];
                    dup_y[i + 1][j + 1] = 0;
                    dup_x[i + 1][j + 1] = dup_x[i + 1][j] + 1;
                }
            }
        }
    }

    vector<vector<int> > paths;
    int i = r - 1;
    int j = c - 1;
    double v1, v2, v3;
    vector<int> p;
    vector<int> q;
    p.push_back(i);
    q.push_back(j);
    cout << i << "," << j << endl;
    while(i > 0 || j > 0) {
        v1 = D0[i][j];
        v2 = D0[i][j+1];
        v3 = D0[i+1][j];
        if (v1 <= v2 && v1 <= v3) {
            i -= 1;
            j -= 1;
        }
        else if (v2 <= v1 && v2 <= v3) {
            i -= 1;
        }
        else
            j -= 1;

        p.insert(p.begin(), i);
        q.insert(q.begin(), j);
    }
    paths.push_back(q);
    paths.push_back(p);
    cout << q[0];
    for (int i = 1; i < q.size(); i++)
        cout << "," << q[i];
    cout << endl;
    cout << p[0];
    for (int i = 1; i < p.size(); i++)
        cout << "," << p[i];
    cout << endl;
    return paths;
}


double dist(vector<int> X, vector<int> Y, double *coords, bool popCoords, int laplacian) {
    double num, den = 0;
    for (int i = 0; i < X.size(); i++) {
        if (popCoords) {
            num += (coords[X[i]] - coords[Y[i]]);
            den += (coords[X[i]] + coords[Y[i]]);
        }
        else {
            num += (X[i] - Y[i]);
            den += (X[i] + Y[i]);
        }
    }
    double out;
    out = num / (den + laplacian);
    return out;
}