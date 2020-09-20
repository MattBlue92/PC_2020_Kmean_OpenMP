//
// Created by matteo on 25/08/20.
//

#include "SaveToCSV.h"


void SaveToCSV::saveToCsv(double **matrixData, int nRow, int nCol, string path) {
    ofstream myFile;

    myFile.open(path);
    myFile<< "Number_of_threads" << "," << "Sequential_time" <<","<< "Parallel_time" <<","<< "Speed_up" << endl;
    int numThread;
    double sequentialTime;
    double parallelTime;
    double speedUp;

    for(int i = 0; i<nRow; i++){
        numThread = matrixData[i][0];
        parallelTime = matrixData[i][1];
        sequentialTime = matrixData[i][2];
        speedUp = matrixData[i][3];

        myFile<< numThread<< ","<< parallelTime<<","<< sequentialTime<<","<< speedUp<<endl;
    }

    myFile.close();

}