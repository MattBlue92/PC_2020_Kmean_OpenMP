//
// Created by matteo on 25/08/20.
//

#ifndef PC_KMEANS_SAVETOCSV_H
#define PC_KMEANS_SAVETOCSV_H
#include <string>
#include <iostream>
#include <fstream>

using namespace std;


class SaveToCSV{
public:
    void saveToCsv(double **matrixData, int nRow, int nCol, string path);
};



#endif //PC_KMEANS_SAVETOCSV_H