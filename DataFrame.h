//
// Created by matteo on 19/08/20.
//


#ifndef PC_KMEANS_DATAFRAME_H
#define PC_KMEANS_DATAFRAME_H
#include <fstream>
#include <string>
using namespace std;
class DataFrame{
public:
    double** dataFrame(string path, int nRow, int nCol);

private:
    double **initData(int nRow, int nCol) const;

    bool isNumber(string s);
};
#endif //PC_KMEANS_DATAFRAME_H