//
// Created by matteo on 19/08/20.
//

#include "DataFrame.h"
#include <fstream>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>


using namespace std;

double **DataFrame::dataFrame(string path, int nRow, int nCol) {
    double **data = initData(nRow, nCol);
    std::ifstream file(path);

    std::string line;
    std::getline(file, line);

    for(int i= 0; i< nRow; i++)
    {
        std::string line;
        std::getline(file, line);
        if ( !file.good() )
            break;

        std::stringstream iss(line);

        for (int j = 0; j < nCol; j++)
        {
            std::string val;
            std::getline(iss, val, ',');
            if ( !iss.good() )
                break;

            if(!isNumber(val)){
                std::stringstream convertor(val);
                convertor >> data[i][j];
            }

        }
    }

    return data;
}

double **DataFrame::initData(int nRow, int nCol) const {
    double** data;

    data = new double *[nRow];
    for(int i = 0; i <nRow; i++){
        data[i] = new double[nCol];
    }
    for(int i =0;i<nRow;i++){
        for(int j=0;j<nCol;j++){
            data[i][j]= 0;
        }
    }
    return data;
}

bool DataFrame::isNumber(string s) {
    for (int i = 0; i < s.length(); i++)
        if (isdigit(s[i]) == false)
            return false;

    return true;
}