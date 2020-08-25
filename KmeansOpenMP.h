//
// Created by Matteo Marulli and Matteo Gemignani on 21/08/20.
//

#ifndef PC_KMEANS_KMEANSOPENMP_H
#define PC_KMEANS_KMEANSOPENMP_H


#include <math.h>
#include <float.h>
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <string>
#include <omp.h>
#include "KmeansOpenMP.h"

using namespace std;

class KmeansOpenMP {

private:
    double **centroids = nullptr;
    int *labels = nullptr;
    double tol = 0.0001;
    double SSE = 0;
    int it = 100;
    int seed = 0;
    int k = 2;

    void centroidUpdate(int *labels, int *labelsCount, double **centroids, double **dataset, int nCol, int nRow);

    int clusterAssignment(double *point, double **centroids, int nCol);

    double getDistance(double *point, double *centroid, int nCol);

    void initCentroids(double **dataMatrix, int nRow, int nCol);

    void saveCentroid(double **save, double **centroid, int dim);

    void statistics(double SSE, double iterations, double **centroids, int nCol);

    void printMatrix(double **matrix, int nRow, int nCol);

    double computeSSE(int nCol, double *const *centroids, double *const *oldCentroids);

public:
    KmeansOpenMP(int k);

    KmeansOpenMP(int k, int it, int seed, double tol);

    ~KmeansOpenMP();//metodo distruttore
    int *fitPredict(double **dataset, int nRow, int nCol);

    void restart(int k);

    void restart(int k, int it, int seed, double tol);

    int *getLabels();

    double **clusterCenters();

    double getSSE();
};


#endif //PC_KMEANS_KMEANSOPENMP_H
