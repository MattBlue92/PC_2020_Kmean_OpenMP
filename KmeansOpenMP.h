//
// Created by matteo on 21/08/20.
//

#ifndef PC_KMEANS_KMEANSOPENMP_H
#define PC_KMEANS_KMEANSOPENMP_H

class KmeansOpenMP{

private:
    double  **centroids = nullptr;
    int *labels = nullptr;
    double tol = 0.0001;
    double distanceCentroids = 0;
    int it = 100;
    int seed = 0;
    int k = 2;

    void centroidUpdate(int *labels,int* labelsCount,double **centroids,double **dataset,int nCol, int nRow);
    int clusterAssignment(double* point, double** centroids, int nCol);
    double getDistance(double *point, double *centroid, int nCol);
    void initCentroids(double** dataMatrix, int nRow, int nCol);
    //double* getRow(double** dataMatrix, int i, int j);
    void  saveCentroid(double **save, double **centroid,int dim);
    void statistics(double SSE, double iterations, double ** centroids, int nCol);
    void printMatrix(double** matrix, int nRow, int nCol);
    double compute( int nCol, double **centroids, double **oldCentroids);

public:
    KmeansOpenMP(int k);
    KmeansOpenMP(int k, int it, int seed, double tol);
    ~KmeansOpenMP();//metodo distruttore
    int* fitPredict(double** dataset, int nRow, int nCol,int nThreads);
    void restart(int k);
    void restart(int k, int it, int seed, double tol);
    int* getLabels();
    double** clusterCenters();
    double getDistanceCentroids();

};

#endif //PC_KMEANS_KMEANSOPENMP_H