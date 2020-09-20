//
// Created by matteo on 21/08/20.
//

#include <math.h>
#include "DataFrame.cpp"
#include "SaveToCSV.cpp"
#include <float.h>
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <string>
#include <omp.h>
#include "KmeansOpenMP.h"

using namespace std;

void KmeansOpenMP::initCentroids(double** dataMatrix, int nRow,int nCol){
    centroids = new double *[k];

    for (int i = 0; i < k; i++) {
        centroids[i] = new double[nCol];
    }
    srand(seed);
    for(int i = 0;i<k;i++){
        int row = int(rand()%nRow);
        for(int j=0;j<nCol;j++){
            centroids[i][j]=  dataMatrix[row][j];
        }
    }
}

double KmeansOpenMP::getDistance(double *point, double *centroid, int nCol) {
    double distance = 0;
    for (int i = 0; i < nCol; i++) {
        distance +=  pow(point[i] - centroid[i], 2);
    }
    return sqrt(distance);
}


int KmeansOpenMP::clusterAssignment(double* point, double** centroids, int nCol){
    int index = 0;
    double distance = 0;
    double oldDistance = DBL_MAX;
    for(int i =0; i<k;i++){
        double* centroid_i = centroids[i];
        distance = getDistance(point, centroid_i, nCol);
        if(distance<oldDistance){
            index = i;
            oldDistance = distance;
        }
    }
    return index;
}

void KmeansOpenMP::centroidUpdate(int *labels,int* labelsCount,double **centroids,double **dataset,int nCol, int nRow){
    int pointForCluster;
#pragma omp parallel default(none) shared(centroids,labelsCount) firstprivate (nCol,nRow,labels,dataset)
    {
#pragma omp for collapse(2)
        for(int c=0; c<k;c++){
            for(int i=0;i<nCol;i++){
                centroids[c][i]=0; // azzera il centroide c-esimo
            }
        }
#pragma omp for
        for(int i=0;i<k;i++){
            labelsCount[i]=0;
        }

#pragma omp for
        for(int c=0;c<k;c++){
            for(int j=0;j<nRow;j++){
                if(c==labels[j]){
                    labelsCount[c]++; // conta il numero di elementi nel cluster c-esimo
                    for(int i=0;i<nCol;i++){
                        centroids[c][i]+=dataset[j][i]; // aggiornamento centroide c-esimo componente per componente
                    }

                }
            }
        }
#pragma omp for collapse(2)
        for(int c=0; c<k;c++){
            for(int i=0;i<nCol;i++){
                centroids[c][i]=centroids[c][i]/labelsCount[c]; // dividi ogni componente per il numero di oggetti nel cluster
            }
        }
    }
}

void KmeansOpenMP::saveCentroid(double **save, double **centroid,int dim){
#pragma omp parallel for collapse(2) shared(save) firstprivate(centroid,dim)
    for(int i=0;i<k;i++){
        for(int j=0;j<dim;j++){
            save[i][j]=centroid[i][j];
        }
    }
}

double KmeansOpenMP::compute( int nCol, double **centroids, double **oldCentroids) {
    double sum=0;
#pragma omp parallel for reduction(+:sum) firstprivate(centroids,oldCentroids,nCol)
    for(int c=0;c<k;c++){
        sum += pow(getDistance(oldCentroids[c], centroids[c], nCol), 2);
    }
    return sum;
}

int* KmeansOpenMP::fitPredict(double** dataset,  int nRow, int nCol,int nThreads){
    int t = 0;
    double distanceCentroids = 0;
    int *labels = new int[nRow];

    int* labelsCount= new int [k];


    initCentroids(dataset, nRow, nCol);

    double** oldCentroids;
    oldCentroids = new double *[k];
    for(int i = 0; i <k; i++){
        oldCentroids[i] = new double[nCol];
    }

    do {
        t++;
#pragma omp parallel for schedule(static,nRow/nThreads) default(none) shared(labels) firstprivate(nThreads,centroids,nCol,nRow,dataset)
        for (int indexPoint = 0; indexPoint<nRow; indexPoint++){
            labels[indexPoint] = clusterAssignment(dataset[indexPoint], centroids, nCol);
        }

        saveCentroid(oldCentroids, centroids, nCol);
        centroidUpdate(labels,labelsCount, centroids, dataset, nCol, nRow);



        distanceCentroids = compute(nCol, centroids, oldCentroids);
        cout<<t<<"/"<<it<<" distance centroids:"<<distanceCentroids<<endl;
    }while (distanceCentroids>tol and t<it);
    statistics(distanceCentroids, t, centroids, nCol);
    //destroy obj with *

    for(int i = 0; i<k;i++)
        delete oldCentroids[i];
    delete [] oldCentroids;

    return labels;
}

void KmeansOpenMP::printMatrix(double **matrix, int nRow, int nCol) {
    for (int  i=0; i < nRow; i++) {
        cout<<"Centroid "<<i<<" : [";
        for (int j = 0; j < nCol; j++)
            cout << matrix[i][j] << ", ";
        cout << "]"<<endl;
    }
}

void KmeansOpenMP::statistics(double distanceCentroids, double iterations, double ** centroids, int nCol) {
    cout<<"<<<<<<Statistics<<<<<<"<<endl;
    cout<<"distance centroids: "<<distanceCentroids<<endl;
    cout<<"Iterations: "<< iterations<<endl;
    cout<<"Centroids: "<<endl;
    printMatrix(centroids, k, nCol);
    cout<<"<<<<<<End-Statistics<<<<<<"<<endl;
}

void KmeansOpenMP::restart(int k) {
    centroids = nullptr;
    labels = nullptr;
    tol = 0.0001;
    distanceCentroids= 0;
    it = 10;
    this->k = k;
}

double KmeansOpenMP::getDistanceCentroids() {
    return distanceCentroids;
}

double **KmeansOpenMP::clusterCenters() {
    return centroids;
}

int *KmeansOpenMP::getLabels() {
    return labels;
}

KmeansOpenMP::KmeansOpenMP(int k) {
    this->k = k;
}

KmeansOpenMP::KmeansOpenMP(int k, int it,  int seed, double tol) {
    this->k = k;
    this->it = it;
    this->seed = seed;
    this->tol = tol;
}

void KmeansOpenMP::restart(int k, int it, int seed ,double tol) {
    delete [] centroids;
    delete [] labels;
    centroids = nullptr;
    labels = nullptr;
    distanceCentroids = 0;
    this->tol = tol;
    this->it = it;
    this->seed = seed;
    this->k = k;
}

KmeansOpenMP::~KmeansOpenMP(){
    delete [] centroids;
    delete [] labels;
}



int main()
{
#ifdef _OPENMP
    std::cout << "_OPNMP defined" << std::endl;
    int num_nested = omp_get_nested();
    std::cout << "Nested: " << num_nested << std::endl;
    // try commenting/uncommenting the following lines and check the value of fthreads
    if (num_nested == 0) {
        omp_set_nested(1);
        std::cout << "Nested after set: " << omp_get_nested() << std::endl;
    }
#endif

    double start,end;

    double  tol=0.0001;
    int iter=1000;
    int k=10;
    int seed=0;
    int n_threads=1;
    int nCol=10;
    int nRow=10000000;

    double **dataMatrix;
    DataFrame *df = new DataFrame();
    SaveToCSV *saveToCsv = new SaveToCSV();
    string path = "./dataset/dataset_4.csv";
    string pathResult = "./result/resultDatasetNew2_4.csv";
    dataMatrix = df->dataFrame(path, nRow,nCol);
    cout<<"CSV Read"<<endl;

    double **result = new double *[12];
    for(int i = 0; i <12; i++){
        result[i] = new double[4];
    }

    cout<<"Start program:"<<endl;
    for (int i=0;i<12;i++)
    {
        cout<<"Number of Threads: "<<n_threads<<endl;
        omp_set_num_threads(n_threads);
        start = omp_get_wtime();
        KmeansOpenMP* k2 = new KmeansOpenMP(k,iter,seed,tol);
        k2->KmeansOpenMP::fitPredict(dataMatrix, nRow, nCol,n_threads);
        end = omp_get_wtime( );
        cout << "Execution time: " << end-start <<" seconds"<<endl << endl;
        if (i==0)
        {
            for(int j=0;j<12;j++)
            {
                result[j][1]=end-start;
            }
        }
        result[i][0]=n_threads;
        result[i][2]=end-start;
        result[i][3]=result[i][1]/result[i][2];
        delete k2;
        n_threads += 1;
    }
    saveToCsv->saveToCsv(result,12,4,pathResult);

    for(int i = 0; i<nRow; i++){
        delete (dataMatrix[i]);
    }
    delete (dataMatrix);

    for(int i = 0; i<12; i++){
        delete (result[i]);
    }
    delete (result);

    delete saveToCsv;
    delete df;
    return 0;


}
