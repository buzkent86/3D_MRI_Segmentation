//
//  DataSmoothCost.h
//  Graph_Cut
//
//  Created by Burak Uzkent on 7/30/14.
//  Copyright (c) 2014 company. All rights reserved.
//

#ifndef Graph_Cut_DataSmoothCost_h
#define Graph_Cut_DataSmoothCost_h

#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include "GCoptimization.h"

using namespace cv;
using namespace std;

class DataSmoothCost {
    
public:
    
    DataSmoothCost(int image_width, int image_height, int num_Labels, Vector<Mat> im);    // Constructor for the base class
    
    int* DataCost_IntensityDiff(vector<Mat>, int, int, int, int*);                          // Data term, modeling as intensity difference

    int* SmoothCost (int*, int);                                                         // Smooth Cost term
    
    int* DataCost_NormalDist(int, int*);                                                 // Data term, modeling as normal distribution

    int* EuclideanDist_Cost(int, int*);                                           // Spatial distance term

    int* HorizontalDist_Cost(int, int*);                                           // Spatial distance term

    int* VerticalDist_Cost(int, int*);                                           // Spatial distance term

    int width, height, numLabels;
    
    Vector<Mat> Image_Stack;
    
};


class GraphFormer: public DataSmoothCost{
    
public:
    // Constuctor for the child class
    GraphFormer(int width,int height,int numLabels,Vector<Mat> Image_Stack,GCoptimizationGeneralGraph* gc_input):DataSmoothCost(width,height,numLabels,Image_Stack){
        gc = gc_input;
    }
    
    void VerticalGrids ();          // Form the vertical grids of the graph
    
    void HorizontalGrids ();        // Form the horizontal grids of the graph
    
    void ThreeD_Neighbors ();       // Form the 3D neighbors

    void LabelAssigner (int*);      // Assign labels to each node
    
    GCoptimizationGeneralGraph* gc;
};

#endif
