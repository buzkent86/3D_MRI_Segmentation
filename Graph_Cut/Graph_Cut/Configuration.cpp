//
//  Configuration.cpp
//  Graph_Cut
//
//  Created by Burak Uzkent on 9/11/14.
//  Copyright (c) 2014 company. All rights reserved.
//
#include "DataSmoothCost.h"
#include "Configuration.h"
#include "GCoptimization.h"
#include <iostream>
#include <vector>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std; 

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// smoothness and data costs are set up one by one, individually grid neighborhood structure is assumed
// in this version, set data and smoothness terms using arrays grid neighborhood is set up "manually"
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GeneralGraph_DArraySArray(vector<Mat> im_stack, int width, int height, int num_labels)
{
    int num_pixels = width * height;                    // Number of pixels in the graph
    int *result = new int[num_pixels];                  // stores result of optimization
    int *data = new int[num_pixels*num_labels];         // array that stores data cost values
    int *smooth = new int[num_labels*num_labels];       // array that stores smooth cost values
    int DataWeight = 20000; int SmoothWeight = 5 * DataWeight;
    
    DataSmoothCost EnergyTerms(width/im_stack.size(),height,num_labels,im_stack);
    EnergyTerms.DataCost_NormalDist(DataWeight, data);      // Include Data Term
    EnergyTerms.EuclideanDist_Cost(DataWeight, data);
    // EnergyTerms.HorizontalDist_Cost(DataWeight, data);      // Include Horizontal Distance Term
    // EnergyTerms.VerticalDist_Cost(DataWeight, data);       // Include Vertical Distance Term
    EnergyTerms.SmoothCost(smooth, SmoothWeight);        // Include Smoothness Term
    
    try{
        GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels,num_labels);
        gc->setDataCost(data);
        gc->setSmoothCost(smooth);
        
        GraphFormer GraphNeighborhood(width/im_stack.size(),height,num_labels,im_stack,gc);
        GraphNeighborhood.HorizontalGrids();
        GraphNeighborhood.VerticalGrids();
        GraphNeighborhood.ThreeD_Neighbors();
        
        printf("\nBefore optimization energy is %d",gc->compute_energy());            // Print the energy before optimization
        gc->swap(2);                                                   // run expansion for 2 iterations. For swap use gc->swap(num_iterations);
        printf("\nAfter optimization energy is %d",gc->compute_energy());            // Print the energy after optimization
        
        //GetLabels Labels;                                               // Recover the segmentation results for each node of each image
        GraphNeighborhood.LabelAssigner(result);
        
        // Apply colormap and display the segmented image
        Mat cm_image,in_image; double min, max;
        in_image = im_stack[0];
        minMaxIdx(in_image, &min, &max);
        in_image.convertTo(in_image,CV_8UC1,255/(max-min),-min);
        applyColorMap(in_image, cm_image, COLORMAP_JET);
        imshow("Segmented Image", cm_image);
        waitKey(0);
        
        // Save the images for display purposes in Matlab
        //        for (int i = 0; i<im_stack.size() ; i++) {
        //            int img_number = i;
        //            string Result; ostringstream convert;
        //            convert << img_number + 1;
        //            string img_paths = "/Users/burakuzkent/Desktop/Research/Research_MIS/Segmented_Sheep/sheep_" + convert.str() + ".tif";
        //            imwrite(img_paths, 255 * im_stack[i]); }
        
        delete gc;
    }
	catch (GCException e){
		e.Report();
	}
	delete [] result;
	delete [] smooth;
	delete [] data;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
