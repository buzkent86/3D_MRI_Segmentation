///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
// Example illustrating the use of GCoptimization.cpp
//
/////////////////////////////////////////////////////////////////////////////
//
//  Optimization problem:
//  is a set of sites (pixels) of width 10 and hight 5. Thus number of pixels is 50
//  grid neighborhood: each pixel has its left, right, up, and bottom pixels as neighbors
//  7 labels
//  Data costs: D(pixel,label) = 0 if pixel < 25 and label = 0
//            : D(pixel,label) = 10 if pixel < 25 and label is not  0
//            : D(pixel,label) = 0 if pixel >= 25 and label = 5
//            : D(pixel,label) = 10 if pixel >= 25 and label is not  5
// Smoothness costs: V(p1,p2,l1,l2) = min( (l1-l2)*(l1-l2) , 4 )
// Below in the main program, we illustrate different ways of setting data and smoothness costs
// that our interface allow and solve this optimizaiton problem

// For most of the examples, we use no spatially varying pixel dependent terms.
// For some examples, to demonstrate spatially varying terms we use
// V(p1,p2,l1,l2) = w_{p1,p2}*[min((l1-l2)*(l1-l2),4)], with
// w_{p1,p2} = p1+p2 if |p1-p2| == 1 and w_{p1,p2} = p1*p2 if |p1-p2| is not 1

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "GCoptimization.h"
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// smoothness and data costs are set up one by one, individually grid neighborhood structure is assumed
void GridGraph_Individually(cv::Mat image,int width,int height,int num_pixels,int num_labels)
{
    
    int *result = new int[num_pixels];   // stores result of optimization
    int Datacost; float intensity;       // stores data costs and intensity values of my pixels
    
	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);
        
        
        // ------------------------  SEGMENTATION FOR 2 -------------------------------------------------
		// first set up data costs individually
		for ( int i = 0; i < num_pixels; i++ )
			for (int l = 0; l < num_labels; l++ )
                if(  l == 0 ){
                    intensity = (float) image.at<unsigned char>(i);   // Get the intensity values of the pixels
                    Datacost = abs(128 - (int) intensity);   // Estimate the data cost
                    gc->setDataCost(i,l,Datacost);
                }
                else{
                    intensity = (float) image.at<unsigned char>(i);   // Get the intensity values of the pixels
                    Datacost = abs(0 - (int) intensity);     // Estimate the data cost
                    gc->setDataCost(i,l,Datacost);
				}
        //--------------------------------------------------------------------------------------------------
        
        // -----------------------  SEGMENTATION FOR MORE THAN 2----------------------------------------
        // first set up data costs individually
        //        for ( int i = 0; i < num_pixels; i++ )
        //            for (int l = 0; l < num_labels; l++ )
        //                if(  l == 0 ){
        //                    intensity = (float) image.at<unsigned char>(i);   // Get the intensity values of the pixels
        //                    Datacost = abs(50 - (int) intensity);   // Estimate the data cost
        //                    gc->setDataCost(i,l,Datacost);
        //                }
        //                else if(  l == 1 ){
        //                    intensity = (float) image.at<unsigned char>(i);   // Get the intensity values of the pixels
        //                    Datacost = abs(100 - (int) intensity);   // Estimate the data cost
        //                    gc->setDataCost(i,l,Datacost);
        //                }
        //                else if(  l == 2 ){
        //                    intensity = (float) image.at<unsigned char>(i);   // Get the intensity values of the pixels
        //                    Datacost = abs(150 - (int) intensity);   // Estimate the data cost
        //                    gc->setDataCost(i,l,Datacost);
        //                }
        //                else{
        //                    intensity = (float) image.at<unsigned char>(i);   // Get the intensity values of the pixels
        //                    Datacost = abs(200 - (int) intensity);     // Estimate the data cost
        //                    gc->setDataCost(i,l,Datacost);
        //                }
        // ----------------------------------------------------------------------------------------------------
        
		// next set up smoothness costs individually
		for ( int l1 = 0; l1 < num_labels; l1++ )
			for (int l2 = 0; l2 < num_labels; l2++ ){
				int cost = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;    // Set the smoothness costs
				gc->setSmoothCost(l1,l2,cost);
			}
        
        printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		printf("\nAfter optimization energy is %d",gc->compute_energy());
        
        // Get the pixel labels and transfer them to the original image
		for ( int  i = 0; i < num_pixels; i++ ){
			result[i] = gc->whatLabel(i);
            image.at<unsigned char>(i) = (255/(num_labels-1))*(*(result+i));
        }
        
        // Apply colormap and display the segmented image
        Mat cm_image;
        applyColorMap(image, cm_image,COLORMAP_JET);
        imshow("Segmented Image",cm_image);
        waitKey(0);
        
        delete gc;
	}
	catch (GCException e){
		e.Report();
	}
	delete [] result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // Load the cardiac image and convert it to a grayscale image
    Mat image = imread("/Users/burakuzkent/Desktop/Research_MIS/heart_images/shortaxis0095.jpg",0);
    // Mat image = imread("/Users/burakuzkent/Desktop/Research_MIS/Mandrill.jpg",0);
    normalize(image, image, 0, 255, NORM_MINMAX, -1, Mat() );
    
    // Get information about the image
    int height = image.rows; int width = image.cols;
    int num_pixels = width*height; int num_labels = 2;
    
    // Display the original image
    imshow("Original Image", image);
    waitKey(0);
    
    // smoothness and data costs are set up one by one, individually
	GridGraph_Individually(image,width,height,num_pixels,num_labels);
    
	printf("\n  Finished %d (%d) clock per sec %d",clock()/CLOCKS_PER_SEC,clock(),CLOCKS_PER_SEC);
    
	return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

