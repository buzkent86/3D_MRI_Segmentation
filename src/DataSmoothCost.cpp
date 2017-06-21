//
//  DataSmoothCost.cpp
//  Graph_Cut
//
//  Created by Burak Uzkent on 7/30/14.
//  Copyright (c) 2014 company. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "DataSmoothCost.h"
#include "Configuration.h"
#include "GCoptimization.h"
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"


// ------------------------ Constructor for the DataSmoothCost (Base) Class -------------------------------------
DataSmoothCost::DataSmoothCost(int image_width, int image_height, int num_Labels, Vector<Mat> im){
    width = image_width;
    height = image_height;
    Image_Stack = im;
    numLabels = num_Labels;
}
// --------------------------------------------------------------------------------------------------------------

// ------------------------ Data Cost Function for 2 LABELS (Intensity Difference) ------------------------------
int* DataSmoothCost::DataCost_IntensityDiff(vector<Mat> Image_Stack,int width,int height, int DataWeight, int* data){
    
    // Set up data costs individually
    double Datacost, intensity;                       // Intensity and data cost parameters
    int numImages = Image_Stack.size(), index;                // Number of images (slices)
    
    double c[numLabels];
    c[0] = 0.0; c[1] = 0.5; c[2] = 0.2;                // Mean intensity values for the classes

    for (int i_num=0; i_num < numImages; i_num++)                      // Go through each slice
        for (int i = 0; i < width*height; i++ ){           // Go through all the pixels of the slices
            intensity = (double) Image_Stack[i_num].at<unsigned char>(i);      // Get the intensity values of the pixels
            for (int l = 0; l < numLabels; l++ ){                      // Iterate through all the labels
                Datacost = pow(c[l] - intensity,2);                  // Estimate the data cost
                index = i*numLabels+l+i_num*(width*height)*numLabels;
                data[index] = DataWeight * Datacost;
            }
        }
    return data;
}
// ---------------------------------------------------------------------------------------------------------------

// ------------------------ Data Cost Function (Normal Distribution) ---------------------------------------------
int* DataSmoothCost::DataCost_NormalDist(int DataWeight, int* data){
    
    double Datacost, intensity;                       // Intensity and data cost parameters
    int index;
    int numImages = Image_Stack.size();               // Number of images (slices)

    double c[numLabels],s[numLabels];
    c[0] = 0.00; c[1] = 0.35;  c[2] = 0.10;                        // Class mean values
    s[0] = 0.05; s[1] = 0.20; s[2] = 0.05;                        // Class deviation values
    
    for ( int i_num=0; i_num < numImages; i_num++)                           // Go through each slice
        for ( int i = 0; i < width*height; i++ ){                // Go through all the nodes in each slice
            intensity = Image_Stack[i_num].at<double>(i);        // Get the intensity values of the nodes
            for (int l = 0; l < numLabels; l++ ){                            // Iterate through all the labels
                Datacost = -log(1/(s[l]*2.5)*exp(-pow(intensity-c[l],2)/(2*pow(s[l],2))) );    // Compute the pdf value
                index = i*numLabels+l+i_num*(width*height)*numLabels;    // Get the index value
                data[index] = DataWeight * 5 * Datacost;           // Assign Data Cost
            }
        }
    return data;
}
// -----------------------------------------------------------------------------------------------------------------


// ------------------ Spatial EUCLIDEAN Distance Cost Term (Normal Distribution) -----------------------------------
int* DataSmoothCost::EuclideanDist_Cost(int DataWeight, int* data){
    
    double rows[width], cols[height], Distance[width*height];
    double DistanceCost; double weight; double fac=20; double fac2=0.2;
    int numImages = Image_Stack.size();                   // Number of images (slices)
    int index; int it=0;
    
    double c[numLabels],s[numLabels];
    c[0] = 0.50; c[1] = 0.55; c[2] = 4.15;               // Class mean values
    s[0] = 0.50; s[1] = 0.10; s[2] = 3.0;               // Class deviation values
    double inc_r = 2/(double) width;                  // Get the step size to form a distance matrix (rows)
    double inc_c = 2/(double) height;                 // Get the step size to form a distance matrix (columns)
    
    for ( double i = -1; i <= 1; i+=inc_r){
        rows[it] = i;                            // Assign distance value for each node in a row
        it += 1;
    }
    it = 0;
    
    for ( double i = -1; i <= 1; i+=inc_c){
        cols[it] = i * -1;                       // Assign distance value for each node in a column
        it += 1;
    }
    it = 0;
    
    // Calculate the distance value for every pixel location
    for (int i = 0; i < width; i++)
        for( int j = 0; j< height; j++){
            Distance[it] = sqrt(pow(rows[i],2)+pow(cols[j],2));   // Compute the distance value for every pixel location
            it += 1;
        }
    
    for ( int i_num=0; i_num < numImages; i_num++)            // Go through each slice
    {
        //weight = exp(-fac*abs(fac2*(i_num-255)/255));
        if (i_num > 100 && i_num < 400)
            weight = 1;
        else weight = 0;
        cout << 5 * weight << endl;
        for ( int i = 0; i < width*height; i++ )              // Go through all the pixels of the slices
            for (int l = 0; l < numLabels; l++){
                index = (i)*numLabels+l+i_num*(width*height)*numLabels;   // Get the index value
                DistanceCost = -log(1/(s[l]*2.5)*exp(-pow(Distance[i]-c[l],2)/(2*pow(s[l],2))));
                
                data[index] += DataWeight * 3.00 * DistanceCost;     // Assign Data Cost
            }
    }
    return data;
}
// ------------------------------------------------------------------------------------------------------------------

// ------------------ Spatial HORIZONTAL Distance Cost Term (Normal Distribution) ----------------------------------
int* DataSmoothCost::HorizontalDist_Cost(int DataWeight, int* data){
    
    double rows[width], Distance[width*height];
    double DistanceCost;
    int numImages = Image_Stack.size();                   // Number of images (slices)
    int index; int it=0;
    
    double c[numLabels],s[numLabels];
    c[0] = 0.5; c[1] = 0.40; c[2] = 0.60;               // Class mean values
    s[0] = 0.5; s[1] = 0.20; s[2] = 0.30;               // Class deviation values
    double inc = 2/(double) width;                  // Get the step size to form a distance matrix
    
    for ( double i = -1; i <= 1; i+=inc){
        rows[it] = i;                            // Assign distance value for each node in a row
        it += 1;
    }
    it = 0;
    
    // Calculate the distance value for every pixel location
    for (int i = 0; i < width; i++)
        for( int j = 0; j< height; j++){
            Distance[it] = abs(rows[i]);   // Compute the distance value for every pixel location
            it += 1;
        }
    
    for ( int i_num=0; i_num < numImages; i_num++)            // Go through each slice
        for ( int i = 0; i < width*height; i++ )              // Go through all the pixels of the slices
            for (int l = 0; l < numLabels; l++){
                index = (i)*numLabels+l+i_num*(width*height)*numLabels;   // Get the index value
                DistanceCost = -log(1/(s[l]*2.5)*exp(-pow(Distance[i]-c[l],2)/(2*pow(s[l],2))));
                data[index] += DataWeight * 0.10 * DistanceCost;     // Assign Data Cost
            }
    return data;
}
// ------------------------------------------------------------------------------------------------------------------

// ------------------ Spatial VERTICAL Distance Cost Term (Normal Distribution) -----------------------------------
int* DataSmoothCost::VerticalDist_Cost(int DataWeight, int* data){
    
    double cols[height], Distance[width*height];
    double DistanceCost;
    int numImages = Image_Stack.size();                   // Number of images (slices)
    int index; int it=0;
    
    double c[numLabels],s[numLabels];
    c[0] = 0.5; c[1] = 0.30; c[2] = 0.60;               // Class mean values
    s[0] = 0.5; s[1] = 0.15; s[2] = 0.30;              // Class deviation values
    double inc = 2/(double) height;                   // Get the step size to form a distance matrix
    
    for ( double i = -1; i <= 1; i+=inc){
        cols[it] = i * -1;                        // Assign distance value for each node in a column
        it += 1;
    }
    it = 0;
    
    // Calculate the distance value for every pixel location
    for (int i = 0; i < width; i++)
        for( int j = 0; j< height; j++){
            Distance[it] = abs(cols[j]);   // Compute the distance value for every pixel location
            it += 1;
        }
    
    for ( int i_num=0; i_num < numImages; i_num++)            // Go through each slice
        for ( int i = 0; i < width*height; i++ )              // Go through all the pixels of the slices
            for (int l = 0; l < numLabels; l++){
                index = (i)*numLabels+l+i_num*(width*height)*numLabels;   // Get the index value
                DistanceCost = -log(1/(s[l]*2.5)*exp(-pow(Distance[i]-c[l],2)/(2*pow(s[l],2))));
                data[index] += DataWeight * 0.00 * DistanceCost;     // Assign Data Cost
            }
    return data;
}
// ------------------------------------------------------------------------------------------------------------------

// ------------------------ Smooth Cost Function for n Labels ------------------------------------------------------
int* DataSmoothCost::SmoothCost(int* smooth, int SmoothWeight){
    
    // Next set up the array for smooth costs
    for ( int l1 = 0; l1 < numLabels; l1++ )
        for (int l2 = 0; l2 < numLabels; l2++ ){
            if (l1 == l2)
                smooth[l1+l2*numLabels]  = SmoothWeight * 0;        // If the labels are same
            else
                smooth[l1+l2*numLabels]  = SmoothWeight * 1;        // If the labels are different
        }
    smooth[5] = SmoothWeight * 10;
    smooth[7] = SmoothWeight * 10;
    return smooth;
}
// ----------------------------------------------------------------------------------------------------------------


// ------------------------ Form the Horizontal Grids of the Graph ------------------------------------------------
void GraphFormer::HorizontalGrids(){
    
    // First set up horizontal neighbors
    int numImages = Image_Stack.size(), site_1, site_2;
    double intensity_1, intensity_2; double Wp_n;  //int r_width = width/numImages;
    
    
    for (int i_n = 0; i_n < numImages; i_n++)  // Go through each slice
        for (int y = 0; y < height; y++ )      // Iterate each column and row
            for (int  x = 1; x < (width); x++ ){
                site_1 = x+y*(width)+i_n*(width*height);
                site_2 = (x-1)+y*(width)+i_n*(width*height);
                //intensity_1 = (double) Image_Stack[i_n].at<unsigned char>(site_1-i_n*width*height);    // Intensity of the Site 1
                //intensity_2 = (double) Image_Stack[i_n].at<unsigned char>(site_2-i_n*width*height);    // Intensity of the Site 2
                //Wp_n = exp( pow((intensity_1-intensity_2),2) );                                        // Estimate Beta Term
                gc->setNeighbors(site_1,site_2);
            }

}
// -------------------------------------------------------------------------------------------------------------


// ------------------------ Form the Vertical Grids of the Graph -----------------------------------------------
void GraphFormer::VerticalGrids(){
    
    // Next set up vertical neighbors
    int numImages = Image_Stack.size(), site_1, site_2;
    double intensity_1, intensity_2; double Wp_n;

    for (int i_n = 0; i_n < numImages; i_n++)   // Go through each slice
        for (int y = 1; y < height; y++ )     // Iterate each column and row
            for (int  x = 0; x < (width); x++ ){
                site_1  = x+y*(width)+i_n*(width*height);
                site_2  = x+(y-1)*(width)+i_n*(width*height);
                //intensity_1 = (double) Image_Stack[i_n].at<unsigned char>(site_1-i_n*width*height);    // Intensity of the Site 1
                //intensity_2 = (double) Image_Stack[i_n].at<unsigned char>(site_2-i_n*width*height);    // Intensity of the Site 2
                //Wp_n = exp( -pow((intensity_1-intensity_2),2) );                                  // Estimate Beta Term
                gc->setNeighbors(site_1,site_2);
            }
}
// ----------------------------------------------------------------------------------------------------------------


// ------------------------ Form the 3D Neighbors of the nodes ----------------------------------------------------
void GraphFormer::ThreeD_Neighbors(){
    
    // Next set up vertical neighbors
    int numImages = Image_Stack.size(), site_1, site_2;
    double intensity_1, intensity_2; double Wp_n;
    
    for (int i_n = 0; i_n < numImages-1; i_n++)       // Go through each slice
        for (int y = 0; y < height*width; y++ ){      // Iterate each column and row
            site_1  = y+width*height*i_n;
            site_2  = y+width*height*(i_n+1);
            //intensity_1 = (double) Image_Stack[i_n].at<unsigned char>(site_1);    // Intensity of the Site 1
            //intensity_2 = (double) Image_Stack[i_n+1].at<unsigned char>(site_2);    // Intensity of the Site 2
            //Wp_n = exp( -pow((intensity_1-intensity_2),2) );                       // Estimate Beta Term
            gc->setNeighbors(site_1,site_2);
        
        }
}
// ---------------------------------------------------------------------------------------------------------------


// ------------------------ Get the labels from the Segmentation Result -------------------------------------------
void GraphFormer::LabelAssigner(int* result){
    
    // Get the pixel labels and transfer them to the original image
    int numImages = Image_Stack.size();
    for ( int i_n = 0 ; i_n < numImages; i_n++ )                                    // Iterate through each slice
        for ( int  i = 0 ; i < (width*height); i++ ){                                // Iterate through each node of each slice
            result[i+(i_n*width*height)] = gc->whatLabel(i+(i_n*width*height));       // Get the corresponding label
            Image_Stack[i_n].at<double>(i) = 1/double(numLabels-1) * (*(result+i+(i_n*width*height)));  // Assign it to the corresponding slice
        }
}
// --------------------------------------------------------------------------------------------------------------

