//
//  Preprocessing.cpp
//  Graph_Cut
//
//  Created by Burak Uzkent on 8/9/14.
//  Copyright (c) 2014 company. All rights reserved.
//

#include "Preprocessing.h"
#include <iostream>
#include <math.h>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>

using namespace cv;

// ------------------------ Apply Anisotropic Noise Diffusion Method (Perona&Malik) ---------------------------------------------------
int AnisotropicNoiseDiffusion(Mat image, int T, double dt, double K, int option){

    int rows = image.rows; int cols = image.cols;
    Mat deltaN, deltaS, deltaE, deltaW, image_pad;
    Mat cN, cE, cS, cW;

    for (int i = 0; i < T; i++){
        
        copyMakeBorder(image, image_pad, 1, 1, 1, 1, BORDER_CONSTANT);                   // Pad input array with zeros
        deltaN = image_pad(cv::Rect_<double>(0,1,cols,rows)) - image;              // North Differences (Gradient)
        deltaS = image_pad(cv::Rect_<double>(2,1,cols,rows)) - image;              // South Differences
        deltaE = image_pad(cv::Rect_<double>(1,2,cols,rows)) - image;              // East Differences
        deltaW = image_pad(cv::Rect_<double>(1,0,cols,rows)) - image;              // West Differences
        
        // Conduction  - 1st function priveleges wide regions over small ones
        if (option == 1) {
            cN = 1 / ( 1+ ((deltaN.mul(deltaN))/(K*K)));
            cS = 1 / ( 1+ ((deltaS.mul(deltaS))/(K*K)));
            cE = 1 / ( 1+ ((deltaW.mul(deltaE))/(K*K)));
            cW = 1 / ( 1+ ((deltaW.mul(deltaW))/(K*K)));
        }
        else{   // - 2nd function priveleges high contrast regions over low ones
            exp(-(deltaN.mul(deltaN)/(K*K)),cN);
            exp(-(deltaS.mul(deltaS)/(K*K)),cS);
            exp(-(deltaE.mul(deltaE)/(K*K)),cE);
            exp(-(deltaW.mul(deltaW)/(K*K)),cW);
        }
        image += dt * (cN.mul(deltaN) + cS.mul(deltaS) + cE.mul(deltaE) + cW.mul(deltaW));       // Refine the image
    }
}
// ---------------------------------------------------------------------------------------------------------------



// ------------------------ Apply Image Sharpening Method -------------------------------------------------------
int ImageSharpening(Mat image){
    Mat image_blur;
    cv::GaussianBlur(image, image_blur, cv::Size(0,0), 5);        // Apply Gaussian Mask
    cv::addWeighted(image, 2, image_blur, -1, 0,image);          // Sharpen the Image
}
// ---------------------------------------------------------------------------------------------------------------
