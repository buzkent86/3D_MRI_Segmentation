//////////////////////////////////////////////////////////////////////////////
// Example illustrating the use of GCoptimization.cpp
//
/////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "DataSmoothCost.h"
#include "Preprocessing.h"
#include "Configuration.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // Load the cardiac images and convert them to a grayscale image
    Mat image;  vector<Mat> im_stack;
    int num_labels = 3;    // Define Number of labels

    // Load slices and then stack them in a vector
    for (int i = 450; i < 451 ; i++){
        int img_number = i;
        string Result; ostringstream convert; string img_paths;
        convert << img_number;
        if (i<10)
            img_paths = "/Users/burakuzkent/Desktop/Research/Research_MIS/Canine_Images/shortaxis000" + convert.str() + ".jpg";
        else if (i<100)
            img_paths = "/Users/burakuzkent/Desktop/Research/Research_MIS/Canine_Images/shortaxis00" + convert.str() + ".jpg";
        else
            img_paths = "/Users/burakuzkent/Desktop/Research/Research_MIS/Canine_Images/shortaxis0" + convert.str() + ".jpg";
        image = imread(img_paths,0);
        image.convertTo(image, CV_64FC1);                                   // Convert it to a double type image
        // normalize(image, image, 0, 1, NORM_MINMAX, -1, Mat() );               // Normalize the image to [0,1]
        AnisotropicNoiseDiffusion(image, 4, 0.1, 0.1, 2);              // Apply Anisotropic Noise Diffusion Method (Perona & Malik)
        im_stack.push_back(image/255);
    }

    
    // Get information about the image set
    int height = image.rows; int width = image.cols * im_stack.size();
    
    // Graph is general, we set up a neighborhood system which actually is a grid
    GeneralGraph_DArraySArray(im_stack,width,height,num_labels);  
    
    // printf("\n  Finished %d (%d) clock per sec %d",clock() CLOCKS_PER_SEC,clock(),CLOCKS_PER_SEC);
    return 0;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

