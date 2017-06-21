//
//  Preprocessing.h
//  Graph_Cut
//
//  Created by Burak Uzkent on 8/9/14.
//  Copyright (c) 2014 company. All rights reserved.
//

#ifndef Graph_Cut_Preprocessing_h
#define Graph_Cut_Preprocessing_h

#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>

using namespace cv;

int AnisotropicNoiseDiffusion(cv::Mat image,int T, double dt, double K, int option);         // Anisotropic Noise Diffusion Function (Perona & Malik)

int ImageSharpening(cv::Mat image);         // Anisotropic Noise Diffusion Function (Perona & Malik)

#endif
