//
//  Configuration.h
//  Graph_Cut
//
//  Created by Burak Uzkent on 9/11/14.
//  Copyright (c) 2014 company. All rights reserved.
//

#ifndef __Graph_Cut__Configuration__
#define __Graph_Cut__Configuration__

#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>

using namespace std;
using namespace cv;

void GeneralGraph_DArraySArray(vector<Mat> im_stack, int width, int height, int num_labels);   // Function for the General Graph Structure

#endif /* defined(__Graph_Cut__Configuration__) */
