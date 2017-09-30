/**
 * @file SURF_detector
 * @brief SURF keypoint detection + keypoint drawing with OpenCV functions
 * @author A. Huaman
 */

#include "opencv2/opencv_modules.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#ifndef HAVE_OPENCV_NONFREE

int main(int, char**)
{
    std::cout << "The sample requires nonfree module that is not available in your OpenCV distribution." << std::endl;
    return -1;
}

#else

# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/mat.hpp"

using namespace cv;

void readme();

/**
 * @function main
 * @brief Main function
 */


void symmetryTest(const std::vector<cv::DMatch> &matches1,const std::vector<cv::DMatch> &matches2,std::vector<cv::DMatch>& symMatches )
{
    symMatches.clear();
    for (vector<DMatch>::const_iterator matchIterator1= matches1.begin();matchIterator1!= matches1.end(); ++matchIterator1)
    {
        for (vector<DMatch>::const_iterator matchIterator2= matches2.begin();matchIterator2!= matches2.end();++matchIterator2)
        {
            if ((*matchIterator1).queryIdx ==(*matchIterator2).trainIdx &&(*matchIterator2).queryIdx ==(*matchIterator1).trainIdx)
            {
                symMatches.push_back(DMatch((*matchIterator1).queryIdx,(*matchIterator1).trainIdx,(*matchIterator1).distance));
                break;
            }
        }
    }
}

void removeOutliners(std::vector<cv::DMatch> matches , std::vector< DMatch > &good_matches , std::vector<KeyPoint> keypoints_1 , std::vector<KeyPoint> keypoints_2)
{
   
    std::vector<cv::Point2f> points1, points2;
    good_matches.clear();
    for (std::vector<cv::DMatch>::const_iterator it= matches.begin(); it!= matches.end(); ++it)
    {
        //left keypoints
        float x= keypoints_1[it->queryIdx].pt.x;
        float y= keypoints_1[it->queryIdx].pt.y;
        points1.push_back(cv::Point2f(x,y));
     
        //right keypoints
        x = keypoints_2[it->trainIdx].pt.x;
        y = keypoints_2[it->trainIdx].pt.y;
        points2.push_back(cv::Point2f(x,y));
        
    }
    vector<uchar> inliers(points1.size(),0);
   
    findFundamentalMat(Mat(points1),Mat(points2),CV_FM_RANSAC,1,0.99,inliers);
    
    std::vector<uchar>::const_iterator
    itIn= inliers.begin();
    std::vector<cv::DMatch>::const_iterator
    itM= matches.begin();
    for ( ;itIn!= inliers.end(); ++itIn, ++itM)
    {
        if (*itIn)
        {
            good_matches.push_back(*itM);
        }
    }
    //printf("%s %lu\n", "as" , good_matches.size());
}

double angle(Mat m1 , Mat m2)
{
    double angle = sum(m1.mul(m2))[0];
    Mat m3 , m4;
    pow(m1 , 2 , m3);
    pow(m2 , 2 , m4);
    double a;
    a = std::sqrt((sum(m3)[0])*(sum(m4)[0]));
    angle = angle/a;
    
    return angle;
}

int similarity(std::vector< Mat > v)
{
    std::vector< std::vector< double > > v1(10 , std::vector<double>(11));
    for (int i = 0; i < 10 ; ++i)
    {
        for (int j = 0; j < 10; ++j)
        {
            v1[i][j] = angle(v[i] , v[j]);

            v1[i][10] += v1[i][j];
        }
    }

    int idx = 0;
    double max = 0;
    for (int i = 0; i < 10; ++i)
    {   
        if(v1[i][10] > max)
        {
            max = v1[i][10];
            idx = i;
        }
    }

    return idx;
}

int main()
{
 

    Mat img_1 ;
    Mat img_2 ;
   
    VideoCapture vid(0);
    
   
    while(1)
    {

        Mat im3 = imread("real.jpg", 1);
        Mat im4 = imread("forward.jpg" , 1);
        vid.read(img_1);
        img_1 = im3;
       
        //cvtColor(img_1 , img_1 , CV_RGB2GRAY);
        vid.read(img_2);
        img_2 = im4;
        //cvtColor(img_2 , img_2 , CV_RGB2GRAY);
        GaussianBlur(img_1 , img_1 , Size(3,3) ,0,0);
        GaussianBlur(img_2 , img_2 , Size(3,3),0,0);
        GaussianBlur(img_1 , img_1 , Size(3,3) ,0,0);
        GaussianBlur(img_2 , img_2 , Size(3,3),0,0);
        if( !img_1.data || !img_2.data )
        { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

        //-- Step 1: Detect the keypoints using SURF Detectorb
        int minHessian = 400;

        SurfFeatureDetector detector( minHessian );

        std::vector<KeyPoint> keypoints_1, keypoints_2;
        Mat F;
        Mat img_matches;
        vector<Point2f> selPoints1, selPoints2;
        std::vector< Mat > fmatrix;
        for(int i =0 ; i < 10 ; i++)
        {
            
            detector.detect( img_1, keypoints_1 );
            detector.detect( img_2, keypoints_2 );
            //----------------------------------------------------
            SurfDescriptorExtractor extractor;

            Mat descriptors_1, descriptors_2;
           
            vector< DMatch > matches,matches2  , symMatches;

            extractor.compute( img_1, keypoints_1, descriptors_1 );
            extractor.compute( img_2, keypoints_2, descriptors_2 );
           
            //-- Step 3: Matching descriptor vectors using FLANN matcher
            FlannBasedMatcher matcher;

            matcher.match( descriptors_1, descriptors_2, matches );
            matcher.match(descriptors_2 , descriptors_1 , matches2);
            
            //kill outliers with ransac
            std::vector< DMatch > good_matches , good_matches1;
            removeOutliners(matches , good_matches , keypoints_1 , keypoints_2);
            removeOutliners(good_matches, good_matches , keypoints_1 , keypoints_2);
            removeOutliners(good_matches, good_matches , keypoints_1 , keypoints_2);
            if(good_matches.size() > 8)
            {
                int tt = good_matches.size();
                //std::cout << "tt " << tt << "\n";

                
                drawMatches( img_1, keypoints_1,img_2, keypoints_2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
                imshow("features" , img_matches);
                waitKey(10);
                vector<int> pointIndexes1;
                vector<int> pointIndexes2;
                for (vector<DMatch>::const_iterator it= good_matches.begin();it!= good_matches.end(); ++it)
                {
                    pointIndexes1.push_back(it->queryIdx);
                    pointIndexes2.push_back(it->trainIdx);
                   
                }
               
                KeyPoint::convert(keypoints_1,selPoints1,pointIndexes1);
                KeyPoint::convert(keypoints_2,selPoints2,pointIndexes2);
                //std::cout << selPoints2.size() <<"\n";
                //std::cout << selPoints1.size() << "\n";
              
                
                if(selPoints1.size() >= 8)
                {
                    
                   
                    F = findFundamentalMat(Mat(selPoints1),Mat(selPoints2),CV_FM_RANSAC,1,1);
                    
                        
                }
                else
                {
                    continue;
                }
                
                vector<Point3f> lines1;
                computeCorrespondEpilines(Mat(selPoints1) , 1 , F , lines1);
                Mat img_21 = img_2.clone();
                for (int i = 0; i < lines1.size(); ++i)
                {
                    
                    line(img_21 , Point(0,-lines1[i].z/lines1[i].x ), Point(-lines1[i].z/lines1[i].y,0),Scalar(0) , 1);
                }


                printf("%s\n","check2" );
                if(img_21.data)
                {
                    imshow("ad" , img_21);
                }

                waitKey(10);

                fmatrix.push_back(F);
                
            } 
        }
           
        int idx = similarity(fmatrix);
        printf("%s %lu %d\n", "size" , fmatrix.size() , idx);
        F = fmatrix[idx];
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                std::cout << F.at<double>(i,j) << " ";
            }
            std::cout << "\n";
        }
        printf("%s\n","\n" );
        
        fmatrix.clear();

        vector<Point3f> lines;
        computeCorrespondEpilines(Mat(selPoints1) , 1 , F , lines);

        for (int i = 0; i < lines.size(); ++i)
        {
            
            line(img_2 , Point(0,-lines[i].z/lines[i].x ), Point(-lines[i].z/lines[i].y,0),Scalar(0) , 1);
        }



        imshow("asd" , img_2);
       
        
        waitKey(10);


    }

    return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SURF_detector <img1> <img2>" << std::endl; }

#endif