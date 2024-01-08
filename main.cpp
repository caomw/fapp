#pragma warning(disable:4996)

#include <vector>
#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "fapp.h"
#include "feature_utility.h"



int main(int argc, char* argv[])
{

	std::string file1 = "adam1.jpg";
	std::string file2 = "adam2.jpg";

	cv::Mat img1 = cv::imread(file1, cv::IMREAD_COLOR);
	cv::Mat img2 = cv::imread(file2, cv::IMREAD_COLOR);

	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;
	std::vector<std::vector<cv::DMatch>> knnMatches;
	std::vector<cv::DMatch> putativeMatches;
	std::vector<cv::DMatch> finalMatches;

	featuresDetectAndMatch(img1, img2, keypoints1, keypoints2, knnMatches);
	ratio_match(knnMatches, putativeMatches, 0.7518);

	showMatchResult(img1, img2, keypoints1, keypoints2, putativeMatches, "PutativeMatches");

	std::cout << "Putative Matches = " << putativeMatches.size() << std::endl;

	runFAPP(keypoints1, keypoints2, putativeMatches, finalMatches);

	std::cout << "Final Matches = " << finalMatches.size() << std::endl;

	showMatchResult(img1, img2, keypoints1, keypoints2, finalMatches, "FinalMatches");

	cv::waitKey(-1);

	cv::destroyAllWindows();

	return EXIT_SUCCESS;
}