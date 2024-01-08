
#pragma warning(disable:4996)

#define _USE_MATH_DEFINES

#include <math.h>

#include <iostream>
#include <algorithm>
#include <unordered_map>

#include <opencv2/opencv.hpp>


#include "fapp.h"
#include "feature_utility.h"


void featuresDetectAndMatch(
	cv::Mat img1, cv::Mat img2,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<std::vector<cv::DMatch>>& knnMatches12)
{
	cv::Mat descriptors1, descriptors2;
	cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
	sift->detect(img1, keypoints1);
	sift->detect(img2, keypoints2);
	sift->compute(img1, keypoints1, descriptors1);
	sift->compute(img2, keypoints2, descriptors2);

	cv::FlannBasedMatcher matcher;
	std::vector<std::vector<cv::DMatch>> matches;
	matcher.knnMatch(descriptors1, descriptors2, knnMatches12, 2);
}

void ratio_match(
	std::vector<std::vector<cv::DMatch>>& knnMatches,
	std::vector<cv::DMatch>& goodMatche, float r)
{
	assert(knnMatches.size() > 0);
	std::size_t cnt = knnMatches.size();
	goodMatche.clear();
	for (std::size_t i = 0; i < cnt; i++)
	{
		cv::DMatch match1 = knnMatches[i][0];
		cv::DMatch match2 = knnMatches[i][1];
		if (match1.distance < r * match2.distance)
		{
			goodMatche.push_back(match1);
		}
	}

}

void getPreData(
	std::vector<cv::DMatch>& goodMatches,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	Eigen::MatrixXd& X, Eigen::MatrixXd& Y)
{
	int len1 = goodMatches.size();

	//std::cout << "goodMatches.size() = " << goodMatches.size() << std::endl;

	X.resize(len1, 2);
	Y.resize(len1, 2);

	for (int i = 0; i < len1; i++)
	{
		int x = goodMatches[i].queryIdx;

		X(i, 0) = keypoints1[x].pt.x;
		X(i, 1) = keypoints1[x].pt.y;

		int y = goodMatches[i].trainIdx;
		Y(i, 0) = keypoints2[y].pt.x;
		Y(i, 1) = keypoints2[y].pt.y;
	}

}


void preTreat(
	Eigen::MatrixXd& query,
	Eigen::MatrixXd& train,
	Eigen::MatrixXd& coordinate)
{

	Eigen::MatrixXd y;
	Eigen::MatrixXd sin;
	Eigen::MatrixXd dist;
	Eigen::MatrixXd xtemp;

	coordinate.resize(query.rows(), 2);
	y = train.col(1).array() - query.col(1).array();
	xtemp = (train.col(0).array() + query.col(0).maxCoeff()) - query.col(0).array();//
	dist = (y.array().square() + xtemp.array().square()).sqrt();

	sin = ((y.array() / dist.array()) + 1) * (dist.maxCoeff() / 2);
	coordinate << sin, dist;

}


//计算阈值
int cal_threshold(Eigen::MatrixXd& coordinate, int interval)
{
	int gridLen = std::ceil(coordinate.maxCoeff() / (double)interval);
	Eigen::MatrixXi pointsGridCoordinate;
	Eigen::RowVectorXi pointsGridInd;

	pointsGridCoordinate = (coordinate.array() / gridLen).cast<int>();
	pointsGridInd = pointsGridCoordinate.col(0).array() * interval + pointsGridCoordinate.col(1).array();

	std::vector<int> vecPointsGridInd(pointsGridInd.data(), pointsGridInd.data() + pointsGridInd.cols());

	std::unordered_map<int, std::vector<int>> mapVec;
	for (int i = 0; i < vecPointsGridInd.size(); i++)
	{
		mapVec[vecPointsGridInd[i]].push_back(i);
	}

	std::vector<int> correctIndex;

	std::vector<int> vect;
	std::vector<int> numList;
	for (auto it = mapVec.begin(); it != mapVec.end(); it++)
	{
		vect = it->second;
		numList.push_back(vect.size());
	}

	std::sort(numList.begin(), numList.end(), std::greater<int>());

	int partNum = 0;
	if (numList.size() > 5)
	{
		for (int i = 0; i < 5; i++)
		{
			partNum = partNum + numList[i];
		}
	}
	else
	{
		return 2;
	}


	double dataDensity;
	dataDensity = (double)partNum / coordinate.rows();

	//std::cout << "dataDensity: = " << dataDensity << std::endl;

	int threshold = 0;
	double minAngle = 360;
	double angle = 0;

	double x1 = 0;
	double y1 = (double)numList[0];
	double x2 = (double)(numList.size() - 1);
	double y2 = (double)numList[numList.size() - 1];

	for (int i = 0; i < numList.size() - 2; i++)
	{
		angle = get_angle(x1, y1, x2, y2, (double)(i + 1), (double)numList[i + 1]);
		if (angle < minAngle)
		{
			threshold = numList[i + 1];
			minAngle = angle;
		}
	}

	//std::cout << "knee point:" << threshold << std::endl;

	double dataDensityAlias = dataDensity;

	if (threshold > dataDensityAlias * 10)
	{
		threshold = std::ceil(threshold * dataDensityAlias);
	}

	threshold = threshold + 1;

	return threshold;

}

double get_angle(double x1, double y1, double x2, double y2, double x3, double y3)
{
	double theta1 = std::abs(std::atan2(y1 - y3, x1 - x3) * 180.0 / M_PI);
	double theta2 = std::abs(std::atan2(y2 - y3, x2 - x3) * 180.0 / M_PI);
	return theta1 + theta2;
}

std::vector<int> simple_grid(Eigen::MatrixXd& coordinate, int interval, int threshold)
{
	int gridLen = std::ceil(coordinate.maxCoeff() / (double)interval); //计算每个网格长度
	Eigen::MatrixXi pointsGridCoordinate; //点在网格中的坐标，如某一点在第3行第2列网格
	Eigen::RowVectorXi pointsGridInd; //点在网格的索引

	pointsGridCoordinate = (coordinate.array() / gridLen).cast<int>(); //计算点在网格的坐标
	pointsGridInd = pointsGridCoordinate.col(0).array() * interval + pointsGridCoordinate.col(1).array(); //计算点在网格的索引

	std::vector<int> vecPointsGridInd(pointsGridInd.data(), pointsGridInd.data() + pointsGridInd.cols()); //转换数据类型

	//使用无序图速度快
	//将点所在网格索引作为map的key值，将该网格中所有点的索引放入map的value值，这样就实现了将点划分在网格中
	std::unordered_map<int, std::vector<int>> mapVec;
	for (int i = 0; i < vecPointsGridInd.size(); i++)
	{
		mapVec[vecPointsGridInd[i]].push_back(i);
	}

	std::vector<int> correctIndex;

	//根据网格中的点数量是否>=阈值，来决定是否将该网格中的点视为正确点
	std::vector<int> vect;
	for (auto it = mapVec.begin(); it != mapVec.end(); it++)
	{
		vect = it->second;

		if (vect.size() >= threshold)
		{
			correctIndex.insert(correctIndex.end(), vect.begin(), vect.end());
		}
	}

	return correctIndex;
}


void showMatchResult(
	cv::Mat& img1, cv::Mat& img2,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<cv::DMatch>& finalMatches,
	std::string windName)
{
	cv::Mat finalMatchingResult = cvg::draw_horizontal_matches(img1, keypoints1, img2, keypoints2, finalMatches,
		cvg::LineColor::LINE_COLOR_YELLOW, cvg::LineStyle::NO_POINT_LINE, cvg::LineThickness::LINE_THICKNESS_ONE);

	cv::namedWindow(windName, cv::WINDOW_NORMAL);
	cv::imshow(windName, finalMatchingResult);

	//cv::waitKey(-1);
}

void FAPP(
	Eigen::MatrixXd& query,
	Eigen::MatrixXd& train,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<cv::DMatch>& putativeMatches,
	std::vector<int>& correctInd)
{
	Eigen::MatrixXd coordinate;
	preTreat(query, train, coordinate);

	int gridNum = 0;
	int threshold = 0;
	int pointNum = coordinate.rows();

	gridNum = std::sqrt((double)pointNum) * 2;

	//计算阈值
	threshold = cal_threshold(coordinate, gridNum);

	correctInd = simple_grid(coordinate, gridNum, threshold);
}

void runFAPP(
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<cv::DMatch>& putativeMatches,
	std::vector<cv::DMatch>& finalMatches)
{
	Eigen::MatrixXd query;
	Eigen::MatrixXd train;
	getPreData(putativeMatches, keypoints1, keypoints2, query, train);

	std::vector<int> correctInd;
	FAPP(query, train, keypoints1, keypoints2, putativeMatches, correctInd);

	std::vector<cv::DMatch> CorrectMatchs;
	for (int i = 0; i < correctInd.size(); i++)
	{
		finalMatches.push_back(putativeMatches[correctInd[i]]);
	}
}
