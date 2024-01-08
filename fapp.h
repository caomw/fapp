#pragma once

#include <vector>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>


/**
 * \brief 特征点检测和计算描述子
 * \param img1              输入， 查询图像
 * \param img2              输入， 参考图像
 * \param keypoints1        输出， 查询图像特征点
 * \param keypoints2        输出， 参考图像特征点
 * \param knnMatches12      输出， KNN候选匹配
 */
void featuresDetectAndMatch(
	cv::Mat img1, cv::Mat img2,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<std::vector<cv::DMatch>>& knnMatches12);


/**
 * \brief 比例测试
 * \param knnMatches        输入， KNN候选匹配
 * \param goodMatche        输出， 比例测试后的特征匹配结果
 * \param r                 输入， 比例测试阈值
 */
void ratio_match(
	std::vector<std::vector<cv::DMatch>>& knnMatches,
	std::vector<cv::DMatch>& goodMatche, float r);


/**
  * \brief 转换数据类型
  * \param goodMatches    输入，ratio后的匹配点
  * \param Keypoints1     输入，查询图像关键点
  * \param KeyPoints2     输入，训练图像关键点
  * \param X              输出，查询点坐标
  * \param Y              输出，训练点坐标
 */
void getPreData(
	std::vector<cv::DMatch>& goodMatches,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	Eigen::MatrixXd& X, Eigen::MatrixXd& Y);


/**
  * \brief 创建簇点
  * \param query          输入，查询点坐标
  * \param train          输入，训练点坐标
  * \param coordinate     输出，处理后的坐标
 */
void preTreat(
	Eigen::MatrixXd& query,
	Eigen::MatrixXd& train,
	Eigen::MatrixXd& coordinate);


/**
  * \brief FAPP方法
  * \param query           输入，查询图像点坐标
  * \param train           输入，训练图像点坐标
  * \param Keypoints1      输入，查询图像关键点
  * \param KeyPoints2      输入，训练图像关键点
  * \param putativeMatches 输入，假定匹配
  * \param correctInd      输出，正确匹配索引
 */
void FAPP(
	Eigen::MatrixXd& query,
	Eigen::MatrixXd& train,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<cv::DMatch>& putativeMatches,
	std::vector<int>& correctInd);


/**
  * \brief 计算阈值
  * \param coordinate   输入，经过聚类预处理后的坐标点
  * \param interval     输入，每一维划分的网格数量
 */
int cal_threshold(Eigen::MatrixXd& coordinate, int interval);


/**
  * \brief 计算角度
  * \param x1           输入，固定点1的x值
  * \param y1           输入，固定点1的y值
  * \param x2           输入，固定点2的x值
  * \param y2           输入，固定点2的y值
  * \param x3           输入，活动点3的x值
  * \param y3           输入，活动点3的y值
 */
double get_angle(double x1, double y1, double x2, double y2, double x3, double y3);


/**
  * \brief 划分网格空间
  * \param coordinate   输入，经过聚类预处理后的坐标点
  * \param interval     输入，每一维划分的网格数量
  * \param threshold    输入，簇中点数量的阈值
 */
std::vector<int> simple_grid(Eigen::MatrixXd& coordinate, int interval, int threshold);


/**
  * \brief 显示匹配结果
  * \param img1           输入，查询图像
  * \param img2           输入，训练图像
  * \param Keypoints      输入，查询图像关键点
  * \param KeyPoints      输入，训练图像关键点
  * \param finalMatches   输入，正确匹配结果
 */
void showMatchResult(
	cv::Mat& img1, cv::Mat& img2,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<cv::DMatch>& finalMatches,
	std::string windName);

/**
  * \brief 运行FAPP方法
  * \param Keypoints       输入，查询图像关键点
  * \param KeyPoints       输入，训练图像关键点
  * \param putativeMatches 输入，假定匹配
  * \param finalMatches    输入，正确的匹配结果
 */
void runFAPP(
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<cv::DMatch>& putativeMatches,
	std::vector<cv::DMatch>& finalMatches);










