#pragma once

#pragma warning(disable:4996)

#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>


namespace cvg
{

	enum class LineStyle
	{
		POINT_LINE,//原点和线
		NO_POINT_LINE,//仅有线
		CROSS_POINT_LINE//十字交叉点和线
	};

	enum class LineThickness
	{
		LINE_THICKNESS_ONE = 1,
		LINE_THICKNESS_TWO,
		LINE_THICKNESS_THREE,
		LINE_THICKNESS_FOUR,
		LINE_THICKNESS_FIVE,
		LINE_THICKNESS_SIX,
		LINE_THICKNESS_SEVEN,
		LINE_THICKNESS_EIGHT,
		LINE_THICKNESS_NINE,
		LINE_THICKNESS_TEN
	};


	enum class LineColor
	{
		LINE_COLOR_BLUE_GREEN,//蓝绿
		LINE_COLOR_DEFAULT,//墨绿色
		LINE_COLOR_RED,//红色
		LINE_COLOR_BLUE,//蓝色
		LINE_COLOR_GREEN,//绿色
		LINE_COLOR_PINK,//粉红色
		LINE_COLOR_YELLOW,//黄色
		LINE_COLOR_BLUE_RED,//黄色
		LINE_COLOR_LIGHT_GREEN,//浅绿色
		LINE_COLOR_BLACK,//黑色
		LINE_COLOR_WHILTE,//白色

	};


	/**
	 * \brief 绘制水平匹配线条
	 * \param img1      输入，查询图像
	 * \param ktp1      输入，查询特征点
	 * \param img2      输入，参考图像
	 * \param ktp2      输入，参考特征点
	 * \param matches   输入，特征匹配结果
	 * \param color     输入，线条颜色
	 * \param lineStyle 输入，绘制类型
	 * \param lineThickness 输入，线条粗细
	 */
	cv::Mat draw_horizontal_matches(
		cv::Mat& img1,
		std::vector<cv::KeyPoint>& kpt1,
		cv::Mat& img2,
		std::vector<cv::KeyPoint>& kpt2,
		std::vector<cv::DMatch>& matches,
		const LineColor lineColor = LineColor::LINE_COLOR_DEFAULT,
		const LineStyle lineStyle = LineStyle::NO_POINT_LINE,
		const LineThickness lineThickness = LineThickness::LINE_THICKNESS_ONE);


	/**
	 * \brief 绘制垂直匹配线条
	 * \param img1      输入，查询图像
	 * \param ktp1      输入，查询特征点
	 * \param img2      输入，参考图像
	 * \param ktp2      输入，参考特征点
	 * \param matches   输入，特征匹配结果
	 * \param color     输入，线条颜色
	 * \param lineStyle 输入，绘制类型
	 * \param lineThickness 输入，线条粗细
	 */
	cv::Mat draw_vertical_matches(
		cv::Mat& img1,
		std::vector<cv::KeyPoint>& kpt1,
		cv::Mat& img2,
		std::vector<cv::KeyPoint>& kpt2,
		std::vector<cv::DMatch>& matches,
		LineColor lineColor = LineColor::LINE_COLOR_DEFAULT,
		const LineStyle lineStyle = LineStyle::NO_POINT_LINE,
		const LineThickness lineThickness = LineThickness::LINE_THICKNESS_ONE);


	/**
	 * \brief 改变图像的分辨率
	 * \param src      输入，原始图像
	 * \param height   输入，目标图像的高度
	 * \param dst      输出，目标图像
	 */
	void imresize(const cv::Mat& src, const int height, cv::Mat& dst);



	void draw_cross_flag(
		cv::Mat& img, cv::Point point,
		cv::Scalar color, int size, int thickness);



	/**
	 * \brief 保存特征点
	 * \param fileName 输入，文件名
	 * \param kpts     输入， 特征点
	*/
	void save_feature_points(std::string fileName, const std::vector<cv::KeyPoint>& kpts);


	/**
	 * \brief 保存特征描述子
	 * \param fileName 输入，文件名
	 * \param desc     输入， 特征描述子
	*/
	void save_feature_descriptors(std::string fileName, const cv::Mat& desc);


	/**
	 * \brief 读取特征点
	 * \param fileName 输入，文件名
	 * \param kpts     输出，特征点
	*/
	void load_feature_points(std::string fileName, std::vector<cv::KeyPoint>& kpts);


	/**
	 * \brief 读取特征描述子
	 * \param fileName 输入，文件名
	 * \param desc     输出，特征描述子
	*/
	void load_feature_descriptors(std::string fileName, cv::Mat& desc);


	/**
	 * \brief 保存特征匹配结果
	 * \param fileName 输入，文件名
	 * \param matches  输入，特征匹配结果
	*/
	void save_feature_matches(std::string fileName, const std::vector<cv::DMatch>& matches);



	/**
	 * \brief 读取特征匹配结果
	 * \param fileName 输入，文件名
	 * \param matches  输出，特征匹配结果
	*/
	void load_feature_matches(std::string fileName, std::vector<cv::DMatch>& matches);

	/**
	 * \brief 绘制特征点
	 * \param img    输入，原始图像
	 * \param ktps   输入，特征点
	 * \param outImg 输出，绘制特征点后的图像
	*/
	void draw_keypoints(cv::Mat img, const std::vector<cv::Point2f>& kpts, cv::Mat& outImg);


}//namespace cvg

