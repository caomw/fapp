
#pragma warning(disable:4996)

#include <omp.h>
#include <vector>
#include <cassert>
#include <iostream>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "feature_utility.h"


namespace cvg
{

	cv::Mat draw_horizontal_matches(
		cv::Mat& img1,
		std::vector<cv::KeyPoint>& kpt1,
		cv::Mat& img2,
		std::vector<cv::KeyPoint>& kpt2,
		std::vector<cv::DMatch>& matches,
		const LineColor lineColor /*= LINE_COLOR_DEFAULT*/,
		const LineStyle lineStyle /*= LineStyle::NO_POINT_LINE*/,
		const LineThickness lineThickness /*= LINE_THICKNESS_ONE*/)
	{

		//灰度图像转换为彩色图像

		if (img1.channels() == 1)
		{
			cv::cvtColor(img1, img1, cv::COLOR_GRAY2BGR);
		}

		if (img2.channels() == 1)
		{
			cv::cvtColor(img2, img2, cv::COLOR_GRAY2BGR);
		}

		//确定线条的颜色
		cv::Scalar color = cv::Scalar(0, 200, 100);

		switch (lineColor)
		{

		case LineColor::LINE_COLOR_DEFAULT:

			color = cv::Scalar(0, 200, 100);
			break;

		case LineColor::LINE_COLOR_RED:
			color = cv::Scalar(0, 0, 255);
			break;

		case LineColor::LINE_COLOR_BLUE:
			color = cv::Scalar(255, 0, 0);
			break;

		case LineColor::LINE_COLOR_GREEN:
			color = cv::Scalar(0, 255, 0);
			break;

		case LineColor::LINE_COLOR_PINK://粉红色
			color = cv::Scalar(255, 0, 255);
			break;

		case LineColor::LINE_COLOR_YELLOW://黄色

			color = cv::Scalar(0, 255, 255);
			break;

		case LineColor::LINE_COLOR_BLUE_RED://蓝红色
			color = cv::Scalar(255, 0, 255);
			break;

		case LineColor::LINE_COLOR_LIGHT_GREEN://浅绿色
			color = cv::Scalar(100, 255, 100);
			break;

		case LineColor::LINE_COLOR_BLUE_GREEN://蓝绿色
			color = cv::Scalar(255, 255, 0);
			break;

		case LineColor::LINE_COLOR_BLACK://黑色
			color = cv::Scalar(0, 0, 0);
			break;

		case LineColor::LINE_COLOR_WHILTE://白色
			color = cv::Scalar(255, 255, 255);
			break;

		default:
			color = cv::Scalar(0, 0, 255);

			break;
		}


		const int height = (std::max)(img1.rows, img2.rows);
		const int width = img1.cols + img2.cols;
		cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
		img1.copyTo(output(cv::Rect(0, 0, img1.cols, img1.rows)));
		img2.copyTo(output(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

		int mThreads = omp_get_max_threads();

		if (lineStyle == LineStyle::NO_POINT_LINE)
		{

#pragma omp parallel for num_threads(mThreads)
			for (long long i = 0; i < matches.size(); i++)
			{
				cv::Point2f left = kpt1[matches[i].queryIdx].pt;
				cv::Point2f right = (kpt2[matches[i].trainIdx].pt + cv::Point2f((float)img1.cols, 0.f));

				//画线

				cv::line(output, left, right, color, int(lineThickness), 16);

				/*
				cv::line(output, left, right, cv::Scalar(0, 200, 100), 1);//浅绿色
				cv::line(output, left, right, cv::Scalar(0, 255, 255));
				cv::line(output, left, right, cv::Scalar(0, 255, 255), 1);//黄色
				cv::line(output, left, right, cv::Scalar(0, 0, 255), 1);//红色
				cv::line(output, left, right, cv::Scalar(255, 0, 255), 1);//粉红色
				cv::line(output, left, right, cv::Scalar(255, 0, 0), 1);//蓝红
				cv::line(output, left, right, cv::Scalar(255, 255, 0), 1);//蓝红
				cv::line(output, left, right, cv::Scalar(100, 255, 100), 1);//浅绿色
				*/


			}
		}
		else if (lineStyle == LineStyle::POINT_LINE)
		{
#pragma omp parallel for num_threads(mThreads)
			for (long long i = 0; i < matches.size(); i++)
			{
				cv::Point2f left = kpt1[matches[i].queryIdx].pt;
				cv::Point2f right = (kpt2[matches[i].trainIdx].pt + cv::Point2f((float)img1.cols, 0.f));

				//画线
				cv::line(output, left, right, color, int(lineThickness));

				/*
				cv::line(output, left, right, cv::Scalar(255, 0, 0));
				cv::line(output, left, right, cv::Scalar(0, 255, 255), 1);//黄色
				cv::line(output, left, right, cv::Scalar(0, 0, 255), 1);//红色
				cv::line(output, left, right, cv::Scalar(255, 0, 255), 1);//粉红色
				cv::line(output, left, right, cv::Scalar(255, 0, 0), 1);//蓝红
				cv::line(output, left, right, cv::Scalar(255, 255, 0), 1);//蓝红
				cv::line(output, left, right, cv::Scalar(100, 255, 100), 1);//浅绿色
				cv::line(output, left, right, cv::Scalar(0, 200, 100), 1);//浅绿色
				*/

				//画特征点
				cv::circle(output, left, 1, cv::Scalar(0, 255, 255), 2);
				cv::circle(output, right, 1, cv::Scalar(0, 255, 255), 2);
			}

		}
		else if (lineStyle == LineStyle::CROSS_POINT_LINE)
		{

#pragma omp parallel for num_threads(mThreads)
			for (long long i = 0; i < matches.size(); i++)
			{
				cv::Point2f left = kpt1[matches[i].queryIdx].pt;
				cv::Point2f right = (kpt2[matches[i].trainIdx].pt + cv::Point2f((float)img1.cols, 0.f));

				//画线
				cv::line(output, left, right, color, int(lineThickness));

				/*
				cv::line(output, left, right, cv::Scalar(255, 0, 0));
				cv::line(output, left, right, cv::Scalar(0, 255, 255), 1);//黄色
				cv::line(output, left, right, cv::Scalar(0, 0, 255), 1);//红色
				cv::line(output, left, right, cv::Scalar(255, 0, 255), 1);//粉红色
				cv::line(output, left, right, cv::Scalar(255, 0, 0), 1);//蓝红
				cv::line(output, left, right, cv::Scalar(255, 255, 0), 1);//蓝红
				cv::line(output, left, right, cv::Scalar(100, 255, 100), 1);//浅绿色
				cv::line(output, left, right, cv::Scalar(0, 200, 100), 1);//浅绿色
				*/

				//画特征点
				draw_cross_flag(output, left, cv::Scalar(0, 0, 255), 10, int(lineThickness));
				draw_cross_flag(output, right, cv::Scalar(0, 0, 255), 10, int(lineThickness));
			}
		}

		return output;
	}





	cv::Mat draw_vertical_matches(
		cv::Mat& img1,
		std::vector<cv::KeyPoint>& kpt1,
		cv::Mat& img2,
		std::vector<cv::KeyPoint>& kpt2,
		std::vector<cv::DMatch>& matches,
		LineColor lineColor /*= LINE_COLOR_DEFAULT*/,
		const LineStyle lineStyle /*= LineStyle::NO_POINT_LINE*/,
		const LineThickness lineThickness /*= LINE_THICKNESS_ONE*/)
	{

		//灰度图像转换为彩色图像

		if (img1.channels() == 1)
		{
			cv::cvtColor(img1, img1, cv::COLOR_GRAY2BGR);
		}

		if (img2.channels() == 1)
		{
			cv::cvtColor(img2, img2, cv::COLOR_GRAY2BGR);
		}

		//确定线条的颜色
		cv::Scalar color = cv::Scalar(0, 200, 100);

		switch (lineColor)
		{

		case LineColor::LINE_COLOR_DEFAULT:

			color = cv::Scalar(0, 200, 100);
			break;

		case LineColor::LINE_COLOR_RED:
			color = cv::Scalar(0, 0, 255);
			break;

		case LineColor::LINE_COLOR_BLUE:
			color = cv::Scalar(255, 0, 0);
			break;

		case LineColor::LINE_COLOR_GREEN:
			color = cv::Scalar(0, 0, 255);
			break;

		case LineColor::LINE_COLOR_PINK://粉红色
			color = cv::Scalar(255, 0, 255);
			break;

		case LineColor::LINE_COLOR_YELLOW://黄色

			color = cv::Scalar(0, 255, 255);
			break;

		case LineColor::LINE_COLOR_BLUE_RED://蓝红色
			color = cv::Scalar(255, 255, 0);
			break;

		case LineColor::LINE_COLOR_LIGHT_GREEN://浅绿色
			color = cv::Scalar(100, 255, 100);
			break;

		default:
			color = cv::Scalar(0, 0, 255);

			break;
		}



		const int height = img1.rows + img2.rows;
		const int width = (std::max)(img1.cols, img2.cols);
		cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

		img1.copyTo(output(cv::Rect(0, 0, img1.cols, img1.rows)));
		img2.copyTo(output(cv::Rect(0, img1.rows, img2.cols, img2.rows)));


		int mThreads = omp_get_max_threads();

		if (lineStyle == LineStyle::NO_POINT_LINE)
		{

			//std::cout << "test" << std::endl;

#pragma omp parallel for num_threads(mThreads)
			for (long long i = 0; i < matches.size(); i++)
			{
				cv::Point2f left = kpt1[matches[i].queryIdx].pt;
				cv::Point2f right = (kpt2[matches[i].trainIdx].pt + cv::Point2f(0.f, (float)img1.rows));

				//画线

				cv::line(output, left, right, color, int(lineThickness), 16);//红色

				/*
				cv::line(output, left, right, cv::Scalar(0, 200, 100), lineThickness);//浅绿色
				cv::line(output, left, right, cv::Scalar(0, 255, 255), lineThickness);//黄色
				cv::line(output, left, right, cv::Scalar(0, 0, 255), lineThickness);//红色
				cv::line(output, left, right, cv::Scalar(255, 0, 255), lineThickness);//粉红色
				cv::line(output, left, right, cv::Scalar(255, 0, 0), lineThickness);//蓝红
				cv::line(output, left, right, cv::Scalar(255, 255, 0), lineThickness);//蓝红
				cv::line(output, left, right, cv::Scalar(100, 255, 100), lineThickness);//浅绿色
				cv::line(output, left, right, cv::Scalar(0, 255, 0), lineThickness);//绿色
				*/
			}

		}
		else if (lineStyle == LineStyle::POINT_LINE)
		{

#pragma omp parallel for num_threads(mThreads)
			for (long long i = 0; i < matches.size(); i++)
			{
				cv::Point2f left = kpt1[matches[i].queryIdx].pt;
				cv::Point2f right = (kpt2[matches[i].trainIdx].pt + cv::Point2f(0.f, (float)img1.rows));

				//画线
				cv::line(output, left, right, color, int(lineThickness));//浅绿色

				/*
				cv::line(output, left, right, cv::Scalar(0, 255, 255), lineThickness);//黄色
				cv::line(output, left, right, cv::Scalar(0, 0, 255), lineThickness);//红色
				cv::line(output, left, right, cv::Scalar(255, 0, 255), lineThickness);//粉红色
				cv::line(output, left, right, cv::Scalar(255, 0, 0), lineThickness);//蓝红
				cv::line(output, left, right, cv::Scalar(255, 255, 0), lineThickness);//蓝红
				cv::line(output, left, right, cv::Scalar(100, 255, 100), lineThickness);//浅绿色
				cv::line(output, left, right, cv::Scalar(0, 255, 0), lineThickness);//绿色
				*/

				//画特征点
				cv::circle(output, left, 1, cv::Scalar(0, 0, 255), 2);
				cv::circle(output, right, 1, cv::Scalar(0, 0, 255), 2);
			}
		}
		else if (lineStyle == LineStyle::CROSS_POINT_LINE)
		{

#pragma omp parallel for num_threads(mThreads)
			for (long long i = 0; i < matches.size(); i++)
			{
				cv::Point2f left = kpt1[matches[i].queryIdx].pt;
				cv::Point2f right = (kpt2[matches[i].trainIdx].pt + cv::Point2f(0.f, (float)img1.rows));

				//画线
				cv::line(output, left, right, color, int(lineThickness));//浅绿色

				/*
				cv::line(output, left, right, cv::Scalar(0, 255, 255), lineThickness);//黄色
				cv::line(output, left, right, cv::Scalar(0, 0, 255), lineThickness);//红色
				cv::line(output, left, right, cv::Scalar(255, 0, 255), lineThickness);//粉红色
				cv::line(output, left, right, cv::Scalar(255, 0, 0), lineThickness);//蓝红
				cv::line(output, left, right, cv::Scalar(255, 255, 0), lineThickness);//蓝红
				cv::line(output, left, right, cv::Scalar(100, 255, 100), lineThickness);//浅绿色
				cv::line(output, left, right, cv::Scalar(0, 255, 0), lineThickness);//绿色
				*/

				//画特征点
				draw_cross_flag(output, left, cv::Scalar(0, 0, 255), 10, int(lineThickness));
				draw_cross_flag(output, right, cv::Scalar(0, 0, 255), 10, int(lineThickness));
			}
		}


		return output;
	}


	void imresize(const cv::Mat& src, const int height, cv::Mat& dst)
	{

		assert(src.data != nullptr);
		assert(height > 0);

		double ratio = src.rows * 1.0 / height;
		int width = static_cast<int>(src.cols * 1.0 / ratio);
		cv::resize(src, dst, cv::Size(width, height));
	}


	void draw_cross_flag(
		cv::Mat& img, cv::Point point,
		cv::Scalar color, int size, int thickness)
	{
		cv::line(img, cv::Point(point.x - size / 2, point.y),
			cv::Point(point.x + size / 2, point.y), color, thickness, 8, 0);

		cv::line(img, cv::Point(point.x, point.y - size / 2),
			cv::Point(point.x, point.y + size / 2), color, thickness, 8, 0);
	}



	void save_feature_points(std::string fileName, const std::vector<cv::KeyPoint>& kpts)
	{

		assert(fileName.empty() != true);

		assert(kpts.size() > 0);

		cv::FileStorage outFile;
		outFile.open(fileName, cv::FileStorage::WRITE);

		if (outFile.isOpened())
		{
			outFile << "FeaturePoints" << kpts;

			outFile.release();
		}


	}//end of function


	void save_feature_descriptors(std::string fileName, const cv::Mat& desc)
	{
		assert(fileName.empty() != true);

		assert(desc.data != nullptr);

		cv::FileStorage outFile;
		outFile.open(fileName, cv::FileStorage::WRITE);

		if (outFile.isOpened())
		{

			outFile << "descriptors" << desc;

			outFile.release();

		}
	}


	void load_feature_points(
		std::string fileName,
		std::vector<cv::KeyPoint>& kpts)
	{
		assert(fileName.empty() != true);

		kpts.clear();

		cv::FileStorage readFile;
		readFile.open(fileName, cv::FileStorage::READ);

		if (readFile.isOpened())
		{
			readFile.getFirstTopLevelNode() >> kpts;

			readFile.release();
		}

	}



	void load_feature_descriptors(
		std::string fileName, cv::Mat& desc)
	{

		assert(fileName.empty() != true);

		cv::FileStorage readFile;
		readFile.open(fileName, cv::FileStorage::READ);

		if (readFile.isOpened())
		{
			readFile.getFirstTopLevelNode() >> desc;

			readFile.release();
		}

	}//end of function


	void save_feature_matches(
		std::string fileName,
		const std::vector<cv::DMatch>& matches)
	{

		assert(fileName.empty() != true);

		assert(matches.size() > 0);

		cv::FileStorage outFile;
		outFile.open(fileName, cv::FileStorage::WRITE);

		if (outFile.isOpened())
		{
			outFile << "matches" << matches;

			outFile.release();
		}


	}//end of function



	void load_feature_matches(
		std::string fileName,
		std::vector<cv::DMatch>& matches)
	{
		assert(fileName.empty() != true);

		matches.clear();

		cv::FileStorage readFile;
		readFile.open(fileName, cv::FileStorage::READ);

		if (readFile.isOpened())
		{
			readFile.getFirstTopLevelNode() >> matches;

			readFile.release();
		}


	}//end of function

	void draw_keypoints(cv::Mat img, const std::vector<cv::Point2f>& kpts, cv::Mat& outImg)
	{
		assert(img.data != nullptr);

		assert(kpts.size() > 0);

		img.copyTo(outImg);

		std::size_t cnt = kpts.size();

		for (std::size_t i = 0; i < cnt; i++)
		{
			cv::circle(outImg, kpts[i], 2, cv::Scalar(0, 0, 255), 2);
		}
	}

}//namespace cvg
