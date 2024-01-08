#pragma once

#include <vector>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>


/**
 * \brief ��������ͼ���������
 * \param img1              ���룬 ��ѯͼ��
 * \param img2              ���룬 �ο�ͼ��
 * \param keypoints1        ����� ��ѯͼ��������
 * \param keypoints2        ����� �ο�ͼ��������
 * \param knnMatches12      ����� KNN��ѡƥ��
 */
void featuresDetectAndMatch(
	cv::Mat img1, cv::Mat img2,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<std::vector<cv::DMatch>>& knnMatches12);


/**
 * \brief ��������
 * \param knnMatches        ���룬 KNN��ѡƥ��
 * \param goodMatche        ����� �������Ժ������ƥ����
 * \param r                 ���룬 ����������ֵ
 */
void ratio_match(
	std::vector<std::vector<cv::DMatch>>& knnMatches,
	std::vector<cv::DMatch>& goodMatche, float r);


/**
  * \brief ת����������
  * \param goodMatches    ���룬ratio���ƥ���
  * \param Keypoints1     ���룬��ѯͼ��ؼ���
  * \param KeyPoints2     ���룬ѵ��ͼ��ؼ���
  * \param X              �������ѯ������
  * \param Y              �����ѵ��������
 */
void getPreData(
	std::vector<cv::DMatch>& goodMatches,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	Eigen::MatrixXd& X, Eigen::MatrixXd& Y);


/**
  * \brief �����ص�
  * \param query          ���룬��ѯ������
  * \param train          ���룬ѵ��������
  * \param coordinate     ���������������
 */
void preTreat(
	Eigen::MatrixXd& query,
	Eigen::MatrixXd& train,
	Eigen::MatrixXd& coordinate);


/**
  * \brief FAPP����
  * \param query           ���룬��ѯͼ�������
  * \param train           ���룬ѵ��ͼ�������
  * \param Keypoints1      ���룬��ѯͼ��ؼ���
  * \param KeyPoints2      ���룬ѵ��ͼ��ؼ���
  * \param putativeMatches ���룬�ٶ�ƥ��
  * \param correctInd      �������ȷƥ������
 */
void FAPP(
	Eigen::MatrixXd& query,
	Eigen::MatrixXd& train,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<cv::DMatch>& putativeMatches,
	std::vector<int>& correctInd);


/**
  * \brief ������ֵ
  * \param coordinate   ���룬��������Ԥ�����������
  * \param interval     ���룬ÿһά���ֵ���������
 */
int cal_threshold(Eigen::MatrixXd& coordinate, int interval);


/**
  * \brief ����Ƕ�
  * \param x1           ���룬�̶���1��xֵ
  * \param y1           ���룬�̶���1��yֵ
  * \param x2           ���룬�̶���2��xֵ
  * \param y2           ���룬�̶���2��yֵ
  * \param x3           ���룬���3��xֵ
  * \param y3           ���룬���3��yֵ
 */
double get_angle(double x1, double y1, double x2, double y2, double x3, double y3);


/**
  * \brief ��������ռ�
  * \param coordinate   ���룬��������Ԥ�����������
  * \param interval     ���룬ÿһά���ֵ���������
  * \param threshold    ���룬���е���������ֵ
 */
std::vector<int> simple_grid(Eigen::MatrixXd& coordinate, int interval, int threshold);


/**
  * \brief ��ʾƥ����
  * \param img1           ���룬��ѯͼ��
  * \param img2           ���룬ѵ��ͼ��
  * \param Keypoints      ���룬��ѯͼ��ؼ���
  * \param KeyPoints      ���룬ѵ��ͼ��ؼ���
  * \param finalMatches   ���룬��ȷƥ����
 */
void showMatchResult(
	cv::Mat& img1, cv::Mat& img2,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<cv::DMatch>& finalMatches,
	std::string windName);

/**
  * \brief ����FAPP����
  * \param Keypoints       ���룬��ѯͼ��ؼ���
  * \param KeyPoints       ���룬ѵ��ͼ��ؼ���
  * \param putativeMatches ���룬�ٶ�ƥ��
  * \param finalMatches    ���룬��ȷ��ƥ����
 */
void runFAPP(
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<cv::DMatch>& putativeMatches,
	std::vector<cv::DMatch>& finalMatches);










