// Try_2D_ICP_OpenCV3.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "core.hpp"

#include <iostream>
#include <fstream>
#include <float.h>

using namespace std;
using namespace cv;

// 降采样参数
#define REF_DOWN_SAMPLE_RATE 5
#define NEW_DOWN_SAMPLE_RATE 5
// ICP迭代阈值
#define ITER_MAX 100
#define ITER_THRESH 0.001


int main()
{
	// 0. 得到边缘二维点云
	// 0.1 载入两幅刀具图像
	Mat I1 = imread("knife1.bmp", IMREAD_UNCHANGED);
	Mat I2 = imread("knife2.bmp", IMREAD_UNCHANGED);

	// 0.2 边角黑边去除 - 注意，不管是 colRange(a, b) 还是 rolRange(a, b)，
	// 所选的部分都是包含 a 不包含 b 的。
	I1.colRange(0, 80) = 255;
	I1.colRange(1224, I1.cols) = 255;
	I2.colRange(0, 80) = 255;
	I2.colRange(1224, I1.cols) = 255;

	// 0.3 阈值处理
	// 注：
	//		在 THRESH_BINARY 模式下， 128 是阈值 255 是高于阈值应该赋的值
	//		与 MATLAB 不同，OpenCV 的阈值处理输出的矩阵会与输入矩阵类型相同，所以可以直接边缘检测
	Mat T1, T2;
	threshold(I1, T1, 128, 255, THRESH_BINARY);
	threshold(I2, T2, 128, 255, THRESH_BINARY);
	
	// 0.4 边缘检测
	Mat Edge1, Edge2;
	int edge_thresh = 50;
	Canny(T1, Edge1, edge_thresh, edge_thresh * 3);
	Canny(T2, Edge2, edge_thresh, edge_thresh * 3);


	// 0.5 提取所有非零点作为点云数据备用值
	vector<Point> ref_pt_vec;
	vector<Point> new_pt_vec;
	for (int i = 0; i < Edge1.rows; i++)
	{
		for (int j = 0; j < Edge1.cols; j++)
		{
			if (Edge1.at<uchar>(i, j) != 0) // 这个地方这个char请验证
			{
				ref_pt_vec.push_back(Point(j, i));
			}
			if (Edge2.at<uchar>(i, j) != 0) // 这个地方这个char请验证
			{
				new_pt_vec.push_back(Point(j, i));
			}
		}
	}

	
	// 0.6 根据这两个点云的尺寸及预设降采样参数进行降采样
	int ref_pt_num, new_pt_num;
	ref_pt_num = ref_pt_vec.size();
	new_pt_num = new_pt_vec.size();

	Mat ICP_ref_pts = Mat(ref_pt_num/REF_DOWN_SAMPLE_RATE, 2, CV_32FC1);
	Mat ICP_new_pts = Mat(new_pt_num/NEW_DOWN_SAMPLE_RATE, 2, CV_32FC1);

	for (int i = 0; i < ref_pt_num / REF_DOWN_SAMPLE_RATE; i++)
	{
		ICP_ref_pts.at<float>(i, 0) = ref_pt_vec[i * REF_DOWN_SAMPLE_RATE].x;
		ICP_ref_pts.at<float>(i, 1) = ref_pt_vec[i * REF_DOWN_SAMPLE_RATE].y;
	}
	for (int i = 0; i < new_pt_num / NEW_DOWN_SAMPLE_RATE; i++)
	{
		ICP_new_pts.at<float>(i, 0) = new_pt_vec[i * NEW_DOWN_SAMPLE_RATE].x;
		ICP_new_pts.at<float>(i, 1) = new_pt_vec[i * NEW_DOWN_SAMPLE_RATE].y;
	}

	// 调试用
	//ofstream ref_data_file("ref.txt", ios_base::out);
	//ofstream new_data_file("new.txt", ios_base::out);

	//for (int r = 0; r < ICP_ref_pts.rows; r++)
	//{
	//	for (int c = 0; c < ICP_ref_pts.cols; c++)
	//	{
	//		float data = ICP_ref_pts.at<float>(r, c);  //读取数据，at<type> - type 是矩阵元素的具体数据格式  
	//		ref_data_file << data << "\t";   //每列数据用 tab 隔开  
	//	}
	//	ref_data_file << endl;  //换行  
	//}
	//ref_data_file.flush();
	//ref_data_file.close();

	//for (int r = 0; r < ICP_new_pts.rows; r++)
	//{
	//	for (int c = 0; c < ICP_new_pts.cols; c++)
	//	{
	//		float data = ICP_new_pts.at<float>(r, c);  //读取数据，at<type> - type 是矩阵元素的具体数据格式  
	//		new_data_file << data << "\t";   //每列数据用 tab 隔开  
	//	}
	//	new_data_file << endl;  //换行  
	//}
	//new_data_file.flush();
	//new_data_file.close();

	/*cout << "ICP_ref_pts" << endl;
	cout << ICP_ref_pts << endl;

	cout << "ICP_new_pts" << endl;
	cout << ICP_new_pts << endl;

	cin.get();*/

	/*
	// 0. 准备工作
	// 在 MATLAB 端导出一个数据，这里采用的是激光所刀具定位的二维图像
	// save('ref_points.txt', 'ref_points', '-ascii');
	// 拷贝这个文件到：C:\Users\jiajun\Documents\GitHub\Try_2D_ICP
	// 利用OpenCV读取
	fstream data_file;
	Mat ref_points = Mat::zeros(192, 2, CV_32FC1); // 由于参数已知，所以不想费过多力气做自适应了

	double temp_test;

	data_file.open("ref_points.txt");

	for (int i = 0; i < 192; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			data_file >> ref_points.at<float>(i, j);
		}
	}

	data_file.close();
	*/

	// 1. 建立 k-d tree - 还可以把这个k-d tree保存起来
	flann::Index My_Kdtree;
	My_Kdtree.build(ICP_ref_pts, flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_EUCLIDEAN);


	//// 2. 在 tree 中查找最近邻点
	//
	//Mat res_ind, res_dist;
	//
	//My_Kdtree.knnSearch(ICP_new_pts.row(0), res_ind, res_dist, 1, flann::SearchParams(-1));

	//cout << "Find nearest index is: " << res_ind << endl;

	//// 让控制台暂停
	//cin.get();

	// 2. ICP迭代
	// 2.1 准备
	float pre_err = FLT_MAX;	// 迭代需要的误差量存储1
	float now_err = 0;			// 迭代需要的误差量存储2
	float delta_err;
	Mat res_ind, res_dist;
	int res_ind_int = 0;
	float near_x, near_y;
	Mat nearest_pts_from_ref = Mat(ICP_new_pts.rows, 2, CV_32FC1);
	Mat mean_new = Mat(1, 2, CV_32FC1);		// ICP 计算过程中求解各个步骤的R、t的中间量，待匹配点云的重心
	Mat mean_near = Mat(1, 2, CV_32FC1);	// ICP 计算过程中求解各个步骤的R、t的中间量，迭代中点云的重心
	Mat AXY, BXY; // 中间量
	Mat H, U, S, Vt; // 中间量
	Mat Mid_eye = Mat::eye(2, 2, CV_32FC1);
	Mat temp_new_pts;
	Mat R, t; // 计算结果

	


	// 2.2 迭代开始
	for (int iter_num = 0; iter_num < ITER_MAX; iter_num++)
	{
		now_err = 0;


		for (int i = 0; i < ICP_new_pts.rows; i++)
		{
			My_Kdtree.knnSearch(ICP_new_pts.row(i), res_ind, res_dist, 1, flann::SearchParams(-1));
			res_ind_int = res_ind.at<int>(0, 0); // 确认过变量类型应该是对的
			near_x = ICP_ref_pts.at<float>(res_ind_int, 0);
			near_y = ICP_ref_pts.at<float>(res_ind_int, 1);
			nearest_pts_from_ref.at<float>(i, 0) = near_x;
			nearest_pts_from_ref.at<float>(i, 1) = near_y;

			now_err = now_err + sqrtf((ICP_new_pts.at<float>(i, 0) - near_x) * (ICP_new_pts.at<float>(i, 0) - near_x) +
									  (ICP_new_pts.at<float>(i, 1) - near_y) * (ICP_new_pts.at<float>(i, 1) - near_y));
		}

		cout << "nearest: " << endl;
		cout << nearest_pts_from_ref << endl;

		delta_err = abs(now_err - pre_err);

		if (delta_err < ITER_THRESH)
		{
			break;
		}
		else
			pre_err = now_err;

		// 求重心，注意：cv::mean 的返回值是一个 cv::scalar 它由四个元素构成，但是我们只用到第一个，所以后面多了个[0]
		mean_new.at<float>(0, 0) = mean(ICP_new_pts.col(0))[0];
		mean_new.at<float>(0, 1) = mean(ICP_new_pts.col(1))[0];
		mean_near.at<float>(0, 0) = mean(nearest_pts_from_ref.col(0))[0];
		mean_near.at<float>(0, 1) = mean(nearest_pts_from_ref.col(1))[0];


		cout << "mean_new:" << endl;
		cout << mean_new << endl;

		cout << "mean_near" << endl;
		cout << mean_near << endl;


		// 所有点按重心归一化
		AXY = ICP_new_pts - repeat(mean_new, ICP_new_pts.rows, 1);
		BXY = nearest_pts_from_ref - repeat(mean_near, nearest_pts_from_ref.rows, 1);

		cout << "AXY:" << endl;
		cout << AXY << endl;

		cout << "BXY:" << endl;
		cout << BXY << endl;



		// 求出待SVD分解的H矩阵
		H = AXY.t() * BXY;
		SVD::compute(H, S, U, Vt);


		cout << "H:" << endl;
		cout << H << endl;

		cout << "U" << endl;
		cout << U << endl;

		cout << "Vt" << endl;
		cout << Vt << endl;


		Mid_eye.at<float>(1, 1) = determinant(Vt.t()*U.t());
		R = Vt.t() * Mid_eye * U.t();
		t = mean_near.t() - R * mean_new.t();

		cout << "R:" << endl;
		cout << R << endl;

		cout << "t" << endl;
		cout << t << endl;

		transpose((R * ICP_new_pts.t() + repeat(t, 1, ICP_new_pts.rows)), temp_new_pts);
		temp_new_pts.copyTo(ICP_new_pts);

		cout << "ICP_new_pts: " << endl;
		cout << ICP_new_pts << endl;

	}

	cout << "R:" << endl;
	cout << R << endl;

	cout << "t" << endl;
	cout << t << endl;

	cout << "now_err" << endl;
	cout << now_err << endl;

	cin.get();

    return 0;
}

