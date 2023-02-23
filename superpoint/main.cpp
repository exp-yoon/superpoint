#pragma once
#include <filesystem>
#include <iostream>
#include <assert.h>
#include <vector>
#include "ATI_ONNX.h"
#include "ONNXCore.h"
#include "opencv2/opencv.hpp"
#include "SpRun.h"
#include "Tracker.h"
#include "Align.h"

using namespace std;

int main()
{
	int rsize = 500;


	ATI_ONNX::AI* ai = new ATI_ONNX::AI(SUPERPOINT);

	//Load Model
	const wchar_t* modelPath = L"D:/ATI/MODEL/SuperPoint_500.onnx";
	bool bTensorRT = false;
	bool bUseCache = false;
	ai->LoadModel(modelPath, bTensorRT, bUseCache);

	long long** inputDims = ai->GetInputDims();
	long long** outputDims = ai->GetOutputDims(); //근데 이게..

	//Load Images
	std::string dir_path = "C:/Users/jsyoon/PycharmProjects/Pytorch/venv/SuperPoint/data_/";
	std::vector<std::string> vec;
	for (const auto& file : std::filesystem::directory_iterator(dir_path))
		vec.push_back(file.path().u8string());
	int imgNum = 2;
	//int imgNum = vec.size(); //imgNum : 폴더 내의 data 개수

	//top, bot img array (1 x 1 x H x W)
	// 
	// 
	// 
	//float*** imgArr_t = new float** [1];
	//float*** imgArr_b = new float** [1];


	//imgArr_t[0] = new float* [imgNum];
	//imgArr_b[0] = new float* [imgNum];


	float*** imgArr = new float** [1];
	imgArr[0] = new float* [imgNum];//top
	//imgArr[1] = new float* [imgNum];//bot

	//int pixNum = rsize * rsize;
	int inpixNum = inputDims[0][2] * inputDims[0][3];
	int outpixNum = outputDims[0][2] * outputDims[0][3];  //ㄹㅇ pixel 개수 가로x세로
	
	cv::Mat top_img(rsize, rsize, CV_32FC1);
	cv::Mat bot_img(rsize, rsize, CV_32FC1);

	int Height;
	int Width;

	for (int imgIdx = 0; imgIdx < 1; ++imgIdx)
	{
		cv::Mat img_mat = cv::imread(vec[imgIdx], cv::IMREAD_GRAYSCALE);
		Height = img_mat.size().height;
		Width = img_mat.size().width;

		//top, bot crop
		cv::Rect rect_top(0, 0, Width, Width);
		cv::Mat top_crop = img_mat(rect_top);

		cv::Rect rect_bot(0, (Height - Width), Width, Width);
		cv::Mat bot_crop = img_mat(rect_bot);

		cv::Mat top_r, bot_r;
		/*cv::Mat top_g(rsize, rsize, CV_32FC1);
		cv::Mat bot_g(rsize, rsize, CV_32FC1);*/
	

		cv::resize(top_crop, top_r, cv::Size(rsize, rsize), 0, 0, CV_INTER_AREA);
		cv::resize(bot_crop, bot_r, cv::Size(rsize, rsize), 0, 0, CV_INTER_AREA);
		//cv::GaussianBlur(top_r, top_g, cv::Size(3, 3), 2);
		//cv::GaussianBlur(bot_r, bot_g, cv::Size(3, 3), 2);
		top_r.convertTo(top_img, CV_32FC1);
		bot_r.convertTo(bot_img, CV_32FC1);
		//cv::divide(top_img, 255., top_n);
		//cv::divide(bot_img, 255., bot_n);
		float* top = new float[inpixNum];
		memcpy(top, top_img.data, sizeof(float) * inpixNum);
		imgArr[0][0] = top;

		float* bot = new float[inpixNum];
		memcpy(bot, bot_img.data, sizeof(float) * inpixNum);
		imgArr[0][1] = bot;
	}

	//Run Inference , top, bot 각각 결과 내야함. batch = 1
	ai->Run(imgArr, 2, 1, true);

	//Get Result
	float*** locresult = new float** [2];
	float*** descresult = new float** [2];

	int loc_channel = 65;
	int desc_channel = 256;

	locresult[0] = new float* [loc_channel];
	descresult[0] = new float* [desc_channel];
	locresult[1] = new float* [loc_channel];
	descresult[1] = new float* [desc_channel];

	for (int chanIdx = 0; chanIdx < loc_channel; ++chanIdx)
	{
		float* single_loc_result_t = new float[outpixNum];
		float* single_loc_result_b = new float[outpixNum];
		locresult[0][chanIdx] = single_loc_result_t;
		locresult[1][chanIdx] = single_loc_result_b;
	}
	for (int chanIdx = 0; chanIdx < desc_channel; ++chanIdx)
	{
		float* single_desc_result_t = new float[outpixNum];
		float* single_desc_result_b = new float[outpixNum];
		descresult[0][chanIdx] = single_desc_result_t;
		descresult[1][chanIdx] = single_desc_result_b;
	}

	bool bRes = ai->GetSuperpointResults(locresult, descresult);

	SpRun* sp = new SpRun();
	Tracker* tracker = new Tracker();

	//top img
	float*** locresult_t = new float** [1];
	locresult_t[0] = locresult[0];
	float*** descresult_t = new float** [1];
	descresult_t[0] = descresult[0];

	sp->calc(locresult_t, descresult_t , top_img);
	int top_count = sp->get_count();

	//SpRun의 결과 받아올 곳
	int** top_pts = new int* [2];
	int* top_pts_x = new int[top_count];
	int* top_pts_y = new int[top_count];
	top_pts[0] = top_pts_y;
	top_pts[1] = top_pts_x;

	float* top_score = new float[top_count];

	float** top_desc = new float* [desc_channel];
	for (int i = 0; i < desc_channel; i++) {
		float* top_desc_ = new float[top_count];
		top_desc[i] = top_desc_;
	}
	sp->get_sp_result(top_pts, top_score, top_desc);


	//bot img
	float*** locresult_b = new float** [1];
	locresult_b[0] = locresult[1];
	float*** descresult_b = new float** [1];
	descresult_b[0] = descresult[1];

	sp->calc(locresult_b, descresult_b, bot_img);
	int bot_count = sp->get_count();

	//SpRun의 결과 받아올 곳
	int** bot_pts = new int* [2];
	int* bot_pts_x = new int[bot_count];
	int* bot_pts_y = new int[bot_count];
	bot_pts[0] = bot_pts_y;
	bot_pts[1] = bot_pts_x;

	float* bot_score = new float[bot_count];

	float** bot_desc = new float* [desc_channel];
	for (int i = 0; i < desc_channel; i++) {
		float* bot_desc_ = new float[bot_count];
		bot_desc[i] = bot_desc_;
	}

	sp->get_sp_result(bot_pts, bot_score, bot_desc);

	//tracker->update(bot_pts, bot_score, bot_desc);

	tracker->match_twoway(top_desc, bot_desc);
	tracker->match_point_idx(top_pts, bot_pts);
	int match_count = tracker->get_keep_count();

	float** matches = new float* [3];
	for (int i = 0; i < 3; i++) {
		float* match = new float[match_count];
		matches[i] = match;
	}

	tracker->get_match_result(matches);

	Align* align = new Align(Height, Width, rsize, rsize);
	align->get_alignment(matches, match_count);


	delete[] imgArr;
	//delete[] locresult_t;
	//delete[] descresult_t;
	delete[] locresult_b;
	delete[] descresult_b;

	imgArr = nullptr;
	//locresult_t = nullptr;
	//descresult_t = nullptr;
	locresult_b = nullptr;
	descresult_b = nullptr;
	return 0;
}