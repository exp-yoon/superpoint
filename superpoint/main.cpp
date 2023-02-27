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
	long long rsize = 500;


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

	//int imgNum = vec.size(); //imgNum : 폴더 내의 data 개수
	int imgNum = 2; //사용할 이미지는 top,bot 항상 2개
	

	//Onnx model에 입력할 ImgArr -> Shape : (1, 2(imgNum), inpixNum)
	float*** imgArr = new float** [1];
	imgArr[0] = new float* [imgNum];//top,bot

	//모델의 입력이미지의 총 픽셀 개수 inpixNum = rsize * rsize;
	long long inpixNum = inputDims[0][2] * inputDims[0][3];
	//모델의 출력이미지의 총 픽셀 개수 inpixNum = (rsize/8) *(rsize/8)
	long long outpixNum = outputDims[0][2] * outputDims[0][3];
	
	cv::Mat top_img(rsize, rsize, CV_32FC1);
	cv::Mat bot_img(rsize, rsize, CV_32FC1);

	//폴더에 넣은 길쭉한 원본 이미지의 H,W
	long long Height;
	long long Width;

	for (size_t imgIdx = 0; imgIdx < 1; ++imgIdx)
	{
		cv::Mat img_mat = cv::imread(vec[imgIdx], cv::IMREAD_GRAYSCALE);
		Height = img_mat.size().height;
		Width = img_mat.size().width;

		//top, bot crop
		cv::Rect rect_top(0, 0, Width, Width);
		cv::Mat top_crop = img_mat(rect_top);

		cv::Rect rect_bot(0, (Height - Width), Width, Width);
		cv::Mat bot_crop = img_mat(rect_bot);

		cv::Mat top_r, bot_r, top_g, bot_g;
		//cv::Mat top_g(rsize, rsize, CV_32FC1);
		//cv::Mat bot_g(rsize, rsize, CV_32FC1);
	

		cv::resize(top_crop, top_r, cv::Size(rsize, rsize), 0, 0, CV_INTER_AREA);
		cv::resize(bot_crop, bot_r, cv::Size(rsize, rsize), 0, 0, CV_INTER_AREA);
		//가우시안 블러 꼭 해줘야 결과값이 좋음. 근데 이거 파이썬이랑 똑같은 값으로 했는데 왜
		//블러 결과가 다를까용?
		cv::GaussianBlur(top_r, top_g, cv::Size(3, 3), 2);
		cv::GaussianBlur(bot_r, bot_g, cv::Size(3, 3), 2);
		top_g.convertTo(top_img, CV_32FC1);
		bot_g.convertTo(bot_img, CV_32FC1);
		//cv::divide(top_img, 255., top_n);
		//cv::divide(bot_img, 255., bot_n);
		float* top = new float[inpixNum];
		memcpy(top, top_img.data, sizeof(float) * inpixNum);
		imgArr[0][0] = top;

		float* bot = new float[inpixNum];
		memcpy(bot, bot_img.data, sizeof(float) * inpixNum);
		imgArr[0][1] = bot;

	}

	std::cout << " Inference start " << std::endl;
	//Run Inference , top, bot 각각 결과 내야함. batch = 1
	ai->Run(imgArr, 2, 1, true);

	//Get Result
	float*** locresult = new float** [2];
	float*** descresult = new float** [2];

	long long loc_channel = 65;
	long long desc_channel = 256;

	locresult[0] = new float* [loc_channel];
	descresult[0] = new float* [desc_channel];
	locresult[1] = new float* [loc_channel];
	descresult[1] = new float* [desc_channel];

	for (size_t chanIdx = 0; chanIdx < loc_channel; ++chanIdx)
	{
		float* single_loc_result_t = new float[outpixNum];
		float* single_loc_result_b = new float[outpixNum];
		locresult[0][chanIdx] = single_loc_result_t;
		locresult[1][chanIdx] = single_loc_result_b;
	}
	for (size_t chanIdx = 0; chanIdx < desc_channel; ++chanIdx)
	{
		float* single_desc_result_t = new float[outpixNum];
		float* single_desc_result_b = new float[outpixNum];
		descresult[0][chanIdx] = single_desc_result_t;
		descresult[1][chanIdx] = single_desc_result_b;
	}

	bool bRes = ai->GetSuperpointResults(locresult, descresult);

	SpRun* sp = new SpRun();
	Tracker* tracker = new Tracker();

	std::cout << " SpRun start " << std::endl;

	//top img
	float*** locresult_t = new float** [1];
	locresult_t[0] = locresult[0];
	float*** descresult_t = new float** [1];
	descresult_t[0] = descresult[0];

	sp->calc(locresult_t, descresult_t , top_img);
	long long top_count = sp->get_count();

	//SpRun의 결과 받아올 곳
	long long** top_pts = new long long* [2];
	long long* top_pts_x = new long long[top_count];
	long long* top_pts_y = new long long[top_count];
	top_pts[0] = top_pts_y;
	top_pts[1] = top_pts_x;

	double* top_score = new double[top_count];

	double** top_desc = new double* [desc_channel];
	for (size_t i = 0; i < desc_channel; i++) {
		double* top_desc_ = new double[top_count];
		top_desc[i] = top_desc_;
	}
	sp->get_sp_result(top_pts, top_score, top_desc);

	//bot img
	float*** locresult_b = new float** [1];
	locresult_b[0] = locresult[1];
	float*** descresult_b = new float** [1];
	descresult_b[0] = descresult[1];

	sp->calc(locresult_b, descresult_b, bot_img);
	size_t bot_count = sp->get_count();

	//SpRun의 결과 받아올 곳
	long long** bot_pts = new long long* [2];
	long long* bot_pts_x = new long long[bot_count];
	long long* bot_pts_y = new long long[bot_count];
	bot_pts[0] = bot_pts_y;
	bot_pts[1] = bot_pts_x;

	double* bot_score = new double[bot_count];

	double** bot_desc = new double* [desc_channel];
	for (size_t i = 0; i < desc_channel; i++) {
		double* bot_desc_ = new double[bot_count];
		bot_desc[i] = bot_desc_;
	}

	sp->get_sp_result(bot_pts, bot_score, bot_desc);

	std::cout << " matching start " << std::endl;

	tracker->match_twoway(top_desc, bot_desc);
	tracker->match_point_idx(top_pts, bot_pts);
	long long match_count = tracker->get_keep_count();

	long long** matches_pts = new long long* [match_count];
	for (size_t i = 0; i < match_count; i++) {
		long long* match = new long long[4];
		matches_pts[i] = match;
	}

	tracker->get_match_result(matches_pts);


	std::cout << " alignment start " << std::endl;

 	Align* align = new Align(Height, Width, rsize, rsize);
	align->get_alignment(matches_pts, match_count);

	//delete
	for (size_t i = 0; i < 1; i++) {
		for (size_t j = 0; j < imgNum; j++) {
			delete[] imgArr[i][j];
		}
		delete[] imgArr[i];
	}
	delete[] imgArr;

	for (size_t i = 0; i < 2; i++) {
		for (size_t j = 0; j < loc_channel; j++) {
			delete[] locresult[i][j]; 
		}
		delete[] locresult[i];
	}
	delete[] locresult;

	for (size_t i = 0; i < 2; i++) {
		for (size_t j = 0; j < desc_channel; j++) {
			delete[] descresult[i][j];
		}
		delete[] descresult[i];
	}
	delete[] descresult;


	for (size_t i = 0; i < 2; i++) {
		delete[] top_pts[i];
		delete[] bot_pts[i];
	}
	delete[] top_pts;
	delete[] top_score;
	delete[] bot_pts;
	delete[] bot_score;


	for (size_t i = 0; i < 2; i++) {
		delete[] top_desc[i];
		delete[] bot_desc[i];
	}
	delete[] top_desc;
	delete[] bot_desc;

	for (size_t i = 0; i < match_count; i++) {
		delete[] matches_pts[i];
	}
	delete[] matches_pts;


	imgArr = nullptr;
	locresult = nullptr;
	descresult = nullptr;
	top_pts = nullptr;
	top_score = nullptr;
	bot_pts = nullptr;
	bot_score = nullptr;
	top_desc = nullptr;
	bot_desc = nullptr;
	matches_pts = nullptr;

	return 0;
}