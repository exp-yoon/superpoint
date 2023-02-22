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
	//int imgNum = vec.size();
	int imgNum = vec.size(); //imgNum : 폴더 내의 data 개수

	//top, bot img array (1 x 1 x H x W)
	float*** imgArr_t = new float** [1];
	float*** imgArr_b = new float** [1];


	imgArr_t[0] = new float* [imgNum];
	imgArr_b[0] = new float* [imgNum];

	//int pixNum = rsize * rsize;
	int inpixNum = inputDims[0][2] * inputDims[0][3];
	int outpixNum = outputDims[0][2] * outputDims[0][3];  //ㄹㅇ pixel 개수 가로x세로
	
	cv::Mat top_img(rsize, rsize, CV_32FC1);
	cv::Mat bot_img(rsize, rsize, CV_32FC1);
	cv::Mat top_g(rsize, rsize, CV_8UC1);;
	cv::Mat bot_g(rsize, rsize, CV_8UC1);;

	for (int imgIdx = 0; imgIdx < imgNum; ++imgIdx)
	{
		cv::Mat img_mat = cv::imread(vec[imgIdx], cv::IMREAD_GRAYSCALE);
		int Height = img_mat.size().height;
		int Width = img_mat.size().width;

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
		imgArr_t[0][imgIdx] = top;

		float* bot = new float[inpixNum];
		memcpy(bot, bot_img.data, sizeof(float) * inpixNum);
		imgArr_b[0][imgIdx] = bot;
	}

	//Run Inference , top, bot 각각 결과 내야함. batch = 1
	ai->Run(imgArr_t, 1, 1, true);
	//ai->Run(imgArr_b, imgNum, 1, true);

	//Get Result
	float*** locresult = new float** [1];
	float*** descresult = new float** [1];

	int loc_channel = 65;
	int desc_channel = 256;

	locresult[0] = new float* [loc_channel];
	descresult[0] = new float* [desc_channel];

	for (int chanIdx = 0; chanIdx < loc_channel; ++chanIdx)
	{
		float* single_loc_result = new float[outpixNum];
		locresult[0][chanIdx] = single_loc_result;
	}
	for (int chanIdx = 0; chanIdx < desc_channel; ++chanIdx)
	{
		float* single_desc_result = new float[outpixNum];
		descresult[0][chanIdx] = single_desc_result;
	}


	bool bRes = ai->GetSuperpointResults(locresult, descresult);
	//for (int chanIdx = 0; chanIdx < loc_channel; ++chanIdx)
	//{
	//	cv::Mat loc_result_mat(outputDims[0][2], outputDims[0][3], CV_32FC1);
	//	memcpy(loc_result_mat.data, locresult[0][chanIdx], sizeof(float) * outpixNum);

	//}
	//for (int chanIdx = 0; chanIdx < desc_channel; ++chanIdx)
	//{
	//	cv::Mat desc_result_mat(outputDims[1][2], outputDims[1][3], CV_32FC1);
	//	memcpy(desc_result_mat.data, descresult[0][chanIdx], sizeof(float) * outpixNum);
	//}


	//for (int i = 0; i < loc_channel; i++) {
	//	for (int j = 0; j < outpixNum; j++){
	//		std::cout << i<<","<<j<<" "<<locresult[0][i][j] << std::endl;
	//	}
	//}

	SpRun* sp = new SpRun();
	sp->calc(locresult, descresult,top_img);
	int count = sp->get_count();


	//SpRun의 결과 받아올 곳
	int** pts = new int* [2];
	int* pts_x = new int[count];
	int* pts_y = new int[count];
	pts[0] = pts_y;
	pts[1] = pts_x;

	float* score = new float[count];

	float** desc = new float* [desc_channel];
	for (int i = 0; i < desc_channel; i++) {
		float* desc_ = new float[count];
		desc[i] = desc_;
	}

	sp->get_sp_result(pts, score, desc);

	Tracker* tracker = new Tracker();
	tracker->update(pts, score, desc);


	delete[] imgArr_t;
	delete[] imgArr_b;
	delete[] locresult;
	delete[] descresult;

	imgArr_t = nullptr;
	imgArr_b = nullptr;
	locresult = nullptr;
	descresult = nullptr;

	return 0;
}