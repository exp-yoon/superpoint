#pragma once
#include <cmath>
#include "opencv2/opencv.hpp"

class SpRun
{

protected:
	float nn_thresh;
	float nms_dist;
	long long loc_channel = 65;
	long long desc_channel = 256;
	long long pixnum = 3844;
	long long org_h = 500;
	long long org_w = 500;
	int pad = 4;
	int border = 4;
	long long h = 62;
	long long w = 62;
	int cell = 8;
	float conf_thresh = 0.015;

	int** pts_save;
	float* score_save;
	float** desc_save;
	
public:
	SpRun();
	~SpRun();
	int count;
	void set_count(int value);
	int get_count();
	void get_sp_result(int** pts_result, float* score_result, float** desc_result);
	void grid_sample(float*** coarse_desc, float** samp_pts, long long count, float** desc);
	void calc(float*** semi, float*** coarse_desc_, cv::Mat img);

};