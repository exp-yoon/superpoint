#pragma once
#include <cmath>
#include "opencv2/opencv.hpp"

class Tracker
{
protected:
	double nn_thresh = 0.7;
	int top_count;
	int bot_count;
	int desc_channel = 256;
	int keep_count;

	double** matches;
	long long** match_point;

public:
	Tracker();
	Tracker(int t_cnt, int b_cnt);
	~Tracker();
	void get_match_result(long long** match_result);
	void set_keep_count(long long val);
	long long get_keep_count();
	void match_point_idx(long long** top_pt, long long** bot_pt);
	void match_twoway(double** top_desc, double** bot_desc);

};