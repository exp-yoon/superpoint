#pragma once
#include <cmath>
#include "opencv2/opencv.hpp"

class Tracker
{
protected:
	int max_length = 5;
	double nn_thresh = 0.7;
	int max_score = 9999;
	double** last_desc;
	double*** all_pts;
	int top_count = 118;
	int bot_count = 117;
	int desc_channel = 256;
	int keep_count;

	double** matches;
	long long** match_point;

public:
	Tracker();
	~Tracker();
	void get_match_result(long long** match_result);
	void set_keep_count(long long val);
	long long get_keep_count();
	void match_point_idx(long long** top_pt, long long** bot_pt);
	void match_twoway(double** top_desc, double** bot_desc);

};