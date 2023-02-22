#pragma once
#include <cmath>
#include "opencv2/opencv.hpp"

class Tracker
{
protected:
	int max_length = 5;
	float nn_thresh = 0.7;
	int max_score = 9999;
	float** last_desc;
	float*** all_pts;
	int top_count = 118;
	int bot_count = 117;
	int desc_channel = 256;
	int keep_count;

	float** matches;

public:
	Tracker();
	~Tracker();
	void set_keep_count(int val);
	void update(int** pts, float* score, float** desc);
	void match_twoway(float** top_desc, float** bot_desc);

};