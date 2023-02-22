#pragma once
#include <cmath>
#include "opencv2/opencv.hpp"

class Tracker
{
protected:
	int max_length = 5;
	float nn_thresh = 0.7;
	int max_score = 9999;


public:
	Tracker();
	~Tracker();
	void update(int** pts, float* score, float** desc);

};