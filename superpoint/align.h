#pragma once
#include <cmath>
#include <vector>
#include "opencv2/opencv.hpp"
#include <algorithm>

class Align
{
protected:

	double H;
	double W;
	double rsize_h = 500;
	double rsize_w = 500;


public:
	Align();
	Align(int H, int W, int rsize_h, int rsize_w);
	~Align();
	void get_alignment(long long** matches, long long match_count);

};