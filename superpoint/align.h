#pragma once
#include <cmath>
#include "opencv2/opencv.hpp"

class Align
{
protected:

	long long H;
	long long W;
	long long rsize_h = 500;
	long long rsize_w = 500;


public:
	Align();
	Align(int H, int W, int rsize_h, int rsize_w);
	~Align();
	void get_alignment(int** point);

};