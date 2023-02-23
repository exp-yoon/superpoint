#pragma once
#include "align.h"
#include <iostream>

Align::Align() {

}

Align::Align(int height, int width, int r_height, int r_width)
{
	H = height;
	W = width;
	rsize_h = r_height;
	rsize_w = r_width;

}

Align::~Align()
{
}


void Align::get_alignment(float** matches, int match_count) {
	//dx,dy 구하고, 반올림하여 degree count

	float resize_y = H * (rsize_w / W);

	int* dx = new int[match_count];
	int* dy = new int[match_count];
	double* degree = new double[match_count];
	const double pi = 3.14159265358979;
	std::vector<double> v_degree;

	for (int i = 0; i < match_count; i++) {

		dx[i] = int(-(matches[i][0] - matches[i][2]));
		dy[i] = int(resize_y - rsize_h + (matches[i][1] - matches[i][3]));
		degree[i] = round((atan2(dy[i], dx[i]) * 180 / pi)*pow(10,2)) / pow(10,2); 
		v_degree.push_back(degree[i]);
	}

	sort(v_degree.begin(), v_degree.end());

	float first = 0.0;
	float second = 0.0;
	int first_num = 0 ;
	int second_num = 0;

	for (int i = 0; i < match_count-1; i++) {
		if (v_degree[i] != v_degree[i + 1]) {
			second = first;
			second_num = first_num;
			first = v_degree[i];
			first_num = i + 1;
		}
	}

	std::cout << "최빈값 : " << first << " , count:" << first_num << std::endl;
	std::cout << "차빈값 : " << second << " , count:" << second_num << std::endl;

}
