#pragma once
#include "align.h"
#include <iostream>

Align::Align() {

}

Align::Align(int height, int width, int r_height, int r_width)
{
	H = double(height);
	W = double(width);
	rsize_h = double(r_height);
	rsize_w = double(r_width);

}

Align::~Align()
{
}


void Align::get_alignment(long long** matches, long long match_count) {
	//dx,dy 구하고, 반올림하여 degree count

	double resize_y = double(H) * (double(rsize_w) / double(W));

	double* dx = new double[match_count];
	double* dy = new double[match_count];
	double* degree = new double[match_count];
	const double pi = 3.14159265358979;
	std::vector<double> v_degree;
	std::vector<double> round_degree;
	std::vector<long long> num_degree;

	for (size_t i = 0; i < match_count; i++) {

		dx[i] = -(matches[i][0] - matches[i][2]);
		dy[i] = (resize_y - rsize_h + (matches[i][1] - matches[i][3]));
		degree[i] = round((atan2(dx[i], dy[i]) * 180 / pi)*pow(10,2)) / pow(10,2); 
		v_degree.push_back(degree[i]);
	}

	sort(v_degree.begin(), v_degree.end());

	for (size_t i = 0; i < match_count-1; i++) {
		if ((v_degree[i] != v_degree[i + 1]) && find(round_degree.begin(),round_degree.end(),v_degree[i]) == round_degree.end()) {
			round_degree.push_back(v_degree[i]);
			num_degree.push_back(count(v_degree.begin(), v_degree.end(), v_degree[i]));
		}
	}

	for (size_t j = 0; j < round_degree.size(); j++) {
		for (size_t i = 0; i < round_degree.size() - 1; i++) {
			if (num_degree[i] < num_degree[i + 1]) {
				long long temp = num_degree[i];
				num_degree[i] = num_degree[i + 1];
				num_degree[i + 1] = temp;

				double temp1 = round_degree[i];
				round_degree[i] = round_degree[i + 1];
				round_degree[i + 1] = temp1;
			}
		}
	}


	for (size_t i = 0; i < round_degree.size(); i++) {
		std::cout << "Align : " << round_degree[i] << " , count:" << num_degree[i] << std::endl;
	}


	//delete
	delete[] dx;
	delete[] dy;
	delete[] degree;

	dx = nullptr;
	dy = nullptr;
	degree = nullptr;
}
