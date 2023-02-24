#pragma once
#include "Tracker.h"
#include <iostream>

Tracker::Tracker()
{
}

Tracker::~Tracker()
{
}

void Tracker::get_match_result(long long** match_result) {

	for (size_t kc = 0; kc < keep_count; ++kc) {
		memcpy(match_result[kc], match_point[kc], sizeof(long long) *4);
	}


	//delete
	for (size_t i = 0; i < keep_count; i++) {
		delete[] match_point[i];
	}
	delete[] match_point;

	for (size_t i = 0; i < 3; i++) {
		delete[] matches[i];
	}
	delete[] matches;

	match_point = nullptr;
	matches = nullptr;
}

void Tracker::set_keep_count(long long val) {

	keep_count = val;

}

long long Tracker::get_keep_count() {

	return keep_count;

}

void Tracker::match_point_idx(long long** top_pt, long long** bot_pt){

	match_point = new long long* [keep_count];

	for (size_t kc = 0; kc < keep_count; kc++) {
		long long* point_4 = new long long[4];

		int top_idx = matches[0][kc];
		int bot_idx = matches[1][kc];

		point_4[0] = top_pt[0][top_idx];
		point_4[1] = top_pt[1][top_idx];
		point_4[2] = bot_pt[0][bot_idx];
		point_4[3] = bot_pt[1][bot_idx];

		match_point[kc] = point_4;
	}
		
}


void Tracker::match_twoway(double** top_desc, double** bot_desc) {

	//top_desc의 transpose와 bot_desc를 dot product
	double** dmat = new double* [top_count];

	for (size_t tc = 0; tc < top_count; tc++) {
		double* dmat_ = new double[bot_count];
		for (size_t bc = 0; bc < bot_count; bc++) {
			double val = 0;
			for (size_t dc = 0; dc < desc_channel; dc++){
				val += top_desc[dc][tc] * bot_desc[dc][bc];
			}
			//dot product 결과가 -1~1 범위를 넘으면 clip
			if (val > 1) {
				val = 1;
			}
			else if (val < -1) {
				val = -1;
			}
			val = sqrt(2 - 2 * val);
			dmat_[bc] = val;
		}
		dmat[tc] = dmat_;

	}

	long long* min_idx = new long long[top_count]; // 한 행에서의 최소값 인덱스
	for (size_t tc = 0; tc < top_count; tc++) {
		double mval = dmat[tc][0];
		int midx = 0;
		for (size_t bc = 1; bc < bot_count; bc++) {
			if (mval > dmat[tc][bc]) {
				mval = dmat[tc][bc];
				midx = bc;
			}
		}
		min_idx[tc] = midx;
	}

	bool* keep01 = new bool[top_count];
	for (size_t tc = 0; tc < top_count; tc++) {
		//행 기준 오름차순으로 정렬한 score = dmat[tc][min_idx[tc]]가
		//threshold보다 작으면 keep
		if (dmat[tc][min_idx[tc]] < nn_thresh) { 
			keep01[tc] = true;
		}
		else
			keep01[tc] = false;
	}

	long long* min_idx2 = new long long[bot_count]; //한 열에서의 최소값 인덱스
	for (size_t bc = 0; bc < bot_count; bc++) {
		double mval = dmat[0][bc];
		int midx = 0;
		for (size_t tc = 1; tc < top_count; tc++) {
			if (mval > dmat[tc][bc]) {
				mval = dmat[tc][bc];
				midx = tc;
			}
		}
		min_idx2[bc] = midx;
	}

	bool* keep_bi = new bool[top_count];
	for (size_t tc = 0; tc < top_count; tc++) {
		if (tc == min_idx2[min_idx[tc]])
			keep_bi[tc] = true;
		else
			keep_bi[tc] = false;
	}	

	bool* keep02 = new bool[top_count];
	int keep_cnt = 0;
	for (size_t tc = 0; tc < top_count; tc++) {
		keep02[tc] = (keep01[tc] && keep_bi[tc]);
		if (keep02[tc] == true)
			keep_cnt++;
	}

	set_keep_count(keep_cnt);

	double* keep_min_idx = new double[keep_count]; //python->m_idx2
	double* keep_score = new double[keep_count]; //python -> scores
	int keepidx = 0;
	for (size_t tc = 0; tc < top_count; tc++) {
		if (keep02[tc] == true) {
			keep_min_idx[keepidx] = double(min_idx[tc]);
			keep_score[keepidx] = dmat[tc][min_idx[tc]];
			keepidx++;
		}
	}

	matches = new double* [3];
	double* match01 = new double[keep_count];
	int midx = 0;
	for (size_t tc = 0; tc < top_count; tc++) {
		if (keep02[tc] == true) {
			match01[midx] = double(tc);
			midx++;
		}
	}
	matches[0] = match01;
	matches[1] = keep_min_idx;
	matches[2] = keep_score;


	//delete
	for (size_t i = 0; i < top_count; i++) {
		delete[] dmat[i];
	}
	delete[] dmat;
	delete[] min_idx;
	delete[] keep01;
	delete[] min_idx2;
	delete[] keep_bi;
	delete[] keep02;
	delete[] keep_min_idx;
	delete[] keep_score;

	dmat = nullptr;
	min_idx = nullptr;
	keep01 = nullptr;
	min_idx2 = nullptr;
	keep_bi = nullptr;
	keep02 = nullptr;
	keep_min_idx = nullptr;
	keep_score = nullptr;
}