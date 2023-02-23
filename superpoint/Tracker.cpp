#pragma once
#include "Tracker.h"
#include <iostream>

Tracker::Tracker()
{
}

Tracker::~Tracker()
{
}


void Tracker::set_keep_count(int val) {

	keep_count = val;

}

void Tracker::get_match_point(int** top_pt, int** bot_pt){

	int** match_point = new int* [keep_count];

	for (int kc = 0; kc < keep_count; kc++) {
		int* point_4 = new int[4];

		int top_idx = matches[0][kc];
		int bot_idx = matches[1][kc];

		point_4[0] = top_pt[0][top_idx];
		point_4[1] = top_pt[1][top_idx];
		point_4[2] = bot_pt[0][bot_idx];
		point_4[3] = bot_pt[1][bot_idx];

		match_point[kc] = point_4;
	}
		
}


void Tracker::match_twoway(float** top_desc, float** bot_desc) {

	//top_desc의 transpose와 bot_desc를 dot product
	float** dmat = new float* [top_count];

	for (int tc = 0; tc < top_count; tc++) {
		float* dmat_ = new float[bot_count];
		for (int bc = 0; bc < bot_count; bc++) {
			float val = 0;
			for (int dc = 0; dc < desc_channel; dc++){
				val += top_desc[dc][tc] * bot_desc[bc][dc];
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

	int* min_idx = new int[top_count]; // 한 행에서의 최소값 인덱스
	for (int tc = 0; tc < top_count; tc++) {
		int mval = dmat[tc][0];
		int midx = 0;
		for (int bc = 1; bc < bot_count; bc++) {
			if (mval > dmat[tc][bc]) {
				mval = dmat[tc][bc];
				midx = bc;
			}
		}
		min_idx[tc] = midx;
	}

	bool* keep01 = new bool[top_count];
	for (int tc = 0; tc < top_count; tc++) {
		//행 기준 오름차순으로 정렬한 score = dmat[tc][min_idx[tc]]가
		//threshold보다 작으면 keep
		if (dmat[tc][min_idx[tc]] < nn_thresh) { 
			keep01[tc] = true;
		}
		else
			keep01[tc] = false;
	}

	int* min_idx2 = new int[bot_count]; //한 열에서의 최소값 인덱스
	for (int bc = 0; bc < bot_count; bc++) {
		int mval = dmat[0][bc];
		int midx = 0;
		for (int tc = 1; tc < top_count; tc++) {
			if (mval > dmat[tc][bc]) {
				mval = dmat[tc][bc];
				midx = tc;
			}
		}
		min_idx2[bc] = midx;
	}

	bool* keep_bi = new bool[top_count];
	for (int tc = 0; tc < top_count; tc++) {
		if (tc == min_idx2[min_idx[tc]])
			keep_bi[tc] = true;
		else
			keep_bi[tc] = false;
	}	

	bool* keep02 = new bool[top_count];
	int keep_cnt = 0;
	for (int tc = 0; tc < top_count; tc++) {
		keep02[tc] = (keep01[tc] && keep_bi[tc]);
		if (keep02[tc] == true)
			keep_cnt++;
	}

	set_keep_count(keep_cnt);


	float* keep_min_idx = new float[keep_count]; //python->m_idx2
	float* keep_score = new float[keep_count]; //python -> scores
	int keepidx = 0;
	for (int tc = 0; tc < top_count; tc++) {
		if (keep02[tc] == true) {
			keep_min_idx[keepidx] = float(min_idx[tc]);
			keep_score[keepidx] = dmat[tc][min_idx[tc]];
			keepidx++;
		}
	}

	matches = new float* [3];
	float* match01 = new float[keep_count];
	int midx = 0;
	for (int tc = 0; tc < top_count; tc++) {
		if (keep02[tc] == true) {
			match01[midx] = float(tc);
		}
	}
	matches[0] = match01;
	matches[1] = keep_min_idx;
	matches[2] = keep_score;



}