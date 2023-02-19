#pragma once
#include "SpRun.h"
#include <iostream>


SpRun::SpRun()
{
}

SpRun::~SpRun()
{
}

void SpRun::grid_sample(float*** coarse_desc, float** samp_pts, long long count, float** desc) {

	float** result = new float* [desc_channel]; //(desc_channel,count)
	for (int i = 0; i < desc_channel; i++) {
		float* result_2 = new float[count];
		result[i] = result_2;
	}

	for (int channel = 0; channel < desc_channel; channel++) {
		for (int p_cnt = 0; p_cnt < count; p_cnt++) {
			float cvt_x = ((samp_pts[p_cnt][0] + 1) / 2) * (w-1);
			float cvt_y = ((samp_pts[p_cnt][1] + 1) / 2) * (h-1);
			
			float a = (cvt_x)-floor(cvt_x);
			float b = (cvt_y)-floor(cvt_y);
			float x = coarse_desc[channel][int(floor(cvt_y))][int(floor(cvt_x))];
			float y = coarse_desc[channel][int(floor(cvt_y))][int(ceil(cvt_x))];
			float w = coarse_desc[channel][int(ceil(cvt_y))][int(floor(cvt_x))];
			float z = coarse_desc[channel][int(ceil(cvt_y))][int(ceil(cvt_x))];

			float out = (a * b * z) + (1 - a) * (1 - b) * x + (1 - a) * b * w + (1 - b) * a * y;
			result[channel][p_cnt] = out;
		}
	}

	for (size_t chanIdx = 0; chanIdx < desc_channel; ++chanIdx) {
		memcpy(desc[chanIdx], result[chanIdx], sizeof(float) * count);
		int a = 0;
	}

}

void SpRun::calc(float*** semi, float*** coarse_desc_, cv::Mat img) {

	//point location pixel-wise softmax
	float** nodust = new float* [loc_channel - 1];
	float** dense_p = new float* [loc_channel]; //(3,62*62)형태의 semi에 exp계
	float* expSum = new float[pixnum]; //pixel wise expsum
	for (size_t chanIdx = 0; chanIdx < loc_channel; ++chanIdx) {

		float* dense_c = new float[pixnum];
		float* nodust_c = new float[pixnum];
		for (size_t pixIdx = 0; pixIdx < pixnum; pixIdx += 2) {
			float val0 = exp(semi[0][chanIdx][pixIdx]);
			float val1 = exp(semi[0][chanIdx][pixIdx + 1]);


			if (chanIdx == 0) {
				expSum[pixIdx] = val0;
				expSum[pixIdx + 1] = val1;
			}
			else {
				expSum[pixIdx] = expSum[pixIdx] + val0;
				expSum[pixIdx + 1] = expSum[pixIdx] + val1;
			}

			dense_c[pixIdx] = val0;
			dense_c[pixIdx + 1] = val1;

		}
		dense_p[chanIdx] = dense_c;
		nodust[chanIdx] = nodust_c;
	}

	for (size_t chanIdx = 0; chanIdx < loc_channel - 1; ++chanIdx) {
		for (size_t pixIdx = 0; pixIdx < pixnum; pixIdx += 2) {
			nodust[chanIdx][pixIdx] = dense_p[chanIdx][pixIdx] / (expSum[pixIdx]);
			nodust[chanIdx][pixIdx + 1] = dense_p[chanIdx][pixIdx + 1] / (expSum[pixIdx + 1]);
		}
	}

	// nodust (1,2,0) transpose
	float*** nodust_t = new float** [h];
	for (int i = 0; i < h; i++) {
		float** single_h = new float* [w];
		nodust_t[i] = single_h;
		for (int j = 0; j < w; j++) {
			float* single_c = new float[loc_channel - 1];
			nodust_t[i][j] = single_c;
		}
	}

	for (int nc = 0; nc < loc_channel - 1; nc++) {
		for (int pn = 0; pn < pixnum; pn++) {
			int nh = pn / h;
			int nw = pn % h;
			nodust_t[nh][nw][nc] = nodust[nc][pn]; //(62,62,64)
			int a = 0;
		}
	} //dz

	//heatmap
	float**** heatmap = new float*** [h];
	for (int i = 0; i < h; i++) {
		float*** heatmap1d = new float** [cell];
		heatmap[i] = heatmap1d;
		for (int j = 0; j < cell; j++) {
			float** heatmap2d = new float* [w];
			heatmap[i][j] = heatmap2d;
			for (int k = 0; k < w; k++) {
				float* heatmap3d = new float[cell];
				heatmap[i][j][k] = heatmap3d;
			}
		}
	}

	//reshape(62,62,64)->(62,62,8,8), transpose(0,2,1,3) : (62,62,8,8) -> (62,8,62,8)
	for (int nh = 0; nh < h; nh++) {
		for (int nw = 0; nw < w; nw++) {
			for (int nc = 0; nc < loc_channel - 1; nc++) {
				int cell1 = nc / cell;
				int cell2 = nc % cell;
				heatmap[nh][cell1][nw][cell2] = nodust_t[nh][nw][nc]; //(62,8,62,8);
			}
		}
	}

	//reshape(62,8,62,8)->(496,496)
	//float** heatmap_r = new  float* [h * cell];
	//for (int nh = 0; nh < h; nh++) {
	//	float* heat_h = new float[w * cell];
	//	heatmap_r[nh] = heat_h;
	//}
	//for (int nh = 0;nh < h; nh++) {
	//	for (int c1 = 0; c1 < cell; c1++) {
	//		for (int nw = 0; nw < w; nw++) {
	//			for (int c2 = 0; c2 < cell; c2++) {
	//				heatmap_r[nh*c1+c1][nw*c2+c2] = heatmap[nh][c1][nw][c2];
	//			}
	//		}
	//	}
	//}


	//reshape(62,8,62,8)->(496,496)
	float* heatmap_r = new  float[h * w * cell * cell];
	int cnt = 0; //heatmap에서 conf_threshold 이상인 좌표 개수 
	for (int nh = 0; nh < h; nh++) {
		for (int c1 = 0; c1 < cell; c1++) {
			for (int nw = 0; nw < w; nw++) {
				for (int c2 = 0; c2 < cell; c2++) {
					heatmap_r[(nh * w * cell * cell) + (c1 * w * cell) + (nw * cell) + (c2)] = heatmap[nh][c1][nw][c2];
					if (heatmap[nh][c1][nw][c2] >= conf_thresh)
						cnt += 1;
					//std::cout << (nh * w * cell * cell) + (c1 * w * cell) + (nw * cell) + (c2) << std::endl;
				}
			}
		}
	}

	int** pts_xy = new int* [2];
	int* pt_x = new int[cnt];
	int* pt_y = new int[cnt];
	float* pt_score = new float[cnt];

	int cnt_idx = 0;
	for (int pixIdx = 0; pixIdx < h * w * cell * cell; pixIdx++) {
		if (heatmap_r[pixIdx] >= conf_thresh) {
			int idx_x = pixIdx / (h * cell);
			int idx_y = pixIdx % (w * cell);

			pt_x[cnt_idx] = idx_x;
			pt_y[cnt_idx] = idx_y;
			pt_score[cnt_idx] = heatmap_r[pixIdx];
			cnt_idx++;
		}
	}
	pts_xy[0] = pt_x;
	pts_xy[1] = pt_y;


	//nms 수행
	//score 내림차순 정렬

	int* sorted_idx = new int[cnt]; //score 내림차순 정렬한 인덱스 -> python : inds1
	for (int i = 0; i < cnt; i++)
		sorted_idx[i] = i;

	for (int i = 0; i < cnt; i++) {
		for (int j = 0; j < cnt - 1; j++) {
			if (pt_score[sorted_idx[j]] < pt_score[sorted_idx[j + 1]]) {
				int temp = sorted_idx[j];
				sorted_idx[j] = sorted_idx[j + 1];
				sorted_idx[j + 1] = temp;
				int a = 0;
			}
		}
	}


	// 내림차순 정렬한 index로 x,y도 정렬 
	int** sorted_xy = new int* [2];
	int* sorted_x = new int[cnt]; //내림차순 정렬된 x좌표
	int* sorted_y = new int[cnt]; //내림차순 정렬된 y좌표
	float* sorted_score = new float[cnt]; //내림차순 정렬된 score
	for (int i = 0; i < cnt; i++) {
		sorted_x[i] = pts_xy[0][sorted_idx[i]];
		sorted_y[i] = pts_xy[1][sorted_idx[i]];
		sorted_score[i] = pt_score[sorted_idx[i]];
	}

	sorted_xy[0] = sorted_x;
	sorted_xy[1] = sorted_y;

	int** grid = new int* [org_h + 2 * pad]; // Track NMS data ,zero padding 
	int** inds = new int* [org_h]; // store indices of points
	int* grid_w = new int[org_w + 2 * pad]();
	int* inds_w = new int[org_w]();

	for (int i = 0; i < org_h + 2 * pad; i++) {
		grid[i] = grid_w;
	}

	for (int i = 0; i < org_h; i++) {
		inds[i] = inds_w;
	}

	//정렬한 좌표순서대로 grid엔 1값을, inds엔 score 순위(해당좌표에 score 순위가 있음)
	for (int i = 0; i < cnt; i++) {
		grid[sorted_xy[1][i] + pad][sorted_xy[0][i] + pad] = 1; //padding 고려
		inds[sorted_xy[1][i]][sorted_xy[0][i]] = i;
	}

	//nms해서 주변 포인트 지워줌(남은애들은 grid에 -1로 마킹)
	int count = 0;
	for (int i = 0; i < cnt; i++) {
		int pt_x = sorted_xy[0][i] + pad;
		int pt_y = sorted_xy[1][i] + pad;
		for (int p = 0; p < 2 * pad + 1; p++) {
			if (grid[pt_y][pt_x] == 1) {
				grid[pt_y - pad + p][pt_y - pad + p] = 0;
				grid[pt_y][pt_x] = -1;
				count++;
			}
		}
	}

	for (int i = 0; i < cnt; i++) {
		for (int j = 0; j < cnt; j++){
			std::cout << i << "," << j << ":" << grid[i][j] << std::endl;
		}
	}

	///////////////////////////////////////////////////////////여기 뭔가 이상해!!////////////////////////////////////
	////////////////////////////////////////////////////////////nms가 문젠가!!!!!////////////////////////////////////

	//nms해서 남은 애들(grid = -1)의 좌표만 구해줌
	int* keepx = new int[count];
	int* keepy = new int[count];

	int keepcnt = 0;
	for (int i = 4; i < org_h + pad; i++) {//
		for (int j = 4; j < org_w + pad; j++) {
			if (grid[i][j] == -1) {
				keepx[keepcnt] = j - pad;
				keepy[keepcnt] = i - pad;
				keepcnt++;
				int a = 0;
			}
		}
	}

	//좌표에 score 순위가 매겨졌던 inds에서도 nms해서 남은 좌표들의 순위만 inds_keep에 저장
	int* inds_keep = new int[count]; //nms 적용 후 남은 좌표의 score 순위
	for (int i = 0; i < count; i++) {
		inds_keep[i] = inds[keepy[i]][keepx[i]];
	}

	//nms로 남은 score의 내림차순 인덱스 구하기 
	int* sorted_keep_idx = new int[count];
	for (int i = 0; i < count; i++)
		sorted_keep_idx[i] = i;

	for (int i = 0; i < count; i++) {
		for (int j = 0; j < count - 1; j++) {
			if (pt_score[sorted_idx[j]] < pt_score[sorted_idx[j + 1]]) {
				int temp = sorted_idx[j];
				sorted_idx[j] = sorted_idx[j + 1];
				sorted_idx[j + 1] = temp;
			}
		}
	}


	//nms적용후 score별로 내림차순된 x,y,score 
	int** nms_sorted_xy = new int* [2];
	int* nms_sorted_x = new int[count];
	int* nms_sorted_y = new int[count];
	float* nms_sorted_score = new float[count];
	nms_sorted_xy[0] = nms_sorted_x;
	nms_sorted_xy[1] = nms_sorted_y;

	for (int i = 0; i < count; i++) {
		nms_sorted_x[i] = sorted_x[sorted_keep_idx[i]];
		nms_sorted_y[i] = sorted_y[sorted_keep_idx[i]];
		nms_sorted_score[i] = sorted_score[sorted_keep_idx[i]];
	}

	//int* out_inds = new int[count];
	//for (int i = 0; i < count; i++) {
	//	out_inds[i] = sorted_idx[inds_keep[sorted_keep_idx[i]]];
	//}



	bool* toremove = new bool[count];
	for (int i = 0; i < count; i++) {
		if (((nms_sorted_xy[0][i] < border) || (nms_sorted_xy[0][i] >= (org_w - border))) || ((nms_sorted_xy[1][i] < border) || (nms_sorted_xy[1][i] >= (org_h - border))))
			toremove[i] = false;
		else
			toremove[i] = true;
		std::cout << toremove[i] << std::endl;
	}


	for (int i = 0; i < count; i++) {
		std::cout << nms_sorted_xy[1][i] << " , " << nms_sorted_xy[0][i]<<  " , " << nms_sorted_score[i] << std::endl;
	}

	//feature point 확인용 출력
	//cv::Mat imgmat(cv::Size(500, 500), CV_32SC3);
	//cv::cvtColor(img, imgmat, cv::COLOR_GRAY2BGR);
	//cv::Mat imgcp;
	//imgmat.copyTo(imgcp);
	//for (int i = 0; i < count; i++) {
	//	cv::circle(imgcp, cv::Point(nms_sorted_xy[1][i], nms_sorted_xy[0][i]),1,cv::Scalar(0,0,255),1,8,0);
	//}
	//cv::imwrite("C:/Users/jsyoon/PycharmProjects/Pytorch/venv/SuperPoint/cppresult/point.bmp", imgcp);

	
	float** samp_pts= new float* [count]; //(count,2);
	for (int i = 0; i < count; i++) {
		float* samp_xy = new float[2];
		samp_xy[0] = (nms_sorted_xy[0][i] / (org_w / 2.)) - 1.; //normalized(-1~1)
		samp_xy[1] = (nms_sorted_xy[1][i] / (org_h / 2.)) - 1.; //normalized(-1~1)
		samp_pts[i] = samp_xy;
	}

	//grid_sample한 결과 descriptor 저장할 곳
	float** desc = new float* [desc_channel]; //(desc_channel,count)
	for (int i = 0; i < desc_channel; i++) {
		float* desc_2d = new float[count];
		desc[i] = desc_2d;
	}

	//coarse_desc reshape
	float*** coarse_desc = new float** [desc_channel];
	float** desc_h = new float* [h];
	float* desc_w = new float[w];

	for (int i = 0; i < h; i++) {
		desc_h[i] = desc_w;
	}
	for (int i = 0; i < desc_channel; i++) {
		coarse_desc[i] = desc_h;
	}
	for (int c = 0; c < desc_channel; c++) {
		for (int p = 0; p < h * w; p++) {
			int dh = p / h;
			int dw = p % h;
			coarse_desc[c][dh][dw] = coarse_desc_[0][c][p];
		}
	}

	//grid_sample -> interpolation 
	grid_sample(coarse_desc, samp_pts, count, desc);

	// delete[] nodust;
	delete[] dense_p;
	delete[] expSum;
	delete[] nodust_t;
	delete[] heatmap;
	delete[] heatmap_r;
	delete[] pts_xy;
	delete[] pt_score;

	delete[] sorted_idx;
	delete[] sorted_xy;
	delete[] sorted_score;
	delete[] grid;
	delete[] inds;
	delete[] keepx;
	delete[] keepy;
	delete[] inds_keep;
	delete[] sorted_keep_idx;
	delete[] nms_sorted_xy;
	delete[] toremove;
	delete[] samp_pts;
	delete[] desc;
	delete[] coarse_desc;

	// nodust = nullptr;
	dense_p = nullptr;
	expSum = nullptr;
	nodust_t = nullptr;
	heatmap = nullptr;
	heatmap_r = nullptr;
	pts_xy = nullptr;
	pt_score = nullptr;
	sorted_idx = nullptr;
	sorted_xy = nullptr;
	sorted_score = nullptr;
	grid = nullptr;
	inds = nullptr;
	keepx = nullptr;
	keepy = nullptr;
	inds_keep = nullptr;
	sorted_keep_idx = nullptr;
	nms_sorted_xy = nullptr;
	toremove = nullptr;
	samp_pts = nullptr;
	desc = nullptr;
	coarse_desc = nullptr;
}