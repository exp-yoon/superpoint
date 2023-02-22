#pragma once
#include "SpRun.h"
#include <iostream>

SpRun::SpRun()
{
}

SpRun::~SpRun()
{
}

void SpRun::set_count(int value) {
	count = value;
}

int SpRun::get_count() {
	return count;
}

void SpRun::get_sp_result(int** pts_result, float* score_result, float** desc_result) {

	memcpy(pts_result[0], pts_save[0], sizeof(int) * count);
	memcpy(pts_result[1], pts_save[1], sizeof(int) * count);

	memcpy(score_result, score_save, sizeof(float) * count);

	for (size_t chanIdx = 0; chanIdx < desc_channel; ++chanIdx) {
		memcpy(desc_result[chanIdx], desc_save[chanIdx], sizeof(float) * count);
		int tttttt = 0;
	}
}

void SpRun::grid_sample(float*** coarse_desc, float** samp_pts, long long count, float** desc) {

	float** result = new float* [desc_channel]; //(desc_channel,count)
	float* norm = new float[count];

	for (int i = 0; i < desc_channel; i++) {
		float* result_2 = new float[count];
		result[i] = result_2;
	}

	for (int channel = 0; channel < desc_channel; channel++) {
		for (int p_cnt = 0; p_cnt < count; p_cnt++) {
			float cvt_x = ((samp_pts[p_cnt][0] + 1.) * w - 1.) / 2.;
			float cvt_y = ((samp_pts[p_cnt][1] + 1.) * h - 1.) / 2.;

			float grid_a = (cvt_x)-floor(cvt_x);
			float grid_b = (cvt_y)-floor(cvt_y);
			float grid_x = coarse_desc[channel][int(floor(cvt_y))][int(floor(cvt_x))];
			float grid_y = coarse_desc[channel][int(floor(cvt_y))][int(ceil(cvt_x))];
			float grid_w = coarse_desc[channel][int(ceil(cvt_y))][int(floor(cvt_x))];
			float grid_z = coarse_desc[channel][int(ceil(cvt_y))][int(ceil(cvt_x))];

			float out = (grid_a * grid_b * grid_z) + ((1 - grid_a) * (1 - grid_b) * grid_x) + ((1 - grid_a) * grid_b * grid_w) + ((1 - grid_b) * grid_a * grid_y);
			result[channel][p_cnt] = out;
			int ttttttttttt = 0;
		}
	}

	//norm ��� (count�� ����)
	for (int p_cnt = 0 ; p_cnt < count ; p_cnt++){
		float norm_acc= 0;
		for (int channel = 0; channel < desc_channel; channel++) {
			norm_acc += pow(result[channel][p_cnt],2);
			if (channel == desc_channel - 1) {
				norm[p_cnt] = sqrt(norm_acc);
			}
		}
	}

	for (int channel = 0; channel < desc_channel; channel++) {
		for (int p_cnt = 0; p_cnt < count; p_cnt++) {
			result[channel][p_cnt] = result[channel][p_cnt] / norm[p_cnt];
		}
	}

	for (size_t chanIdx = 0; chanIdx < desc_channel; ++chanIdx) {
		memcpy(desc[chanIdx], result[chanIdx], sizeof(float) * count);
	}

	delete[] result;
	result = nullptr;
}

void SpRun::calc(float*** semi, float*** coarse_desc_, cv::Mat img) {

	//point location pixel-wise softmax
	float** nodust = new float* [loc_channel - 1];
	float** dense_p = new float* [loc_channel]; //(3,62*62)������ semi�� exp���
	float* expSum = new float[pixnum]; //pixel wise expsum
	for (size_t chanIdx = 0; chanIdx < loc_channel; chanIdx++) {

		float* dense_c = new float[pixnum];
		float* nodust_c = new float[pixnum]();
		for (size_t pixIdx = 0; pixIdx < pixnum; pixIdx++) {
			float val0 = exp(semi[0][chanIdx][pixIdx]);

			if (chanIdx == 0) {
				expSum[pixIdx] = val0;
			}
			else {
				expSum[pixIdx] = expSum[pixIdx] + val0;
			}

			dense_c[pixIdx] = val0;
			int t = 0;
		}
		dense_p[chanIdx] = dense_c;
		nodust[chanIdx] = nodust_c;
	}

	for (size_t chanIdx = 0; chanIdx < loc_channel - 1; chanIdx++) {
		for (size_t pixIdx = 0; pixIdx < pixnum; pixIdx++) {
			nodust[chanIdx][pixIdx] = dense_p[chanIdx][pixIdx] / (expSum[pixIdx]);
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
	int cnt = 0; //heatmap���� conf_threshold �̻��� ��ǥ ���� 
	for (int nh = 0; nh < h; nh++) {
		for (int c1 = 0; c1 < cell; c1++) {
			for (int nw = 0; nw < w; nw++) {
				for (int c2 = 0; c2 < cell; c2++) {
					heatmap_r[(nh * w * cell * cell) + (c1 * w * cell) + (nw * cell) + (c2)] = heatmap[nh][c1][nw][c2];
					if (heatmap[nh][c1][nw][c2] >= conf_thresh)
						cnt += 1;
				}
			}
		}
	}

	//���Ⱑ python�� superpoint.run���� pts.
	int** pts_xy = new int* [2];
	int* pt_y = new int[cnt]; //pts[0]
	int* pt_x = new int[cnt]; //pts[1]
	float* pt_score = new float[cnt]; //pts[2]

	int cnt_idx = 0;
	for (long long pixIdx = 0; pixIdx < h * w * cell * cell; pixIdx++) {
		if (heatmap_r[pixIdx] >= conf_thresh) {
			long long idx_y = pixIdx / (h * cell);
			long long idx_x = pixIdx % (w * cell);

			pt_x[cnt_idx] = idx_x;
			pt_y[cnt_idx] = idx_y;
			pt_score[cnt_idx] = heatmap_r[pixIdx];
			cnt_idx++;
		}
	}
	pts_xy[0] = pt_y;
	pts_xy[1] = pt_x;


	//nms ����
	//score �������� ����

	int* sorted_idx = new int[cnt]; //score �������� ������ �ε��� -> python : inds1
	for (int i = 0; i < cnt; i++)
		sorted_idx[i] = i;

	for (int i = 0; i < cnt; i++) {
		for (int j = 0; j < cnt - 1; j++) {
			if (pt_score[sorted_idx[j]] < pt_score[sorted_idx[j + 1]]) {
				int temp = sorted_idx[j];
				sorted_idx[j] = sorted_idx[j + 1];
				sorted_idx[j + 1] = temp;
			}
		}
	}


	// �������� ������ index�� x,y�� ���� 
	int** sorted_xy = new int* [2];
	int* sorted_x = new int[cnt]; //�������� ���ĵ� x��ǥ
	int* sorted_y = new int[cnt]; //�������� ���ĵ� y��ǥ
	float* sorted_score = new float[cnt]; //�������� ���ĵ� score
	for (int i = 0; i < cnt; i++) {
		sorted_x[i] = pts_xy[0][sorted_idx[i]];
		sorted_y[i] = pts_xy[1][sorted_idx[i]];
		sorted_score[i] = pt_score[sorted_idx[i]];
	}

	sorted_xy[0] = sorted_x;
	sorted_xy[1] = sorted_y;

	int** grid = new int* [org_h + 2 * pad]; // Track NMS data ,zero padding 
	int** inds = new int* [org_h]; // store indices of points

	for (int i = 0; i < org_h + 2 * pad; i++) {
		int* grid_w = new int[org_w + 2 * pad]();
		grid[i] = grid_w;
	}

	for (int i = 0; i < org_h; i++) {
		int* inds_w = new int[org_w]();
		inds[i] = inds_w;
	}

	//������ ��ǥ������� grid�� 1����, inds�� score ����(�ش���ǥ�� score ������ ����)
	for (int i = 0; i < cnt; i++) {
		grid[sorted_xy[0][i]+pad][sorted_xy[1][i] + pad] = 1; //padding ���
		inds[sorted_xy[0][i]][sorted_xy[1][i]] = i;
	}

	//nms�ؼ� �ֺ� ����Ʈ ������(�����ֵ��� grid�� -1�� ��ŷ)
	int nms_count = 0;
	for (int i = 0; i < cnt; i++) {
		int pt_y = sorted_xy[0][i] + pad;
		int pt_x = sorted_xy[1][i] + pad;
		int a = 0; 
		if (grid[pt_y][pt_x] == 1) {
			
			for (int ph = 0; ph < 2*pad+1; ph++) {
				for (int pw = 0; pw < 2*pad+1 ; pw++) {
					grid[pt_y-pad+ph][pt_x-pad+ pw] = 0;
				}
			}
			grid[pt_y][pt_x] = -1;
			nms_count += 1;
		}
	}

	set_count(nms_count);

	//nms�ؼ� ���� �ֵ�(grid = -1)�� ��ǥ�� ������
	int* keepx = new int[count];
	int* keepy = new int[count];

	int keepcnt = 0;
	for (long long i = 4; i < org_h + pad; i++) {//
		for (long long j = 4; j < org_w + pad; j++) {
			if (grid[i][j] == -1) {
				keepx[keepcnt] = j - pad;
				keepy[keepcnt] = i - pad;
				keepcnt++;
			}
		}
	}

	//��ǥ�� score ������ �Ű����� inds������ nms�ؼ� ���� ��ǥ���� ������ inds_keep�� ����
	int* inds_keep = new int[count]; //nms ���� �� ���� ��ǥ�� score ����
	for (int i = 0; i < count; i++) {
		inds_keep[i] = inds[keepy[i]][keepx[i]];
	}

	//nms�� ���� score��
	float* keep_score = new float[count];
	for (int i = 0; i < count; i++) {
		keep_score[i] = sorted_score[inds_keep[i]];
	}

	//nms�� ���� score�� �������� �ε��� ���ϱ� 
	int* sorted_keep_idx = new int[count];
	for (int i = 0; i < cnt; i++)
		sorted_keep_idx[i] = i;

	for (int i = 0; i < count; i++) {
		for (int j = 0; j < count - 1; j++) {
			if (keep_score[sorted_keep_idx[j]] < keep_score[sorted_keep_idx[j + 1]]) {
				int temp = sorted_keep_idx[j];
				sorted_keep_idx[j] = sorted_keep_idx[j + 1];
				sorted_keep_idx[j + 1] = temp;
			}
		}
	}

	//nms������ score���� ���������� x,y,score 
	int** nms_sorted_xy = new int* [2];
	int* nms_sorted_x = new int[count];
	int* nms_sorted_y = new int[count];
	float* nms_sorted_score = new float[count];

	for (int i = 0; i < count; i++) {
		nms_sorted_x[i] = keepx[sorted_keep_idx[i]];
		nms_sorted_y[i] = keepy[sorted_keep_idx[i]];
		nms_sorted_score[i] = keep_score[sorted_keep_idx[i]];
	}

	//����..���� xy�ٲ�Ű����� Ȯ���غ��ध��..
	nms_sorted_xy[1] = nms_sorted_y;
	nms_sorted_xy[0] = nms_sorted_x;

	//nms ��//

	//�̹����� ����� ��ǥ ����
	bool* toremove = new bool[count];
	int tm_count = 0;
	for (int i = 0; i < count; i++) {
		if (((nms_sorted_xy[0][i] < border) || (nms_sorted_xy[0][i] >= (org_w - border))) || ((nms_sorted_xy[1][i] < border) || (nms_sorted_xy[1][i] >= (org_h - border))))
			toremove[i] = false;
		else {
			toremove[i] = true;
			tm_count++;
		}
	}

	set_count(tm_count);

	pts_save = new int* [2];
	int* pts_save_x = new int[count];
	int* pts_save_y = new int[count];
	score_save = new float[count];

	pts_save[1] = pts_save_y;
	pts_save[0] = pts_save_x;

	int idx = 0;
	for (int cnt = 0; cnt < nms_count; cnt++) {
		if (toremove[cnt] == true) {
			pts_save[0][idx] = nms_sorted_xy[0][idx];
			pts_save[1][idx] = nms_sorted_xy[1][idx];
			score_save[idx] = nms_sorted_score[idx];
			
			int hjj = 0;
			idx++;
		}
	}

	//feature point Ȯ�ο� ���
	//cv::Mat imgmat(cv::Size(500, 500), CV_8UC3);
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
		samp_xy[0] = ((nms_sorted_xy[0][i]) / (org_w / 2.)) - 1.; //normalized(-1~1)
		samp_xy[1] = ((nms_sorted_xy[1][i]) / (org_h / 2.)) - 1.; //normalized(-1~1)
		samp_pts[i] = samp_xy;
	}

	//grid_sample�� ��� descriptor ������ ��
	desc_save = new float* [desc_channel]; //(desc_channel,count)
	for (int i = 0; i < desc_channel; i++) {
		float* desc_2d = new float[count];
		desc_save[i] = desc_2d;
	}

	//coarse_desc reshape
	float*** coarse_desc = new float** [desc_channel]; //(desc_channel , h,w )
	

	for (int c = 0; c < desc_channel; c++) {
		float** desc_h = new float* [h];
		for (int dh = 0; dh < h; dh++) {
			float* desc_w = new float[w];
			memcpy(desc_w, coarse_desc_[0][c] + w * dh, sizeof(float) * w);
			desc_h[dh] = desc_w;
			int a = 0;
		}
		coarse_desc[c] = desc_h;
		int a = 0;
	}

	//grid_sample -> interpolation 
	grid_sample(coarse_desc, samp_pts, count, desc_save);

	//delete[] nodust;
	//delete[] dense_p;
	/*delete[] expSum;
	delete[] nodust_t;
	delete[] heatmap;
	delete[] heatmap_r;
	delete[] pts_xy;
	delete[] pt_score;
	delete[] sorted_keep_idx;
	delete[] sorted_idx;
	delete[] sorted_xy;
	delete[] sorted_score;
	delete[] sorted_keep_idx;
	delete[] grid;
	delete[] inds;
	delete[] keepx;
	delete[] keepy;
	delete[] keep_score;
	delete[] inds_keep;
	delete[] nms_sorted_xy;
	delete[] toremove;
	delete[] samp_pts;
	delete[] coarse_desc;*/

	//nodust = nullptr;
	//dense_p = nullptr;
	//expSum = nullptr;
	//nodust_t = nullptr;
	//heatmap = nullptr;
	//heatmap_r = nullptr;
	//pts_xy = nullptr;
	//pt_score = nullptr;
	//sorted_keep_idx = nullptr;
	//sorted_idx = nullptr;
	//sorted_xy = nullptr;
	//sorted_score = nullptr;
	//grid = nullptr;
	//inds = nullptr;
	//keepx = nullptr;
	//keepy = nullptr;
	//inds_keep = nullptr;
	//nms_sorted_xy = nullptr;
	//toremove = nullptr;
	//samp_pts = nullptr;
	//coarse_desc = nullptr;

}