#pragma once

struct DetectionResult
{
	int x;
	int y;
	int w;
	int h;
	float Objectness;
	int BestClass;
	float Score;
};