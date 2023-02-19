#pragma once
#include "ONNXCore.h"


class Segmentation : public ONNXCore
{
public:
	Segmentation();
	~Segmentation();
	bool GetWholeImageSegmentationResults(unsigned char*, float, bool);
};