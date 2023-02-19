#pragma once
#include "ONNXCore.h"


class Twinnet : public ONNXCore
{
public:
	Twinnet();
	~Twinnet();
	bool GetOutput(float** twinnetResultArray, float scoreCut, bool bIsOutputNormalized);
};