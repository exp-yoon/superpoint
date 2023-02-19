#pragma once
#include "ONNXCore.h"


class Classification : public ONNXCore
{
public:
	Classification();
	~Classification();

	bool GetOutput(float** pClassificationResultArray);
};