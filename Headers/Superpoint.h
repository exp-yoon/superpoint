#pragma once
#include "ONNXCore.h"


class Superpoint : public ONNXCore
{
public:
	Superpoint();
	~Superpoint();

	bool GetOutput(float*** pSuperpointLocationArray, float*** pSuperpointDescArray);

};