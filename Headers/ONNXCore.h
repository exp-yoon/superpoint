#pragma once
#include <iostream>
#include <atltypes.h>
#include <Windows.h>
#include <array>
#include <vector>
#include <string>
#include <time.h>
#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"
#include "Eigen/Eigen"
#include "ResultStructure.h"


class ONNXCore
{
protected:
	const wchar_t* mModelPath;

	bool mIsModelLoaded;
	bool mIsDataLoaded;

	Ort::Session* mSession;
	Ort::RunOptions mRunOptions;
	Ort::SessionOptions* mSessionOptions;
	Ort::Env* mEnv;

	size_t mInputOpNum;
	size_t mOutputOpNum;

	size_t* mInputDimLenArr; //ex) mInputDimLenArr[0] = 4
	size_t* mOutputDimLenArr; //ex) mOutputDimLenArr[0] = 4 mOutputDimLenArr[1] = 2
	long long** mInputDimArr; //ex) mInputDimArr[1][1] = 640  mInputDimArr[1] = [-1, 640, 640, 1]
	long long** mOutputDimArr;

	size_t* mInputUnitSizeArr;
	size_t* mOutputUnitSizeArr;

	std::vector<std::vector<float>> mOutputValues;
	std::vector<std::vector<uint16_t>> mOutputValues16bitFloat;
	std::vector<const char*> mInputOpNames;
	std::vector<const char*> mOutputOpNames;

	CPoint mImageSize;
	CPoint mCropSize;
	CPoint mOverlapSize;

	ONNXTensorElementDataType mInputType;
	ONNXTensorElementDataType mOutputType;

	bool mbIs16bitModel;

public:
	ONNXCore();
	~ONNXCore();

	bool LoadModel(const wchar_t*, bool, bool, const char*);

	bool Run(float*** inputImgArr, int imgNum, int batch, bool bNormalize);
	bool Run(float** inputImgArr, int imgNum, int batch, bool bNormalize);
	bool Run(unsigned char** inputImgArr, int imgNum, int batch, bool bNormalize);

	bool Run(unsigned char**, CPoint, CPoint, CPoint, CPoint, int, bool, bool);
	bool Run(unsigned char**, CPoint, int, CPoint, CPoint, CPoint, int, bool, bool);

	bool FreeModel();
	bool IsModelLoaded();

	long long** GetInputDims();
	long long** GetOutputDims();

	void SetInputDims(long long** inputDims, size_t* inputDimLens);
	void SetOutputDims(long long** outputDims, size_t* outputDimLens);

private:
	bool _Run();
};