#pragma once
#include <vector>
#include "ResultStructure.h"

#define __DLLEXPORT __declspec(dllexport)

#define CLASSIFICATION	0
#define SEGMENTATION	1
#define DETECTION		2
#define TWINNET			3
#define SUPERPOINT		4

class Classification;
class Segmentation;
class Detection;
class Twinnet;
class Superpoint;

template class __DLLEXPORT std::vector<DetectionResult>;

namespace ATI_ONNX
{
	class AI
	{
	private:
		int mTaskType = -1;

		Classification* mClassification;
		Segmentation* mSegmentation;
		Detection* mDetection;
		Twinnet* mTwinnet;
		Superpoint* mSuperpoint;
		const char* mVersion;

	public:
		/**
			@details AI Class 생성자
			@param int taskType : AI Task 타입 (0 : Classification, 1 : Segmentation, 2 : Detection, 3 : Twinnet)
			@return 없음
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-06-02|김규태|onnx runtime 기반 library 구현|-
			@warning 없음
		**/
		__DLLEXPORT AI(int taskType);

		/**
			@details AI Class 소멸자
			@param 없음
			@return 없음
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-06-02|김규태|onnx runtime 기반 library 구현|-
			@warning 없음
		**/
		__DLLEXPORT ~AI();

		/**
			@details ONNX 기반 Model파일 로드
			@param const wchar_t* modelPath : 학습된 AI 모델 Path
			@param bool bTensorRT : Tensor RT 사용 여부
			@param bool bUseCache : Tensor RT 사용 시 Cache 사용 여부
			@param const char* cachePath : Cache 경로. 경로 주지 않으면 model 경로에 cache 생성됨.
			@return AI 모델 로드 성공 여부 Bool값 반환
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-06-02|김규태|onnx runtime 기반 AI load 함수 구현|-
			2022-07-12|김규태|float16 model load시 TensorRT Library Executor 설정되도록 수정|-
			@warning 없음
		**/
		__DLLEXPORT bool LoadModel(const wchar_t* modelPath, bool bTensorRT = false, bool bUseCache = true, const char* cachePath = nullptr);

		//Run Session : Deprecated on onnx runtime.
		__DLLEXPORT bool Run(float** inputImgArr, bool bNormalize = false);

		//Run Session : Deprecated on onnx runtime.
		__DLLEXPORT bool Run(float*** inputImgArr, bool bNormalize = false);

		//Run Session : Deprecated on onnx runtime.
		__DLLEXPORT bool Run(unsigned char** inputImgArr, bool bNormalize = false);

		//Run Session : Deprecated on onnx runtime.
		__DLLEXPORT bool Run(unsigned char*** inputImgArr, bool bNormalize = false);

		//Run Session : Deprecated on onnx runtime.
		__DLLEXPORT bool Run(unsigned char** inputImg, int imgSizeX, int imgSizeY,
			int cropSizeX, int cropSizeY, int overlapSizeX, int overlapSizeY, int buffPosX, int buffPosY,
			bool bNormalize = false, bool bConvertGrayToColor = false, bool bReloadEveryRun = false);

		/**
			@details ONNX inference session 구동
			@param float*** inputImgArr : 3중 float array, i : operator idx, j : img idx, k : pixel idx
			@param int imgNum : Image 개수
			@param int batch : Inference batch size
			@param bool bNormalize : Image를 0~1로 normalize하여 AI에 입력할지 여부
			@return AI inference 성공 여부 반환	
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-12-28|김규태|함수 구현|-
			2022-12-29|김규태|Image 전처리 SIMD 적용|-
			@warning -
		**/
		__DLLEXPORT bool Run(float*** inputImgArr, int imgNum, int batch, bool bNormalize = false);

		/**
			@details ONNX inference session 구동
			@param float*** inputImgArr : 2중 float array, i : img idx, j : pixel idx
			@param int imgNum : Image 개수
			@param int batch : Inference batch size
			@param bool bNormalize : Image를 0~1로 normalize하여 AI에 입력할지 여부
			@return AI inference 성공 여부 반환
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-12-28|김규태|함수 구현|-
			2022-12-29|김규태|Image 전처리 SIMD 적용|-
			@warning -
		**/
		__DLLEXPORT bool Run(float** inputImgArr, int imgNum, int batch, bool bNormalize = false);

		//Run Session : Deprecated on onnx runtime.
		__DLLEXPORT bool Run(unsigned char*** inputImgArr, int batch, bool bNormalize = false);

		/**
			@details ONNX inference session 구동
			@param float*** inputImgArr : 2중 uchar array, i : img idx, j : pixel idx
			@param int imgNum : Image 개수
			@param int batch : Inference batch size
			@param bool bNormalize : Image를 0~1로 normalize하여 AI에 입력할지 여부
			@return AI inference 성공 여부 반환
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-12-28|김규태|함수 구현|-
			@warning 현재 SIMD를 지원하지 않음
		**/
		__DLLEXPORT bool Run(unsigned char** inputImgArr, int imgNum, int batch, bool bNormalize = false);

		/**
			@details ONNX inference session 구동
			@param unsigned char** inputImg : 2중 uchar array로 정의된 image pointer
			@param int imgSizeX : 전체 Image 크기 (width)
			@param int imgSizeY : 전체 Image 크기 (height)
			@param int cropSizeX : AI inference 시 들어갈 image crop 크기 (width)
			@param int cropSizeY : AI inference 시 들어갈 image crop 크기 (height)
			@param int overlapSizeX : Image crop 시 겹치는 크기 (width)
			@param int overlapSizeY : Image crop 시 겹치는 크기 (height)
			@param int buffPosX : AI 적용될 이미지가 시작되는 Buffer Position (width)
			@param int buffPosY : AI 적용될 이미지가 시작되는 Buffer Position (height)
			@param int batch : 한 batch에 들어갈 image 갯수
			@param bool bNormalize : Image를 0~1로 normalize하여 AI에 입력할지 여부
			@param bool bConvertGraytoColor : 흑백 Image를 3채널 이미지로 입력할지 여부
			@return AI inference 성공 여부 반환
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-06-09|김규태|Session 구동 함수 onnx 기반으로 대체|-
			2022-06-22|김규태|FP16 모델 대응기능 추가|-
			2022-06-24|김규태|Image 처리 시 SIMD 기능 추가|-
			2022-07-12|김규태|FP16 Model TensorRT 대응 기능 추가|-
			@warning 현재 FP16 Model은 SIMD를 지원하지 않음
		**/
		__DLLEXPORT bool Run(unsigned char** inputImg, int imgSizeX, int imgSizeY, int cropSizeX, int cropSizeY,
			int overlapSizeX, int overlapSizeY, int buffPosX, int buffPosY, int batch,
			bool bNormalize = false, bool bConvertGraytoColor = false);

		/**
			@details ONNX 3d inference session 구동
			@param unsigned char** inputImg : 2중 uchar array로 정의된 image pointer 시작점
			@param int imgSizeX : 개별 3D image 크기 (width)
			@param int imgSizeY : 개별 3D image 크기 (height)
			@param int layerNum : 3D Layer 개수
			@param int cropSizeX : AI inference 시 들어갈 image crop 크기 (width)
			@param int cropSizeY : AI inference 시 들어갈 image crop 크기 (height)
			@param int overlapSizeX : Image crop 시 겹치는 크기 (width)
			@param int overlapSizeY : Image crop 시 겹치는 크기 (height)
			@param int buffPosX : AI 적용될 이미지가 시작되는 Buffer Position (width)
			@param int buffPosY : AI 적용될 이미지가 시작되는 Buffer Position (height)
			@param int batch : 한 batch에 들어갈 image 갯수
			@param bool bNormalize : Image를 0~1로 normalize하여 AI에 입력할지 여부
			@param bool bConvertGraytoColor : 흑백 Image를 3채널 이미지로 입력할지 여부
			@return AI inference 성공 여부 반환
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-08-11|김규태|기능 구현|-
		**/
		__DLLEXPORT bool Run(unsigned char** inputImg, int imgSizeX, int imgSizeY, int layerNum,
			int cropSizeX, int cropSizeY, int overlapSizeX, int overlapSizeY, int buffPosX, int buffPosY,
			int batch, bool bNormalize = false, bool bConvertGraytoColor = false);

		/**
			@details Instance 메모리 해제
			@param 없음
			@return 성공 여부 반환
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-06-09|김규태|함수 구현|-
			@warning 없음
		**/
		__DLLEXPORT bool FreeModel();

		/**
			@details Classification 결과 반환
			@param float** classificationResultArr : 결과 반환 받을 float 이중 배열의 포인터
			@return 성공 여부 반환
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-12-28|김규태|함수 구현|-
			@warning 없음
		**/
		__DLLEXPORT bool GetClassificationResults(float** classificationResultArr);

		//Returns segmentation result : Deprecated on onnx runtime.
		__DLLEXPORT bool GetSegmentationResults(float*** SegmentationResultArr);

		/**
			@details Twinnet 결과 반환
			@param float** twinnetResultArray : 결과 반환 받을 float 이중 배열의 포인터
			@param float scoreCut : Softmax threshold 값
			@param bool bIsOutputNormalized : Model의 Output이 Softmax인지 여부
			@return 성공 여부 반환
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-12-28|김규태|함수 구현|-
			2022-12-29|김규태|threshold, softmax 여부 파라미터 추가|-
			@warning 없음
		**/
		__DLLEXPORT bool GetTwinnetResults(float** twinnetResultArray, float scoreCut = 0.45, bool bIsOutputNormalized = false);

		__DLLEXPORT bool GetSuperpointResults(float*** SuperpointLocationArray, float*** SuperpointDescArray);


		//Returns detection result : Deprecated on onnx runtime.
		__DLLEXPORT bool GetDetectionResultsByArray(DetectionResult** detectionResultArr, int* boxNumArr, float iouThresh = 0.5, float scoreThresh = 0.3);

		/**
			@details Detection Inference 결과 반환
			@param DetectionResult* detectionResultArr : 반환될 Detection box 배열
			@param int& boxNum : 반환될 Box 갯수
			@param int clsNum : Model class 갯수
			@param float iouThresh : NMS 적용 시 IOU threshold (default 0.5)
			@param float scoreThresh : NMS 적용 시 score threshold (default 0.25)
			@return AI inference 결과 반환 성공 여부 반환
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-06-10|김규태|Onnx 기반 함수 구현|-
			2022-06-15|김규태|nms 최적화|-
			2022-06-22|김규태|FP16 모델 대응기능 추가|-
			@warning 없음
		**/
		__DLLEXPORT bool GetWholeImageDetectionResults(DetectionResult* detectionResultArr, int& boxNum, int clsNum, float iouThresh = 0.5, float scoreThresh = 0.25);

		/**
			@details Segmentation Inference 결과 반환
			@param unsigned char* outputImg : 반환될 uchar* 형식의 이미지 배열
			@param float scoreCut : Segmentation softmax score threshold, 더 작은 경우 background로 간주 (default 0.5)
			@param bool bIsOutputNormalized : AI segmentation model의 output이 softmaxr 값인지 여부(default true)
			@return AI inference 결과 반환 성공 여부 반환
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-06-10|김규태|Onnx 기반 함수 구현|-
			2022-06-17|김규태|속도 최적화|-
			2022-06-22|김규태|FP16 모델 대응기능 추가|-
			@warning 없음
		**/
		__DLLEXPORT bool GetWholeImageSegmentationResults(unsigned char* outputImg, float scoreCut = 0.5, bool bIsOutputNormalized = false);

		/**
			@details Model load 여부 반환
			@param 없음
			@return Model load 여부 반환
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-12-28|김규태|문서화|-
			@warning 없음
		**/
		__DLLEXPORT bool IsModelLoaded();

		/**
			@details Input dimension array 반환
			@param 없음
			@return Input dimension array 반환
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-12-28|김규태|문서화|-
			@warning 없음
		**/
		__DLLEXPORT long long** GetInputDims();

		/**
			@details Output dimension array 반환
			@param 없음
			@return Output dimension array 반환
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-12-28|김규태|문서화|-
			@warning 없음
		**/
		__DLLEXPORT long long** GetOutputDims();

		/**
			@details Input Dimension 설정
			@param long long** inputDims : operator별 input dimension 배열
			@param size_t* inputDimLengths : operator별 input dimension 길이 배열
			@return 없음
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-12-28|김규태|함수 구현|-
			@warning 없음
		**/
		void SetInputDims(long long** inputDims, size_t* inputDimLengths);

		/**
			@details Output dimension 설정
			@param long long** outputDims : operator별 output dimension 배열
			@param size_t* outputDimLengths : operator별 output dimension 길이 배열
			@return 없음
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-12-28|김규태|함수 구현|-
			@warning 없음
		**/
		void SetOutputDims(long long** outputDims, size_t* outputDimLengths);

		/**
			@details ATI ONNX Runtime version 반환
			@param 없음
			@return ATI ONNX Runtime version 반환
			@note Patch-notes
			날짜|작성자|설명|비고
			-|-|-|-
			2022-12-28|김규태|문서화|-
			@warning 없음
		**/
		__DLLEXPORT const char* GetVersion();
	};
}