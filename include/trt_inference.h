#ifndef TRT_INFERENCE_H
#define TRT_INFERENCE_H

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "utils.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "mish.h"
#include "yololayer.h"
#include <algorithm>

using namespace nvinfer1;

namespace IMXAIEngine
{

    typedef enum{
        TRT_RESULT_SUCCESS,
        TRT_RESULT_ERROR
    } trt_error;


    typedef struct
    {
        uint32_t ClassID;	
		float Confidence;	
        float bbox[4];
    } trt_results;
    

    class TRT_Inference
    {
    private:
        IRuntime* runtime= NULL;
        ICudaEngine* engine= NULL;
        IExecutionContext* context= NULL;

    public:
        TRT_Inference();
        ~TRT_Inference(){
            if (context != NULL)
            {
                context->destroy();
            }
            if (context != NULL)
            {
                engine->destroy();
            }
            if (context != NULL)
            {
                runtime->destroy();
            }
            printf("Da huy Inference \n");
        }
        trt_error init_inference(const char * input_folder, std::vector<std::string> &file_names); 
        trt_error trt_APIModel();
        trt_error trt_detection(std::vector<cv::Mat> &input_img, std::vector< std::vector<trt_results>> &results);
    };

} // namespace IMXAIEngine

#endif