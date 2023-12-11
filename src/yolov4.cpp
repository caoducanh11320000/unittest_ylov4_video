#include <iostream>
#include "trt_inference.h"

#define DEVICE 0  // GPU id


using namespace IMXAIEngine;
using namespace nvinfer1;


TRT_Inference test1;
std::vector<std::string> file_image;
std::vector<cv::Mat> input_img; 
std::vector< std::vector<trt_results>> results;

int main(int argc, char** argv){

    cudaSetDevice(DEVICE);

    if (argc == 2 && std::string(argv[1]) == "-s") {
        // co the goi ham API model o day
        test1.trt_APIModel();     
    } 
    else if (argc == 3 && std::string(argv[1]) == "-d") {
        // goi ham init
        test1.init_inference(argv[2],file_image);
    } 
    else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov4 -s  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov4 -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    /// thuc hien ham do_Inference o day
    std::string folder= std::string(argv[2]);
    for(int i=0; i< (int)file_image.size(); i++){
        cv::Mat img = cv::imread(folder + "/" + file_image[i]);
        if(!img.empty()) input_img.push_back(img);
        else std::cout << "That bai" << std::endl;
    }
    for (int i=0; i< (int)file_image.size(); i++)
    {
        std::cout <<"Ten anh la:" << file_image[i] <<std::endl;
    }
    
  
    test1.trt_detection(input_img, results);
    std::cout << results.size() << std::endl;
    for (int i = 0; i < (int)results.size(); i++)
    {
        auto x= results[i];
        std::cout <<"Anh" << std::endl;
        for(int j=0; j< (int)x.size(); j++){
            std::cout <<"Bounding box: " << x[j].ClassID << x[i].Confidence << x[i].bbox[0]<< x[i].bbox[1] << x[i].bbox[2] << x[i].bbox[3] <<std::endl;
        }
    }
    

}