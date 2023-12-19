#include <iostream>
#include "trt_inference.h"

#define DEVICE 0  // GPU id


using namespace IMXAIEngine;
using namespace nvinfer1;


TRT_Inference test1;
std::vector<std::string> file_image;
std::vector<cv::Mat> input_img; 
std::vector< std::vector<trt_results>> results;
std::vector< IMXAIEngine:: trt_input> trt_inputs;
std::vector< IMXAIEngine:: trt_output> trt_outputs;

// khai bao size cho dau vao
int sizes= 0;


int main(int argc, char** argv){

    cudaSetDevice(DEVICE);

    if (argc == 3 && std::string(argv[1]) == "-s") { // modify argc if you want
        // co the goi ham API model o day
        test1.trt_APIModel( std::string(argv[2]) );     
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

        // de test 
    for (int i=0; i< (int)file_image.size(); i++)
    {
        std::cout <<"Ten anh la:" << file_image[i] <<std::endl;
    }

    std::string folder= std::string(argv[2]);
    for(int i=0; i< (int)file_image.size(); i++){

        std::cout << "Thuc hien voi anh:" << i <<std::endl;

        cv::Mat img = cv::imread(folder + "/" + file_image[i]);
        IMXAIEngine:: trt_input trt_input;
        if(!img.empty()) {
            //input_img.push_back(img); // danh so ID o day luon
            trt_input.input_img= img;
            trt_input.id_img = i;
            trt_inputs.push_back(trt_input);
            std::cout<< "thanh cong voi anh" << i <<std::endl;
            }
        else std::cout << "That bai" << std::endl;
    }

    
    test1.trt_detection(trt_inputs, trt_outputs);

    std::cout << "so luong ket qua:" << trt_outputs.size() << std::endl;

    for (int i = 0; i < (int) trt_outputs.size(); i++) 
    {
    auto x = trt_outputs[i];
    std::cout << "ID anh: " <<x.id << std::endl;
    std::cout << x.results.size() << std::endl;
    for (int j = 0; j < (int)x.results.size(); j++)
    {
        std::cout << "Bounding box: " << x.results[j].ClassID << x.results[j].Confidence << x.results[j].bbox[0] << x.results[j].bbox[1] << x.results[j].bbox[2] << x.results[j].bbox[3] << std::endl;
    }
    }

}