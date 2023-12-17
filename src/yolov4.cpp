#include <iostream>
#include "trt_inference.h"

#define DEVICE 0  // GPU id


using namespace IMXAIEngine;
using namespace nvinfer1;


TRT_Inference test1;
std::vector<std::string> file_image;
std::vector<cv::Mat> input_img; 
std::vector< std::vector<trt_results>> results;
std::vector<IMXAIEngine::input> Input(10); // neu ko khai bao so luong se bi loi
// khai bao size cho dau vao
int sizes= 0;

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

        // de test 
    for (int i=0; i< (int)file_image.size(); i++)
    {
        std::cout <<"Ten anh la:" << file_image[i] <<std::endl;
    }

    std::string folder= std::string(argv[2]);
    for(int i=0; i< (int)file_image.size(); i++){

        std::cout << "Thuc hien voi anh:" << i <<std::endl;

        cv::Mat img = cv::imread(folder + "/" + file_image[i]);
        if(!img.empty()) {
            //input_img.push_back(img); // danh so ID o day luon
            Input[i].input_img.push_back(img);
            Input[i].id_img= i;    
            sizes++;
            std::cout<< "thanh cong voi anh" << i <<std::endl;
            }
        else std::cout << "That bai" << std::endl;
    }

    
    test1.trt_detection(Input, results, sizes);

    std::cout << "so luong ket qua:" << results.size() << std::endl;

    for (int i = 0; i < (int) results.size(); i++) 
    {
    auto x = results[i];
    std::cout << "Anh" << std::endl;
    std::cout << x.size() << std::endl;
    for (int j = 0; j < (int)x.size(); j++)
    {
        std::cout << "Bounding box: " << x[j].ClassID << x[j].Confidence << x[j].bbox[0] << x[j].bbox[1] << x[j].bbox[2] << x[j].bbox[3] << std::endl;
    }
    }

}