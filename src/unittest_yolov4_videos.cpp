#include <iostream>
#include "trt_inference.h"
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>

#define DEVICE 0  // GPU id


using namespace IMXAIEngine;
using namespace nvinfer1;
namespace fs = std::experimental::filesystem;

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
        return 1;
    } 
    else if (argc == 4 && std::string(argv[1]) == "-d") {
        // goi ham init
        test1.init_inference(std::string(argv[2]));  // truyen vao path cua video
     
    } 
    else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov4 -s  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov4 -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // chuyen video thanh anh
    std::string video_path = std::string(argv[3]);
    cv::VideoCapture cap(video_path);
    if(!cap.isOpened()){
        std::cout <<" Khong the mo video" <<std::endl;
    }
    // tao duong dan thu muc
    std::string outputDir = "image";
    fs::create_directories(outputDir);
    int id_img=0;
    IMXAIEngine::trt_input trt_input;
    while(true){
        cv::Mat img;
        cap >> img;
        if(img.empty()){
            std::cout<<"Het video" <<std::endl;
            break;
        }

        // Lưu khung hình thành ảnh trong thư mục cụ thể
        std::string filename = outputDir + "/frame_" + std::to_string(id_img) + ".png";
        cv::imwrite(filename, img);  // ti nua truyen outputDir vao trt_detection

        trt_input.id_img= id_img;
        trt_input.input_img= img;
        trt_inputs.push_back(trt_input);
        id_img ++;
    }

   
    test1.trt_detection(trt_inputs, trt_outputs,outputDir );

    std::cout << "so luong ket qua:" << trt_outputs.size() << std::endl;

    for (int i = 0; i < (int) trt_outputs.size(); i++) 
    {
    auto x = trt_outputs[i];
    std::cout << "ID anh: " <<x.id << std::endl;
    std::cout << x.results.size() << std::endl;
    for (int j = 0; j < (int)x.results.size(); j++)
    {
        std::cout << "Bounding box: " << x.results[j].ClassID<<" " << x.results[j].Confidence<<" " << x.results[j].bbox[0]<<" " << x.results[j].bbox[1]<<" " << x.results[j].bbox[2]<<" " << x.results[j].bbox[3] << std::endl;
    }
    }

}