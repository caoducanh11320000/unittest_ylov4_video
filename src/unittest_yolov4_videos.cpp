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
    std::string outputDir = "images";
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
        id_img ++;
    }


    std::cout<< "So luong anh la: " <<id_img<< std::endl;
    for(int i=0; i< id_img; i++)
    {   std::cout << "Bat dau doc anh" <<std::endl;
        cv::Mat inp_img = cv::imread( outputDir + "/frame_" + std::to_string(i) + ".png" );
        if(!inp_img.empty()){
        trt_input.id_img= i;
        trt_input.input_img= inp_img;
        trt_inputs.push_back(trt_input);
        }
        else{
            std::cout<<"Thuc hien ko thanh cong voi anh: " << i<< std::endl;
        }
        if((i+1) % 8 ==0){
            
            std::cout << "So luong dau vao: " << trt_inputs.size() << std::endl;
            test1.trt_detection(trt_inputs, trt_outputs,outputDir );
            std::cout << "So luong dau ra: " << trt_outputs.size()<< std::endl;
            
            trt_inputs.clear();
            std::vector< IMXAIEngine::trt_input> ().swap(trt_inputs) ;

        }
    }
    
}