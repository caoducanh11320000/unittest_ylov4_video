#ifndef TRT_INFERENCE_HPP
#define TRT_INFERENCE_HPP

#include <vector>

namespace IMXAIEngine
{

    typedef enum{
        TRT_RESULT_SUCCESS,
        TRT_RESULT_ERROR
    } trt_error;


    typedef struct
    {
            
    } trt_results;
    

    class TRT_Inference
    {
    private:
        trt_error trt_release(void);
        
    public:
        TRT_Inference();
        ~TRT_Inference(){
            this->trt_release();
        }
    };
    trt_error init_inference(int argc, char **argv);
    trt_error trt_detection(std::vector<cv::Mat> &input_img, std::vector<trt_results> &results);

} // namespace IMXAIEngine

#endif