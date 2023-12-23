#include "trt_inference.h"


#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define BBOX_CONF_THRESH 0.5

#define BATCH_SIZE 1


using namespace IMXAIEngine;
using namespace nvinfer1;

// do the thay doi sau
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int DETECTION_SIZE = sizeof(Yolo::Detection) / sizeof(float);
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

TRT_Inference::TRT_Inference(){
    printf("khoi tao inference \n");
}

/// Xu li anh 

cv::Mat preprocess_img(cv::Mat& img) { // resize anhr goc ve anh xu  li
    int w, h, x, y;
    float r_w = INPUT_W / (img.cols*1.0);  // Tinh ti le Scale, doj chenh lenh giua dau vao real va dau vao mong muon
    float r_h = INPUT_H / (img.rows*1.0); 
    if (r_h > r_w) {     // cai nao cos do chenh lech nho hon, se duoc uu tien gan gia tri mong muon
        w = INPUT_W;    // W co do chenh lech nho hon
        h = r_w * img.rows;   // H phai dam bao va giu ti le voi W nhu anh goc
        x = 0;                // Toa di theo chieu ngang, muon datj copy
        y = (INPUT_H - h) / 2;  // Toa do theo chieu doc
    } else {
        w = r_h* img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);   // Tao ma tran, 8UC3 tuc la 8 unsignchar 3 kenh mau: R, G, B
    cv::resize(img, re, re.size());   // resize lai anh
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));  //tao ma tran, 128 128 128 tuc la anh nay se co mau xam
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {   // ham de chuyen bounding box tu anh xu li ve anh goc
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2]/2.f;
        r = bbox[0] + bbox[2]/2.f;
        t = bbox[1] - bbox[3]/2.f - (INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3]/2.f - (INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2]/2.f - (INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2]/2.f - (INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3]/2.f;
        b = bbox[1] + bbox[3]/2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r-l, b-t);
}

// thay doi tuy vao model, dung chung dc cho cac model Yolo
float iou(float lbox[4], float rbox[4]) { //tính toán IoU (Intersection over Union) giữa hai bounding box
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) { // ham so sanh do tin cay (kha nang co doi tuong trong khung hinh)
    return a.det_confidence > b.det_confidence;
}

void nms(std::vector<Yolo::Detection>& res, float *output, float nms_thresh = NMS_THRESH) { // ham loc bounding box theo NMS, co su dung ham cmp va iou
    std::map<float, std::vector<Yolo::Detection>> m;                                        
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + DETECTION_SIZE * i + 4] <= BBOX_CONF_THRESH) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + DETECTION_SIZE * i], DETECTION_SIZE * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }
} // -------res se chua cac bounding box da duoc loc

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

/// toi uu hoa sau, thay doi tuy vao model
IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* convBnMish(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, int linx) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-4);

    auto creator = getPluginRegistry()->getPluginCreator("Mish_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin(("mish" + std::to_string(linx)).c_str(), pluginData);
    ITensor* inputTensors[] = {bn1->getOutput(0)};
    auto mish = network->addPluginV2(&inputTensors[0], 1, *pluginObj);
    return mish;
}

ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, int linx) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-4);

    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    return lr;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, std::string model_path) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(model_path);  // path of .weight
    //std::map<std::string, Weights> weightMap = loadWeights("../yolov4.wts"); 
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // define each layer.
    auto l0 = convBnMish(network, weightMap, *data, 32, 3, 1, 1, 0);
    auto l1 = convBnMish(network, weightMap, *l0->getOutput(0), 64, 3, 2, 1, 1);
    auto l2 = convBnMish(network, weightMap, *l1->getOutput(0), 64, 1, 1, 0, 2);
    auto l3 = l1;
    auto l4 = convBnMish(network, weightMap, *l3->getOutput(0), 64, 1, 1, 0, 4);
    auto l5 = convBnMish(network, weightMap, *l4->getOutput(0), 32, 1, 1, 0, 5);
    auto l6 = convBnMish(network, weightMap, *l5->getOutput(0), 64, 3, 1, 1, 6);
    auto ew7 = network->addElementWise(*l6->getOutput(0), *l4->getOutput(0), ElementWiseOperation::kSUM);
    auto l8 = convBnMish(network, weightMap, *ew7->getOutput(0), 64, 1, 1, 0, 8);

    ITensor* inputTensors9[] = {l8->getOutput(0), l2->getOutput(0)};
    auto cat9 = network->addConcatenation(inputTensors9, 2);

    auto l10 = convBnMish(network, weightMap, *cat9->getOutput(0), 64, 1, 1, 0, 10);
    auto l11 = convBnMish(network, weightMap, *l10->getOutput(0), 128, 3, 2, 1, 11);
    auto l12 = convBnMish(network, weightMap, *l11->getOutput(0), 64, 1, 1, 0, 12);
    auto l13 = l11;
    auto l14 = convBnMish(network, weightMap, *l13->getOutput(0), 64, 1, 1, 0, 14);
    auto l15 = convBnMish(network, weightMap, *l14->getOutput(0), 64, 1, 1, 0, 15);
    auto l16 = convBnMish(network, weightMap, *l15->getOutput(0), 64, 3, 1, 1, 16);
    auto ew17 = network->addElementWise(*l16->getOutput(0), *l14->getOutput(0), ElementWiseOperation::kSUM);
    auto l18 = convBnMish(network, weightMap, *ew17->getOutput(0), 64, 1, 1, 0, 18);
    auto l19 = convBnMish(network, weightMap, *l18->getOutput(0), 64, 3, 1, 1, 19);
    auto ew20 = network->addElementWise(*l19->getOutput(0), *ew17->getOutput(0), ElementWiseOperation::kSUM);
    auto l21 = convBnMish(network, weightMap, *ew20->getOutput(0), 64, 1, 1, 0, 21);

    ITensor* inputTensors22[] = {l21->getOutput(0), l12->getOutput(0)};
    auto cat22 = network->addConcatenation(inputTensors22, 2);

    auto l23 = convBnMish(network, weightMap, *cat22->getOutput(0), 128, 1, 1, 0, 23);
    auto l24 = convBnMish(network, weightMap, *l23->getOutput(0), 256, 3, 2, 1, 24);
    auto l25 = convBnMish(network, weightMap, *l24->getOutput(0), 128, 1, 1, 0, 25);
    auto l26 = l24;
    auto l27 = convBnMish(network, weightMap, *l26->getOutput(0), 128, 1, 1, 0, 27);
    auto l28 = convBnMish(network, weightMap, *l27->getOutput(0), 128, 1, 1, 0, 28);
    auto l29 = convBnMish(network, weightMap, *l28->getOutput(0), 128, 3, 1, 1, 29);
    auto ew30 = network->addElementWise(*l29->getOutput(0), *l27->getOutput(0), ElementWiseOperation::kSUM);
    auto l31 = convBnMish(network, weightMap, *ew30->getOutput(0), 128, 1, 1, 0, 31);
    auto l32 = convBnMish(network, weightMap, *l31->getOutput(0), 128, 3, 1, 1, 32);
    auto ew33 = network->addElementWise(*l32->getOutput(0), *ew30->getOutput(0), ElementWiseOperation::kSUM);
    auto l34 = convBnMish(network, weightMap, *ew33->getOutput(0), 128, 1, 1, 0, 34);
    auto l35 = convBnMish(network, weightMap, *l34->getOutput(0), 128, 3, 1, 1, 35);
    auto ew36 = network->addElementWise(*l35->getOutput(0), *ew33->getOutput(0), ElementWiseOperation::kSUM);
    auto l37 = convBnMish(network, weightMap, *ew36->getOutput(0), 128, 1, 1, 0, 37);
    auto l38 = convBnMish(network, weightMap, *l37->getOutput(0), 128, 3, 1, 1, 38);
    auto ew39 = network->addElementWise(*l38->getOutput(0), *ew36->getOutput(0), ElementWiseOperation::kSUM);
    auto l40 = convBnMish(network, weightMap, *ew39->getOutput(0), 128, 1, 1, 0, 40);
    auto l41 = convBnMish(network, weightMap, *l40->getOutput(0), 128, 3, 1, 1, 41);
    auto ew42 = network->addElementWise(*l41->getOutput(0), *ew39->getOutput(0), ElementWiseOperation::kSUM);
    auto l43 = convBnMish(network, weightMap, *ew42->getOutput(0), 128, 1, 1, 0, 43);
    auto l44 = convBnMish(network, weightMap, *l43->getOutput(0), 128, 3, 1, 1, 44);
    auto ew45 = network->addElementWise(*l44->getOutput(0), *ew42->getOutput(0), ElementWiseOperation::kSUM);
    auto l46 = convBnMish(network, weightMap, *ew45->getOutput(0), 128, 1, 1, 0, 46);
    auto l47 = convBnMish(network, weightMap, *l46->getOutput(0), 128, 3, 1, 1, 47);
    auto ew48 = network->addElementWise(*l47->getOutput(0), *ew45->getOutput(0), ElementWiseOperation::kSUM);
    auto l49 = convBnMish(network, weightMap, *ew48->getOutput(0), 128, 1, 1, 0, 49);
    auto l50 = convBnMish(network, weightMap, *l49->getOutput(0), 128, 3, 1, 1, 50);
    auto ew51 = network->addElementWise(*l50->getOutput(0), *ew48->getOutput(0), ElementWiseOperation::kSUM);
    auto l52 = convBnMish(network, weightMap, *ew51->getOutput(0), 128, 1, 1, 0, 52);

    ITensor* inputTensors53[] = {l52->getOutput(0), l25->getOutput(0)};
    auto cat53 = network->addConcatenation(inputTensors53, 2);

    auto l54 = convBnMish(network, weightMap, *cat53->getOutput(0), 256, 1, 1, 0, 54);
    auto l55 = convBnMish(network, weightMap, *l54->getOutput(0), 512, 3, 2, 1, 55);
    auto l56 = convBnMish(network, weightMap, *l55->getOutput(0), 256, 1, 1, 0, 56);
    auto l57 = l55;
    auto l58 = convBnMish(network, weightMap, *l57->getOutput(0), 256, 1, 1, 0, 58);
    auto l59 = convBnMish(network, weightMap, *l58->getOutput(0), 256, 1, 1, 0, 59);
    auto l60 = convBnMish(network, weightMap, *l59->getOutput(0), 256, 3, 1, 1, 60);
    auto ew61 = network->addElementWise(*l60->getOutput(0), *l58->getOutput(0), ElementWiseOperation::kSUM);
    auto l62 = convBnMish(network, weightMap, *ew61->getOutput(0), 256, 1, 1, 0, 62);
    auto l63 = convBnMish(network, weightMap, *l62->getOutput(0), 256, 3, 1, 1, 63);
    auto ew64 = network->addElementWise(*l63->getOutput(0), *ew61->getOutput(0), ElementWiseOperation::kSUM);
    auto l65 = convBnMish(network, weightMap, *ew64->getOutput(0), 256, 1, 1, 0, 65);
    auto l66 = convBnMish(network, weightMap, *l65->getOutput(0), 256, 3, 1, 1, 66);
    auto ew67 = network->addElementWise(*l66->getOutput(0), *ew64->getOutput(0), ElementWiseOperation::kSUM);
    auto l68 = convBnMish(network, weightMap, *ew67->getOutput(0), 256, 1, 1, 0, 68);
    auto l69 = convBnMish(network, weightMap, *l68->getOutput(0), 256, 3, 1, 1, 69);
    auto ew70 = network->addElementWise(*l69->getOutput(0), *ew67->getOutput(0), ElementWiseOperation::kSUM);
    auto l71 = convBnMish(network, weightMap, *ew70->getOutput(0), 256, 1, 1, 0, 71);
    auto l72 = convBnMish(network, weightMap, *l71->getOutput(0), 256, 3, 1, 1, 72);
    auto ew73 = network->addElementWise(*l72->getOutput(0), *ew70->getOutput(0), ElementWiseOperation::kSUM);
    auto l74 = convBnMish(network, weightMap, *ew73->getOutput(0), 256, 1, 1, 0, 74);
    auto l75 = convBnMish(network, weightMap, *l74->getOutput(0), 256, 3, 1, 1, 75);
    auto ew76 = network->addElementWise(*l75->getOutput(0), *ew73->getOutput(0), ElementWiseOperation::kSUM);
    auto l77 = convBnMish(network, weightMap, *ew76->getOutput(0), 256, 1, 1, 0, 77);
    auto l78 = convBnMish(network, weightMap, *l77->getOutput(0), 256, 3, 1, 1, 78);
    auto ew79 = network->addElementWise(*l78->getOutput(0), *ew76->getOutput(0), ElementWiseOperation::kSUM);
    auto l80 = convBnMish(network, weightMap, *ew79->getOutput(0), 256, 1, 1, 0, 80);
    auto l81 = convBnMish(network, weightMap, *l80->getOutput(0), 256, 3, 1, 1, 81);
    auto ew82 = network->addElementWise(*l81->getOutput(0), *ew79->getOutput(0), ElementWiseOperation::kSUM);
    auto l83 = convBnMish(network, weightMap, *ew82->getOutput(0), 256, 1, 1, 0, 83);

    ITensor* inputTensors84[] = {l83->getOutput(0), l56->getOutput(0)};
    auto cat84 = network->addConcatenation(inputTensors84, 2);

    auto l85 = convBnMish(network, weightMap, *cat84->getOutput(0), 512, 1, 1, 0, 85);
    auto l86 = convBnMish(network, weightMap, *l85->getOutput(0), 1024, 3, 2, 1, 86);
    auto l87 = convBnMish(network, weightMap, *l86->getOutput(0), 512, 1, 1, 0, 87);
    auto l88 = l86;
    auto l89 = convBnMish(network, weightMap, *l88->getOutput(0), 512, 1, 1, 0, 89);
    auto l90 = convBnMish(network, weightMap, *l89->getOutput(0), 512, 1, 1, 0, 90);
    auto l91 = convBnMish(network, weightMap, *l90->getOutput(0), 512, 3, 1, 1, 91);
    auto ew92 = network->addElementWise(*l91->getOutput(0), *l89->getOutput(0), ElementWiseOperation::kSUM);
    auto l93 = convBnMish(network, weightMap, *ew92->getOutput(0), 512, 1, 1, 0, 93);
    auto l94 = convBnMish(network, weightMap, *l93->getOutput(0), 512, 3, 1, 1, 94);
    auto ew95 = network->addElementWise(*l94->getOutput(0), *ew92->getOutput(0), ElementWiseOperation::kSUM);
    auto l96 = convBnMish(network, weightMap, *ew95->getOutput(0), 512, 1, 1, 0, 96);
    auto l97 = convBnMish(network, weightMap, *l96->getOutput(0), 512, 3, 1, 1, 97);
    auto ew98 = network->addElementWise(*l97->getOutput(0), *ew95->getOutput(0), ElementWiseOperation::kSUM);
    auto l99 = convBnMish(network, weightMap, *ew98->getOutput(0), 512, 1, 1, 0, 99);
    auto l100 = convBnMish(network, weightMap, *l99->getOutput(0), 512, 3, 1, 1, 100);
    auto ew101 = network->addElementWise(*l100->getOutput(0), *ew98->getOutput(0), ElementWiseOperation::kSUM);
    auto l102 = convBnMish(network, weightMap, *ew101->getOutput(0), 512, 1, 1, 0, 102);

    ITensor* inputTensors103[] = {l102->getOutput(0), l87->getOutput(0)};
    auto cat103 = network->addConcatenation(inputTensors103, 2);

    auto l104 = convBnMish(network, weightMap, *cat103->getOutput(0), 1024, 1, 1, 0, 104);

    // ---------
    auto l105 = convBnLeaky(network, weightMap, *l104->getOutput(0), 512, 1, 1, 0, 105);
    auto l106 = convBnLeaky(network, weightMap, *l105->getOutput(0), 1024, 3, 1, 1, 106);
    auto l107 = convBnLeaky(network, weightMap, *l106->getOutput(0), 512, 1, 1, 0, 107);

    auto pool108 = network->addPoolingNd(*l107->getOutput(0), PoolingType::kMAX, DimsHW{5, 5});
    pool108->setPaddingNd(DimsHW{2, 2});
    pool108->setStrideNd(DimsHW{1, 1});

    auto l109 = l107;

    auto pool110 = network->addPoolingNd(*l109->getOutput(0), PoolingType::kMAX, DimsHW{9, 9});
    pool110->setPaddingNd(DimsHW{4, 4});
    pool110->setStrideNd(DimsHW{1, 1});

    auto l111 = l107;

    auto pool112 = network->addPoolingNd(*l111->getOutput(0), PoolingType::kMAX, DimsHW{13, 13});
    pool112->setPaddingNd(DimsHW{6, 6});
    pool112->setStrideNd(DimsHW{1, 1});

    ITensor* inputTensors113[] = {pool112->getOutput(0), pool110->getOutput(0), pool108->getOutput(0), l107->getOutput(0)};
    auto cat113 = network->addConcatenation(inputTensors113, 4);

    auto l114 = convBnLeaky(network, weightMap, *cat113->getOutput(0), 512, 1, 1, 0, 114);
    auto l115 = convBnLeaky(network, weightMap, *l114->getOutput(0), 1024, 3, 1, 1, 115);
    auto l116 = convBnLeaky(network, weightMap, *l115->getOutput(0), 512, 1, 1, 0, 116);
    auto l117 = convBnLeaky(network, weightMap, *l116->getOutput(0), 256, 1, 1, 0, 117);

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts118{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer* deconv118 = network->addDeconvolutionNd(*l117->getOutput(0), 256, DimsHW{2, 2}, deconvwts118, emptywts);
    assert(deconv118);
    deconv118->setStrideNd(DimsHW{2, 2});
    deconv118->setNbGroups(256);
    weightMap["deconv118"] = deconvwts118;

    auto l119 = l85;
    auto l120 = convBnLeaky(network, weightMap, *l119->getOutput(0), 256, 1, 1, 0, 120);

    ITensor* inputTensors121[] = {l120->getOutput(0), deconv118->getOutput(0)};
    auto cat121 = network->addConcatenation(inputTensors121, 2);

    auto l122 = convBnLeaky(network, weightMap, *cat121->getOutput(0), 256, 1, 1, 0, 122);
    auto l123 = convBnLeaky(network, weightMap, *l122->getOutput(0), 512, 3, 1, 1, 123);
    auto l124 = convBnLeaky(network, weightMap, *l123->getOutput(0), 256, 1, 1, 0, 124);
    auto l125 = convBnLeaky(network, weightMap, *l124->getOutput(0), 512, 3, 1, 1, 125);
    auto l126 = convBnLeaky(network, weightMap, *l125->getOutput(0), 256, 1, 1, 0, 126);
    auto l127 = convBnLeaky(network, weightMap, *l126->getOutput(0), 128, 1, 1, 0, 127);

    Weights deconvwts128{DataType::kFLOAT, deval, 128 * 2 * 2};
    IDeconvolutionLayer* deconv128 = network->addDeconvolutionNd(*l127->getOutput(0), 128, DimsHW{2, 2}, deconvwts128, emptywts);
    assert(deconv128);
    deconv128->setStrideNd(DimsHW{2, 2});
    deconv128->setNbGroups(128);

    auto l129 = l54;
    auto l130 = convBnLeaky(network, weightMap, *l129->getOutput(0), 128, 1, 1, 0, 130);

    ITensor* inputTensors131[] = {l130->getOutput(0), deconv128->getOutput(0)};
    auto cat131 = network->addConcatenation(inputTensors131, 2);

    auto l132 = convBnLeaky(network, weightMap, *cat131->getOutput(0), 128, 1, 1, 0, 132);
    auto l133 = convBnLeaky(network, weightMap, *l132->getOutput(0), 256, 3, 1, 1, 133);
    auto l134 = convBnLeaky(network, weightMap, *l133->getOutput(0), 128, 1, 1, 0, 134);
    auto l135 = convBnLeaky(network, weightMap, *l134->getOutput(0), 256, 3, 1, 1, 135);
    auto l136 = convBnLeaky(network, weightMap, *l135->getOutput(0), 128, 1, 1, 0, 136);
    auto l137 = convBnLeaky(network, weightMap, *l136->getOutput(0), 256, 3, 1, 1, 137);
    IConvolutionLayer* conv138 = network->addConvolutionNd(*l137->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.138.Conv2d.weight"], weightMap["module_list.138.Conv2d.bias"]);
    assert(conv138);
    // 139 is yolo layer

    auto l140 = l136;
    auto l141 = convBnLeaky(network, weightMap, *l140->getOutput(0), 256, 3, 2, 1, 141);

    ITensor* inputTensors142[] = {l141->getOutput(0), l126->getOutput(0)};
    auto cat142 = network->addConcatenation(inputTensors142, 2);

    auto l143 = convBnLeaky(network, weightMap, *cat142->getOutput(0), 256, 1, 1, 0, 143);
    auto l144 = convBnLeaky(network, weightMap, *l143->getOutput(0), 512, 3, 1, 1, 144);
    auto l145 = convBnLeaky(network, weightMap, *l144->getOutput(0), 256, 1, 1, 0, 145);
    auto l146 = convBnLeaky(network, weightMap, *l145->getOutput(0), 512, 3, 1, 1, 146);
    auto l147 = convBnLeaky(network, weightMap, *l146->getOutput(0), 256, 1, 1, 0, 147);
    auto l148 = convBnLeaky(network, weightMap, *l147->getOutput(0), 512, 3, 1, 1, 148);
    IConvolutionLayer* conv149 = network->addConvolutionNd(*l148->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.149.Conv2d.weight"], weightMap["module_list.149.Conv2d.bias"]);
    assert(conv149);
    // 150 is yolo layer

    auto l151 = l147;
    auto l152 = convBnLeaky(network, weightMap, *l151->getOutput(0), 512, 3, 2, 1, 152);

    ITensor* inputTensors153[] = {l152->getOutput(0), l116->getOutput(0)};
    auto cat153 = network->addConcatenation(inputTensors153, 2);

    auto l154 = convBnLeaky(network, weightMap, *cat153->getOutput(0), 512, 1, 1, 0, 154);
    auto l155 = convBnLeaky(network, weightMap, *l154->getOutput(0), 1024, 3, 1, 1, 155);
    auto l156 = convBnLeaky(network, weightMap, *l155->getOutput(0), 512, 1, 1, 0, 156);
    auto l157 = convBnLeaky(network, weightMap, *l156->getOutput(0), 1024, 3, 1, 1, 157);
    auto l158 = convBnLeaky(network, weightMap, *l157->getOutput(0), 512, 1, 1, 0, 158);
    auto l159 = convBnLeaky(network, weightMap, *l158->getOutput(0), 1024, 3, 1, 1, 159);
    IConvolutionLayer* conv160 = network->addConvolutionNd(*l159->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.160.Conv2d.weight"], weightMap["module_list.160.Conv2d.bias"]);
    assert(conv160);
    // 161 is yolo layer

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    ITensor* inputTensors_yolo[] = {conv138->getOutput(0), conv149->getOutput(0), conv160->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine, phan nay co the dung chung giua cac Model
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB //khac nhau
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);     // khac nhau
#endif
    std::cout << "Building tensorrt engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel( unsigned int maxBatchSize, IHostMemory** modelStream, std::string model_path) { // giong het nhau
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT, model_path);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

trt_error TRT_Inference::trt_APIModel(std::string model_path){
    IHostMemory* modelStream{nullptr};
    APIToModel(BATCH_SIZE, &modelStream, model_path);
    assert(modelStream != nullptr);
    // them duong dan tuy chon o day 
    std::ofstream p("yolov4.engine", std::ios::binary); 
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return TRT_RESULT_ERROR;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return TRT_RESULT_SUCCESS;
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float))); // khac nhau
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host, KHAC NHAU
    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
}

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) { //luu cac tep trong thu muc vao mang, mang se chua duong dan toi cac thu muc
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
                strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

std::string extractPrefix(const std::string& input) {
    // Tìm vị trí của ".wts" trong chuỗi
    size_t pos = input.find(".wts");
    
    // Kiểm tra xem ".wts" có xuất hiện trong chuỗi hay không
    if (pos != std::string::npos) {
        // Lấy phần trước ".wts"
        std::string prefix = input.substr(0, pos);
        // Tìm vị trí cuối cùng của '/' để lấy phần cuối cùng
        size_t lastSlash = prefix.find_last_of('/');
        if (lastSlash != std::string::npos) {
            prefix = prefix.substr(lastSlash + 1);
        }
        return prefix;
    } else {
        // Không tìm thấy ".wts" trong chuỗi
        return "";
    }
}

trt_error TRT_Inference::init_inference(std::string engine_name , const char * input_folder, std::vector<std::string> &file_names){
    // create a model using the API directly and serialize it to a stream
    printf("Thuc hien ham Init\n");
    char *trtModelStream{nullptr};
    size_t size{0};

    //std::ifstream file("yolov4.engine", std::ios::binary);
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    //     /// doc anh tu thu muc
    if (read_files_in_dir(input_folder, file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return TRT_RESULT_ERROR;
    }
    // static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    // static float prob[BATCH_SIZE * OUTPUT_SIZE];

    this->runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    this->engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    this->context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    printf("ket thuc ham Init\n");
    return TRT_RESULT_SUCCESS;
}

// bool isImage(const std::string& filename) {
//     // Danh sách các phần mở rộng cho ảnh
//     std::vector<std::string> imageExtensions = {".jpg", ".jpeg", ".png", ".bmp"};

//     // Tìm phần mở rộng của tệp
//     size_t dotPosition = filename.find_last_of(".");
//     if (dotPosition != std::string::npos) {
//         std::string extension = filename.substr(dotPosition);
        
//         // Chuyển đổi phần mở rộng thành chữ thường để so sánh không phân biệt chữ hoa chữ thường
//         std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

//         // Kiểm tra xem phần mở rộng có trong danh sách phần mở rộng của ảnh không
//         if (std::find(imageExtensions.begin(), imageExtensions.end(), extension) != imageExtensions.end()) return true;
//         return false;
//     }

//     // Nếu không có phần mở rộng, giả định là không phải ảnh
//     return false;
// }

// bool isVideo(const std::string& filename) {
//     // Danh sách các phần mở rộng cho video
//     std::vector<std::string> videoExtensions = {".mp4", ".avi", ".mkv"};

//     // Tìm phần mở rộng của tệp
//     size_t dotPosition = filename.find_last_of(".");
//     if (dotPosition != std::string::npos) {
//         std::string extension = filename.substr(dotPosition);
        
//         // Chuyển đổi phần mở rộng thành chữ thường để so sánh không phân biệt chữ hoa chữ thường
//         std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

//         // Kiểm tra xem phần mở rộng có trong danh sách phần mở rộng của video không
//         if( std::find(videoExtensions.begin(), videoExtensions.end(), extension) != videoExtensions.end()) return true;
//         return false;
//     }

//     // Nếu không có phần mở rộng, giả định là không phải video
//     return false;
// }



trt_error TRT_Inference::trt_detection(std::vector<IMXAIEngine::trt_input> &trt_inputs, std::vector<IMXAIEngine::trt_output> &trt_outputs ){
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_SIZE];

    int fcount = 0;
    int img_id=0;
    for (int f = 0; f < (int) trt_inputs.size(); f++) {    
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (int)trt_inputs.size() ) continue;
        // xu li anh
        for (int b = 0; b < fcount; b++) {
            //cv::Mat img = trt_inputs[f - fcount + 1 + b].input_img;
            cv::Mat img = trt_inputs[img_id + b].input_img;
            if(!img.empty()){
            cv::Mat pr_img = preprocess_img(img); // goi ham su li anh
            for (int i = 0; i < INPUT_H * INPUT_W; i++) {
                data[b * 3 * INPUT_H * INPUT_W + i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
                data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
                data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
            }
            }
        }
    

        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE]);
        }

        std::cout <<"kich thuoc cua fcount:" <<fcount <<std::endl;

        for (int b = 0; b < fcount; b++) {
        auto& res = batch_res[b];
        std::vector<trt_results> image_result;
        IMXAIEngine::trt_output out_img;

            for (size_t j = 0; j < res.size(); j++) {
                trt_results boundingbox_result;
                boundingbox_result.ClassID = res[j].class_id;
                boundingbox_result.Confidence = res[j].det_confidence;
                boundingbox_result.bbox[0] = res[j].bbox[0];
                boundingbox_result.bbox[1] = res[j].bbox[1];
                boundingbox_result.bbox[2] = res[j].bbox[2];
                boundingbox_result.bbox[3] = res[j].bbox[3];

            //image_result.push_back(boundingbox_result);
            out_img.results.push_back(boundingbox_result);
        }


        // Thêm image_result vào results
        out_img.id= img_id + b ; /// phan ID nay can xem lai, voi batch size=1 thi dung, con neu batchsize khac thi chua chac
        trt_outputs.push_back(out_img);
        }

        // Di chuyển việc reset fcount xuống cuối vòng lặp
        img_id= img_id + fcount;
        fcount = 0;
    }
    

    return TRT_RESULT_SUCCESS;
}