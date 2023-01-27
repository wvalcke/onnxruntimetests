#include <opencv2/opencv.hpp>

#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <chrono>
#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>

using namespace cv;

struct Detection
{
    cv::Rect box;
    float conf{};
    int classId{};
};

template <typename T>
T clip(const T& n, const T& lower, const T& upper)
{
    return std::max(lower, std::min(n, upper));
}

cv::Rect2f scaleCoords(const cv::Size& imageShape, cv::Rect2f coords, const cv::Size& imageOriginalShape, bool p_Clip = false)
{
    cv::Rect2f l_Result;
    float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
                          (float)imageShape.width / (float)imageOriginalShape.width);

    int pad[2] = { (int)std::round((( (float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f)-0.1f),
                 (int)std::round((( (float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f)-0.1f)};

    l_Result.x = (int) std::round(((float)(coords.x - pad[0]) / gain));
    l_Result.y = (int) std::round(((float)(coords.y - pad[1]) / gain));

    l_Result.width = (int) std::round(((float)coords.width / gain));
    l_Result.height = (int) std::round(((float)coords.height / gain));

    // // clip coords, should be modified for width and height
    if (p_Clip)
    {
        l_Result.x = clip(l_Result.x, (float)0, (float)imageOriginalShape.width);
        l_Result.y = clip(l_Result.y, (float)0, (float)imageOriginalShape.height);
        l_Result.width = clip(l_Result.width, (float)0, (float)(imageOriginalShape.width-l_Result.x));
        l_Result.height = clip(l_Result.height, (float)0, (float)(imageOriginalShape.height-l_Result.y));
    }
    return l_Result;
}

void getBestClassInfo(const cv::Mat& p_Mat, const int& numClasses,
                                    float& bestConf, int& bestClassId)
{
    bestClassId = 0;
    bestConf = 0;

    if (p_Mat.rows && p_Mat.cols)
    {
        for (int i = 0; i < numClasses; i++)
        {
            if (p_Mat.at<float>(0, i+4) > bestConf)
            {
                bestConf = p_Mat.at<float>(0, i+4);
                bestClassId = i;
            }
        }
    }
}

std::vector<Detection> postprocessing(const cv::Size& resizedImageShape,
                                                    const cv::Size& originalImageShape,
                                                    std::vector<Ort::Value>& outputTensors,
                                                    const float& confThreshold, const float& iouThreshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<cv::Rect> nms_boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    auto* rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    cv::Mat l_Mat = cv::Mat(outputShape[1], outputShape[2], CV_32FC1, (void*)rawOutput);
    cv::Mat l_Mat_t = l_Mat.t();
    //std::vector<float> output(rawOutput, rawOutput + count);

    // for (const int64_t& shape : outputShape)
    //     std::cout << "Output Shape: " << shape << std::endl;

    // first 5 elements are box[4] and obj confidence
    int numClasses = l_Mat_t.cols - 4;

    // only for batch size = 1

    for (int l_Row = 0; l_Row < l_Mat_t.rows; l_Row++)
    {
        cv::Mat l_MatRow = l_Mat_t.row(l_Row);
        float objConf;
        int classId;

        getBestClassInfo(l_MatRow, numClasses, objConf, classId);

        if (objConf > confThreshold)
        {
            float centerX = (l_MatRow.at<float>(0, 0));
            float centerY = (l_MatRow.at<float>(0, 1));
            float width = (l_MatRow.at<float>(0, 2));
            float height = (l_MatRow.at<float>(0, 3));
            float left = centerX - width / 2;
            float top = centerY - height / 2;

            float confidence = objConf;
            cv::Rect2f l_Scaled = scaleCoords(resizedImageShape, cv::Rect2f(left, top, width, height), originalImageShape, true);

            // Prepare NMS filtered per class id's
            nms_boxes.emplace_back((int)std::round(l_Scaled.x)+classId*7680, (int)std::round(l_Scaled.y)+classId*7680, 
                (int)std::round(l_Scaled.width), (int)std::round(l_Scaled.height));
            boxes.emplace_back((int)std::round(l_Scaled.x), (int)std::round(l_Scaled.y), 
                (int)std::round(l_Scaled.width), (int)std::round(l_Scaled.height));
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(nms_boxes, confs, confThreshold, iouThreshold, indices);
    // std::cout << "amount of NMS indices: " << indices.size() << std::endl;

    std::vector<Detection> detections;

    for (int idx : indices)
    {
        Detection det;
        det.box = boxes[idx];
        det.conf = confs[idx];
        det.classId = classIds[idx];
        detections.emplace_back(det);
    }

    return detections;
}


void letterbox(const cv::Mat& image, cv::Mat& outImage,
                      const cv::Size& newShape = cv::Size(640, 640),
                      const cv::Scalar& color = cv::Scalar(114, 114, 114),
                      bool auto_ = true,
                      bool scaleFill = false,
                      bool scaleUp = true,
                      int stride = 32)
{
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2] {r, r};
    int newUnpad[2] {(int)std::round((float)shape.width * r),
                     (int)std::round((float)shape.height * r)};

    auto dw = (float)(newShape.width - newUnpad[0]);
    auto dh = (float)(newShape.height - newUnpad[1]);

    if (auto_)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] || shape.height != newUnpad[1])
    {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}


int calculate_product(const std::vector<int64_t> &v)
{
    int total = 1;
    for (auto &i : v)
        total *= i;
    return total;
}

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<int64_t> &v)
{
    std::stringstream ss("");
    for (size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("usage: DisplayImage.out <Image_Path> <onnxmodel> CPU|GPU\n");
        return -1;
    }

    bool useGPU = false;
    std::string l_GpuOption(argv[3]);
    std::transform(l_GpuOption.begin(), l_GpuOption.end(), l_GpuOption.begin(), [](unsigned char c)
                   { return std::tolower(c); });
    if (l_GpuOption == "gpu")
    {
        useGPU = true;
        std::cout << "Using GPU" << std::endl;
    }

#ifdef _WIN32
    std::string str = argv[2];
    std::wstring wide_string = std::wstring(str.begin(), str.end());
    std::basic_string<ORTCHAR_T> model_file = std::basic_string<ORTCHAR_T>(wide_string);
#else
    std::string model_file = argv[2];
#endif

    // onnxruntime setup
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");

    auto providers = Ort::GetAvailableProviders();
    std::cout << "Available providers" << std::endl;
    for (auto provider : providers)
    {
        std::cout << provider << std::endl;
    }
    Ort::SessionOptions session_options;

    if (useGPU)
    {
        OrtCUDAProviderOptions l_CudaOptions;
        l_CudaOptions.device_id = 0;
        std::cout << "Before setting session options" << std::endl;
        session_options.AppendExecutionProvider_CUDA(l_CudaOptions);
        std::cout << "set session options" << std::endl;
    }
    else
    {
        // session_options.SetIntraOpNumThreads(12);
    }

    Ort::Experimental::Session session = Ort::Experimental::Session(env, model_file, session_options); // access experimental components via the Experimental namespace

    // print name/shape of inputs
    std::vector<std::string> input_names = session.GetInputNames();
    std::vector<std::vector<int64_t>> input_shapes = session.GetInputShapes();
    std::cout << "Input Node Name/Shape (" << input_names.size() << "):" << std::endl;
    for (size_t i = 0; i < input_names.size(); i++)
    {
        std::cout << "\t" << input_names[i] << " : " << print_shape(input_shapes[i]) << std::endl;
    }

    // print name/shape of outputs
    std::vector<std::string> output_names = session.GetOutputNames();
    std::vector<std::vector<int64_t>> output_shapes = session.GetOutputShapes();
    std::cout << "Output Node Name/Shape (" << output_names.size() << "):" << std::endl;
    for (size_t i = 0; i < output_names.size(); i++)
    {
        std::cout << "\t" << output_names[i] << " : " << print_shape(output_shapes[i]) << std::endl;
    }

    if (useGPU)
    {
        std::cout << "Perform wamup on CUDA inference" << std::endl;
        // Create a single Ort tensor of random numbers
        auto input_shape = input_shapes[0];
        int total_number_elements = calculate_product(input_shape);
        std::vector<float> input_tensor_values(total_number_elements);
        std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&]
                      { return 0.0f; }); // generate random numbers in the range [0, 255]
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(input_tensor_values.data(), input_tensor_values.size(), input_shape));
        for (int count = 0; count < 20; count++)
        {
            auto output_tensors = session.Run(session.GetInputNames(), input_tensors, session.GetOutputNames());
        }
        std::cout << "Warmup done" << std::endl;
        //std::cout << "output_tensor_shape: " << print_shape(output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << std::endl;
    }

    Mat image;
    image = imread(argv[1], 1);
    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }

    int l_Number = 500;
    std::cout << "FPS testing" << std::endl;
    float *blob = new float[640 * 640 * 3];
    cv::Mat resizedImage, floatImage;

    auto startTime = std::chrono::steady_clock::now();

    for (int count = 0; count < l_Number; count++)
    {
        cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
        letterbox(resizedImage, resizedImage, cv::Size(640, 640),
                  cv::Scalar(114, 114, 114), false,
                  false, true, 32);
        resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
        cv::Size floatImageSize{floatImage.cols, floatImage.rows};

        std::vector<cv::Mat> chw(floatImage.channels());
        for (int i = 0; i < floatImage.channels(); ++i)
        {
            chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
        }
        cv::split(floatImage, chw);
        std::vector<float> inputTensorValues(blob, blob + 3 * floatImageSize.width * floatImageSize.height);
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(inputTensorValues.data(), inputTensorValues.size(), input_shapes[0]));
        auto output_tensors = session.Run(session.GetInputNames(), input_tensors, session.GetOutputNames());
        std::vector<Detection> result = postprocessing(cv::Size(640, 640), image.size(), output_tensors, 0.5, 0.45);
    }
    auto endTime = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    std::cout << "FPS " << l_Number / elapsedTime.count() << std::endl;

    return 0;
}