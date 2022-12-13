#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <chrono>
#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>

int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= i;
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
    std::cout << "Hello ONNX runtime" << std::endl;
    if (argc < 3)
    {
        std::cout << "Usage: ./onnx-api-example <onnx_model.onnx> CPU|GPU" << std::endl;
        return -1;
    }
    bool useGPU = false;
    std::string l_GpuOption(argv[2]);
    std::transform(l_GpuOption.begin(), l_GpuOption.end(), l_GpuOption.begin(), [](unsigned char c){return std::tolower(c);});
    if (l_GpuOption == "gpu")
    {
        useGPU = true;
        std::cout << "Using GPU" << std::endl;
    }

#ifdef _WIN32
  std::string str = argv[1];
  std::wstring wide_string = std::wstring(str.begin(), str.end());
  std::basic_string<ORTCHAR_T> model_file = std::basic_string<ORTCHAR_T>(wide_string);
#else
  std::string model_file = argv[1];
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
        //session_options.SetIntraOpNumThreads(12);
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

    // Create a single Ort tensor of random numbers
    auto input_shape = input_shapes[0];
    int total_number_elements = calculate_product(input_shape);
    std::vector<float> input_tensor_values(total_number_elements);
    std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&]
                  { return rand() % 255; }); // generate random numbers in the range [0, 255]
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(input_tensor_values.data(), input_tensor_values.size(), input_shape));

    // double-check the dimensions of the input tensor
    assert(input_tensors[0].IsTensor() &&
           input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_shape);
    std::cout << "\ninput_tensor shape: " << print_shape(input_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << std::endl;

    // pass data through model
    std::cout << "Running model...";
    try
    {
        for (int count = 0; count < 2; count++)
        {
            auto output_tensors = session.Run(session.GetInputNames(), input_tensors, session.GetOutputNames());
            std::cout << "done" << std::endl;

            // double-check the dimensions of the output tensors
            // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
            assert(output_tensors.size() == session.GetOutputNames().size() &&
                   output_tensors[0].IsTensor());
            std::cout << "output_tensor_shape: " << print_shape(output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << std::endl;
        }
        int l_Number = 500;
        std::cout << "FPS testing" << std::endl;
        auto startTime = std::chrono::steady_clock::now();
        for (int count=0; count < l_Number; count++)
        {
            auto output_tensors = session.Run(session.GetInputNames(), input_tensors, session.GetOutputNames());
        }
        auto endTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedTime = endTime - startTime;
        std::cout << "FPS " << l_Number/elapsedTime.count() << std::endl;
    }
    catch (const Ort::Exception &exception)
    {
        std::cout << "ERROR running model inference: " << exception.what() << std::endl;
        exit(-1);
    }
}
