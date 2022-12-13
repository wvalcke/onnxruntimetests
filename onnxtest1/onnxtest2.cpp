#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <chrono>
#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>

int main(int argc, char *argv[])
{
    try
    {
        Ort::Env env;
        Ort::SessionOptions session_options;
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
#ifdef _WIN32
        Ort::Session session = Ort::Session(env, L"rim.onnx", session_options);
#else
        Ort::Session session = Ort::Session(env, "rim.onnx", session_options);
#endif
    }

    catch (const Ort::Exception &exception)
    {
        std::cout << "ERROR running model inference: " << exception.what() << std::endl;
        exit(-1);
    }

    std::cout << "End reached" << std::endl;
}
