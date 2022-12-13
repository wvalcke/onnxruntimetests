#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <chrono>
#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>

int main(int argc, char *argv[])
{
    Ort::Env env;
    Ort::SessionOptions session_options;
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    session_options.AppendExecutionProvider_CUDA(cuda_options);
    Ort::Session session = Ort::Session(env, "rim.onnx", session_options);
    std::cout << "End reached" << std::endl;
}
