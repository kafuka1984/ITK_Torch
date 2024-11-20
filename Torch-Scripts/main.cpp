#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>

int main(int argc, const char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try
    {
        // 使用以下命令从文件中反序列化脚本模块: torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the module\n";
        return -1;
    }
    module.eval();

    std::cout << "ok\n";
    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    torch::Device device(device_type);

    module.to(device);
    // 创建输入向量
    auto input_tensor = torch::ones({1, 1, 512, 512, 80});

    // Move the input tensor to the same device as the module
    input_tensor = input_tensor.to(device);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    // Move the input tensor to the same device as the module
    torch::NoGradGuard no_guard;
    torch::jit::getProfilingMode() = false;
    // 执行模型并将输出转化为张量
    auto output = module.forward(inputs);
    std::cout << "推理完成" << std::endl;
    // std::cout << output << '\n';
}