#include "itkImage.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"
#include <iostream>
#include <tuple>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <cstring> // for strdup
#include <cstdlib> // for free
#include <cxxabi.h>
#include <Eigen/Dense>

// 定义影像类型
using PixelType = signed short;
constexpr unsigned int Dimension = 3;
using ImageType = itk::Image<PixelType, Dimension>;

// 读取 DICOM 系列文件的函数
std::tuple<std::vector<PixelType>, ImageType::SpacingType, ImageType::PointType, ImageType::SizeType, ImageType::DirectionType> ITKLoadDICOMSeries(const std::string &dirName, const std::string &seriesIdentifier = "")
{
    using NamesGeneratorType = itk::GDCMSeriesFileNames;
    auto nameGenerator = NamesGeneratorType::New();

    nameGenerator->SetUseSeriesDetails(true);
    nameGenerator->AddSeriesRestriction("0008|0021");
    nameGenerator->SetGlobalWarningDisplay(false);
    nameGenerator->SetDirectory(dirName);

    try
    {
        using SeriesIdContainer = std::vector<std::string>;
        const SeriesIdContainer &seriesUID = nameGenerator->GetSeriesUIDs();
        auto seriesItr = seriesUID.begin();
        auto seriesEnd = seriesUID.end();
        std::cout << "Series " << *seriesItr << std::endl;

        if (seriesItr == seriesEnd)
        {
            std::cerr << "No DICOMs in: " << dirName << std::endl;
            throw std::runtime_error("No DICOMs found in the specified directory.");
        }

        std::cout << "Reading series: " << seriesIdentifier << std::endl;

        using FileNamesContainer = std::vector<std::string>;
        FileNamesContainer fileNames = nameGenerator->GetFileNames(*seriesItr);

        using ReaderType = itk::ImageSeriesReader<ImageType>;
        auto reader = ReaderType::New();
        using ImageIOType = itk::GDCMImageIO;
        auto dicomIO = ImageIOType::New();
        reader->SetImageIO(dicomIO);
        reader->SetFileNames(fileNames);
        reader->ForceOrthogonalDirectionOff(); // properly read CTs with gantry tilt

        reader->Update();

        ImageType::Pointer image = reader->GetOutput();
        ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();
        ImageType::SpacingType spacing = image->GetSpacing();
        ImageType::PointType origin = image->GetOrigin();
        ImageType::DirectionType direction = image->GetDirection();

        std::vector<PixelType> imageData(size[0] * size[1] * size[2]);
        std::copy(image->GetBufferPointer(), image->GetBufferPointer() + imageData.size(), imageData.begin());

        // 打印 spacing 和 origin
        std::cout << "Spacing: " << spacing[0] << "  " << spacing[1] << "  " << spacing[2] << std::endl;
        std::cout << "Origin: " << origin[0] << "  " << origin[1] << "  " << origin[2] << std::endl;
        std::cout << "Dims: " << " size 0 " << size[0] << "  size 1 " << size[1] << "  size 2 " << size[2] << std::endl;

        return std::make_tuple(imageData, spacing, origin, size, direction);
    }
    catch (itk::ExceptionObject &excp)
    {
        std::cerr << "ExceptionObject caught: " << excp << std::endl;
        throw;
    }
}

// 将影像数据转换为 PyTorch 张量
torch::Tensor ConvertToTensor(const std::vector<float> &imageData, const itk::Size<3> &dims)
{
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);

    // 创建张量并按 Z, Y, X 顺序加载数据
    torch::Tensor image_tensor = torch::from_blob(
        (void *)imageData.data(),
        {static_cast<int>(dims[2]), static_cast<int>(dims[1]), static_cast<int>(dims[0])},
        options);

    // 使用 permute 调整张量维度顺序为 (X, Y, Z)
    torch::Tensor permuted_tensor = image_tensor.permute({2, 1, 0}).clone();

    std::cout << "Tensor Shape: " << permuted_tensor.sizes() << std::endl;
    std::cout << std::endl;

    return permuted_tensor;
}

// 解码类型名称
std::string demangle(const char *mangled_name)
{
    int status;
    char *demangled = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);
    if (status == 0)
    {
        std::string result(demangled);
        free(demangled);
        return result;
    }
    else
    {
        free(demangled);
        return mangled_name;
    }
}

// HU值转换为uint8
std::vector<float> HU2uint8(const std::vector<PixelType> &image, const itk::Size<3> &size, float HU_min = -1024.0, float HU_max = 300.0, float HU_nan = -2000.0)
{
    // 计算总元素数量
    size_t total_elements = size[0] * size[1] * size[2];

    // 创建一个 float 类型的向量来存储转换后的数据
    std::vector<float> image_float(total_elements);

    // 将 short int 类型的数据转换为 float 类型
    for (size_t i = 0; i < total_elements; ++i)
    {
        image_float[i] = static_cast<float>(image[i]);
    }

    // 将输入的 std::vector<float> 映射为 Eigen::MatrixXf
    Eigen::Map<Eigen::MatrixXf> image_matrix(image_float.data(), total_elements, 1);

    // 使用 unaryExpr 对每个元素进行操作，将 NaN 值替换为 HU_nan
    image_matrix = image_matrix.unaryExpr([HU_nan](float x)
                                          { return std::isnan(x) ? HU_nan : x; });

    // 将 HU 值归一化到 [0, 1] 范围内
    image_matrix.array() = (image_matrix.array() - HU_min) / (HU_max - HU_min);

    // 使用 cwiseMax 和 cwiseMin 将所有值限制在 [0, 1] 范围内
    image_matrix = image_matrix.cwiseMax(0.0f).cwiseMin(1.0f);

    // 创建一个 std::vector<float> 来存储结果
    std::vector<float> result(total_elements);

    // 将处理后的数据从 Eigen::MatrixXf 复制到 std::vector<float> 中
    for (size_t i = 0; i < total_elements; ++i)
    {
        result[i] = image_matrix(i, 0);
    }

    return result;
}
int main(int argc, const char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <directory> <path-to-exported-script-module>\n";
        return -1;
    }

    std::string dirName = argv[1];
    std::string modelPath = argv[2];

    try
    {
        // 读取 DICOM 系列文件
        auto [imageData, spacing, origin, size, direction] = ITKLoadDICOMSeries(dirName);

        // 打印一些信息
        std::cout << "Image data size: " << imageData.size() << std::endl;
        std::cout << "Spacing: " << spacing[0] << "  " << spacing[1] << "  " << spacing[2] << std::endl;
        std::cout << "Origin: " << origin[0] << "  " << origin[1] << "  " << origin[2] << std::endl;
        std::cout << "Size: " << " size 0 " << size[0] << "  size 1 " << size[1] << "  size 2 " << size[2] << std::endl;
        std::cout << "imageData type: " << demangle(typeid(imageData).name()) << std::endl;
        std::cout << "spacing type: " << demangle(typeid(spacing).name()) << std::endl;
        std::cout << "origin type: " << demangle(typeid(origin).name()) << std::endl;
        std::cout << "size type: " << demangle(typeid(size).name()) << std::endl;
        std::cout << "direction type: " << demangle(typeid(direction).name()) << std::endl;
        std::cout << "Direction matrix:" << std::endl;
        for (unsigned int i = 0; i < 3; ++i)
        {
            for (unsigned int j = 0; j < 3; ++j)
            {
                std::cout << direction[i][j] << " ";
            }
            std::cout << std::endl;
        }

        // 预处理影像数据
        std::vector<float> floatImageData = HU2uint8(imageData, size);

        // 将影像数据转换为 PyTorch 张量
        torch::Tensor tensorImage = ConvertToTensor(floatImageData, size);

        // 加载预训练模型
        torch::jit::script::Module module;
        try
        {
            module = torch::jit::load(modelPath);
        }
        catch (const c10::Error &e)
        {
            std::cerr << "Error loading the module\n";
            return -1;
        }
        module.eval();

        std::cout << "Model loaded successfully.\n";

        torch::DeviceType device_type;
        if (torch::cuda::is_available())
        {
            std::cout << "CUDA available! Running on GPU." << std::endl;
            device_type = torch::kCUDA;
        }
        else
        {
            std::cout << "Running on CPU." << std::endl;
            device_type = torch::kCPU;
        }

        torch::Device device(device_type);

        module.to(device);

        // 将输入张量移动到与模型相同的设备上
        tensorImage = tensorImage.to(device);

        // 准备输入数据
        std::vector<torch::jit::IValue> inputs;
        tensorImage = tensorImage.unsqueeze(0).unsqueeze(0); // 形状变为 [1, 1, depth, height, width]
        inputs.push_back(tensorImage);

        // 执行模型推理
        torch::NoGradGuard no_grad;
        torch::jit::getProfilingMode() = false;
        auto output = module.forward(inputs);

        std::cout << "Inference completed." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}