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

        if (seriesItr == seriesEnd)
        {
            std::cerr << "No DICOMs in: " << dirName << std::endl;
            throw std::runtime_error("No DICOMs found in the specified directory.");
        }

        // if (seriesIdentifier.empty())
        // {
        //     seriesIdentifier = *seriesItr;
        // }

        std::cout << "Reading series: " << seriesIdentifier << std::endl;

        using FileNamesContainer = std::vector<std::string>;
        FileNamesContainer fileNames = nameGenerator->GetFileNames(seriesIdentifier);

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

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <directory> [<seriesIdentifier>]" << std::endl;
        return EXIT_FAILURE;
    }

    std::string dirName = argv[1];
    std::string seriesIdentifier = (argc > 2) ? argv[2] : "";

    try
    {
        auto [imageData, spacing, origin, size, direction] = ITKLoadDICOMSeries(dirName, seriesIdentifier);

        // 打印一些信息
        std::cout << "Image data size: " << imageData.size() << std::endl;
        std::cout << "Spacing: " << spacing[0] << "  " << spacing[1] << "  " << spacing[2] << std::endl;
        std::cout << "Origin: " << origin[0] << "  " << origin[1] << "  " << origin[2] << std::endl;
        std::cout << "Size: " << " size 0 " << size[0] << "  size 1 " << size[1] << "  size 2 " << size[2] << std::endl;
        // 打印变量类型
        std::cout << "imageData type: " << demangle(typeid(imageData).name()) << std::endl;
        std::cout << "spacing type: " << demangle(typeid(spacing).name()) << std::endl;
        std::cout << "origin type: " << demangle(typeid(origin).name()) << std::endl;
        std::cout << "size type: " << demangle(typeid(size).name()) << std::endl;
        std::cout << "direction type: " << demangle(typeid(direction).name()) << std::endl;
        // 打印方向矩阵
        std::cout << "Direction matrix:" << std::endl;
        for (unsigned int i = 0; i < 3; ++i)
        {
            for (unsigned int j = 0; j < 3; ++j)
            {
                std::cout << direction[i][j] << " ";
            }
            std::cout << std::endl;
        }

        // 将影像数据转换为 float 类型
        std::vector<float> floatImageData(imageData.begin(), imageData.end());

        // 将影像数据转换为 PyTorch 张量
        torch::Tensor tensorImage = ConvertToTensor(floatImageData, size);

        // 这里可以对 `tensorImage` 进行进一步处理
        // 例如，计算影像的平均值
        double mean = tensorImage.mean().item<double>();
        std::cout << "Mean intensity: " << mean << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}