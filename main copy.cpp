#include "itkImage.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"
#include <iostream>
#include <tuple>
#include <vector>

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
        using SeriesIdContatiner = std::vector<std::string>;
        const SeriesIdContationer &seriesUID = nameGenerator->GetSeriesUIDs();
        auto seriesItr = seriesUID.begin();
        auto seriesEnd = seriesUID.end();

        if (seriesItr == seriesEnd)
        {
            std::cerr << "No DICOMs in: " << dirName << std::endl;
            throw std::runtime_error("No DICOMS found in the specified directory.");
        }

        std::cout << "Reading series: " << seriesIdentifier << std::endl;

        using FileNamesContainer = std::vector<std::string>;
        FileNamesContainer fileNames = nameGenerator->GetFileNames(seriesIdentifier);

        using ReaderType = itk::ImageSeriesReader<ImageType>;
        auto reader = ReaderType::New();
        using ImageIOType = itk::GDCMImageIO;
        auto dicomIO = ImageIOType::New();
        reader->SetImageIO(dicomIO);
        reader->SetFileNames(fileNames);
        reader->ForceOrthogonalDirectionOff(); // 读取CTs with gantry tilt

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

        return std::make_tuple(iamgeData, spacing, origin, size, direction);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}

torch::Tensor ConverToTensor(const std::vector<float> &imageData, const itk::Size<3> &dims)
{
    troch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);

    torch::Tensor image_tensor = torch::from_blob(
        (void*)imageData.data(),
        {static_cast<int>(dims[2],  static_cast<int>(dims[1]), static_cast<int>(dims[0])},
        options);
    torch::Tensor permuted_tensor = image_tensor.permute({2, 1, 0}).clone();

    std::cout << "Tensor Shape:" << permuted_tensor.sizes() << std::endl;
    return permuted_tensor;
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

        std::cout << "Image data size: " << imageData.size() << std::endl;
        std::cout << "Spacing: " << spacing[0] << "  " << spacing[1] << "  " << spacing[2] << std::endl;
        std::cout << "Origin: " << origin[0] << "  " << origin[1] << "  " << origin[2] << std::endl;
        std::cout << "Size: " << " size 0 " << size[0] << "  size 1 " << size[1] << "  size 2 " << size[2] << std::endl;
    }

    std::cout << "Direction matrix:" << std::endl;
    for (unsigned int i = 0; i < 3; ++i)
    {
        for (unsigned int j = 0; j < 3; ++j)
        {
            std::cout << direction[i][j];
        }
        std::cout << std::endl;
    }

    std::vector<float> floatImageData(imageData.begin(), imageData.end());

    torch::Tensor tensorImage = ConvertToTensor(floatImageData, size);

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