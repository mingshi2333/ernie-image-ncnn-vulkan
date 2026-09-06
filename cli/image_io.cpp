// SPDX-License-Identifier: MIT
#include "image_io.h"
#include <png.h>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>
namespace ernie::cli
{
namespace
{
constexpr size_t max_pixels = 100000000;
constexpr size_t max_file_bytes = 512u * 1024u * 1024u;

size_t image_bytes(int width, int height, int channels)
{
    if (width < 1 || height < 1 || channels < 1 || width > 32768 || height > 32768 ||
        size_t(width) > max_pixels / size_t(height) ||
        size_t(width) * size_t(height) > std::numeric_limits<size_t>::max() / size_t(channels))
        throw std::invalid_argument("Image dimensions are invalid or too large");
    return size_t(width) * size_t(height) * size_t(channels);
}

std::vector<uint8_t> read_file(const std::filesystem::path &path)
{
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size == 0 || size > max_file_bytes || size > std::numeric_limits<size_t>::max())
        throw std::invalid_argument("Image file is empty, inaccessible, or too large");
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    std::ifstream input(path, std::ios::binary);
    if (!input.read(reinterpret_cast<char *>(bytes.data()), std::streamsize(bytes.size())))
        throw std::runtime_error("Image file read failed");
    return bytes;
}

std::string extension(const std::filesystem::path &path)
{
    std::string result = path.extension().string();
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    if (result != ".png" && result != ".jpg" && result != ".jpeg" && result != ".bmp" && result != ".tga")
        throw std::invalid_argument("Image extension must be PNG, JPEG, BMP, or TGA");
    return result;
}

void append_bytes(void *context, void *data, int size)
{
    auto &bytes = *static_cast<std::vector<uint8_t> *>(context);
    if (size < 0 || bytes.size() > max_file_bytes - size_t(size))
        return;
    const auto *first = static_cast<const uint8_t *>(data);
    bytes.insert(bytes.end(), first, first + size);
}

void save_new(const std::filesystem::path &path, const std::vector<uint8_t> &bytes)
{
    if (std::filesystem::exists(path))
        throw std::invalid_argument("Output exists; use a new image path");
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::out);
    if (!output.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size())))
        throw std::runtime_error("Image write failed");
}
} // namespace

RgbImage read_image(const std::filesystem::path &path, const std::array<uint8_t, 3> &background)
{
    const auto kind = extension(path);
    const auto encoded = read_file(path);
    int width = 0, height = 0;
    std::vector<uint8_t> rgba;
    if (kind == ".png")
    {
        png_image image{};
        image.version = PNG_IMAGE_VERSION;
        if (!png_image_begin_read_from_memory(&image, encoded.data(), encoded.size()))
            throw std::runtime_error(std::string("PNG read failed: ") + image.message);
        try
        {
            width = int(image.width); height = int(image.height);
            rgba.resize(image_bytes(width, height, 4));
            image.format = PNG_FORMAT_RGBA;
            if (!png_image_finish_read(&image, nullptr, rgba.data(), 0, nullptr))
                throw std::runtime_error(std::string("PNG decode failed: ") + image.message);
        }
        catch (...)
        {
            png_image_free(&image);
            throw;
        }
        png_image_free(&image);
    }
    else
    {
        int source_channels = 0;
        if (!stbi_info_from_memory(encoded.data(), int(encoded.size()), &width, &height, &source_channels))
            throw std::runtime_error(std::string("Image header read failed: ") + stbi_failure_reason());
        image_bytes(width, height, 4);
        stbi_uc *decoded = stbi_load_from_memory(encoded.data(), int(encoded.size()), &width, &height,
                                                 &source_channels, 4);
        if (!decoded)
            throw std::runtime_error(std::string("Image decode failed: ") + stbi_failure_reason());
        rgba.assign(decoded, decoded + image_bytes(width, height, 4));
        stbi_image_free(decoded);
    }
    RgbImage result{width, height, std::vector<uint8_t>(image_bytes(width, height, 3))};
    for (size_t source = 0, target = 0; source < rgba.size(); source += 4, target += 3)
    {
        const unsigned alpha = rgba[source + 3];
        for (int channel = 0; channel < 3; ++channel)
            result.pixels[target + channel] = uint8_t((unsigned(rgba[source + channel]) * alpha +
                                                       unsigned(background[channel]) * (255 - alpha) + 127) / 255);
    }
    return result;
}

void write_image(const std::filesystem::path &path, const RgbImage &rgb)
{
    const auto bytes = image_bytes(rgb.width, rgb.height, 3);
    if (rgb.pixels.size() != bytes)
        throw std::invalid_argument("Invalid RGB image");
    const auto kind = extension(path);
    std::vector<uint8_t> encoded;
    int ok = 0;
    if (kind == ".png")
    {
        png_image image{};
        image.version = PNG_IMAGE_VERSION; image.width = rgb.width; image.height = rgb.height;
        image.format = PNG_FORMAT_RGB;
        png_alloc_size_t size = 0;
        if (!png_image_write_to_memory(&image, nullptr, &size, 0, rgb.pixels.data(), 0, nullptr) || size > max_file_bytes)
            throw std::runtime_error(std::string("PNG sizing failed: ") + image.message);
        encoded.resize(size);
        if (!png_image_write_to_memory(&image, encoded.data(), &size, 0, rgb.pixels.data(), 0, nullptr))
            throw std::runtime_error(std::string("PNG encode failed: ") + image.message);
        encoded.resize(size); ok = 1;
    }
    else if (kind == ".jpg" || kind == ".jpeg")
        ok = stbi_write_jpg_to_func(append_bytes, &encoded, rgb.width, rgb.height, 3, rgb.pixels.data(), 95);
    else if (kind == ".bmp")
        ok = stbi_write_bmp_to_func(append_bytes, &encoded, rgb.width, rgb.height, 3, rgb.pixels.data());
    else
        ok = stbi_write_tga_to_func(append_bytes, &encoded, rgb.width, rgb.height, 3, rgb.pixels.data());
    if (!ok || encoded.empty())
        throw std::runtime_error("Image encode failed");
    save_new(path, encoded);
}

void write_png(const std::filesystem::path &path, const RgbImage &rgb)
{
    if (extension(path) != ".png")
        throw std::invalid_argument("write_png requires a PNG path");
    write_image(path, rgb);
}
} // namespace ernie::cli
