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
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
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

struct EncodedBuffer
{
    std::vector<uint8_t> bytes;
    bool failed = false;
};

void append_bytes(void *context, void *data, int size)
{
    auto &buffer = *static_cast<EncodedBuffer *>(context);
    auto &bytes = buffer.bytes;
    if (buffer.failed)
        return;
    if (size < 0 || size_t(size) > max_file_bytes || bytes.size() > max_file_bytes - size_t(size))
    {
        buffer.failed = true;
        return;
    }
    const auto *first = static_cast<const uint8_t *>(data);
    try { bytes.insert(bytes.end(), first, first + size); }
    catch (...) { buffer.failed = true; }
}

struct EncodedInput
{
    const std::vector<uint8_t> &bytes;
    size_t position = 0;
    bool truncated = false;
};

int input_read(void *context, char *data, int size)
{
    auto &input = *static_cast<EncodedInput *>(context);
    if (size <= 0)
    {
        input.truncated = input.truncated || size < 0;
        return 0;
    }
    const size_t count = std::min(size_t(size), input.bytes.size() - input.position);
    // A short prefetch is legal. Requesting another byte after EOF is not:
    // stb's memory reader otherwise supplies zero pixels for truncated BMP/TGA.
    if (size > 0 && count == 0)
        input.truncated = true;
    std::memcpy(data, input.bytes.data() + input.position, count);
    input.position += count;
    return int(count);
}

void input_skip(void *context, int size)
{
    auto &input = *static_cast<EncodedInput *>(context);
    if (size < 0 || size_t(size) > input.bytes.size() - input.position)
    {
        input.truncated = true;
        input.position = input.bytes.size();
    }
    else
        input.position += size_t(size);
}

int input_eof(void *context)
{
    const auto &input = *static_cast<EncodedInput *>(context);
    return input.position == input.bytes.size();
}

std::runtime_error decode_error(const char *action)
{
    const auto *reason = stbi_failure_reason();
    return std::runtime_error(std::string(action) + (reason ? reason : "invalid or truncated image"));
}

void save_new(const std::filesystem::path &path, const std::vector<uint8_t> &bytes)
{
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
#ifdef _WIN32
    const int fd = _wopen(path.c_str(), _O_BINARY | _O_WRONLY | _O_CREAT | _O_EXCL, _S_IREAD | _S_IWRITE);
#else
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
#endif
    if (fd < 0)
    {
        if (errno == EEXIST)
            throw std::invalid_argument("Output exists; use a new image path");
        throw std::runtime_error("Cannot create image output");
    }
    bool failed = false;
    for (size_t offset = 0; offset < bytes.size();)
    {
        const auto count = unsigned(std::min(bytes.size() - offset, size_t(std::numeric_limits<int>::max())));
#ifdef _WIN32
        const auto written = _write(fd, bytes.data() + offset, count);
#else
        const auto written = write(fd, bytes.data() + offset, count);
#endif
        if (written < 0 && errno == EINTR)
            continue;
        if (written <= 0) { failed = true; break; }
        offset += size_t(written);
    }
#ifdef _WIN32
    const int closed = _close(fd);
#else
    const int closed = close(fd);
#endif
    if (failed || closed != 0)
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
            throw decode_error("Image header read failed: ");
        image_bytes(width, height, 4);
        const stbi_io_callbacks callbacks{input_read, input_skip, input_eof};
        EncodedInput input{encoded};
        const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> decoded(
            stbi_load_from_callbacks(&callbacks, &input, &width, &height, &source_channels, 4), stbi_image_free);
        if (!decoded)
            throw decode_error("Image decode failed: ");
        if (input.truncated)
            throw std::runtime_error("Image decode failed: truncated image payload");
        rgba.assign(decoded.get(), decoded.get() + image_bytes(width, height, 4));
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
    EncodedBuffer buffer;
    auto &encoded = buffer.bytes;
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
        ok = stbi_write_jpg_to_func(append_bytes, &buffer, rgb.width, rgb.height, 3, rgb.pixels.data(), 95);
    else if (kind == ".bmp")
        ok = stbi_write_bmp_to_func(append_bytes, &buffer, rgb.width, rgb.height, 3, rgb.pixels.data());
    else
        ok = stbi_write_tga_to_func(append_bytes, &buffer, rgb.width, rgb.height, 3, rgb.pixels.data());
    if (!ok || buffer.failed || encoded.empty())
        throw std::runtime_error("Image encode failed");
    save_new(path, encoded);
}

RgbImage resize_image(const RgbImage &source, int width, int height, const std::string &mode,
                      const std::array<uint8_t, 3> &background)
{
    image_bytes(source.width, source.height, 3);
    if (source.pixels.size() != size_t(source.width) * source.height * 3)
        throw std::invalid_argument("Invalid source RGB image");
    image_bytes(width, height, 3);
    if (mode != "stretch" && mode != "fit" && mode != "crop")
        throw std::invalid_argument("Resize mode must be stretch, fit, or crop");
    int scaled_width = width, scaled_height = height, offset_x = 0, offset_y = 0;
    if (mode != "stretch")
    {
        const double sx = double(width) / source.width, sy = double(height) / source.height;
        const double scale = mode == "fit" ? std::min(sx, sy) : std::max(sx, sy);
        scaled_width = std::max(1, int(std::lround(source.width * scale)));
        scaled_height = std::max(1, int(std::lround(source.height * scale)));
        offset_x = (width - scaled_width) / 2;
        offset_y = (height - scaled_height) / 2;
    }
    RgbImage result{width, height, std::vector<uint8_t>(image_bytes(width, height, 3))};
    for (size_t i = 0; i < result.pixels.size(); i += 3)
        std::copy(background.begin(), background.end(), result.pixels.begin() + i);
    const int x0 = std::max(0, offset_x), y0 = std::max(0, offset_y);
    const int x1 = std::min(width, offset_x + scaled_width), y1 = std::min(height, offset_y + scaled_height);
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x)
        {
            const float fx = (float(x - offset_x) + .5f) * source.width / scaled_width - .5f;
            const float fy = (float(y - offset_y) + .5f) * source.height / scaled_height - .5f;
            const int floor_x=int(std::floor(fx)),floor_y=int(std::floor(fy));
            const int ax = std::clamp(floor_x, 0, source.width - 1);
            const int ay = std::clamp(floor_y, 0, source.height - 1);
            const int bx = std::clamp(floor_x + 1, 0, source.width - 1);
            const int by = std::clamp(floor_y + 1, 0, source.height - 1);
            const float wx = std::clamp(fx - std::floor(fx), 0.f, 1.f);
            const float wy = std::clamp(fy - std::floor(fy), 0.f, 1.f);
            for (int c = 0; c < 3; ++c)
            {
                auto pixel = [&](int px, int py) { return source.pixels[(size_t(py) * source.width + px) * 3 + c]; };
                const float top = pixel(ax, ay) * (1.f - wx) + pixel(bx, ay) * wx;
                const float bottom = pixel(ax, by) * (1.f - wx) + pixel(bx, by) * wx;
                result.pixels[(size_t(y) * width + x) * 3 + c] = uint8_t(std::lround(top * (1.f - wy) + bottom * wy));
            }
        }
    return result;
}

void write_png(const std::filesystem::path &path, const RgbImage &rgb)
{
    if (extension(path) != ".png")
        throw std::invalid_argument("write_png requires a PNG path");
    write_image(path, rgb);
}
} // namespace ernie::cli
