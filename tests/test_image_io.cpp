// SPDX-License-Identifier: MIT
#include "image_io.h"
#include <png.h>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <class Function> void require_error(Function function, const char *message)
{
    bool rejected = false;
    try { function(); } catch (const std::exception &) { rejected = true; }
    require(rejected, message);
}

std::vector<uint8_t> png_bytes(int width, int height, int format, const std::vector<uint8_t> &pixels)
{
    png_image image{};
    image.version = PNG_IMAGE_VERSION; image.width = width; image.height = height; image.format = format;
    png_alloc_size_t size = 0;
    require(png_image_write_to_memory(&image, nullptr, &size, 0, pixels.data(), 0, nullptr), "PNG fixture sizing failed");
    std::vector<uint8_t> bytes(size);
    require(png_image_write_to_memory(&image, bytes.data(), &size, 0, pixels.data(), 0, nullptr), "PNG fixture encoding failed");
    bytes.resize(size);
    return bytes;
}

void save(const std::filesystem::path &path, const std::vector<uint8_t> &bytes)
{
    std::ofstream stream(path, std::ios::binary);
    require(bool(stream.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()))),
            "Fixture write failed");
}

int main()
{
    namespace fs = std::filesystem;
    try
    {
        const fs::path root = fs::temp_directory_path() / "ernie-image-io-contract";
        fs::remove_all(root); fs::create_directories(root);
        ernie::RgbImage source; source.width = 32; source.height = 24;
        source.pixels.resize(size_t(source.width) * source.height * 3);
        for (int y = 0; y < source.height; ++y)
            for (int x = 0; x < source.width; ++x)
            {
                const size_t i = size_t(y * source.width + x) * 3;
                source.pixels[i] = uint8_t(x * 7); source.pixels[i + 1] = uint8_t(y * 10);
                source.pixels[i + 2] = uint8_t((x * 3 + y * 5) & 255);
            }
        for (const char *name : {"roundtrip.png", "roundtrip.bmp", "roundtrip.tga"})
        {
            const auto path = root / name;
            ernie::cli::write_image(path, source);
            const auto actual = ernie::cli::read_image(path);
            require(actual.width == source.width && actual.height == source.height && actual.pixels == source.pixels,
                    "Lossless image roundtrip differs");
        }
        const auto jpeg = root / "roundtrip.jpg";
        ernie::cli::write_image(jpeg, source);
        const auto jpeg_image = ernie::cli::read_image(jpeg);
        require(jpeg_image.width == source.width && jpeg_image.height == source.height, "JPEG dimensions differ");
        double error = 0;
        for (size_t i = 0; i < source.pixels.size(); ++i)
            error += std::abs(int(source.pixels[i]) - int(jpeg_image.pixels[i]));
        require(error / source.pixels.size() < 8., "JPEG quality-95 mean error is unreasonable");

        const auto gray = root / "gray.png";
        save(gray, png_bytes(2, 1, PNG_FORMAT_GRAY, {17, 201}));
        require(ernie::cli::read_image(gray).pixels == std::vector<uint8_t>({17,17,17,201,201,201}),
                "Grayscale channels were not replicated");
        const auto alpha = root / "alpha.png";
        save(alpha, png_bytes(2, 1, PNG_FORMAT_RGBA, {255,0,0,128, 1,2,3,0}));
        require(ernie::cli::read_image(alpha).pixels == std::vector<uint8_t>({255,127,127,255,255,255}),
                "Default white alpha composition differs");
        require(ernie::cli::read_image(alpha, {0,0,0}).pixels == std::vector<uint8_t>({128,0,0,0,0,0}),
                "Explicit alpha background differs");

        const auto unicode = root / u8"中文图像.tga";
        ernie::cli::write_image(unicode, source);
        require(ernie::cli::read_image(unicode).pixels == source.pixels, "Unicode path roundtrip differs");
        require_error([&] { ernie::cli::write_image(unicode, source); }, "Existing output was overwritten");
        save(root / "broken.jpg", {0xff,0xd8,0,1,2,3});
        require_error([&] { ernie::cli::read_image(root / "broken.jpg"); }, "Corrupt image was accepted");
        std::vector<uint8_t> huge(54); huge[0]='B'; huge[1]='M'; huge[2]=54; huge[10]=54; huge[14]=40;
        const uint32_t declared = 50000;
        for (int shift=0;shift<4;++shift)
        {
            huge[18+shift]=uint8_t(declared>>(shift*8)); huge[22+shift]=uint8_t(declared>>(shift*8));
        }
        huge[26]=1; huge[28]=24; save(root / "huge.bmp", huge);
        require_error([&] { ernie::cli::read_image(root / "huge.bmp"); }, "Huge declared image was accepted");
        ernie::RgbImage invalid{std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), {}};
        require_error([&] { ernie::cli::write_image(root / "overflow.png", invalid); }, "Overflowing RGB dimensions accepted");
        require_error([&] { ernie::cli::write_png(root / "wrong.jpg", source); }, "write_png accepted non-PNG path");
        fs::remove_all(root);
        std::cout << "PNG/JPEG/BMP/TGA image I/O contracts passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
