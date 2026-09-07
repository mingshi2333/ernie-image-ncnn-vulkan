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
#ifndef _WIN32
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

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
        const fs::path root = fs::temp_directory_path() / fs::u8path(u8"ernie-image-io-\u56fe\u50cf \U0001f5bc");
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
        const ernie::RgbImage wide{2,1,{255,0,0,0,0,255}};
        const auto stretched=ernie::cli::resize_image(wide,4,1,"stretch");
        require(stretched.pixels[0]==255 && stretched.pixels[1]==0 && stretched.pixels[2]==0 &&
                stretched.pixels[9]==0 && stretched.pixels[10]==0 && stretched.pixels[11]==255,
                "Bilinear resize did not clamp edge samples");
        const auto fitted=ernie::cli::resize_image(wide,4,4,"fit",{3,4,5});
        require(fitted.width==4 && fitted.height==4 && fitted.pixels[0]==3 && fitted.pixels[1]==4 && fitted.pixels[2]==5,
                "Fit resize did not letterbox with the explicit background");
        require(fitted.pixels[(size_t(1)*4)*3]!=3,"Fit resize omitted centered content");
        const ernie::RgbImage solid{1,1,{9,17,33}};
        require(ernie::cli::resize_image(solid,3,2,"stretch").pixels==std::vector<uint8_t>({9,17,33,9,17,33,9,17,33,9,17,33,9,17,33,9,17,33}),
                "Stretch resize changed a constant image");
        const auto cropped=ernie::cli::resize_image(wide,1,2,"crop");
        require(cropped.width==1 && cropped.height==2 && cropped.pixels.size()==6,"Crop resize shape differs");
        require_error([&]{ernie::cli::resize_image(wide,4,4,"implicit");},"Unknown resize mode accepted");

        const auto unicode = root / u8"中文图像.tga";
        ernie::cli::write_image(unicode, source);
        require(ernie::cli::read_image(unicode).pixels == source.pixels, "Unicode path roundtrip differs");
        require_error([&] { ernie::cli::write_image(unicode, source); }, "Existing output was overwritten");
        require(ernie::cli::read_image(unicode).pixels == source.pixels, "Rejected write changed existing output");
        // Valid headers and plausible sizes do not prove that pixels/entropy
        // are complete. stb's unbounded memory reader formerly fabricated zeros.
        const ernie::RgbImage small{2, 2, std::vector<uint8_t>(12, 123)};
        for (const auto &entry : std::vector<std::pair<std::string, size_t>>{{"bmp",54},{"tga",18},{"jpg",10}})
        {
            const auto full = root / ("complete." + entry.first);
            ernie::cli::write_image(full, small);
            std::ifstream stream(full, std::ios::binary);
            std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(stream)), {});
            const size_t cut = entry.first == "jpg" ? bytes.size() - entry.second : entry.second;
            bytes.resize(cut);
            const auto truncated = root / ("truncated." + entry.first);
            save(truncated, bytes);
            require_error([&] { ernie::cli::read_image(truncated); }, "Truncated valid image was accepted");
        }
#ifndef _WIN32
        // Deterministic write failure, isolated from the test runner's limits.
        const pid_t child = fork();
        require(child >= 0, "Cannot fork write-failure probe");
        if (child == 0)
        {
            signal(SIGXFSZ, SIG_IGN);
            const rlimit limit{0, 0};
            if (setrlimit(RLIMIT_FSIZE, &limit)) _exit(2);
            try { ernie::cli::write_image(root / "write-failure.png", small); }
            catch (const std::exception &) { _exit(0); }
            _exit(1);
        }
        int status = 0;
        require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                "Incomplete output was reported as successfully written");
#endif
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
