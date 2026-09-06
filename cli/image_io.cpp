// SPDX-License-Identifier: MIT
#include "image_io.h"
#include <png.h>
#include <stdexcept>
namespace ernie::cli
{
void write_png(const std::filesystem::path &path, const RgbImage &rgb)
{
    if (rgb.width < 1 || rgb.height < 1 || rgb.pixels.size() != size_t(rgb.width) * rgb.height * 3)
        throw std::invalid_argument("Invalid RGB image");
    if (std::filesystem::exists(path))
        throw std::invalid_argument("Output exists; use a new PNG path");
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    image.width = rgb.width;
    image.height = rgb.height;
    image.format = PNG_FORMAT_RGB;
    if (!png_image_write_to_file(&image, path.string().c_str(), 0, rgb.pixels.data(), 0, nullptr))
        throw std::runtime_error(std::string("PNG write failed: ") + image.message);
}
} // namespace ernie::cli
