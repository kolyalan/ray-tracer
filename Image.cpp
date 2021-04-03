//
// Created by kolyalan on 27.03.2021.
//

#include "Image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

MyImage::MyImage(const std::string &a_path, bool hdr) : hdr(hdr)
{

    int ch_num;
    if (hdr) {
        data = (cl_uchar4*)stbi_loadf(a_path.c_str(), &width, &height, &ch_num, 4);
    } else {
        data = (cl_uchar4*)stbi_load(a_path.c_str(), &width, &height, &ch_num, sizeof(cl_uchar4));
    }
    if (data != nullptr) {
        if (ch_num != sizeof(cl_uchar4)) {
            std::cout << "Image channels incoorect: " << ch_num << " path: "<< a_path << std::endl;
        }
        size = width * height * sizeof(cl_uchar4);
        self_allocated = false;
    } else {
        std::cout << "Unable to load image path: " << a_path << std::endl;
    }
}

MyImage::MyImage(const MyImage &other) {
  width = other.width;
  height = other.height;
  size = other.size;
  self_allocated = other.self_allocated;
  if (other.self_allocated) {
    data = new cl_uchar4[width * height] {};
  } else {
    data = (cl_uchar4 *)malloc(size);
  }
  if (data != nullptr) {
    memcpy(data, other.data, size);
  }
}

MyImage::MyImage(MyImage &&other) {
  width = other.width;
  height = other.height;
  size = other.size;
  self_allocated = other.self_allocated;
  data = other.data;
  other.data = nullptr;
  other.size = 0;
  other.width = -1;
  other.height = -1;
  other.self_allocated = false;
}

MyImage &MyImage::operator= (const MyImage &other) {
  if (this == &other) {
    return *this;
  }
  this->~MyImage();
  width = other.width;
  height = other.height;
  size = other.size;
  self_allocated = other.self_allocated;
  if (other.self_allocated) {
    data = new cl_uchar4[width * height] {};
  } else {
    data = (cl_uchar4 *)malloc(size);
  }
  if (data != nullptr) {
    memcpy(data, other.data, size);
  }
  return *this;
}

MyImage &MyImage::operator= (MyImage &&other) {
  if (this == &other) {
    return *this;
  }
  this->~MyImage();
  width = other.width;
  height = other.height;
  size = other.size;
  self_allocated = other.self_allocated;
  data = other.data;
  other.data = nullptr;
  other.size = 0;
  other.width = -1;
  other.height = -1;
  other.self_allocated = false;
  return *this;
}


MyImage::MyImage(int width, int height) : width(width), height(height) {
    data = new cl_uchar4[width * height];
    size = width * height * sizeof(*data);
    self_allocated = true;
}

MyImage::~MyImage()
{
  if (data == nullptr) {
    return;
  }
  if(self_allocated) {
    delete [] data;
  } else {
    stbi_image_free(data);
  }
  data = nullptr;
}

int MyImage::getWidth() const {
    return width;
}

int MyImage::getHeight() const {
    return height;
}

size_t MyImage::getSize() const {
    return size;
}

cl_uchar4 *MyImage::getData() const {
    return data;
}

int MyImage::save(const std::string &path) const {
    return !stbi_write_png(path.c_str(), width, height, sizeof(cl_uchar4), data, sizeof(cl_uchar4) * width);
}
