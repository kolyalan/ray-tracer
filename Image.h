//
// Created by kolyalan on 27.03.2021.
//

#ifndef GRAPHICS0_IMAGE_H
#define GRAPHICS0_IMAGE_H

#include <string>
#include <iostream>
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_VERSION_3_0
#include "CL/opencl.hpp"

class MyImage {

public:
    explicit MyImage(const std::string &a_path, bool hdr = 0);
    MyImage() {}
    MyImage(int width, int height);
    MyImage(const MyImage &other);
    MyImage(MyImage &&other);
    MyImage &operator= (const MyImage &other);
    MyImage &operator= (MyImage &&other);
    ~MyImage();

    int getWidth() const;
    int getHeight() const;
    int isHDR() const {return hdr;};
    size_t getSize() const;
    cl_uchar4 *getData() const;

    int save(const std::string &path) const;

private:
    int width = -1;
    int height = -1;
    size_t size = 0;
    cl_uchar4 *data = nullptr;
    bool self_allocated = false;
    bool hdr = 0;
};


#endif //GRAPHICS0_IMAGE_H
