#ifndef FILE_IO_HPP_
#define FILE_IO_HPP_

#include <fstream>

#include <stddef.h>
#include <unistd.h>

#include "core/utils.hpp"

namespace IO {

template <class T>
bool save_to_file(const char* path, const T* a, const size_t& a_size, const T* b,
                  const size_t& b_size, const T* c, const size_t& c_size, const T* d,
                  const size_t& d_size, const T* e, const size_t& e_size);

template <class T>
bool save_to_file(const char* path, const T* a, const T* b, const T* c, const size_t& n);

template <class T>
bool read_from_file(const char* path, T* a, const size_t& a_size, T* b, const size_t& b_size, T* c,
                    const size_t& c_size, T* d, const size_t& d_size, T* e, const size_t& e_size,
                    bool trunc);

template <class T>
bool read_from_file(const char* path, T* a, T* b, T* c, const size_t& n, bool trunc = true);

} // namespace IO

template <class T>
bool IO::save_to_file(const char* path, const T* a, const size_t& a_size, const T* b,
                      const size_t& b_size, const T* c, const size_t& c_size, const T* d,
                      const size_t& d_size, const T* e, const size_t& e_size) {
    std::fstream file(path, std::ios_base::out | std::ios_base::app | std::ios_base::binary);

    if (!file.is_open())
        return false;

    file.write((char*)a, a_size * sizeof(T));
    file.write((char*)b, b_size * sizeof(T));
    file.write((char*)c, c_size * sizeof(T));
    file.write((char*)d, d_size * sizeof(T));
    file.write((char*)e, e_size * sizeof(T));

    file.close();
    return !file.fail();
}

template <class T>
bool IO::save_to_file(const char* path, const T* a, const T* b, const T* c, const size_t& n) {
    std::fstream file(path, std::ios_base::out | std::ios_base::app | std::ios_base::binary);

    if (!file.is_open())
        return false;

    file.write((char*)a, n * sizeof(T));
    file.write((char*)b, n * sizeof(T));
    file.write((char*)c, n * sizeof(T));

    file.close();
    return !file.fail();
}

template <class T>
bool IO::read_from_file(const char* path, T* a, const size_t& a_size, T* b, const size_t& b_size,
                        T* c, const size_t& c_size, T* d, const size_t& d_size, T* e,
                        const size_t& e_size, bool trunc) {
    auto total = (a_size + b_size + c_size + d_size + e_size) * sizeof(T);

    if (!total) return true;

    std::fstream file;
    file.open(path, std::ios_base::in | std::ios_base::ate | std::ios_base::binary);
    if (!file.is_open()) {
        Utils::log(Utils::Level::FAILED, "Couldn't open: ", path);
        return false;
    }

    size_t size = file.tellg();
    if (file.fail()) {
        Utils::log(Utils::Level::ERROR, "Couln't read file size");
    }

    if (total > size) {
        file.close();
        Utils::log(Utils::Level::ERROR, "file too small");
        return false;
    }

    file.seekg(0, std::ios::beg);

    file.read((char*)a, a_size * sizeof(T));
    file.read((char*)b, b_size * sizeof(T));
    file.read((char*)c, c_size * sizeof(T));
    file.read((char*)d, d_size * sizeof(T));
    file.read((char*)e, e_size * sizeof(T));

    file.close();

    if (trunc) {
        if (size == total) {
            if (remove(path) != 0) {
                std::perror("Truncation failed");
            }
            return true;
        }

        file.open(path, std::ios::in | std::ios::out | std::ios::binary);
        if (file.is_open()) {
            std::vector<char> buffer(size - total);
            file.seekg(total, std::ios::beg);
            file.read(buffer.data(), buffer.size());
            file.seekg(0, std::ios::beg);
            file.write(buffer.data(), buffer.size());
            file.close();
        }

        int res = truncate(path, size - total);
        if (res != 0)
            std::perror("Truncation failed");
    }

    return !file.fail();
}

template <class T>
bool IO::read_from_file(const char* path, T* a, T* b, T* c, const size_t& n, bool trunc) {
    std::fstream file;
    file.open(path, std::ios_base::in | std::ios_base::ate | std::ios_base::binary);
    if (!file.is_open()) {
        Utils::log(Utils::Level::FAILED, "Couldn't open: ", path);
        return false;
    }

    size_t size = file.tellg();
    if (file.fail()) {
        Utils::log(Utils::Level::ERROR, "Couln't read file size");
    }

    std::cout << "SIZE: " << size << "\n";
    std::cout << "n: " << n << "\n";
    if (n * sizeof(T) * 3 > size) {
        file.close();
        Utils::log(Utils::Level::ERROR, "file too small");
        return false;
    }

    file.seekg(0, std::ios::beg);

    file.read((char*)a, n * sizeof(T));
    file.read((char*)b, n * sizeof(T));
    file.read((char*)c, n * sizeof(T));

    file.close();

    if (trunc) {
        size_t to_trunc = n * 3 * sizeof(T);

        file.open(path, std::ios::in | std::ios::out | std::ios::binary);
        if (file.is_open()) {
            std::vector<char> buffer(size - to_trunc);
            file.seekg(to_trunc, std::ios::beg);
            file.read(buffer.data(), buffer.size());
            file.seekg(0, std::ios::beg);
            file.write(buffer.data(), buffer.size());
            file.close();
        }

        if (size == to_trunc) {
            if (remove(path) != 0) {
                std::perror("Truncation failed");
            }
        } else if (size > n * 3 * sizeof(T)) {
            int res = truncate(path, size - to_trunc);
            if (res != 0)
                std::perror("Truncation failed");
        }
    }

    return !file.fail();
}

#endif