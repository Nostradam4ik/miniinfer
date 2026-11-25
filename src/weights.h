#pragma once
// Reads the tensors written by tools/export_reference.py:
//   <base>.manifest.txt  - lines of "name dtype ndim d0 d1 ... offset"
//   <base>.bin           - raw data
// A tensor is read from disk only when requested (the weights take hundreds of MB).
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "tensor.cuh"

class TensorFile {
public:
    struct Entry {
        std::string dtype;  // "float32" or "int32"
        std::vector<int> shape;
        uint64_t offset = 0;
        size_t numel() const {
            size_t n = 1;
            for (int d : shape) n *= static_cast<size_t>(d);
            return n;
        }
    };

    explicit TensorFile(const std::string& base) : bin_path_(base + ".bin") {
        std::ifstream man(base + ".manifest.txt");
        if (!man) throw std::runtime_error("cannot open " + base + ".manifest.txt");
        std::string line;
        while (std::getline(man, line)) {
            if (line.empty()) continue;
            std::istringstream is(line);
            std::string name;
            Entry e;
            int ndim = 0;
            is >> name >> e.dtype >> ndim;
            e.shape.resize(ndim);
            for (int& d : e.shape) is >> d;
            is >> e.offset;
            if (!is) throw std::runtime_error("bad manifest line: " + line);
            entries_[name] = e;
        }
    }

    bool has(const std::string& name) const { return entries_.count(name) > 0; }

    const Entry& entry(const std::string& name) const {
        auto it = entries_.find(name);
        if (it == entries_.end()) throw std::runtime_error("tensor not found: " + name);
        return it->second;
    }

    std::vector<float> floats(const std::string& name) const {
        const Entry& e = entry(name);
        if (e.dtype != "float32") throw std::runtime_error(name + ": not float32");
        std::vector<float> v(e.numel());
        read_raw(e.offset, v.data(), v.size() * sizeof(float));
        return v;
    }

    std::vector<int> ints(const std::string& name) const {
        const Entry& e = entry(name);
        if (e.dtype != "int32") throw std::runtime_error(name + ": not int32");
        std::vector<int> v(e.numel());
        read_raw(e.offset, v.data(), v.size() * sizeof(int));
        return v;
    }

    Tensor to_gpu(const std::string& name) const {
        return Tensor::from_host(floats(name), entry(name).shape);
    }

private:
    void read_raw(uint64_t offset, void* dst, size_t bytes) const {
        std::ifstream f(bin_path_, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + bin_path_);
        f.seekg(static_cast<std::streamoff>(offset));
        f.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
        if (!f) throw std::runtime_error("short read from " + bin_path_);
    }

    std::string bin_path_;
    std::map<std::string, Entry> entries_;
};
