#ifndef __SIM_UTILS_H__
#define __SIM_UTILS_H__
// Simulation utilities

#include <H5Cpp.h>
#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <vector>

const std::size_t L1_CACHE_SIZE = 64;

static std::size_t TOTAL_ALLOC = 0;

template <typename Type> Type *alloc_mem(const std::size_t n) {
  auto alloc_size = sizeof(Type) * n;
  auto *ret = std::aligned_alloc(L1_CACHE_SIZE, alloc_size);
  assert(ret && "Failed to allocate memory");

  return static_cast<Type *>(ret);
}

template <typename Type> struct StaticArray {
  Type *_data;
  std::size_t _size;

  explicit StaticArray(std::size_t n) : _data{nullptr}, _size{0} {
    _data = alloc_mem<Type>(n);
    _size = n;
  }

  ~StaticArray() {
    std::free(_data);
    _data = nullptr;
    _size = 0;
  }

  StaticArray() = delete;                               // default constructor
  StaticArray(const StaticArray &) = delete;            // copy constructor
  StaticArray(StaticArray &&) = delete;                 // move constructor
  StaticArray &operator=(const StaticArray &) = delete; // copy assignment
  StaticArray &operator=(StaticArray &&) = delete;      // move assignment

  [[nodiscard]] Type &operator[](std::size_t i) { return _data[i]; }
  [[nodiscard]] Type operator[](std::size_t i) const { return _data[i]; }

  [[nodiscard]] Type *begin() const { return _data; }
  [[nodiscard]] Type *end() const { return _data + _size; }
  [[nodiscard]] std::size_t size() const { return _size; }
  [[nodiscard]] Type *data() const { return _data; }

  Type get(std::size_t i) const { return _data[i]; }

  void set(std::size_t i, Type x) { _data[i] = x; }
  void set_plus(std::size_t i, Type x) { _data[i] += x; }
  void set_minus(std::size_t i, Type x) { _data[i] -= x; }
  void set_times(std::size_t i, Type x) { _data[i] *= x; }
  void set_div(std::size_t i, Type x) { _data[i] /= x; }
};

template <typename Type> struct Range {
  Type _start;
  Type _stop;

  struct iterator {
    Type cur;

    explicit iterator(Type c = 0) : cur{c} {}
    iterator &operator++() {
      ++cur;
      return *this;
    }
    bool operator==(iterator rhs) const { return cur == rhs.cur; }
    bool operator!=(iterator rhs) const { return cur != rhs.cur; }
    Type operator*() const { return cur; }
    Type operator-(iterator rhs) const { return cur - rhs.cur; }
    iterator &operator+=(Type rhs) {
      cur += rhs;
      return *this;
    }
  };

  Range() : _start{0}, _stop{0} {}
  Range(Type a, Type b) : _start{a}, _stop{b} {}

  Range(const Range &) = default;            // copy constructor
  Range(Range &&) = default;                 // move constructor
  Range &operator=(const Range &) = default; // copy assignment
  Range &operator=(Range &&) = default;      // move assignment

  [[nodiscard]] iterator begin() const { return iterator(_start); }
  [[nodiscard]] iterator end() const { return iterator(_stop); }
  [[nodiscard]] std::size_t size() const { return _stop - _start; }
};

template <typename Type>
Type get_attribute(const H5::H5File &file, const char *attr_name) {
  Type ret;

  H5::Attribute attr = file.openAttribute(attr_name);
  attr.read(attr.getDataType(), &ret);
  attr.close();

  return ret;
}

template <typename Type>
void read_dataset(const H5::H5File &file, const char *dataset_name,
                  const H5::PredType &h5_type, const std::size_t count,
                  Type *const out) {

  H5::DataSet dataset = file.openDataSet(dataset_name);

  // Ensure datatype is correct
  H5::DataType dataset_type = dataset.getDataType();
  assert(dataset_type == h5_type);

  // Ensure count is correct
  H5::DataSpace dataspace = dataset.getSpace();
  const auto ndims = dataspace.getSimpleExtentNdims();
  std::vector<hsize_t> dims(ndims);
  dataspace.getSimpleExtentDims(&dims[0]);
  assert(dims.size() == 1);
  assert(dims[0] == count);

  dataset.read(out, h5_type);

  dataspace.close();
  dataset_type.close();
  dataset.close();
}

#endif // __SIM_UTILS_H__
