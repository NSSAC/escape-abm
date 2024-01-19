#ifndef __SIM_UTILS_H__
#define __SIM_UTILS_H__
// Simulation utilities

#include <H5Cpp.h>
#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <vector>

const std::size_t L1_CACHE_SIZE = 64;

template <typename IntType> IntType aligned_size(IntType n) {
  auto rem = n % L1_CACHE_SIZE;
  if (rem) {
    n += L1_CACHE_SIZE - rem;
  }
  return n;
}

static std::size_t TOTAL_ALLOC = 0;

template <typename Type> Type *alloc_mem(const std::size_t n) {
  auto alloc_size = sizeof(Type) * n;
  alloc_size = aligned_size(alloc_size);
  auto *ret = std::aligned_alloc(L1_CACHE_SIZE, alloc_size);
  assert(ret && "Failed to allocate memory");

  TOTAL_ALLOC += alloc_size;

  return static_cast<Type *>(ret);
}

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

  void init(const std::size_t start, const std::size_t end) {
    for (std::size_t i = start; i < end; i++) {
      _data[i] = 0;
    }
  }

  template <typename Type1>
  void init(const Range<Type1> &range) {
    const auto [start, stop] = range;
    init(start, stop);
  }

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

  void load(const H5::H5File &file, const char *dataset_name,
            const H5::PredType &h5_type) {
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
    assert(dims[0] == _size);

    dataset.read(_data, h5_type);

    dataspace.close();
    dataset_type.close();
    dataset.close();
  }

  void save(const H5::H5File &file, const std::string &dataset_name,
            const H5::PredType &h5_type) {
    hsize_t dims[1] = {_size};
    H5::DataSpace dataspace(1, dims);

    H5::DataSet dataset = file.createDataSet(dataset_name, h5_type, dataspace);
    dataset.write(_data, h5_type);

    dataset.close();
    dataspace.close();
  }
};

template <typename Type> struct DynamicArrayList {
  Type **_data;
  std::size_t *_size;
  std::size_t *_cap;
  std::size_t _n;

  DynamicArrayList(const StaticArray<std::size_t> &caps)
      : _data{nullptr}, _size{nullptr}, _cap{nullptr}, _n{0} {
    const auto n = caps.size();

    _data = alloc_mem<Type *>(n);
    for (std::size_t i = 0; i < n; i++) {
      _data[i] = alloc_mem<Type>(caps[i]);
    }

    _size = alloc_mem<std::size_t>(n);
    for (std::size_t i = 0; i < n; i++) {
      _size[i] = 0;
    }

    _cap = alloc_mem<std::size_t>(n);
    for (std::size_t i = 0; i < n; i++) {
      _cap[i] = caps[i];
    }

    _n = n;
  }

  ~DynamicArrayList() {
    for (std::size_t i = 0; i < _n; i++) {
      std::free(_data[i]);
      _data[i] = nullptr;
    }
    std::free(_data);
    _data = nullptr;

    std::free(_size);
    _size = nullptr;

    std::free(_cap);
    _cap = nullptr;

    _n = 0;
  }

  DynamicArrayList() = delete;                         // default constructor
  DynamicArrayList(const DynamicArrayList &) = delete; // copy constructor
  DynamicArrayList(DynamicArrayList &&) = delete;      // move constructor
  DynamicArrayList &
  operator=(const DynamicArrayList &) = delete;              // copy assignment
  DynamicArrayList &operator=(DynamicArrayList &&) = delete; // move assignment

  [[nodiscard]] Type *begin(std::size_t i) const { return _data[i]; }
  [[nodiscard]] Type *end(std::size_t i) const { return _data[i] + _size[i]; }
  [[nodiscard]] std::size_t size(std::size_t i) const { return _size[i]; }
  [[nodiscard]] Type *data(std::size_t i) const { return _data[i]; }

  void init(std::size_t i) {
    const auto a = _data[i];
    const auto n = _cap[i];
    for (std::size_t j = 0; j < n; j++) {
      a[j] = 0;
    }
  }

  void append(std::size_t i, const Type &v) {
    _data[i][_size[i]] = v;
    _size[i]++;
  }

  void clear(std::size_t i) { _size[i] = 0; }

  void save(const H5::H5File &file, const std::string &dataset_name,
            const H5::PredType &h5_type) {
    std::size_t dataset_size = 0;
    for (std::size_t i = 0; i < _n; i++) {
      dataset_size += _size[i];
    }

    hsize_t dims[] = {dataset_size};
    H5::DataSpace file_space(1, dims);

    H5::DataSet dataset = file.createDataSet(dataset_name, h5_type, file_space);

    std::size_t write_offset = 0;
    for (std::size_t i = 0; i < _n; i++) {
      hsize_t dims[] = {_size[i]};
      H5::DataSpace mem_space(1, dims);

      hsize_t count[] = {_size[i]};
      hsize_t offset[] = {write_offset};
      file_space.selectHyperslab(H5S_SELECT_SET, count, offset);

      dataset.write(_data[i], h5_type, mem_space, file_space);

      write_offset += count[0];
      mem_space.close();
    }

    dataset.close();
    file_space.close();
  }
};

template <typename Type>
Type get_attribute(const H5::H5File &file, const char *attr_name) {
  Type ret;

  H5::Attribute attr = file.openAttribute(attr_name);
  attr.read(attr.getDataType(), &ret);
  attr.close();

  return ret;
}

static void create_group(H5::H5File &output_file,
                         const std::string &group_name) {
  H5::Group group = output_file.createGroup(group_name);
  group.close();
}

#endif // __SIM_UTILS_H__
