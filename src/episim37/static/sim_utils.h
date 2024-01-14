#ifndef __SIM_UTILS_H__
#define __SIM_UTILS_H__
// Simulation utilities

#include <H5Cpp.h>
#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <vector>

static std::size_t TOTAL_ALLOC = 0;

template <typename T> T *alloc_mem(const std::size_t n) {
  const auto alloc_size = sizeof(T) * n;
  TOTAL_ALLOC += alloc_size;
  T *ret = static_cast<T *>(std::malloc(alloc_size));
  assert(ret != nullptr);

  return ret;
}

template <typename T> void free_mem(T **p) {
  if (p == nullptr || *p == nullptr) {
    return;
  }
  std::free(*p);
  *p = nullptr;
}

template <typename T>
T get_attribute(const H5::H5File &file, const char *attr_name) {
  T ret;

  H5::Attribute attr = file.openAttribute(attr_name);
  attr.read(attr.getDataType(), &ret);
  attr.close();

  return ret;
}

template <typename T>
void read_dataset(const H5::H5File &file, const char *dataset_name,
                  const H5::PredType &h5_type, const std::size_t count,
                  T *const out) {

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
