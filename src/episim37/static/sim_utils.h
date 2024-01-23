#ifndef __SIM_UTILS_H__
#define __SIM_UTILS_H__
// Simulation utilities

#include <H5Cpp.h>
#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <omp.h>
#include <random>
#include <vector>

// ----------------------------------------------------------------------------
// Useful constants
// ----------------------------------------------------------------------------

const int L1_CACHE_SIZE = 64;
const int CHUNK_SIZE = 128;

// ----------------------------------------------------------------------------
// THREAD_IDX and NUM_THREADS
// ----------------------------------------------------------------------------

static int THREAD_IDX = 0;
#pragma omp threadprivate(THREAD_IDX)

static int NUM_THREADS = 0;

static void init_thread_idx() {
    NUM_THREADS = omp_get_max_threads();

#pragma omp parallel
    { THREAD_IDX = omp_get_thread_num(); }
}

// ----------------------------------------------------------------------------
// Memory allocation utils
// ----------------------------------------------------------------------------

template <typename IntType> IntType aligned_size(IntType n) {
    auto rem = n % L1_CACHE_SIZE;
    if (rem) {
        n += L1_CACHE_SIZE - rem;
    }
    return n;
}

static std::size_t TOTAL_ALLOC = 0;

template <typename Type> Type* alloc_mem(const std::size_t n) {
    auto alloc_size = sizeof(Type) * n;
    alloc_size = aligned_size(alloc_size);
    auto* ret = std::aligned_alloc(L1_CACHE_SIZE, alloc_size);
    assert(ret && "Failed to allocate memory");

    TOTAL_ALLOC += alloc_size;

    return static_cast<Type*>(ret);
}

// ----------------------------------------------------------------------------
// OMP + HDF5 aware data structures
// ----------------------------------------------------------------------------

template <typename Type> struct StaticArray {
    Type* _data;
    std::size_t _size;

    explicit StaticArray(std::size_t size)
        : _data{nullptr}
        , _size{0} {
        _data = alloc_mem<Type>(size);
        _size = size;
    }

    ~StaticArray() {
        std::free(_data);
        _data = nullptr;
        _size = 0;
    }

    StaticArray() = delete;                              // default constructor
    StaticArray(const StaticArray&) = delete;            // copy constructor
    StaticArray(StaticArray&&) = delete;                 // move constructor
    StaticArray& operator=(const StaticArray&) = delete; // copy assignment
    StaticArray& operator=(StaticArray&&) = delete;      // move assignment

    void range_init(std::size_t start, std::size_t stop) {
        for (std::size_t i = start; i < stop; i++) {
            _data[i] = 0;
        }
    }

    [[nodiscard]] Type& operator[](std::size_t i) { return _data[i]; }
    [[nodiscard]] Type operator[](std::size_t i) const { return _data[i]; }

    [[nodiscard]] Type* begin() const { return _data; }
    [[nodiscard]] Type* end() const { return _data + _size; }
    [[nodiscard]] std::size_t size() const { return _size; }
    [[nodiscard]] Type* data() const { return _data; }

    Type get(std::size_t i) const { return _data[i]; }

    void set(std::size_t i, Type x) { _data[i] = x; }
    void set_plus(std::size_t i, Type x) { _data[i] += x; }
    void set_minus(std::size_t i, Type x) { _data[i] -= x; }
    void set_times(std::size_t i, Type x) { _data[i] *= x; }
    void set_div(std::size_t i, Type x) { _data[i] /= x; }

    void load(const H5::H5File& file, const char* dataset_name, const H5::PredType& h5_type) {
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

    void save(const H5::H5File& file, const std::string& dataset_name, const H5::PredType& h5_type) {
        hsize_t dims[1] = {_size};
        H5::DataSpace dataspace(1, dims);

        H5::DataSet dataset = file.createDataSet(dataset_name, h5_type, dataspace);
        dataset.write(_data, h5_type);

        dataset.close();
        dataspace.close();
    }
};

template <typename Type> struct PerThreadDynamicArray {
    Type** _data;
    std::size_t* _size; // size of each array
    std::size_t* _cap;  // capacity of each array

    template <typename SizeType>
    PerThreadDynamicArray(const StaticArray<SizeType>& caps)
        : _data{nullptr}
        , _size{nullptr}
        , _cap{nullptr} {

        assert(caps.size() == NUM_THREADS);

        _data = alloc_mem<Type*>(NUM_THREADS);
        for (std::size_t i = 0; i < NUM_THREADS; i++) {
            _data[i] = alloc_mem<Type>(caps[i]);
        }

        _size = alloc_mem<std::size_t>(NUM_THREADS);
        for (std::size_t i = 0; i < NUM_THREADS; i++) {
            _size[i] = 0;
        }

        _cap = alloc_mem<std::size_t>(NUM_THREADS);
        for (std::size_t i = 0; i < NUM_THREADS; i++) {
            _cap[i] = caps[i];
        }
    }

    ~PerThreadDynamicArray() {
        for (std::size_t i = 0; i < NUM_THREADS; i++) {
            std::free(_data[i]);
            _data[i] = nullptr;
        }
        std::free(_data);
        _data = nullptr;

        std::free(_size);
        _size = nullptr;

        std::free(_cap);
        _cap = nullptr;
    }

    PerThreadDynamicArray() = delete;                                        // default constructor
    PerThreadDynamicArray(const PerThreadDynamicArray&) = delete;            // copy constructor
    PerThreadDynamicArray(PerThreadDynamicArray&&) = delete;                 // move constructor
    PerThreadDynamicArray& operator=(const PerThreadDynamicArray&) = delete; // copy assignment
    PerThreadDynamicArray& operator=(PerThreadDynamicArray&&) = delete;      // move assignment

    [[nodiscard]] Type* begin(std::size_t i) const { return _data[i]; }
    [[nodiscard]] Type* end(std::size_t i) const { return _data[i] + _size[i]; }
    [[nodiscard]] std::size_t size(std::size_t i) const { return _size[i]; }
    [[nodiscard]] Type* data(std::size_t i) const { return _data[i]; }

    void par_init() {
#pragma omp parallel
        {
            const auto a = _data[THREAD_IDX];
            const auto cap = _cap[THREAD_IDX];
            for (std::size_t j = 0; j < cap; j++) {
                a[j] = 0;
            }
        }
    }

    void thread_append(const Type& v) {
        auto i = _size[THREAD_IDX];
        auto a = _data[THREAD_IDX];
        a[i] = v;
        _size[THREAD_IDX] = i + 1;
    }

    void thread_clear() { _size[THREAD_IDX] = 0; }

    void save(const H5::H5File& file, const std::string& dataset_name, const H5::PredType& h5_type) {
        std::size_t dataset_size = 0;
        for (std::size_t i = 0; i < NUM_THREADS; i++) {
            dataset_size += _size[i];
        }

        hsize_t dims[] = {dataset_size};
        H5::DataSpace file_space(1, dims);

        H5::DataSet dataset = file.createDataSet(dataset_name, h5_type, file_space);

        std::size_t write_offset = 0;
        for (std::size_t i = 0; i < NUM_THREADS; i++) {
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

// ----------------------------------------------------------------------------
// Per thread random number generator
// ----------------------------------------------------------------------------

typedef std::mt19937 rnd_state_type;

static rnd_state_type* THREAD_RND_STATE = nullptr;
#pragma omp threadprivate(THREAD_RND_STATE)

struct ThreadRndState {
    ThreadRndState() {
        // get the default random device
        std::random_device rd;

        // make seed generator using the random device
        std::seed_seq seq{rd(), rd(), rd(), rd(), rd()};

        // make seeds for each thread
        StaticArray<std::uint32_t> seeds(NUM_THREADS);
        seq.generate(seeds.begin(), seeds.end());

#pragma omp parallel
        {
            // allocate a generator for each thread
#pragma omp critical
            THREAD_RND_STATE = alloc_mem<rnd_state_type>(1);

            // assign the seeds to each thread
            THREAD_RND_STATE->seed(seeds[THREAD_IDX]);
        }
    }

    ~ThreadRndState() {
#pragma omp parallel
        {
#pragma omp critical
            std::free(THREAD_RND_STATE);

            THREAD_RND_STATE = nullptr;
        }
    }
};

// ----------------------------------------------------------------------------
// HDF5 helpers
// ----------------------------------------------------------------------------

template <typename Type> Type get_attribute(const H5::H5File& file, const char* attr_name) {
    Type ret;

    H5::Attribute attr = file.openAttribute(attr_name);
    attr.read(attr.getDataType(), &ret);
    attr.close();

    return ret;
}

static void create_group(H5::H5File& output_file, const std::string& group_name) {
    H5::Group group = output_file.createGroup(group_name);
    group.close();
}

#endif // __SIM_UTILS_H__
