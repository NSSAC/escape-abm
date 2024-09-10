#ifndef __SIMULATION_COMMON_OPENMP_H__
#define __SIMULATION_COMMON_OPENMP_H__
// Common code for simulations
//
// Conventions
// init_* methods / functions use their own parallel section
// thread_* and par_* methods / functions should be called from within parallel section

#include <H5Cpp.h>
#include <H5public.h>
#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <fmt/core.h>
#include <omp.h>
#include <random>
#include <vector>

// ----------------------------------------------------------------------------
// Common types
// ----------------------------------------------------------------------------

typedef int64_t int_type;
typedef uint64_t uint_type;
typedef float float_type;
typedef uint8_t bool_type;
typedef int64_t size_type;
typedef int32_t node_index_type;
typedef int64_t edge_index_type;

// ----------------------------------------------------------------------------
// HDF5 Pred Type from common types
// ----------------------------------------------------------------------------

template <typename Type> const H5::PredType& h5_type() = delete;

template <> inline const H5::PredType& h5_type<int8_t>() { return H5::PredType::NATIVE_INT8; }
template <> inline const H5::PredType& h5_type<int16_t>() { return H5::PredType::NATIVE_INT16; }
template <> inline const H5::PredType& h5_type<int32_t>() { return H5::PredType::NATIVE_INT32; }
template <> inline const H5::PredType& h5_type<int64_t>() { return H5::PredType::NATIVE_INT64; }

template <> inline const H5::PredType& h5_type<uint8_t>() { return H5::PredType::NATIVE_UINT8; }
template <> inline const H5::PredType& h5_type<uint16_t>() { return H5::PredType::NATIVE_UINT16; }
template <> inline const H5::PredType& h5_type<uint32_t>() { return H5::PredType::NATIVE_UINT32; }
template <> inline const H5::PredType& h5_type<uint64_t>() { return H5::PredType::NATIVE_UINT64; }

template <> inline const H5::PredType& h5_type<float>() { return H5::PredType::NATIVE_FLOAT; }
template <> inline const H5::PredType& h5_type<double>() { return H5::PredType::NATIVE_DOUBLE; }

// ----------------------------------------------------------------------------
// Parse Cstring
// ----------------------------------------------------------------------------

template <typename Type> Type parse_cstr(const char* s) = delete;

template <> inline int8_t parse_cstr<int8_t>(const char* s) { return std::stol(s); }
template <> inline int16_t parse_cstr<int16_t>(const char* s) { return std::stol(s); }
template <> inline int32_t parse_cstr<int32_t>(const char* s) { return std::stol(s); }
template <> inline int64_t parse_cstr<int64_t>(const char* s) { return std::stol(s); }

template <> inline uint8_t parse_cstr<uint8_t>(const char* s) { return std::stoul(s); }
template <> inline uint16_t parse_cstr<uint16_t>(const char* s) { return std::stoul(s); }
template <> inline uint32_t parse_cstr<uint32_t>(const char* s) { return std::stoul(s); }
template <> inline uint64_t parse_cstr<uint64_t>(const char* s) { return std::stoul(s); }

template <> inline float parse_cstr<float>(const char* s) { return std::stod(s); }
template <> inline double parse_cstr<double>(const char* s) { return std::stod(s); }

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

static size_type TOTAL_ALLOC = 0;

template <typename Type> Type* alloc_mem(const size_type n) {
    auto alloc_size = sizeof(Type) * n;
    alloc_size = aligned_size(alloc_size);
    auto* ret = std::aligned_alloc(L1_CACHE_SIZE, alloc_size);
    assert(ret);

    TOTAL_ALLOC += alloc_size;

    return static_cast<Type*>(ret);
}

// ----------------------------------------------------------------------------
// OMP + HDF5 aware data structures
// ----------------------------------------------------------------------------

template <typename Type> struct StaticArray {
    Type* _data;
    size_type _size;

    explicit StaticArray(size_type size)
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

    void range_init(size_type start, size_type stop) {
        for (size_type i = start; i < stop; i++) {
            _data[i] = 0;
        }
    }

    StaticArray() = delete;                              // default constructor
    StaticArray(const StaticArray&) = delete;            // copy constructor
    StaticArray(StaticArray&&) = delete;                 // move constructor
    StaticArray& operator=(const StaticArray&) = delete; // copy assignment
    StaticArray& operator=(StaticArray&&) = delete;      // move assignment

    [[nodiscard]] Type& operator[](size_type i) { return _data[i]; }
    [[nodiscard]] Type operator[](size_type i) const { return _data[i]; }

    [[nodiscard]] Type* begin() const { return _data; }
    [[nodiscard]] Type* end() const { return _data + _size; }
    [[nodiscard]] size_type size() const { return _size; }
    [[nodiscard]] Type* data() const { return _data; }

    Type get(size_type i) const { return _data[i]; }

    void set(size_type i, Type x) { _data[i] = x; }
    void set_plus(size_type i, Type x) { _data[i] += x; }
    void set_minus(size_type i, Type x) { _data[i] -= x; }
    void set_times(size_type i, Type x) { _data[i] *= x; }
    void set_div(size_type i, Type x) { _data[i] /= x; }

    void load(const H5::H5File& file, const char* dataset_name) {
        H5::DataSet dataset = file.openDataSet(dataset_name);

        // Ensure datatype is correct
        H5::DataType dataset_type = dataset.getDataType();
        assert(dataset_type == h5_type<Type>());

        // Ensure count is correct
        H5::DataSpace dataspace = dataset.getSpace();
        const auto ndims = dataspace.getSimpleExtentNdims();
        std::vector<hsize_t> dims(ndims);
        dataspace.getSimpleExtentDims(&dims[0]);
        assert(dims.size() == 1);
        assert(static_cast<size_type>(dims[0]) == _size);

        dataset.read(_data, h5_type<Type>());

        dataspace.close();
        dataset_type.close();
        dataset.close();
    }

    void save(const H5::H5File& file, const std::string& dataset_name) {
        hsize_t dims[1] = {static_cast<hsize_t>(_size)};
        H5::DataSpace dataspace(1, dims);
        H5::DSetCreatPropList prop;
        if (_size > 0) {
            prop.setChunk(1, dims);
            prop.setDeflate(9);
        }

        H5::DataSet dataset = file.createDataSet(dataset_name, h5_type<Type>(), dataspace, prop);
        dataset.write(_data, h5_type);

        dataset.close();
        prop.close();
        dataspace.close();
    }
};

template <typename Type> struct PerThreadDynamicArray {
    Type** _data;
    size_type* _size; // size of each array
    size_type* _cap;  // capacity of each array

    template <typename SizeType>
    PerThreadDynamicArray(const StaticArray<SizeType>& caps)
        : _data{nullptr}
        , _size{nullptr}
        , _cap{nullptr} {

        assert(caps.size() == NUM_THREADS);

        _data = alloc_mem<Type*>(NUM_THREADS);
        for (size_type i = 0; i < NUM_THREADS; i++) {
            _data[i] = alloc_mem<Type>(caps[i]);
        }

        _size = alloc_mem<size_type>(NUM_THREADS);
        for (size_type i = 0; i < NUM_THREADS; i++) {
            _size[i] = 0;
        }

        _cap = alloc_mem<size_type>(NUM_THREADS);
        for (size_type i = 0; i < NUM_THREADS; i++) {
            _cap[i] = caps[i];
        }
    }

    ~PerThreadDynamicArray() {
        for (size_type i = 0; i < NUM_THREADS; i++) {
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

    void par_init() {
        const auto a = _data[THREAD_IDX];
        const auto cap = _cap[THREAD_IDX];
        for (size_type j = 0; j < cap; j++) {
            a[j] = 0;
        }
    }

    PerThreadDynamicArray() = delete;                                        // default constructor
    PerThreadDynamicArray(const PerThreadDynamicArray&) = delete;            // copy constructor
    PerThreadDynamicArray(PerThreadDynamicArray&&) = delete;                 // move constructor
    PerThreadDynamicArray& operator=(const PerThreadDynamicArray&) = delete; // copy assignment
    PerThreadDynamicArray& operator=(PerThreadDynamicArray&&) = delete;      // move assignment

    [[nodiscard]] Type* begin(size_type i) const { return _data[i]; }
    [[nodiscard]] Type* end(size_type i) const { return _data[i] + _size[i]; }
    [[nodiscard]] size_type size(size_type i) const { return _size[i]; }
    [[nodiscard]] Type* data(size_type i) const { return _data[i]; }

    void thread_append(const Type& v) {
        auto i = _size[THREAD_IDX];
        auto a = _data[THREAD_IDX];
        a[i] = v;
        _size[THREAD_IDX] = i + 1;
    }

    void thread_clear() { _size[THREAD_IDX] = 0; }

    void save(const H5::H5File& file, const std::string& dataset_name) {
        size_type dataset_size = 0;
        for (size_type i = 0; i < NUM_THREADS; i++) {
            dataset_size += _size[i];
        }

        hsize_t dims[] = {static_cast<hsize_t>(dataset_size)};
        H5::DataSpace file_space(1, dims);
        H5::DSetCreatPropList prop;
        if (dataset_size > 0) {
            prop.setChunk(1, dims);
            prop.setDeflate(9);
        }

        H5::DataSet dataset = file.createDataSet(dataset_name, h5_type<Type>(), file_space, prop);

        size_type write_offset = 0;
        for (size_type i = 0; i < NUM_THREADS; i++) {
            hsize_t dims[] = {static_cast<hsize_t>(_size[i])};
            H5::DataSpace mem_space(1, dims);

            hsize_t count[] = {static_cast<hsize_t>(_size[i])};
            hsize_t offset[] = {static_cast<hsize_t>(write_offset)};
            file_space.selectHyperslab(H5S_SELECT_SET, count, offset);

            dataset.write(_data[i], h5_type<Type>(), mem_space, file_space);

            write_offset += count[0];
            mem_space.close();
        }

        dataset.close();
        prop.close();
        file_space.close();
    }
};

// ----------------------------------------------------------------------------
// Random number generation
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

static std::uniform_real_distribution<float_type> uniform01{0.0, 1.0};
static std::normal_distribution<float_type> normal01{0.0, 1.0};

// ----------------------------------------------------------------------------
// HDF5 helpers
// ----------------------------------------------------------------------------

template <typename Type> Type read_attribute(const H5::H5File& file, const char* attr_name) {
    Type ret;

    H5::Attribute attr = file.openAttribute(attr_name);
    attr.read(attr.getDataType(), &ret);
    attr.close();

    return ret;
}

template <typename Type> void write_attribute(const H5::H5File& file, const char* attr_name, Type value) {
    H5::DataSpace attr_dataspace(H5S_SCALAR);
    H5::Attribute attribute = file.createAttribute(attr_name, h5_type<Type>(), attr_dataspace);

    attribute.write(h5_type<Type>(), &value);

    attribute.close();
    attr_dataspace.close();
}

template <typename Type>
void write_attribute(const H5::Group& group, const char* attr_name, Type value) {
    H5::DataSpace attr_dataspace(H5S_SCALAR);
    H5::Attribute attribute = group.createAttribute(attr_name, h5_type<Type>(), attr_dataspace);

    attribute.write(h5_type<Type>(), &value);

    attribute.close();
    attr_dataspace.close();
}

static void create_group(H5::H5File& output_file, const std::string& group_name) {
    H5::Group group = output_file.createGroup(group_name);
    group.close();
}

// ----------------------------------------------------------------------------
// Global variables
// ----------------------------------------------------------------------------

static size_type NUM_NODES = 0;
static size_type NUM_EDGES = 0;

static int_type NUM_TICKS = 0;
static int_type CUR_TICK = -1;
static float_type TICK_ELAPSED = 1.0;

// ----------------------------------------------------------------------------
// Node and edge count getters
// ----------------------------------------------------------------------------

static size_type get_num_nodes(const H5::H5File& file) { return read_attribute<size_type>(file, "num_nodes"); }

static size_type get_num_edges(const H5::H5File& file) { return read_attribute<size_type>(file, "num_edges"); }

// ----------------------------------------------------------------------------
// Incidence network (target_node_index, edge_index)
// ----------------------------------------------------------------------------

struct IncomingIncidenceCSR {
    StaticArray<edge_index_type> indptr;

    IncomingIncidenceCSR(const H5::H5File& file)
        : indptr{NUM_NODES + 1} {
        indptr.load(file, "/incoming/incidence/csr/indptr");
    }
};

static IncomingIncidenceCSR* IN_INC_CSR = nullptr;

// ----------------------------------------------------------------------------
// Thread to node range mapping
// ----------------------------------------------------------------------------

static size_type TN_START = 0; // thread node range start
static size_type TN_END = 0;   // thread node range end
#pragma omp threadprivate(TN_START)
#pragma omp threadprivate(TN_END)

static void init_thread_node_range() {
    StaticArray<size_type> ptr(NUM_THREADS + 1);

    auto thread_load = NUM_NODES / NUM_THREADS;
    thread_load += (NUM_NODES % NUM_THREADS != 0);

    ptr[0] = 0;
#pragma omp parallel
    {
        auto end = thread_load * (THREAD_IDX + 1);
        end = aligned_size(end);
        if (end > NUM_NODES) {
            end = NUM_NODES;
        }
        ptr[THREAD_IDX + 1] = end;
    }

    assert(ptr[0] == 0);
    assert(ptr[NUM_THREADS] == NUM_NODES);
    for (size_type i = 0; i < NUM_THREADS; i++) {
        assert(ptr[i] <= ptr[i + 1]);
    }

    // for (size_type i = 0; i < NUM_THREADS; i++) {
    //    std::cout << "thread node count: " << i << " " << ptr[i+1] - ptr[i] << "\n";
    // }

#pragma omp parallel
    {
        TN_START = ptr[THREAD_IDX];
        TN_END = ptr[THREAD_IDX + 1];
    }
}

// ----------------------------------------------------------------------------
// Thread to edge range mapping
// ----------------------------------------------------------------------------

static size_type TE_START = 0; // thread edge range start
static size_type TE_END = 0;   // thread edge range end
#pragma omp threadprivate(TE_START)
#pragma omp threadprivate(TE_END)

static void init_thread_edge_range() {
    StaticArray<size_type> ptr(NUM_THREADS + 1);

    auto thread_load = NUM_EDGES / NUM_THREADS;
    thread_load += (NUM_EDGES % NUM_THREADS != 0);

    ptr[0] = 0;
#pragma omp parallel
    {
        auto end = thread_load * (THREAD_IDX + 1);
        end = aligned_size(end);
        if (end > NUM_EDGES) {
            end = NUM_EDGES;
        }
        ptr[THREAD_IDX + 1] = end;
    }

    assert(ptr[0] == 0);
    assert(ptr[NUM_THREADS] == NUM_EDGES);
    for (size_type i = 0; i < NUM_THREADS; i++) {
        assert(ptr[i] <= ptr[i + 1]);
    }

    // for (size_type i = 0; i < NUM_THREADS; i++) {
    //    std::cout << "thread edge count: " << i << " " << ptr[i+1] - ptr[i] << "\n";
    // }

#pragma omp parallel
    {
        TE_START = ptr[THREAD_IDX];
        TE_END = ptr[THREAD_IDX + 1];
    }
}

// ----------------------------------------------------------------------------
// Node or Edge Set
// ----------------------------------------------------------------------------

struct Set {
    bool_type is_node_set;
    StaticArray<bool_type> is_in;
    StaticArray<size_type> thread_k;    // for absolute sampling
    StaticArray<size_type> thread_size; // size per thread
    size_type size;                     // total size

    explicit Set(bool_type is_node_set_)
        : is_node_set{is_node_set_}
        , is_in{is_node_set ? NUM_NODES : NUM_EDGES}
        , thread_k{size_type(NUM_THREADS)}
        , thread_size{size_type(NUM_THREADS)}
        , size{0} {}

    void par_init() {
#pragma omp master
        {
            thread_k.range_init(0, NUM_THREADS);
            thread_size.range_init(0, NUM_THREADS);
        }

        is_in.range_init(thread_start(), thread_end());
    }

    size_type thread_start() { return is_node_set ? TN_START : TE_START; }
    size_type thread_end() { return is_node_set ? TN_END : TE_END; }

    void update_size() {
        size = 0;
        for (int tid = 0; tid < NUM_THREADS; tid++) {
            size += thread_size[tid];
        }
    }

    size_type max_size() { return is_node_set ? NUM_NODES : NUM_EDGES; }
    size_type thread_max_size() { return is_node_set ? (TN_END - TN_START) : (TE_END - TE_START); }
};

const static Set* ALL_NODES = static_cast<Set*>((void*)1);
const static Set* ALL_EDGES = static_cast<Set*>((void*)2);

static size_type len(const Set* set) {
    if (set == ALL_NODES) {
        return NUM_NODES;
    } else if (set == ALL_EDGES) {
        return NUM_EDGES;
    } else {
        return set->size;
    }
}

static void par_sample_rel(Set* set, float_type prob) {
    size_type size = 0;
    for (size_type x = set->thread_start(); x < set->thread_end(); x++) {
        set->is_in[x] = 0;
        if (uniform01(*THREAD_RND_STATE) < prob) {
            set->is_in[x] = 1;
            size += 1;
        }
    }

    set->thread_size[THREAD_IDX] = size;

#pragma omp barrier

#pragma omp master
    set->update_size();
}

static void par_sample_rel(Set* set, Set* parent, float_type prob) {
    size_type size = 0;
    for (size_type x = set->thread_start(); x < set->thread_end(); x++) {
        set->is_in[x] = 0;
        if (parent->is_in[x]) {
            if (uniform01(*THREAD_RND_STATE) < prob) {
                set->is_in[x] = 1;
                size += 1;
            }
        }
    }

    set->thread_size[THREAD_IDX] = size;

#pragma omp barrier

#pragma omp master
    set->update_size();
}

static void allocate_k(Set* set, size_type total_k) {
    if (total_k > set->max_size()) {
        total_k = set->max_size();
    }

    size_type remaining_n = set->max_size();
    size_type remaining_k = total_k;
    for (int tid = 0; tid < NUM_THREADS; tid++) {
        float_type p = float_type(remaining_k) / remaining_n;
        size_type k_max = set->thread_max_size();
        remaining_n -= k_max;

        if (k_max > remaining_k) {
            k_max = remaining_k;
        }

        size_type k_min = 0;
        if (remaining_n < remaining_k) {
            k_min = remaining_k - remaining_n;
        }
        assert(k_min <= k_max);

        size_type samp_space = k_max - k_min;
        std::binomial_distribution<size_type> sampler(samp_space, p);
        size_type tid_k = sampler(*THREAD_RND_STATE) + k_min;

        set->thread_k[tid] = tid_k;
        remaining_k -= tid_k;
    }

    size_type assigned_k = 0;
    for (int tid = 0; tid < NUM_THREADS; tid++) {
        assigned_k += set->thread_k[tid];
        assert(set->thread_k[tid] <= set->thread_max_size());
    }
    assert(assigned_k == total_k);
}

static void par_sample_abs(Set* set, size_type total_k) {
#pragma omp master
    allocate_k(set, total_k);

#pragma omp barrier

    size_type size = 0;
    size_type remaining_n = set->thread_max_size();
    size_type remaining_k = set->thread_k[THREAD_IDX];
    assert(remaining_k <= remaining_n);
    for (size_type x = set->thread_start(); x < set->thread_end(); x++) {
        set->is_in[x] = 0;
        float_type p = float_type(remaining_k) / remaining_n;
        if (uniform01(*THREAD_RND_STATE) < p) {
            set->is_in[x] = 1;
            size += 1;
            remaining_k -= 1;
        }
        remaining_n -= 1;
    }

    assert(remaining_k == 0);
    assert(remaining_n == 0);
    assert(size == set->thread_k[THREAD_IDX]);

    set->thread_size[THREAD_IDX] = size;

#pragma omp barrier

#pragma omp master
    set->update_size();
}

static void allocate_k(Set* set, Set* parent, size_type total_k) {
    if (total_k > parent->size) {
        total_k = parent->size;
    }

    size_type remaining_n = parent->size;
    size_type remaining_k = total_k;
    for (int tid = 0; tid < NUM_THREADS; tid++) {
        float_type p = float_type(remaining_k) / remaining_n;
        size_type k_max = parent->thread_size[tid];
        remaining_n -= k_max;

        if (k_max > remaining_k) {
            k_max = remaining_k;
        }

        size_type k_min = 0;
        if (remaining_n < remaining_k) {
            k_min = remaining_k - remaining_n;
        }
        assert(k_min <= k_max);

        size_type samp_space = k_max - k_min;
        std::binomial_distribution<size_type> sampler(samp_space, p);
        size_type tid_k = sampler(*THREAD_RND_STATE) + k_min;

        set->thread_k[tid] = tid_k;
        remaining_k -= tid_k;
    }

    size_type assigned_k = 0;
    for (int tid = 0; tid < NUM_THREADS; tid++) {
        assigned_k += set->thread_k[tid];
        assert(set->thread_k[tid] <= parent->thread_size[tid]);
    }
    assert(assigned_k == total_k);
}

static void par_sample_abs(Set* set, Set* parent, size_type total_k) {
#pragma omp master
    allocate_k(set, parent, total_k);

#pragma omp barrier

    size_type size = 0;
    size_type remaining_n = parent->thread_size[THREAD_IDX];
    size_type remaining_k = set->thread_k[THREAD_IDX];
    assert(remaining_k <= remaining_n);
    for (size_type x = set->thread_start(); x < set->thread_end(); x++) {
        set->is_in[x] = 0;
        if (parent->is_in[x]) {
            float_type p = float_type(remaining_k) / remaining_n;
            if (uniform01(*THREAD_RND_STATE) < p) {
                set->is_in[x] = 1;
                size += 1;
                remaining_k -= 1;
            }
            remaining_n -= 1;
        }
    }

    assert(remaining_k == 0);
    assert(remaining_n == 0);
    assert(size == set->thread_k[THREAD_IDX]);

    set->thread_size[THREAD_IDX] = size;

#pragma omp barrier

#pragma omp master
    set->update_size();
}

// ----------------------------------------------------------------------------
// Contagion output container
// ----------------------------------------------------------------------------

static H5::H5File* OUTPUT_FILE = nullptr;

template <typename StateType> struct ContagionOutputContainer {
    std::string name;
    PerThreadDynamicArray<node_index_type> transition_node_idx;
    PerThreadDynamicArray<StateType> transition_state;
    PerThreadDynamicArray<edge_index_type> transmission_edge_idx;
    PerThreadDynamicArray<StateType> transmission_state;

    ContagionOutputContainer(const std::string n, const StaticArray<size_type>& thread_node_count)
        : name{n}
        , transition_node_idx{thread_node_count}
        , transition_state{thread_node_count}
        , transmission_edge_idx{thread_node_count}
        , transmission_state{thread_node_count} {}

    void par_init() {
        transition_node_idx.par_init();
        transition_state.par_init();
        transmission_edge_idx.par_init();
        transmission_state.par_init();
    }

    void reset_transitions() {
        transition_node_idx.thread_clear();
        transition_state.thread_clear();
    }

    void add_transition(const size_type v, const StateType state) {
        transition_node_idx.thread_append(v);
        transition_state.thread_append(state);
    }

    void do_save_transitions(const std::string& transition_type) {
        auto group_name = fmt::format("/{}/{}/tick_{}", name, transition_type, CUR_TICK);
        create_group(*OUTPUT_FILE, group_name);

        {
            auto dataset_name = fmt::format("/{}/{}/tick_{}/node_index", name, transition_type, CUR_TICK);
            transition_node_idx.save(*OUTPUT_FILE, dataset_name);
        }

        {
            auto dataset_name = fmt::format("/{}/{}/tick_{}/state", name, transition_type, CUR_TICK);
            transition_state.save(*OUTPUT_FILE, dataset_name);
        }
    }

    void save_interventions() { do_save_transitions("interventions"); }

    void save_transitions() { do_save_transitions("transitions"); }

    void reset_transmissions() {
        transmission_edge_idx.thread_clear();
        transmission_state.thread_clear();
    }

    void add_transmission(const size_type e, const StateType state) {
        transmission_edge_idx.thread_append(e);
        transmission_state.thread_append(state);
    }

    void save_transmissions() {
        auto group_name = fmt::format("/{}/transmissions/tick_{}", name, CUR_TICK);
        create_group(*OUTPUT_FILE, group_name);

        {
            auto dataset_name = fmt::format("/{}/transmissions/tick_{}/edge_index", name, CUR_TICK);
            transmission_edge_idx.save(*OUTPUT_FILE, dataset_name);
        }

        {
            auto dataset_name = fmt::format("/{}/transmissions/tick_{}/state", name, CUR_TICK);
            transmission_state.save(*OUTPUT_FILE, dataset_name);
        }
    }

    void create_output_groups() {
        auto group_name = fmt::format("/{}", name);
        create_group(*OUTPUT_FILE, group_name);

        group_name = fmt::format("/{}/interventions", name);
        create_group(*OUTPUT_FILE, group_name);

        group_name = fmt::format("/{}/transitions", name);
        create_group(*OUTPUT_FILE, group_name);

        group_name = fmt::format("/{}/transmissions", name);
        create_group(*OUTPUT_FILE, group_name);

        group_name = fmt::format("/{}/state_count", name);
        create_group(*OUTPUT_FILE, group_name);
    }
};

#endif // __SIMULATION_COMMON_OPENMP_H__
