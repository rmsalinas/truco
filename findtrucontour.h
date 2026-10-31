#pragma once
// TRUCO - race-free version, with optional contour hierarchy.
//
//   cv::findTRUContours(img, contours [, minSize, binarize])                 no hierarchy (any order)
//   cv::findTRUContours(img, contours, hierarchy [, mode, minSize, binarize]) RETR_LIST / RETR_CCOMP /
//       RETR_TREE: same contours, same order and same hierarchy as cv::findContours (Suzuki-Abe)
//       with CHAIN_APPROX_NONE when minSize == 0.
//
// Tracing:
// Key design decisions (vs. the repository version):
//  1. The padded binary image is the shared label board. Background pixels are never
//     written, so the tracing geometry (background / non-background) never changes.
//  2. Thread k writes VISITED / EAST marks ONLY on its own rows [s_k, e_k) of the
//     shared buffer, plus a PRIVATE copy of row e_k (the first row of the next stripe,
//     scanned by thread k for holes only). Traces still run freely downward, but do
//     not write outside those rows. Hence every decision of thread k (outer/hole
//     start tests) reads only marks written by thread k itself, and its output is a
//     pure function of the input image (deterministic for any schedule).
//     Other threads may read thread k's rows while tracing, but only to test
//     background vs. non-background, which never changes; those cross-thread
//     accesses use relaxed atomics, so the program is data-race-free under the
//     C++ memory model (relaxed atomic byte loads/stores compile to plain moves).
//  3. Ownership is purely geometric. Let minrow(C) be the topmost row of contour C.
//       - outer contour: accepted only from an attempt on row minrow(C)
//       - hole  contour: accepted only from an attempt on row minrow(C)+1
//     Outer attempts are made on rows [s_k, e_k), hole attempts on rows (s_k, e_k],
//     so every contour can only be accepted by the thread whose own rows contain
//     minrow(C). Any trace that moves above s_k is aborted.
//  4. Quick NE test for holes: the Suzuki-Abe start pixel of a hole always has a
//     foreground NE neighbour (the pixel above the hole's top-left background pixel).
//
// Hierarchy: the hierarchy of Suzuki-Abe is topological (containment of 1- and 0-components), so it is
// enough to know, for every contour C, ONE neighbouring border D found on the row where C starts:
//  - outer C, started at run start a of row r: D is the border through the end b of the previous
//    run of row r, on its EAST side (the background between b and a is a single 0-component).
//    No previous run -> C is a root.
//  - hole C, started at run end b of row r: D is the border through the start a of the same run,
//    on its WEST side.
// Then parent(C) = (type(D) != type(C)) ? D : parent(D)   (the rules of Suzuki-Abe's table).
//  Phase 1 (parallel): the tracer; the owner of a contour knows a or b from its own row scan, so
//                      every accepted contour carries a reference (pixel, side).
//  Phase 2 (parallel): resolve the references. A pair (pixel, side whose neighbour is background)
//                      lies on exactly one border: walk the contours (in chunks of points), recompute
//                      which neighbours the trace examined at each point, and look the pairs up in a
//                      mask of referenced pixels.
//  Phase 3:            parents, Suzuki-Abe (raster) order, tree of the requested mode, laid out as
//                      cv::findContours does (children prepended, pre-order traversal).
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <atomic>
#include <bit>

#if !defined(__cpp_lib_atomic_ref)
#error "TRUCO requires C++20 std::atomic_ref"
#endif

// relaxed atomic byte access on the (non-atomic) cv::Mat buffer, standard C++20
static_assert(std::atomic_ref<unsigned char>::required_alignment == 1, "atomic_ref<uchar> must not need extra alignment");
static_assert(std::atomic_ref<unsigned char>::is_always_lock_free, "atomic_ref<uchar> must be lock-free");
inline unsigned char truco_ld(const unsigned char* p) {
    // atomic_ref<const T> is C++26; a relaxed load does not modify the object
    return std::atomic_ref<unsigned char>(*const_cast<unsigned char*>(p)).load(std::memory_order_relaxed);
}
inline void truco_st(unsigned char* p, unsigned char v) { std::atomic_ref<unsigned char>(*p).store(v, std::memory_order_relaxed); }

namespace truco_detail {

template <size_t BLOCK_SIZE = 2048>
class TRUCOPagedContour {
public:
    struct Block { cv::Point data[BLOCK_SIZE]; };
    TRUCOPagedContour() { allocateBlock(); curr_ptr_ = all_blocks_[0]->data; end_ptr_ = curr_ptr_ + BLOCK_SIZE; }
    ~TRUCOPagedContour() { for (Block* b : all_blocks_) cv::fastFree(b); }
    TRUCOPagedContour(const TRUCOPagedContour&) = delete;
    TRUCOPagedContour& operator=(const TRUCOPagedContour&) = delete;

    inline void push_back(const cv::Point& pt) {
        if (curr_ptr_ == end_ptr_) {
            current_block_idx_++;
            if (current_block_idx_ == all_blocks_.size()) allocateBlock();
            curr_ptr_ = all_blocks_[current_block_idx_]->data;
            end_ptr_  = curr_ptr_ + BLOCK_SIZE;
        }
        *curr_ptr_++ = pt;
    }
    inline void pop_back() {
        if (current_block_idx_ == 0 && curr_ptr_ == all_blocks_[0]->data) return;
        if (curr_ptr_ == all_blocks_[current_block_idx_]->data) {
            current_block_idx_--;
            curr_ptr_ = all_blocks_[current_block_idx_]->data + BLOCK_SIZE;
            end_ptr_  = curr_ptr_;
        }
        curr_ptr_--;
    }
    inline const cv::Point& back() const {
        if (curr_ptr_ == all_blocks_[current_block_idx_]->data)
            return all_blocks_[current_block_idx_ - 1]->data[BLOCK_SIZE - 1];
        return *(curr_ptr_ - 1);
    }
    inline const cv::Point& front() const { return all_blocks_[0]->data[0]; }
    size_t size() const {
        return current_block_idx_ * BLOCK_SIZE + (size_t)(curr_ptr_ - all_blocks_[current_block_idx_]->data);
    }
    void clear() { current_block_idx_ = 0; curr_ptr_ = all_blocks_[0]->data; end_ptr_ = curr_ptr_ + BLOCK_SIZE; }
    void copyTo(std::vector<cv::Point>& out) const {
        size_t total = size(); out.resize(total); if (!total) return;
        cv::Point* dst = out.data();
        for (size_t i = 0; i < current_block_idx_; ++i) {
            std::memcpy(dst, all_blocks_[i]->data, BLOCK_SIZE * sizeof(cv::Point)); dst += BLOCK_SIZE;
        }
        size_t last = (size_t)(curr_ptr_ - all_blocks_[current_block_idx_]->data);
        if (last) std::memcpy(dst, all_blocks_[current_block_idx_]->data, last * sizeof(cv::Point));
    }
private:
    void allocateBlock() { all_blocks_.push_back((Block*)cv::fastMalloc(sizeof(Block))); }
    std::vector<Block*> all_blocks_;
    size_t current_block_idx_ = 0;
    cv::Point* curr_ptr_ = nullptr;
    cv::Point* end_ptr_  = nullptr;
};

using AccumulatorT = std::vector<std::vector<cv::Point>>;

// per contour data needed to build the hierarchy
struct ContourInfo {
    int64_t order;   // Suzuki-Abe order: start pixel (unpadded linear index) * 2 + isHole
    int64_t refPix;  // linear index (unpadded) of the reference pixel, -1: none (root); 64-bit: images of more than 2^31 pixels
    uchar refSide;   // REF_WEST or REF_EAST
    bool isHole;
};
static constexpr uchar REF_WEST = 1, REF_EAST = 2;
struct Accumulator {
    AccumulatorT contours;
    std::vector<ContourInfo> info;   // only filled when the hierarchy is requested
};

class TRUCOntourTracer : public cv::ParallelLoopBody
{
public:
    // img    : padded binary image (0 / 255). Shared label board: thread k writes only rows [s_k, e_k)
    // ranges : stripe k owns rows [ranges[k].start, ranges[k].end)
    // extra  : one private row per thread (copy of row e_k), extra.row(k)
    // withInfo: also store the ContourInfo of every contour (hierarchy); minSize is then applied later
    TRUCOntourTracer(cv::Mat& img, const std::vector<cv::Range>& ranges,
                     std::vector<Accumulator>& acc, cv::Mat& extra, size_t minSize, bool withInfo)
        : img_(img), extra_(extra), ranges_(ranges), acc_(acc), minSize_(withInfo ? 0 : minSize), withInfo_(withInfo)
    {
        step_ = img_.step;
        const int s = (int)step_;
        const int o[8] = { 1, -s + 1, -s, -s - 1, -1, s - 1, s, s + 1 };
        for (int i = 0; i < 16; ++i) offsets_[i] = o[i & 7];
    }

    void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        TRUCOPagedContour<4096> buffer;
        const int cols = img_.cols;
        const int W = cols - 2;
        for (int k = range.start; k < range.end; ++k)
        {
            const int s = ranges_[k].start, e = ranges_[k].end;  // scans rows [s, e]
            Accumulator& out = acc_[k];
            out.contours.reserve(2048);
            if (withInfo_) out.info.reserve(2048);

            // private copy of row e (normalized: any non-background -> FOREGROUND;
            // the owner of row e may be writing marks on it concurrently)
            uchar* ext = extra_.data + (size_t)k * extra_.step;
            const uchar* src = img_.data + (size_t)e * step_;
            for (int x = 0; x < cols; ++x) ext[x] = truco_ld(src + x) ? FOREGROUND : BACKGROUND;
            Labels L{ img_.data, ext, step_, e };

            for (int r = s; r <= e; ++r)
            {
                uchar* lrow = (r < e) ? img_.data + (size_t)r * step_ : ext;
                const int64_t rowBase = (int64_t)(r - 1) * W - 1;   // unpadded index of padded column x: rowBase + x
                int prevEnd = -1;                          // padded column of the end of the previous run
                for (int c = 1; c < cols - 1; )
                {
                    if ((c = findStartContourPoint(lrow, cols, c)) == cols) break;
                    const int runStart = c;

                    // OUTER attempt: rows [s, e)
                    if (r < e && lrow[c] == FOREGROUND) {
                        if (traceContour(buffer, r, c, s, e, L, true))
                            emit(buffer, out, { (rowBase + runStart) * 2,
                                                prevEnd < 0 ? -1 : rowBase + prevEnd, REF_EAST, false });
                    }
                    c = findEndContourPoint(lrow, cols, c + 1);
                    if (c >= cols) break;
                    prevEnd = c - 1;
                    // HOLE attempt: rows (s, e]
                    if (r > s && lrow[c - 1] != EAST && neIsForeground(r, c - 1)) {
                        if (traceContour(buffer, r, c - 1, s, e, L, false))
                            emit(buffer, out, { (rowBase + c - 1) * 2 + 1, rowBase + runStart, REF_WEST, true });
                    }
                }
            }
        }
    }

private:
    // where the labels of a pixel (given by its image index and row) live for this thread
    struct Labels {
        uchar* shared; uchar* ext; size_t step; int e;
        inline uchar* at(size_t idx, int row) const {     // only valid for row <= e
            return row < e ? shared + idx : ext + (idx - (size_t)e * step);
        }
    };
    inline bool neIsForeground(int r, int c) const {   // row r-1 is always an own row
        return img_.data[(size_t)(r - 1) * step_ + c + 1] != BACKGROUND;
    }

    inline void emit(TRUCOPagedContour<4096>& buffer, Accumulator& out, const ContourInfo& info) const {
        if (buffer.size() > 1 && buffer.back() == buffer.front()) buffer.pop_back();
        if (buffer.size() < minSize_) return;
        out.contours.emplace_back(); buffer.copyTo(out.contours.back());
        if (withInfo_) out.info.push_back(info);
    }

    // Returns true iff the contour was traced completely AND this attempt is its owner
    // (outer: started on its top row; hole: started one row below its top row).
    bool traceContour(TRUCOPagedContour<4096>& buffer, int r, int c, int s, int e,
                      const Labels& L, bool isExternal) const
    {
        buffer.clear();
        const uchar* g = img_.data;                 // shared board: only bg / non-bg is read here
        int curr_x = c, curr_y = r, minrow = r;
        int start_dir = -1, dir = -1;
        int search_idx = isExternal ? 5 : 1;
        size_t curr = (size_t)r * step_ + c;
        const size_t start = curr;
        bool is_first_move = true;
        while (true)
        {
            buffer.push_back({ curr_x - 1, curr_y - 1 });
            for (int n = 0; n < 8; ++n)
            {
                const int idx = search_idx + n;
                const size_t nb = curr + offsets_[idx];
                if (truco_ld(g + nb) == BACKGROUND) continue;
                dir = idx & 7;
                curr_y += dy_[dir];
                curr_x += dx_[dir];
                if (curr_y < s) return false;       // upward termination
                if (curr_y < minrow) minrow = curr_y;
                // mark the pixel we are leaving (its row is in [s, ...]); only rows <= e are stored
                const int py = curr_y - dy_[dir];
                if (py <= e) {
                    uchar* lp = L.at(curr, py);
                    if ((search_idx <= 1) || (dir <= search_idx - 2)) truco_st(lp, EAST);
                    else if (*lp == FOREGROUND)                       truco_st(lp, VISITED);
                }
                if (curr == start && !is_first_move && dir == start_dir)
                    return isExternal ? (minrow == r) : (minrow == r - 1);
                curr = nb;
                search_idx = (dir + 6) & 7;
                break;
            }
            if (is_first_move) {
                if (dir == -1) {                     // isolated pixel
                    truco_st(L.at(curr, r), EAST);
                    return isExternal;               // an isolated pixel is an outer contour only
                }
                start_dir = dir;
                is_first_move = false;
            }
        }
    }

    static inline int findStartContourPoint(const uchar* src, int width, int j)
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        cv::v_uint8 v_zero = cv::vx_setzero_u8();
        for (; j <= width - cv::VTraits<cv::v_uint8>::vlanes(); j += cv::VTraits<cv::v_uint8>::vlanes()) {
            cv::v_uint8 vmask = cv::v_ne(cv::vx_load(src + j), v_zero);
            if (cv::v_check_any(vmask)) return j + cv::v_scan_forward(vmask);
        }
#endif
        for (; j < width && !src[j]; ++j) ;
        return j;
    }
    static inline int findEndContourPoint(const uchar* src, int width, int j)
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        if (j < width && !src[j]) return j;
        cv::v_uint8 v_zero = cv::vx_setzero_u8();
        for (; j <= width - cv::VTraits<cv::v_uint8>::vlanes(); j += cv::VTraits<cv::v_uint8>::vlanes()) {
            cv::v_uint8 vmask = cv::v_eq(cv::vx_load(src + j), v_zero);
            if (cv::v_check_any(vmask)) return j + cv::v_scan_forward(vmask);
        }
#endif
        for (; j < width && src[j]; ++j) ;
        return j;
    }

    cv::Mat& img_;
    cv::Mat& extra_;
    const std::vector<cv::Range>& ranges_;
    std::vector<Accumulator>& acc_;
    size_t minSize_;
    bool withInfo_;
    size_t step_;
    int offsets_[16];
    // 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
    const int dx_[8] = { 1,  1,  0, -1, -1, -1,  0,  1 };
    const int dy_[8] = { 0, -1, -1, -1,  0,  1,  1,  1 };
    static constexpr uchar FOREGROUND = 255;
    static constexpr uchar BACKGROUND = 0;
    static constexpr uchar EAST       = 100;
    static constexpr uchar VISITED    = 200;
};

inline std::vector<cv::Range> splitStripes(int inner, int nstripes)
{
    if (nstripes < 1) nstripes = 1;
    if (nstripes > inner) nstripes = std::max(1, inner);
    std::vector<cv::Range> ranges;
    int cur = 1;
    for (int t = 0; t < nstripes; ++t) {
        int end = cur + inner / nstripes + (t < inner % nstripes ? 1 : 0);
        ranges.emplace_back(cur, end);
        cur = end;
    }
    return ranges;
}

// Phase 1 (both versions): trace the contours of the padded image in parallel stripes
inline void traceStripes(cv::Mat& padded, std::vector<Accumulator>& acc, int minSize, int nstripes, bool withInfo)
{
    const std::vector<cv::Range> ranges = splitStripes(padded.rows - 2, nstripes);
    cv::Mat extra((int)ranges.size(), padded.cols, CV_8UC1);   // one private row per thread
    acc.assign(ranges.size(), Accumulator());
    TRUCOntourTracer worker(padded, ranges, acc, extra, (size_t)minSize, withInfo);
    cv::parallel_for_(cv::Range(0, (int)ranges.size()), worker);
}

inline void findTRUContoursImpl(cv::Mat& padded, std::vector<std::vector<cv::Point>>& outContours,
                                int minSize, int nstripes)
{
    std::vector<Accumulator> acc;
    traceStripes(padded, acc, minSize, nstripes, false);
    size_t total = 0;
    for (auto& a : acc) total += a.contours.size();
    outContours.clear();
    outContours.reserve(total);
    for (auto& a : acc)
        outContours.insert(outContours.end(), std::make_move_iterator(a.contours.begin()), std::make_move_iterator(a.contours.end()));
}

// direction (0=E, 1=NE, ..., 7=SE, y down) of the move a -> b between 8-neighbours
inline int moveDir(const cv::Point& a, const cv::Point& b) {
    static const int lut[3][3] = { { 3, 2, 1 }, { 4, -1, 0 }, { 5, 6, 7 } };   // [dy+1][dx+1]
    return lut[b.y - a.y + 1][b.x - a.x + 1];
}
// Background neighbours examined by the Suzuki-Abe search at a point where the code search
// started at sc and found the next point in direction d: the cyclic range [sc-1, d)
// (sc-1 is known background and is the first examined pixel of Suzuki-Abe's step 3.3).
inline uchar examinedSides(int sc, int d) {
    const int from = (sc + 7) & 7;
    const int len = ((d - sc) & 7) + 1;
    uchar sides = 0;
    if (((4 - from) & 7) < len) sides |= REF_WEST;
    if (((0 - from) & 7) < len) sides |= REF_EAST;
    return sides;
}

// Sorts pairs (pixel*2 + bit, index) that come from the concatenation of the stripe lists: each
// list is sorted, and two consecutive lists only overlap on the row they share, so only the
// windows of those rows are sorted.
inline void sortStripeConcat(std::vector<std::pair<int64_t, int>>& v, int W) {
    auto row = [&](size_t i) { return (v[i].first >> 1) / W; };
    const size_t n = v.size();
    for (size_t i = 1; i < n; ++i) {
        if (!(v[i] < v[i - 1])) continue;
        const int64_t r = row(i);
        size_t l = i, h = i;
        while (l > 0 && row(l - 1) == r) --l;
        while (h < n && row(h) == r) ++h;
        std::sort(v.begin() + l, v.begin() + h);
        i = h;
    }
}

// Phase 2: ref[i] = index of the contour that answers the reference of contour i (-1: root).
// Each (pixel, side) is referenced at most once (by the next run start, or by the end of the same
// run), and answered by exactly one contour. The walk over the contour points runs in parallel, in
// chunks of points so that a single long contour does not serialize it.
// The referenced pairs are a bitmap with one bit per key = pixel*2 + (side == REF_EAST), plus, for
// every 64-bit word with a key, the index of its first key in the sorted references (sparse rank):
// a lookup is O(1), and only the words of referenced pixels are touched.
class ReferenceResolver : public cv::ParallelLoopBody
{
public:
    static inline size_t chunkPoints = 1 << 15;   // points per task (small values only for testing)

    ReferenceResolver(const AccumulatorT& contours, const std::vector<ContourInfo>& info, int W,
                      const uint64_t* bits, const uint32_t* rank, const std::vector<std::pair<int64_t, int>>& refs,
                      const std::vector<size_t>& firstPoint, std::vector<int>& ref)
        : contours_(contours), info_(info), W_(W), bits_(bits), rank_(rank), refs_(refs), firstPoint_(firstPoint), ref_(ref)
    {
        // examined sides at a point entered in direction din and left in direction dout
        for (int din = 0; din < 8; ++din)
            for (int dout = 0; dout < 8; ++dout) sides_[din][dout] = examinedSides((din + 6) & 7, dout);
    }

    void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        const size_t p0 = (size_t)range.start * chunkPoints;
        const size_t p1 = std::min((size_t)range.end * chunkPoints, firstPoint_.back());
        // contour that contains point p0
        size_t i = std::upper_bound(firstPoint_.begin(), firstPoint_.end(), p0) - firstPoint_.begin() - 1;
        for (size_t p = p0; p < p1; ++i) {
            const auto& c = contours_[i];
            const size_t m = c.size();
            const size_t j0 = p - firstPoint_[i], j1 = std::min(m, j0 + (p1 - p));
            p += j1 - j0;
            if (m == 0) continue;
            if (m == 1) { answer(c[0], REF_WEST | REF_EAST, (int)i); continue; }   // isolated pixel
            if (j0 == 0)   // first visit of the start point (search started at SW for outer, NE for holes)
                answer(c[0], examinedSides(info_[i].isHole ? 1 : 5, moveDir(c[0], c[1])), (int)i);
            // every visit, including the closing one of the start point
            int din = moveDir(c[j0 == 0 ? m - 1 : j0 - 1], c[j0]);
            for (size_t j = j0; j < j1; ++j) {
                const cv::Point& next = c[j + 1 == m ? 0 : j + 1];
                const int dout = moveDir(c[j], next);
                answer(c[j], sides_[din][dout], (int)i);
                din = dout;
            }
        }
    }

private:
    inline void answer(const cv::Point& pt, uchar sides, int contour) const {
        const uint64_t key0 = ((uint64_t)pt.y * W_ + pt.x) * 2;   // even: both sides in the same word
        const uint64_t word = bits_[key0 >> 6];
        const int sh = (int)(key0 & 63);
        const uchar hit = (uchar)((word >> sh) & 3) & sides;
        if (!hit) return;
        for (int side = REF_WEST; side <= REF_EAST; ++side) {
            if (!(hit & side)) continue;
            const int b = sh + side - 1;
            const uint32_t k = rank_[key0 >> 6] + (uint32_t)std::popcount(word & ((uint64_t(1) << b) - 1));
            // a contour may visit a pixel several times, possibly from two chunks: same value
            std::atomic_ref<int>(ref_[refs_[k].second]).store(contour, std::memory_order_relaxed);
        }
    }

    const AccumulatorT& contours_;
    const std::vector<ContourInfo>& info_;
    const int W_;
    const uint64_t* bits_;
    const uint32_t* rank_;
    const std::vector<std::pair<int64_t, int>>& refs_;
    const std::vector<size_t>& firstPoint_;
    std::vector<int>& ref_;
    uchar sides_[8][8];
};

inline void resolveReferences(const AccumulatorT& contours, const std::vector<ContourInfo>& info,
                              int W, int H, std::vector<int>& ref)
{
    const size_t n = contours.size();
    CV_Assert(n < UINT32_MAX);
    ref.assign(n, -1);
    // bitmap of referenced keys, and for every word with a key the index of its first key in the
    // sorted refs (rank). calloc gives zeroed pages lazily: only the words of referenced pixels are touched
    const size_t nwords = ((size_t)W * H * 2 + 63) / 64;
    std::unique_ptr<uint64_t, decltype(&std::free)> bits((uint64_t*)std::calloc(nwords, sizeof(uint64_t)), &std::free);
    std::unique_ptr<uint32_t, decltype(&std::free)> rank((uint32_t*)std::calloc(nwords, sizeof(uint32_t)), &std::free);
    CV_Assert(bits && rank);
    std::vector<std::pair<int64_t, int>> refs;   // (key, contour)
    refs.reserve(n);
    std::vector<size_t> firstPoint(n + 1);
    firstPoint[0] = 0;
    for (size_t i = 0; i < n; ++i) {
        firstPoint[i + 1] = firstPoint[i] + contours[i].size();
        if (info[i].refPix < 0) continue;
        const uint64_t key = (uint64_t)info[i].refPix * 2 + (info[i].refSide - 1);
        bits.get()[key >> 6] |= uint64_t(1) << (key & 63);
        refs.emplace_back((int64_t)key, (int)i);
    }
    sortStripeConcat(refs, W);
    for (size_t k = refs.size(); k-- > 0; ) rank.get()[refs[k].first >> 6] = (uint32_t)k;

    const int nchunks = (int)((firstPoint[n] + ReferenceResolver::chunkPoints - 1) / ReferenceResolver::chunkPoints);
    ReferenceResolver body(contours, info, W, bits.get(), rank.get(), refs, firstPoint, ref);
    cv::parallel_for_(cv::Range(0, nchunks), body);
}

// optional per-phase timings (ms), for benchmarking
struct PhaseTimes { double trace = 0, merge = 0, references = 0, parents = 0, output = 0; };

// mode: cv::RETR_LIST, cv::RETR_CCOMP or cv::RETR_TREE.
// minSize: contours with fewer points are dropped; RETR_TREE: their children go to the nearest kept
// ancestor; RETR_CCOMP: a hole whose outer border is dropped becomes a root.
inline void findTRUContoursHierarchyImpl(cv::Mat& padded, std::vector<std::vector<cv::Point>>& outContours,
                                         std::vector<cv::Vec4i>& outHierarchy, int mode, int minSize,
                                         int nstripes, PhaseTimes* times = nullptr)
{
    CV_Assert(mode == cv::RETR_LIST || mode == cv::RETR_CCOMP || mode == cv::RETR_TREE);
    int64 t0 = cv::getTickCount();
    auto lap = [&](double PhaseTimes::* field) {
        if (!times) return;
        const int64 t1 = cv::getTickCount();
        times->*field = (t1 - t0) * 1000.0 / cv::getTickFrequency();
        t0 = t1;
    };
    const int H = padded.rows - 2, W = padded.cols - 2;

    // Phase 1
    std::vector<Accumulator> acc;
    traceStripes(padded, acc, minSize, nstripes, true);
    lap(&PhaseTimes::trace);

    size_t total = 0;
    for (auto& a : acc) total += a.contours.size();
    AccumulatorT contours;
    std::vector<ContourInfo> info;
    contours.reserve(total);
    info.reserve(total);
    for (auto& a : acc) {
        contours.insert(contours.end(), std::make_move_iterator(a.contours.begin()), std::make_move_iterator(a.contours.end()));
        info.insert(info.end(), a.info.begin(), a.info.end());
    }
    acc.clear();
    const int n = (int)total;
    lap(&PhaseTimes::merge);

    // Phase 2 (not needed for RETR_LIST)
    std::vector<int> ref(n, -1);
    if (mode != cv::RETR_LIST) resolveReferences(contours, info, W, H, ref);
    lap(&PhaseTimes::references);

    // Phase 3: parents (RETR_TREE semantics). ref[i] precedes i in raster order, but not necessarily
    // in stripe order (rows shared by two stripes), so resolve with an explicit stack.
    std::vector<int> parent(n, -2);   // -2: unresolved, -1: root
    std::vector<int> stack;
    for (int i = 0; i < n; ++i) {
        int j = i;
        while (parent[j] == -2) {
            const int d = ref[j];
            if (d < 0) { parent[j] = -1; break; }
            if (info[d].isHole != info[j].isHole) { parent[j] = d; break; }
            stack.push_back(j);
            j = d;
        }
        while (!stack.empty()) {
            const int k = stack.back(); stack.pop_back();
            parent[k] = parent[ref[k]];
        }
    }
    lap(&PhaseTimes::parents);

    // Suzuki-Abe order: start pixel in raster order, outer before hole on the same pixel
    // (the stripe concatenation is already sorted except at the rows shared by two stripes)
    std::vector<std::pair<int64_t, int>> sorted(n);
    for (int i = 0; i < n; ++i) sorted[i] = { info[i].order, i };
    sortStripeConcat(sorted, W);

    // tree of the requested mode, minSize applied. Same layout as cv::findContours (OpenCV 4.x):
    // every new child is prepended to its parent's list (CTree::addChild), and the output is the
    // pre-order traversal of the tree.
    auto small = [&](int i) { return (int)contours[i].size() < minSize; };
    std::vector<int> firstChild(n + 1, -1), next(n, -1), treeParent(n, -1);   // index n: virtual root
    for (const auto& s : sorted) {
        const int i = s.second;
        if (small(i)) continue;
        int p = parent[i];
        if (mode == cv::RETR_LIST || (mode == cv::RETR_CCOMP && !info[i].isHole)) p = -1;
        else if (mode == cv::RETR_CCOMP) { if (small(p)) p = -1; }
        else while (p >= 0 && small(p)) p = parent[p];
        const int pp = p < 0 ? n : p;
        treeParent[i] = p;
        next[i] = firstChild[pp];
        firstChild[pp] = i;
    }
    std::vector<int> newIdx(n, -1), preorder;
    preorder.reserve(n);
    auto pushChildren = [&](int node) {
        const size_t mark = stack.size();
        for (int c = firstChild[node]; c >= 0; c = next[c]) stack.push_back(c);
        std::reverse(stack.begin() + mark, stack.end());
    };
    pushChildren(n);
    while (!stack.empty()) {
        const int i = stack.back(); stack.pop_back();
        newIdx[i] = (int)preorder.size();
        preorder.push_back(i);
        pushChildren(i);
    }

    const int kept = (int)preorder.size();
    outContours.clear();
    outContours.reserve(kept);
    outHierarchy.assign(kept, cv::Vec4i(-1, -1, -1, -1));
    for (int ni = 0; ni < kept; ++ni) {
        const int i = preorder[ni];
        cv::Vec4i& hv = outHierarchy[ni];
        hv[0] = next[i] < 0 ? -1 : newIdx[next[i]];
        if (hv[0] >= 0) outHierarchy[hv[0]][1] = ni;
        hv[2] = firstChild[i] < 0 ? -1 : newIdx[firstChild[i]];
        hv[3] = treeParent[i] < 0 ? -1 : newIdx[treeParent[i]];
        outContours.push_back(std::move(contours[i]));
    }
    lap(&PhaseTimes::output);
}

} // namespace truco_detail

namespace cv {
namespace truco_internal {
// false: sequential cv::copyMakeBorder (only for benchmarking)
inline bool parallelPad = true;

// Padded copy (frame of background pixels) of a CV_8UC1 image, made in parallel by row blocks: a single
// thread does not saturate the memory bandwidth, and the pages of the new buffer are touched by all threads.
inline Mat padImage(InputArray _src, bool binarize) {
    Mat src = _src.getMat();
    CV_Assert(!src.empty() && src.type() == CV_8UC1);
    Mat padded;
    if (!parallelPad) {
        copyMakeBorder(src, padded, 1, 1, 1, 1, BORDER_CONSTANT | BORDER_ISOLATED, 0);   // ISOLATED: ignore the pixels outside a ROI
        if (binarize) threshold(padded, padded, 0, 255, THRESH_BINARY);
        return padded;
    }
    padded.create(src.rows + 2, src.cols + 2, CV_8UC1);
    const int H = src.rows, W = src.cols;
    parallel_for_(Range(0, H + 2), [&](const Range& rows) {
        for (int y = rows.start; y < rows.end; ++y) {
            uchar* d = padded.ptr<uchar>(y);
            if (y == 0 || y == H + 1) { std::memset(d, 0, W + 2); continue; }
            const uchar* s = src.ptr<uchar>(y - 1);
            d[0] = d[W + 1] = 0;
            if (binarize) for (int x = 0; x < W; ++x) d[x + 1] = s[x] ? 255 : 0;
            else std::memcpy(d + 1, s, W);
        }
    }, getNumThreads());
    return padded;
}
inline void copyOut(const std::vector<std::vector<Point>>& tmp, OutputArrayOfArrays _contours) {
    _contours.create((int)tmp.size(), 1, 0, -1, true);
    for (size_t i = 0; i < tmp.size(); i++) {
        _contours.create((int)tmp[i].size(), 1, CV_32SC2, (int)i, true);
        Mat m = _contours.getMat((int)i);
        std::memcpy(m.data, tmp[i].data(), tmp[i].size() * sizeof(Point));
    }
}
inline std::vector<std::vector<Point>>* directOutput(OutputArrayOfArrays _contours, std::vector<std::vector<Point>>& tmp) {
    return (_contours.kind() == _InputArray::STD_VECTOR_VECTOR && _contours.type() == CV_32SC2)
        ? reinterpret_cast<std::vector<std::vector<Point>>*>(_contours.getObj()) : &tmp;
}
}

// Contours without hierarchy (the order is not the one of cv::findContours).
// minSize: contours with fewer points are not returned. binarize: any non-zero pixel is foreground
// (otherwise the image must already be 0 / 255).
inline void findTRUContours(InputArray _src, OutputArrayOfArrays _contours, int minSize = 0, bool binarize = false)
{
    Mat padded = truco_internal::padImage(_src, binarize);
    std::vector<std::vector<Point>> tmp;
    std::vector<std::vector<Point>>* dst = truco_internal::directOutput(_contours, tmp);
    truco_detail::findTRUContoursImpl(padded, *dst, minSize, getNumThreads());
    if (dst == &tmp) truco_internal::copyOut(tmp, _contours);
}

// Contours with hierarchy ([next, prev, first_child, parent], as in cv::findContours).
// mode: RETR_LIST, RETR_CCOMP or RETR_TREE. With minSize == 0 the output is identical to
// cv::findContours(img, contours, hierarchy, mode, CHAIN_APPROX_NONE).
// times: optional per-phase timings (ms).
inline void findTRUContours(InputArray _src, OutputArrayOfArrays _contours, std::vector<Vec4i>& hierarchy,
                            int mode = RETR_TREE, int minSize = 0, bool binarize = false,
                            truco_detail::PhaseTimes* times = nullptr)
{
    Mat padded = truco_internal::padImage(_src, binarize);
    std::vector<std::vector<Point>> tmp;
    std::vector<std::vector<Point>>* dst = truco_internal::directOutput(_contours, tmp);
    truco_detail::findTRUContoursHierarchyImpl(padded, *dst, hierarchy, mode, minSize, getNumThreads(), times);
    if (dst == &tmp) truco_internal::copyOut(tmp, _contours);
}
}
