#pragma once
// Ablation variants of TRUCO stage 1 (contours without hierarchy), for the single-thread analysis of the
// paper: the label image B with 8-bit or 32-bit elements, and the run detection of the scan with or without
// vector instructions. The tracer is a copy of TRUCOntourTracer (findtrucontour.h) templated on the element
// type of B; only the element type of B and the scan change. Regenerate it if the tracer changes.
//   cv::findTRUContoursAblation<uchar, true >(img, contours)   == cv::findTRUContours (8-bit, vector scan)
//   cv::findTRUContoursAblation<uchar, false>(img, contours)   8-bit, scalar scan
//   cv::findTRUContoursAblation<int,   false>(img, contours)   32-bit, scalar scan
#include "findtrucontour.h"

namespace truco_ablation {
using truco_detail::TRUCOPagedContour;
using truco_detail::Accumulator;
using truco_detail::ContourInfo;
using truco_detail::REF_WEST;
using truco_detail::REF_EAST;

template <typename L, bool SIMD>
class TRUCOntourTracerA : public cv::ParallelLoopBody
{
public:
    // img    : padded binary image (0 / 255). Shared label board: thread k writes only rows [s_k, e_k)
    // ranges : stripe k owns rows [ranges[k].start, ranges[k].end)
    // extra  : one private row per thread (copy of row e_k), extra.row(k)
    // withInfo: also store the ContourInfo of every contour (hierarchy); minSize is then applied later
    TRUCOntourTracerA(cv::Mat& img, const std::vector<cv::Range>& ranges,
                     std::vector<Accumulator>& acc, cv::Mat& extra, size_t minSize, bool withInfo)
        : img_(img), extra_(extra), ranges_(ranges), acc_(acc), minSize_(withInfo ? 0 : minSize), withInfo_(withInfo)
    {
        step_ = img_.step / sizeof(L);
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
            L* ext = (L*)extra_.data + (size_t)k * (extra_.step / sizeof(L));
            const L* src = ((L*)img_.data) + (size_t)e * step_;
            for (int x = 0; x < cols; ++x) ext[x] = ld(src + x) ? FOREGROUND : BACKGROUND;
            Labels LB{ ((L*)img_.data), ext, step_, e };

            for (int r = s; r <= e; ++r)
            {
                L* lrow = (r < e) ? ((L*)img_.data) + (size_t)r * step_ : ext;
                const int64_t rowBase = (int64_t)(r - 1) * W - 1;   // unpadded index of padded column x: rowBase + x
                int prevEnd = -1;                          // padded column of the end of the previous run
                for (int c = 1; c < cols - 1; )
                {
                    if ((c = findStartContourPoint(lrow, cols, c)) == cols) break;
                    const int runStart = c;

                    // OUTER attempt: rows [s, e)
                    if (r < e && lrow[c] == FOREGROUND) {
                        if (traceContour(buffer, r, c, s, e, LB, true))
                            emit(buffer, out, { (rowBase + runStart) * 2,
                                                prevEnd < 0 ? -1 : rowBase + prevEnd, REF_EAST, false });
                    }
                    c = findEndContourPoint(lrow, cols, c + 1);
                    if (c >= cols) break;
                    prevEnd = c - 1;
                    // HOLE attempt: rows (s, e]
                    if (r > s && lrow[c - 1] != EAST && neIsForeground(r, c - 1)) {
                        if (traceContour(buffer, r, c - 1, s, e, LB, false))
                            emit(buffer, out, { (rowBase + c - 1) * 2 + 1, rowBase + runStart, REF_WEST, true });
                    }
                }
            }
        }
    }

private:
    static inline L ld(const L* p) { return std::atomic_ref<L>(*const_cast<L*>(p)).load(std::memory_order_relaxed); }
    static inline void st(L* p, L v) { std::atomic_ref<L>(*p).store(v, std::memory_order_relaxed); }
    // where the labels of a pixel (given by its image index and row) live for this thread
    struct Labels {
        L* shared; L* ext; size_t step; int e;
        inline L* at(size_t idx, int row) const {     // only valid for row <= e
            return row < e ? shared + idx : ext + (idx - (size_t)e * step);
        }
    };
    inline bool neIsForeground(int r, int c) const {   // row r-1 is always an own row
        return ((L*)img_.data)[(size_t)(r - 1) * step_ + c + 1] != BACKGROUND;
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
                      const Labels& LB, bool isExternal) const
    {
        buffer.clear();
        const L* g = ((L*)img_.data);                 // shared board: only bg / non-bg is read here
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
                if (ld(g + nb) == BACKGROUND) continue;
                dir = idx & 7;
                curr_y += dy_[dir];
                curr_x += dx_[dir];
                if (curr_y < s) return false;       // upward termination
                if (curr_y < minrow) minrow = curr_y;
                // mark the pixel we are leaving (its row is in [s, ...]); only rows <= e are stored
                const int py = curr_y - dy_[dir];
                if (py <= e) {
                    L* lp = LB.at(curr, py);
                    if ((search_idx <= 1) || (dir <= search_idx - 2)) st(lp, EAST);
                    else if (*lp == FOREGROUND)                       st(lp, VISITED);
                }
                if (curr == start && !is_first_move && dir == start_dir)
                    return isExternal ? (minrow == r) : (minrow == r - 1);
                curr = nb;
                search_idx = (dir + 6) & 7;
                break;
            }
            if (is_first_move) {
                if (dir == -1) {                     // isolated pixel
                    st(LB.at(curr, r), EAST);
                    return isExternal;               // an isolated pixel is an outer contour only
                }
                start_dir = dir;
                is_first_move = false;
            }
        }
    }

    static inline int findStartContourPoint(const L* src, int width, int j)
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        if constexpr (SIMD && sizeof(L) == 1) {
        cv::v_uint8 v_zero = cv::vx_setzero_u8();
        for (; j <= width - cv::VTraits<cv::v_uint8>::vlanes(); j += cv::VTraits<cv::v_uint8>::vlanes()) {
            cv::v_uint8 vmask = cv::v_ne(cv::vx_load((const uchar*)src + j), v_zero);
            if (cv::v_check_any(vmask)) return j + cv::v_scan_forward(vmask);
        }
        }
#endif
        for (; j < width && !src[j]; ++j) ;
        return j;
    }
    static inline int findEndContourPoint(const L* src, int width, int j)
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        if constexpr (SIMD && sizeof(L) == 1) {
        if (j < width && !src[j]) return j;
        cv::v_uint8 v_zero = cv::vx_setzero_u8();
        for (; j <= width - cv::VTraits<cv::v_uint8>::vlanes(); j += cv::VTraits<cv::v_uint8>::vlanes()) {
            cv::v_uint8 vmask = cv::v_eq(cv::vx_load((const uchar*)src + j), v_zero);
            if (cv::v_check_any(vmask)) return j + cv::v_scan_forward(vmask);
        }
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
    static constexpr L FOREGROUND = 255;
    static constexpr L BACKGROUND = 0;
    static constexpr L EAST       = 100;
    static constexpr L VISITED    = 200;
};

template <typename L, bool SIMD>
inline void findContoursImpl(cv::Mat& padded, std::vector<std::vector<cv::Point>>& out, int nstripes)
{
    const std::vector<cv::Range> ranges = truco_detail::splitStripes(padded.rows - 2, nstripes);
    cv::Mat extra((int)ranges.size(), padded.cols, padded.type());   // one private row per thread
    std::vector<Accumulator> acc(ranges.size());
    TRUCOntourTracerA<L, SIMD> worker(padded, ranges, acc, extra, 0, false);
    cv::parallel_for_(cv::Range(0, (int)ranges.size()), worker);
    size_t total = 0;
    for (auto& a : acc) total += a.contours.size();
    out.clear();
    out.reserve(total);
    for (auto& a : acc)
        out.insert(out.end(), std::make_move_iterator(a.contours.begin()), std::make_move_iterator(a.contours.end()));
}
} // namespace truco_ablation

namespace cv {
// same input and output as cv::findTRUContours(img, contours) (0 / 255 image, no hierarchy, minSize 0)
template <typename L, bool SIMD>
inline void findTRUContoursAblation(InputArray _src, std::vector<std::vector<Point>>& contours)
{
    static_assert(std::is_same_v<L, uchar> || std::is_same_v<L, int>);
    Mat padded = truco_internal::padImage(_src, false);
    if constexpr (std::is_same_v<L, int>) padded.convertTo(padded, CV_32S);   // values 0 / 255
    truco_ablation::findContoursImpl<L, SIMD>(padded, contours, getNumThreads());
}
}
