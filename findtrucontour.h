#pragma once
#include "opencv2/core.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <opencv2/imgproc.hpp>
namespace cv{


// Tunable block size. 1024 points = 8KB (Fits easily in L1 Cache)
template <size_t BLOCK_SIZE = 2048>
class TRUCOPagedContour {
public:
    struct Block {
        cv::Point data[BLOCK_SIZE];
    };

    TRUCOPagedContour() {
        allocateBlock();
        // Initialize pointers to the start of the first block
        curr_ptr_ = all_blocks_[0]->data;
        end_ptr_  = curr_ptr_ + BLOCK_SIZE;
    }

    ~TRUCOPagedContour() {
        for (Block* b : all_blocks_) cv::fastFree(b);
    }

    // --- HOT PATH: Minimal instructions ---
    // No counter updates, just raw pointer arithmetic.
    inline void push_back(const cv::Point& pt) {
        if (curr_ptr_ == end_ptr_) {
            current_block_idx_++;
            if (current_block_idx_ == all_blocks_.size()) {
                allocateBlock();
            }
            curr_ptr_ = all_blocks_[current_block_idx_]->data;
            end_ptr_  = curr_ptr_ + BLOCK_SIZE;
        }
        *curr_ptr_++ = pt;
    }

    inline void pop_back() {
        // Safety check: do nothing if completely empty
        if (current_block_idx_ == 0 && curr_ptr_ == all_blocks_[0]->data) return;

        // Check if we are at the start of the current block
        if (curr_ptr_ == all_blocks_[current_block_idx_]->data) {
            // Move to the previous block
            current_block_idx_--;
            // Point to the end of the previous block
            curr_ptr_ = all_blocks_[current_block_idx_]->data + BLOCK_SIZE;
            end_ptr_  = curr_ptr_;
        }
        curr_ptr_--;
    }

    inline const cv::Point& back() const {
        // Handle case where back() crosses block boundary
        if (curr_ptr_ == all_blocks_[current_block_idx_]->data) {
            return all_blocks_[current_block_idx_ - 1]->data[BLOCK_SIZE - 1];
        }
        return *(curr_ptr_ - 1);
    }

    inline const cv::Point& front() const {
        return all_blocks_[0]->data[0];
    }

    // Calculated on demand (O(1) arithmetic, but slightly more math than reading a variable)
    size_t size() const {
        size_t elements_in_last = curr_ptr_ - all_blocks_[current_block_idx_]->data;
        return (current_block_idx_ * BLOCK_SIZE) + elements_in_last;
    }

    void clear() {
        current_block_idx_ = 0;
        if (!all_blocks_.empty()) {
            curr_ptr_ = all_blocks_[0]->data;
            end_ptr_  = curr_ptr_ + BLOCK_SIZE;
        }
    }

    // Optimized Copy: Uses block-wise memcpy
    void copyTo(std::vector<cv::Point>& out) const {
        size_t total = size();
        out.resize(total);
        if (total == 0) return;

        cv::Point* dst = out.data();

        // 1. Copy full blocks
        for (size_t i = 0; i < current_block_idx_; ++i) {
            std::memcpy(dst, all_blocks_[i]->data, BLOCK_SIZE * sizeof(cv::Point));
            dst += BLOCK_SIZE;
        }

        // 2. Copy partial last block
        size_t last_block_count = curr_ptr_ - all_blocks_[current_block_idx_]->data;
        if (last_block_count > 0) {
            std::memcpy(dst, all_blocks_[current_block_idx_]->data, last_block_count * sizeof(cv::Point));
        }
    }

private:
    void grow() {
        current_block_idx_++;
        if (current_block_idx_ == all_blocks_.size()) {
            allocateBlock();
        }
        curr_ptr_ = all_blocks_[current_block_idx_]->data;
        end_ptr_  = curr_ptr_ + BLOCK_SIZE;
    }

    void allocateBlock() {
             Block* b = (Block*)cv::fastMalloc(sizeof(Block));
        all_blocks_.push_back(b);
    }

    std::vector<Block*> all_blocks_;
    size_t current_block_idx_ = 0;

    // Fast pointers for the hot loop
    cv::Point* curr_ptr_ = nullptr;
    cv::Point* end_ptr_  = nullptr;
};

/**
 * @brief Traces the contours of a binary image using the Moore-Neighbor Tracing algorithm.
 *
 * @param _src The input binary image (CV_8UC1).
 * @param _contours Output vector of contours.
 * @param minSize Minimum number of points for a contour to be retained.
 * @param nthreads Number of threads to use for parallel processing. If 0, the number of threads is determined automatically.
 * @param _buffer Optional working buffer (CV_8UC1) sized (src.rows+2 x src.cols+2). It may not be sized, so it will be done here.
 *                Pass a persistent Mat here to avoid memory allocation overhead in loops.
 */
CV_EXPORTS_W  void findTRUContours(InputArray _src, OutputArrayOfArrays _contours, int minSize=0,int nthreads=0,InputOutputArray _buffer = noArray() );

CV_EXPORTS_W  void  findTRUContours(InputArray _src,std::vector<std::vector<cv::Point>>& contours,
                                  int minSize=0,int nthreads=0,
                                  InputOutputArray _buffer= noArray());





////IMPLEMENTATION


#define CV_INSTRUMENT_REGION()//remove when inserted in the OpenCV library


class TRUCOntourTracer : public cv::ParallelLoopBody
{
public:

    // We use a pointer to the accumulator to avoid passing huge objects
    // Accumulator: Vector of (Vector of Contours), where Contour is Vector of Points
    using AccumulatorType = std::vector<std::vector<std::vector<cv::Point>>>;

    TRUCOntourTracer(const cv::Mat& img,
                     const std::vector<cv::Range>& stripRanges,
                     AccumulatorType& accumulator,
                     size_t minSize)
        : padded_(img), ranges_(stripRanges), accumulator_(accumulator), minSize_(minSize)
    {
        step_ = padded_.step;
        // 0: East (Right)
        offsets_[0] = 1;
        // 1: NE (Up-Right)
        offsets_[1] = -step_ + 1;
        // 2: North (Up)
        offsets_[2] = -step_;
        // 3: NW (Up-Left)
        offsets_[3] = -step_ - 1;
        // 4: West (Left)
        offsets_[4] = -1;
        // 5: SW (Down-Left)
        offsets_[5] = step_ - 1;
        // 6: South (Down)
        offsets_[6] = step_;
        // 7: SE (Down-Right)
        offsets_[7] = step_ + 1;

        memcpy(offsets_ + 8, offsets_, 8 * sizeof(int));

    }

    bool traceContour( TRUCOPagedContour<4096>* buffer,  int r,int c,uchar *row_ptr, const cv::Range& rowRange,bool isExternal)const{
        buffer->clear();

        int curr_x = c , curr_y = r;
        int start_dir = -1 ;
        int search_idx = isExternal ? 5 :1;
        uchar* curr_ptr = row_ptr + c , * start_ptr = curr_ptr;
        int dir=-1;
        bool is_first_move = true;
        // 3. TRACING LOOP
        while(true)
        {
            buffer->push_back({curr_x - 1, curr_y - 1});
            // showImage(padded_);
            // Check neighbors
            for (int n = 0; n < 8; ++n)
            {
                int idx = search_idx + n;
                // Use offset cache
                uchar* neighbor = curr_ptr + offsets_[idx];
                if (*neighbor == BACKGROUND) continue;

                dir = idx & 7;
                // --- EXECUTE MOVE ---
                curr_y += dy_[dir];
                curr_x += dx_[dir];
                // Check bounds //we need to move out of the range  //if first line, and internal contour, we let it go, but no further from this line
                if( curr_y < rowRange.start  &&  !(!isExternal && r==rowRange.start && curr_y== rowRange.start -1)){
                    return false;
                }
                if ((search_idx <= 1)  || (dir <= search_idx - 2))
                {
                    *curr_ptr = VISITED_OUTER_RIGHT;
                }
                else if (*curr_ptr == FOREGROUND)
                {
                    *curr_ptr = VISITED_;
                }

                // Short-circuit Jacob's Check
                if (curr_ptr == start_ptr) {
                    if (!is_first_move && dir == start_dir) {
                        return true;//done
                    }
                }

                curr_ptr = neighbor;//move ptr
#ifdef DEBUG_UCONTOUR
                showImage(padded_,cv::Point(curr_x,curr_y));
#endif
                // Reset search index for Moore neighbor
                search_idx = (dir +6) & 7;
                break;

            }
            if (is_first_move) {
                if(dir==-1){//single pixel
                    *curr_ptr = VISITED_OUTER_RIGHT;
                    break;//not moved
                }
                start_dir = dir;
                is_first_move = false;
            }

        }
        return true;
    }
    void operator()(const cv::Range& range) const CV_OVERRIDE
    {


        // Pre-allocate buffer to avoid re-allocation during moves
        TRUCOPagedContour<4096> buffer;

        int cols = padded_.cols;

        for (int i = range.start; i < range.end; ++i)
        {
            const cv::Range& rowRange = ranges_[i];
            auto& local_contours = accumulator_[i];

            // Hint for result vector size
            local_contours.reserve(2048);
#ifdef DEBUG_UCONTOUR
            std::cout<<"Thread Range: "<<rowRange.start<<" - "<<rowRange.end<<std::endl;
#endif
            for (int r = rowRange.start; r < rowRange.end; ++r)
            {
                uchar* row_ptr = padded_.data + r * step_;

                // "c" is updated by the find* functions
                for (int c = 1; c < cols - 1; )
                {
                    // 1. FAST SCAN: Skip background pixels
                    if ((c = findStartContourPoint(row_ptr, cols, c)) == cols) break;

                    // 2. CHECK: Only process if actually FOREGROUND (redundancy check)
                    if (row_ptr[c] == FOREGROUND)
                    {
                        if( traceContour(&buffer,r,c,row_ptr,rowRange,true)){
                            // Post-processing
                            if (buffer.size() > 1 && buffer.back() == buffer.front()) {
                                buffer.pop_back();
                            }
                            if (buffer.size() >= minSize_) {
                                // --- OPTIMIZATION: Move Semantics ---
                                // Instead of copying the vector, we move it.
                                local_contours.emplace_back();
                                buffer.copyTo(local_contours.back());
                            }
                        }
                    }

                    // 4. FAST SCAN: Find end of current component to skip processing it again
                    c = findEndContourPoint(row_ptr, cols, c + 1);
                    if(c>=cols)break;//end of row
                    //internal contour
                    if(row_ptr[c-1]>VISITED_OUTER_RIGHT){// if(row_ptr[c-1] == FOREGROUND || row_ptr[c-1]==VISITED_){


                        if( traceContour(&buffer,r,c-1,row_ptr,rowRange,false)){

                            // Post-processing
                            if (buffer.size() > 1 && buffer.back() == buffer.front()) {
                                buffer.pop_back();

                            }

                            if (buffer.size() >= minSize_) {
                                local_contours.emplace_back();
                                buffer.copyTo(local_contours.back());
                            }
                        }
                    }
                }
            }
        }

    }


    static inline int findStartContourPoint(uchar* src_data, int width, int j)
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        v_uint8 v_zero = vx_setzero_u8();
        for (; j <= width - VTraits<v_uint8>::vlanes(); j += VTraits<v_uint8>::vlanes())
        {
            v_uint8 vmask = (v_ne(vx_load((uchar*)(src_data + j)), v_zero));
            if (v_check_any(vmask))
            {
                j += v_scan_forward(vmask);
                return j;
            }
        }
#endif
        for (; j < width && !src_data[j]; ++j)
            ;
        return j;
    }

    inline static int findEndContourPoint(uchar* src_data,int width, int j)
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        if (j <  width && !src_data[j])
        {
            return j;
        }
        else
        {
            v_uint8 v_zero = vx_setzero_u8();
            for (; j <=  width - VTraits<v_uint8>::vlanes(); j += VTraits<v_uint8>::vlanes())
            {
                v_uint8 vmask = (v_eq(vx_load((uchar*)(src_data + j)), v_zero));
                if (v_check_any(vmask))
                {
                    j += v_scan_forward(vmask);
                    return j;
                }
            }
        }
#endif
        for (; j < width && src_data[j]; ++j)
            ;

        return j;
    }

    /**
 * @brief Generates a 64-bit hash for a contour invariant to starting point.
 * * Logic:
 * 1. Symmetry Breaking: pt.x and pt.y are combined using a large prime multiplier.
 * 2. High Entropy: Each point is scrambled via SplitMix64 to ensure bit dispersion.
 * 3. Commutativity: Point hashes are summed (+) to ensure starting-point invariance.
 * 4. Structural Integrity: The total point count is mixed into the final hash
 * to prevent collisions between contours with different lengths but similar sums.
 */
    static uint64_t __uHashContour(const std::vector<cv::Point>& contour) {
        auto hash_mix = [](uint64_t x) -> uint64_t {
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            x = x ^ (x >> 31);
            return x;
        };

        uint64_t combinedPointHash = 0;
        for (const auto& pt : contour) {
            // Map (x, y) to a unique 64-bit seed
            uint64_t seed = (static_cast<uint64_t>(pt.x) * 0x1f1f1f1f1f1f1f1fULL) ^ static_cast<uint64_t>(pt.y);
            combinedPointHash += hash_mix(seed);
        }

        // Mix the contour size with the accumulated point hash
        // We mix the size first so it acts as a unique 'salt' for the final transform
        uint64_t finalHash = hash_mix(combinedPointHash ^ hash_mix(contour.size()));

        return finalHash;
    }

private:
    cv::Mat padded_;
    const std::vector<cv::Range>& ranges_;
    AccumulatorType& accumulator_;
    size_t minSize_;
    size_t step_;
    int offsets_[16];

    // 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE (CCW Rotation)
    const int dx_[8] = {  1,  1,  0, -1, -1, -1,  0,  1 };
    const int dy_[8] = {  0, -1, -1, -1,  0,  1,  1,  1 };
    // Constants defined once
    const uchar FOREGROUND = 255;
    const uchar BACKGROUND = 0;
    const uchar VISITED_OUTER_RIGHT    = 100;
    const uchar VISITED_    = 200;
};



// ==========================================================
// 1. The Core Implementation (Operates on std::vector directly)
// ==========================================================
void __findTRUContoursImpl(cv::Mat& padded,
                           std::vector<std::vector<cv::Point>>& outContours,
                           int minSize,int nthreads)
{
    // 1. Load Balancing Logic
    std::vector<cv::Range> balancedRanges;
    if (nthreads > 1 ) {
        int rowsPerThread = (padded.rows - 2) / nthreads;
        int remainingRows = (padded.rows - 2) % nthreads;
        int currentRow = 1;
        for (int t = 0; t < nthreads; ++t) {
            int startRow = currentRow;
            int endRow = startRow + rowsPerThread + (t < remainingRows ? 1 : 0);
            balancedRanges.emplace_back(startRow, endRow);
            currentRow = endRow;
        }
    }
    else {
        balancedRanges.emplace_back(1, padded.rows - 1);
    }
    // 2. Parallel Execution
    std::vector<std::vector<std::vector<cv::Point>>> threadAccumulators(balancedRanges.size());
    TRUCOntourTracer worker(padded, balancedRanges, threadAccumulators, minSize);
    cv::parallel_for_(cv::Range(0, (int)balancedRanges.size()), worker, nthreads);


    // 3. ZERO-COPY MERGE
    // Calculate total size
    size_t totalContours = 0;
    for (const auto& vec : threadAccumulators) totalContours += vec.size();

    outContours.clear();
    outContours.reserve(totalContours);

    for (auto& tVec : threadAccumulators) {
        // move_iterator moves the vector internals (pointers) without copying pixel data
        outContours.insert(outContours.end(),
                           std::make_move_iterator(tVec.begin()),
                           std::make_move_iterator(tVec.end()));
    }

}

// ==========================================================
// 2. The Overload for Standard Vector (FAST PATH)
// ==========================================================
void  findTRUContours(InputArray _src,
                     std::vector<std::vector<cv::Point>>& contours,
                     int minsize,int nthreads,
                     InputOutputArray _buffer )
{
    CV_INSTRUMENT_REGION();
    Mat src = _src.getMat();
    CV_Assert(!src.empty() && src.type() == CV_8UC1);
    if (nthreads <= 0) nthreads = cv::getNumThreads();

    // Buffer handling
    cv::Mat padded;
    if (_buffer.needed()) {
        cv::copyMakeBorder(src, _buffer, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
        padded = _buffer.getMat();

    } else {
        cv::copyMakeBorder(src, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    }

    // Call Core Logic directly
    __findTRUContoursImpl(padded, contours, minsize,nthreads);
}

// ==========================================================
// 3. The Overload for Generic OutputArray (SLOW PATH / COMPATIBILITY)
// ==========================================================
void  findTRUContours(InputArray _src, OutputArrayOfArrays _contours, int minsize,int nthreads, InputOutputArray _buffer )
{

    std::vector<std::vector<cv::Point>> tempContours;

    // Call the fast version using our temp vector
    findTRUContours(_src, tempContours, minsize,nthreads, _buffer);

    // Copy to OutputArray (The unavoidable copy if using generic API)
    _contours.create((int)tempContours.size(), 1, 0, -1, true);
    for (size_t i = 0; i < tempContours.size(); i++) {
        _contours.create((int)tempContours[i].size(), 1, CV_32SC2, (int)i, true);
        Mat m = _contours.getMat((int)i);
        Mat(tempContours[i]).copyTo(m);
    }
}


}
