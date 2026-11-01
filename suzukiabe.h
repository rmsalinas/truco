#pragma once
// Suzuki-Abe border following with hierarchy, as in the original paper:
//   S. Suzuki, K. Abe, "Topological structural analysis of digitized binary images by border
//   following", CVGIP 30 (1985) 32-46.
//
//   sa::findContoursSuzukiAbe(img, contours, hierarchy)   with hierarchy (layout of RETR_TREE)
//   sa::findContoursSuzukiAbe(img, contours)              without hierarchy, contours in scan order
//
// One raster scan of an integer label image (frame of background, foreground = 1). Every border is
// followed exactly once, when the scan finds its start pixel; the visited pixels are labeled with
// NBD (or -NBD for east pixels) and the points are stored as they are visited. The parent of a new
// border is obtained from LNBD (the last border met on the current row) with the table of the paper.
// No border is traced twice and no chain code is decoded afterwards.
// The scan skips the runs of background pixels and the interiors of the runs of foreground pixels
// (value 1 with non-zero neighbours, where the scan takes no action) with vector instructions, as
// cv::findContours does; this changes only the speed, not the visited pixels or the output.
//
// The output is the same as cv::findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE):
// same point sequences (same start pixel, direction and stopping rule) and the same layout of the
// tree (every new child is prepended to its parent's list, pre-order traversal).
// Input: CV_8UC1, any non-zero pixel is foreground.
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <vector>
#include <algorithm>
#include <cstring>

namespace sa {

namespace detail {

// 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE (y down)
static constexpr int DX[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
static constexpr int DY[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

// Follows the border that starts at label index `start` (row r, column c of the padded image).
// The first examined neighbour is SW for an outer border and NE for a hole border, which is the
// counter-clockwise neighbour that follows (i2, j2) after the clockwise search of step 3.1.
// Labels the visited pixels with nbd / -nbd (steps 3.3-3.4) and appends the points (unpadded).
inline void followBorder(int* f, const int offsets[16], size_t start, int r, int c, bool isHole, int nbd,
                         std::vector<cv::Point>& pts)
{
    pts.clear();
    size_t curr = start;
    int x = c, y = r;
    int search = isHole ? 1 : 5;
    int startDir = -1, dir = -1;
    bool firstMove = true;
    for (;;) {
        pts.emplace_back(x - 1, y - 1);
        dir = -1;
        for (int n = 0; n < 8; ++n) {
            const int idx = search + n;
            if (f[curr + offsets[idx]] == 0) continue;
            dir = idx & 7;
            break;
        }
        if (dir < 0) {                                   // isolated pixel (step 3.1)
            f[curr] = -nbd;
            return;
        }
        // step 3.4: east pixel if the eastern neighbour is a 0-pixel examined during the search
        if (search <= 1 || dir <= search - 2) f[curr] = -nbd;
        else if (f[curr] == 1)                f[curr] = nbd;
        // step 3.5: back at the start pixel, about to repeat the first move
        if (curr == start && !firstMove && dir == startDir) break;
        if (firstMove) { startDir = dir; firstMove = false; }
        curr += offsets[dir];
        x += DX[dir];
        y += DY[dir];
        search = (dir + 6) & 7;
    }
    if (pts.size() > 1 && pts.back() == pts.front()) pts.pop_back();
}

// First index j' >= j with row[j'] != val, or end if there is none.
inline int skipEqual(const int* row, int j, int end, int val)
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const cv::v_int32 vval = cv::vx_setall_s32(val);
    const int n = cv::VTraits<cv::v_int32>::vlanes();
    for (; j <= end - n; j += n) {
        cv::v_int32 m = cv::v_ne(cv::vx_load(row + j), vval);
        if (cv::v_check_any(m)) return j + cv::v_scan_forward(m);
    }
#endif
    for (; j < end && row[j] == val; ++j) ;
    return j;
}

} // namespace detail

// The scan: contours in scan order and, if withParents, the NBD of the parent of every contour
// (1: frame, i.e. root) indexed by NBD - 1.
inline void scanSuzukiAbe(const cv::Mat& src, std::vector<std::vector<cv::Point>>& found,
                          std::vector<int>& parent, bool withParents)
{
    CV_Assert(!src.empty() && src.type() == CV_8UC1);
    const int H = src.rows, W = src.cols;
    const int cols = W + 2;

    // integer label image with a frame of background: f = 1 for foreground, 0 for background
    cv::Mat lab(H + 2, cols, CV_32SC1);
    int* f = lab.ptr<int>();
    const size_t step = lab.step / sizeof(int);
    std::memset(lab.ptr<int>(0), 0, step * sizeof(int));
    std::memset(lab.ptr<int>(H + 1), 0, step * sizeof(int));
    cv::Mat ones(1, W, CV_8UC1);
    for (int y = 0; y < H; ++y) {
        int* d = lab.ptr<int>(y + 1);
        d[0] = 0; d[cols - 1] = 0;
        cv::min(src.row(y), 1, ones);                                   // vectorized f = (src != 0)
        ones.convertTo(cv::Mat(1, W, CV_32SC1, d + 1), CV_32S);
    }

    int offsets[16];
    {
        const int s = (int)step;
        const int o[8] = { 1, -s + 1, -s, -s - 1, -1, s - 1, s, s + 1 };
        for (int i = 0; i < 16; ++i) offsets[i] = o[i & 7];
    }

    // border NBD (>= 2) is contour NBD - 2; NBD = 1 is the frame (a hole border, no parent)
    found.clear();
    std::vector<char> isHole(1, 1);     // indexed by NBD - 1
    parent.assign(1, -1);              // indexed by NBD - 1, value: parent NBD (1 = frame) or -1
    std::vector<cv::Point> buffer;
    buffer.reserve(4096);
    int nbd = 1;

    for (int r = 1; r <= H; ++r) {
        int lnbd = 1;
        int* row = f + (size_t)r * step;
        for (int c = 1; c <= W; ++c) {
            const int v = row[c];
            if (v == 0) {                                   // run of background; row[W + 1] = 0
                c = detail::skipEqual(row, c + 1, W + 1, 0) - 1;
                continue;
            }
            if (v == 1 && row[c - 1] != 0 && row[c + 1] == 1) {
                // interior of a run of 1s: no action until the last 1, whose right neighbour may be 0
                c = detail::skipEqual(row, c + 1, W + 1, 1) - 2;
                continue;
            }
            bool hole;
            if (v == 1 && row[c - 1] == 0) hole = false;                        // step 1 (a)
            else if (v >= 1 && row[c + 1] == 0) { hole = true; if (v > 1) lnbd = v; }   // step 1 (b)
            else { if (v != 1) lnbd = v < 0 ? -v : v; continue; }              // step 1 (c) -> step 4
            // step 2: parent from the type of border LNBD (table 1 of the paper)
            ++nbd;
            if (withParents) {
                const int b = lnbd - 1;
                const int p = (isHole[b] != (char)hole) ? lnbd : parent[b];
                isHole.push_back((char)hole);
                parent.push_back(p);
            }
            // step 3
            detail::followBorder(f, offsets, (size_t)r * step + c, r, c, hole, nbd, buffer);
            found.emplace_back(buffer.begin(), buffer.end());
            // step 4
            const int nv = row[c];
            if (nv != 1) lnbd = nv < 0 ? -nv : nv;
        }
    }
}

// Contours and hierarchy, same output as cv::findContours(RETR_TREE, CHAIN_APPROX_NONE)
inline void findContoursSuzukiAbe(const cv::Mat& src, std::vector<std::vector<cv::Point>>& contours,
                                  std::vector<cv::Vec4i>& hierarchy)
{
    std::vector<std::vector<cv::Point>> found;
    std::vector<int> parent;
    scanSuzukiAbe(src, found, parent, true);

    // tree with the layout of cv::findContours: contours in scan order, every new child prepended
    // to its parent's list, output in pre-order
    const int n = (int)found.size();
    std::vector<int> firstChild(n + 1, -1), next(n, -1), treeParent(n, -1);   // index n: virtual root
    for (int i = 0; i < n; ++i) {
        const int pn = parent[i + 1];                    // NBD of the parent; 1 or -1: root
        const int p = pn >= 2 ? pn - 2 : -1;
        const int pp = p < 0 ? n : p;
        treeParent[i] = p;
        next[i] = firstChild[pp];
        firstChild[pp] = i;
    }
    std::vector<int> newIdx(n, -1), preorder, stack;
    preorder.reserve(n);
    auto pushChildren = [&](int node) {
        const size_t mark = stack.size();
        for (int ch = firstChild[node]; ch >= 0; ch = next[ch]) stack.push_back(ch);
        std::reverse(stack.begin() + mark, stack.end());
    };
    pushChildren(n);
    while (!stack.empty()) {
        const int i = stack.back(); stack.pop_back();
        newIdx[i] = (int)preorder.size();
        preorder.push_back(i);
        pushChildren(i);
    }

    contours.clear();
    contours.reserve(n);
    hierarchy.assign(n, cv::Vec4i(-1, -1, -1, -1));
    for (int ni = 0; ni < n; ++ni) {
        const int i = preorder[ni];
        cv::Vec4i& hv = hierarchy[ni];
        hv[0] = next[i] < 0 ? -1 : newIdx[next[i]];
        if (hv[0] >= 0) hierarchy[hv[0]][1] = ni;
        hv[2] = firstChild[i] < 0 ? -1 : newIdx[firstChild[i]];
        hv[3] = treeParent[i] < 0 ? -1 : newIdx[treeParent[i]];
        contours.push_back(std::move(found[i]));
    }
}

// Contours only, in scan order (raster order of their start pixels)
inline void findContoursSuzukiAbe(const cv::Mat& src, std::vector<std::vector<cv::Point>>& contours)
{
    std::vector<int> parent;
    scanSuzukiAbe(src, contours, parent, false);
}

} // namespace sa
