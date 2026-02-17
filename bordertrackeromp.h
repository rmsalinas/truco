//This is a rework of the code from
// Parallel border tracking in binary images for multicore computers, Garcia-Molla, Victor and Alonso-Jordá, Pedro,  Parallel border tracking in binary images for multicore computers, 2023
//
// The original code is very inneficient and not adapted to
// for OpenCv. With the help of Gemini 3 Pro we were able to produce this faster and cleaner version


#ifndef BORDER_TRACKER_OMP_H
#define BORDER_TRACKER_OMP_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <omp.h>
namespace btomp{
// Internal structures used by the algorithm
struct coord {
    int i;
    int j;
};

struct VecCont {
    coord act;  // Current point coordinates
    coord sig;  // Next point coordinates
    coord ant;  // Previous point coordinates
    int next;   // Index of next point in vector
};

enum ContourState { OPEN_CONTOUR, CLOSED_CONTOUR, COVERED_CONTOUR };

struct IndCont {
    int ini;    // Index of initial point
    int fin;    // Index of final point
    ContourState sts;    // contour state
};

class BorderTrackerOMP {
public:
    // Constructor allocates memory based on max expected image size
    BorderTrackerOMP(int width, int height);
    ~BorderTrackerOMP();

    // Main entry point: Process image and return contours
    // Expects a binary or grayscale image (will be thresholded if not binary)
    std::vector<std::vector<cv::Point>> findBorders(const cv::Mat& inputImg);

    static std::vector<std::vector<cv::Point>> findContours(const cv::Mat& inputImg );
private:
    // Constants
    static const int N_BLOCKS_ROWS = 32;
    static const int N_BLOCKS_COLS = 32;
    static const int MAX_N_BORDS = 5000;
    static const int MAX_N_CONTOURS = 10000000;
    // Dimensions
    int M, N;   // Original dims
    int Mg, Ng; // Padded dims
    int mfb, nfb; // Blocks
    int numfbl, numcbl; // Block sizes

    // Memory Buffers
    unsigned char* d_A;          // Padded Image
    unsigned char* d_is_bord;    // Border map
    int* d_numconts;             // Contours per block
    int* d_numconts_aux;         // Aux for merging
    int* d_marcado;              // Marker array
    int* d_numconts_glob;        // Global contour count

    VecCont* d_vec_conts;        // Vector of points
    IndCont* d_ind_conts;        // Vector of contour headers
    IndCont* d_ind_conts_glob;   // Global contour headers
    IndCont* d_ind_conts_aux;    // Aux for merging

    // Helper methods (Logic adapted from original C functions)
    void allocateMemory();
    void freeMemory();

    // Core Algorithms
    void preprocessing_cpu();
    void parallel_tracking_cpu();

    void vertical_connection(int num_max_conts, int* numconts, VecCont* vec_conts,
                             IndCont* ind_conts, int* numconts_out, IndCont* ind_conts_out,
                             int* marcado, int current_mbn);

    void horizontal_connection(int num_max_conts, int* numconts, VecCont* vec_conts,
                               IndCont* ind_conts, int* numconts_out, IndCont* ind_conts_out,
                               int* marcado, int current_nbn);

    // Low level helpers
    void track_fw_bkw(int* i_vec_conts_ini, int* numconts_ptr, VecCont* vec_conts,
                      IndCont* ind_conts, int i_ind_conts,
                      int i_ini, int j_ini, int i_fin, int j_fin,
                      coord c_ini_ant, coord c_ini_act, coord c_ini_sig);

    void copy_input_to_padded(const cv::Mat& gray);
};
}
#endif // BORDER_TRACKER_OMP_H
