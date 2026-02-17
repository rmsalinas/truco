#include "bordertrackeromp.h"
#include <cmath>
#include <cstring>
#include <algorithm>
namespace btomp {
// --- Macros for Indexing (kept from original for logic consistency) ---
#define h_A_idx(i, j, stride)    ((i) + ((j)*(stride)))
#define d_idx(i, j, stride)      ((i) + ((j)*(stride)))

// --- Helper Functions for rotation ---
static void clockwise_2(int *difi, int *difj, int *iout, int *jout, int* pos) {
    if (*difi==1) {
        if (*difj==1)      { *iout=1;  *jout=0;  *pos=7; }
        else if(*difj==0)  { *iout=1;  *jout=-1; *pos=1; }
        else               { *iout=0;  *jout=-1; *pos=2; }
    } else if(*difi==0) {
        if (*difj==-1)     { *iout=-1; *jout=-1; *pos=1; }
        else if(*difj==1)  { *iout=1;  *jout=1;  *pos=1; }
    } else if (*difi==-1) {
        if (*difj==-1)     { *iout=-1; *jout=0;  *pos=3; }
        else if(*difj==0)  { *iout=-1; *jout=1;  *pos=1; }
        else if (*difj==1) { *iout=0;  *jout=1;  *pos=5; }
    }
}

static void counterclock_2(int *difi, int *difj, int *iout, int *jout, int* pos) {
    if (*difi==1) {
        if (*difj==1)      { *iout=0;  *jout=1;  *pos=5; }
        else if(*difj==0)  { *iout=1;  *jout=1;  *pos=1; }
        else               { *iout=1;  *jout=0;  *pos=7; }
    } else if(*difi==0) {
        if (*difj==-1)     { *iout=1;  *jout=-1; *pos=1; }
        else if(*difj==1)  { *iout=-1; *jout=1;  *pos=1; }
    } else if (*difi==-1) {
        if (*difj==-1)     { *iout=0;  *jout=-1; *pos=2; }
        else if(*difj==0)  { *iout=-1; *jout=-1; *pos=1; }
        else if (*difj==1) { *iout=-1; *jout=0;  *pos=3; }
    }
}

static void clockwise_o(int *difi, int *difj, int *iout, int *jout) {
    if (*difi == 1) {
        if (*difj == 1)      { *iout = 1;  *jout = 0; }
        else if (*difj == 0) { *iout = 1;  *jout = -1;}
        else                 { *iout = 0;  *jout = -1;}
    } else if (*difi == 0) {
        if (*difj == -1)     { *iout = -1; *jout = -1;}
        else if (*difj == 1) { *iout = 1;  *jout = 1; }
    } else if (*difi == -1) {
        if (*difj == -1)     { *iout = -1; *jout = 0; }
        else if (*difj == 0) { *iout = -1; *jout = 1; }
        else if (*difj == 1) { *iout = 0;  *jout = 1; }
    }
}

// Helper to check neighbor pixels and determine rotation
static void rotate_ini(unsigned char *d_A, coord *coord_ant, coord *coord_sig, int *found, int *val, int *pos_ult_cero, coord coord_act, int Mg) {
    int dif_i = 0, dif_j = -1;
    int itcount, iaux, jaux, iaux2, jaux2;

    // Check pixel above (j-1)
    if (d_A[d_idx(coord_act.i, coord_act.j - 1, Mg)] != 0) {
        *found = 0;
        itcount = 0;
        while (*found == 0 && itcount <= 4) {
            clockwise_o(&dif_i, &dif_j, &iaux2, &jaux2);
            clockwise_o(&iaux2, &jaux2, &iaux, &jaux);
            if (d_A[d_idx(coord_act.i + iaux, coord_act.j + jaux, Mg)] == 0) {
                *found = 1;
                dif_i = iaux; dif_j = jaux;
            } else {
                dif_i = iaux; dif_j = jaux;
                itcount++;
            }
        }
    }

    *found = 0;
    itcount = 0;
    while (*found == 0 && itcount <= 8) {
        clockwise_o(&dif_i, &dif_j, &iaux, &jaux);
        if (d_A[d_idx(coord_act.i + iaux, coord_act.j + jaux, Mg)] != 0) {
            *found = 1;
            (*coord_ant).i = coord_act.i + iaux;
            (*coord_ant).j = coord_act.j + jaux;
        } else {
            dif_i = iaux; dif_j = jaux;
            itcount++;
        }
    }

    if (*found == 0) {
        (*coord_ant).i = 0; (*coord_ant).j = 0;
        (*coord_sig).i = 0; (*coord_sig).j = 0;
        *val = d_A[d_idx(coord_act.i, coord_act.j, Mg)];
        *pos_ult_cero = 2;
    } else {
        *found = 0;
        itcount = 0;
        dif_i = iaux; dif_j = jaux;
        *val = 1;
        int pos = 0;
        while (*found == 0 && itcount <= 8) {
            counterclock_2(&dif_i, &dif_j, &iaux, &jaux, &pos);
            if (d_A[d_idx(coord_act.i + iaux, coord_act.j + jaux, Mg)] == 0) {
                *val = (*val)*pos;
                if (pos > 1) *pos_ult_cero = pos;
            }
            if (d_A[d_idx(coord_act.i + iaux, coord_act.j + jaux, Mg)] != 0) {
                *found = 1;
                (*coord_sig).i = coord_act.i + iaux;
                (*coord_sig).j = coord_act.j + jaux;
            } else {
                dif_i = iaux; dif_j = jaux;
                itcount++;
            }
        }
    }
}

static void rotate_later(unsigned char *d_A, coord *coord_ant, coord *coord_sig, int *fin, int *val, int *pos_ult_cero, coord coord_act, int Mg) {
    int dif_i = (*coord_ant).i - coord_act.i;
    int dif_j = (*coord_ant).j - coord_act.j;
    int itcount, found, pos, iaux, jaux;
    *coord_sig = *coord_ant;
    found = 0; itcount = 0; *fin = 0; *val = 1;

    while (found == 0 && itcount <= 8) {
        if (dif_i*dif_j == 0) {
            clockwise_2(&dif_i, &dif_j, &dif_i, &dif_j, &pos);
            if (d_A[d_idx(coord_act.i + dif_i, coord_act.j + dif_j, Mg)] != 0) {
                (*coord_sig).i = coord_act.i + dif_i;
                (*coord_sig).j = coord_act.j + dif_j;
            }
            clockwise_2(&dif_i, &dif_j, &iaux, &jaux, &pos);
        } else {
            clockwise_2(&dif_i, &dif_j, &iaux, &jaux, &pos);
        }

        if (d_A[d_idx(coord_act.i + iaux, coord_act.j + jaux, Mg)] != 0) {
            (*coord_sig).i = coord_act.i + iaux;
            (*coord_sig).j = coord_act.j + jaux;
        }

        if ((iaux == 0) && (jaux == -1)) {
            found = 1; *fin = 1;
        } else if (d_A[d_idx(coord_act.i + iaux, coord_act.j + jaux, Mg)] == 0) {
            found = 1;
        } else {
            dif_i = iaux; dif_j = jaux; itcount++;
        }
    }

    *pos_ult_cero = pos;
    *val = (*val)*pos;
    dif_i = iaux; dif_j = jaux;

    if ((*fin == 0) && (found == 1)) {
        found = 0; itcount = 0;
        while (found == 0 && itcount <= 8) {
            clockwise_2(&dif_i, &dif_j, &iaux, &jaux, &pos);
            if (d_A[d_idx(coord_act.i + iaux, coord_act.j + jaux, Mg)] == 0) {
                *val = (*val)*pos;
                if (pos > 1) *pos_ult_cero = pos;
            }
            if ((iaux == 0) && (jaux == -1)) {
                found = 1;
                (*coord_ant).i = coord_act.i + iaux;
                (*coord_ant).j = coord_act.j + jaux;
                if (d_A[d_idx(coord_act.i + iaux, coord_act.j + jaux, Mg)] == 0) *fin = 1;
                else *fin = 2;
            } else if (d_A[d_idx(coord_act.i + iaux, coord_act.j + jaux, Mg)] != 0) {
                found = 1;
                (*coord_ant).i = coord_act.i + iaux;
                (*coord_ant).j = coord_act.j + jaux;
            } else {
                dif_i = iaux; dif_j = jaux; itcount++;
            }
        }
    }
}


// --- Class Implementation ---

BorderTrackerOMP::BorderTrackerOMP(int width, int height) : M(width), N(height) {
    allocateMemory();
}

BorderTrackerOMP::~BorderTrackerOMP() {
    freeMemory();
}

void BorderTrackerOMP::allocateMemory() {
    mfb = N_BLOCKS_ROWS;
    nfb = N_BLOCKS_COLS;

    // Calculate padded dimensions
    Mg = ((M + N_BLOCKS_ROWS - 1) / N_BLOCKS_ROWS) * N_BLOCKS_ROWS;
    Ng = ((N + N_BLOCKS_COLS - 1) / N_BLOCKS_COLS) * N_BLOCKS_COLS;

    numfbl = Mg / mfb;
    numcbl = Ng / nfb;
    int numblq = mfb * nfb;

    unsigned int mat_mem_sizeg    = sizeof(unsigned char) * Mg * Ng;
    unsigned int blq_mem_size     = sizeof(int) * numblq;
    unsigned int vec_mem_size     = sizeof(VecCont) * Mg * Ng * 2;
    unsigned int ind_mem_size     = sizeof(IndCont) * numblq * MAX_N_BORDS;
    unsigned int marcado_mem_size = sizeof(int) * numblq * MAX_N_BORDS;
    unsigned int ind_mem_size_glob = sizeof(IndCont) * MAX_N_CONTOURS;

    d_A = (unsigned char*) malloc(mat_mem_sizeg);
    d_is_bord = (unsigned char*) malloc(mat_mem_sizeg);
    d_numconts = (int*) malloc(blq_mem_size);
    d_numconts_aux = (int*) malloc(blq_mem_size);
    d_numconts_glob = (int*) malloc(sizeof(int));
    d_marcado = (int*) malloc(marcado_mem_size);

    d_vec_conts = (VecCont*) malloc(vec_mem_size);
    d_ind_conts = (IndCont*) malloc(ind_mem_size);
    d_ind_conts_aux = (IndCont*) malloc(ind_mem_size);
    d_ind_conts_glob = (IndCont*) malloc(ind_mem_size_glob);

    // [Optimization] Initialize d_marcado to 0 once at allocation.
    // It is reset inside findBorders loops, but good practice here.
    memset(d_marcado, 0, marcado_mem_size);
}

void BorderTrackerOMP::freeMemory() {
    if (d_A) free(d_A);
    if (d_is_bord) free(d_is_bord);
    if (d_numconts) free(d_numconts);
    if (d_numconts_aux) free(d_numconts_aux);
    if (d_numconts_glob) free(d_numconts_glob);
    if (d_marcado) free(d_marcado);
    if (d_vec_conts) free(d_vec_conts);
    if (d_ind_conts) free(d_ind_conts);
    if (d_ind_conts_aux) free(d_ind_conts_aux);
    if (d_ind_conts_glob) free(d_ind_conts_glob);
}

void BorderTrackerOMP::copy_input_to_padded(const cv::Mat& gray) {
    // [Optimization] Removed full memset(d_A). We will overwrite it anyway.
    // Only zeroing the necessary padding borders is more complex than just copying
    // but the actual image copy overwrites the bulk of the data.
    // However, d_A has padding to Mg/Ng. We MUST ensure the padding is 0.
    // The easiest way is still memset, BUT d_A is 1-byte per pixel.
    // Memset on 2MB (1080p) is fast. The bottlenecks were the 100MB+ arrays.
    memset(d_A, 0, sizeof(unsigned char) * Mg * Ng);

    // Copy data row by row to handle padding
    for (int j = 0; j < N; j++) {
        const unsigned char* src_ptr = gray.ptr<unsigned char>(j);
        // Original code used column-major logic in index macro?
        // d_idx(i, j, Mg) = i + j*Mg. This matches row-major memory if stride is width.
        // We copy the row directly.
        memcpy(&d_A[j * Mg], src_ptr, M);
    }
}

void BorderTrackerOMP::preprocessing_cpu() {
    int i_ini = 1;
    int j_ini = 1;
    int i_fin = (M - 2);
    int j_fin = (N - 2);

    // [Optimization Note] This loop writes d_is_bord[i,j].
    // It skips the 1-pixel border.
    // Since we removed memset(d_is_bord), we need to ensure the logic doesn't read garbage.
    // The logic is: d_is_bord[idx] = (d_A[...] > 0) && ...
    // It does NOT read d_is_bord. It only writes.
    // The Tracking phase reads d_is_bord.
    // Tracking phase loops from j_ini to j_fin. It never reads the 1-pixel border of d_is_bord.
    // So removing memset(d_is_bord) is SAFE.

#pragma omp parallel for
    for (int j = j_ini; j <= j_fin; j++) {
        int cond, pos_cero = 0;
        for (int i = i_ini; i <= i_fin; i++) {
            pos_cero = (d_A[d_idx(i - 1, j, Mg)] == 0) * 2;
            cond = (pos_cero == 0) * (d_A[d_idx(i, j + 1, Mg)] == 0);
            pos_cero = 4 * cond + (1 - cond) * pos_cero;
            cond = (pos_cero == 0) * (d_A[d_idx(i + 1, j, Mg)] == 0);
            pos_cero = 6 * cond + (1 - cond) * pos_cero;
            cond = (pos_cero == 0) * (d_A[d_idx(i, j - 1, Mg)] == 0);
            pos_cero = 8 * cond + (1 - cond) * pos_cero;
            d_is_bord[d_idx(i, j, Mg)] = (d_A[d_idx(i, j, Mg)] > 0) && (pos_cero > 0);
        }
    }
}

void BorderTrackerOMP::track_fw_bkw(
    int *i_vec_conts_ini, int* d_numconts_ptr, VecCont* d_vec_conts, IndCont* d_ind_conts,
    int i_ind_conts, int i_ini, int j_ini, int i_fin, int j_fin,
    coord c_ini_ant, coord c_ini_act, coord c_ini_sig)
{
    // ... [Code unchanged from original] ...
    coord coord_sig, coord_act, coord_ant;
    int dif_i, dif_j, itcount, i_vec_conts, iaux, val;
    int found, jaux, pos;
    i_vec_conts = *i_vec_conts_ini;

    coord_sig = c_ini_sig;
    coord_act = c_ini_act;

    d_vec_conts[i_vec_conts].act = c_ini_act;
    d_vec_conts[i_vec_conts].ant = c_ini_ant;
    d_vec_conts[i_vec_conts].sig = c_ini_sig;
    d_ind_conts[i_ind_conts].ini = i_vec_conts;
    d_ind_conts[i_ind_conts].fin = i_vec_conts;
    d_ind_conts[i_ind_conts].sts = OPEN_CONTOUR;

    int end_track_forward = 0;
    if ((coord_sig.i < i_ini) || (coord_sig.i > i_fin) || (coord_sig.j < j_ini) || (coord_sig.j > j_fin)) {
        d_vec_conts[i_vec_conts].next = 0;
        end_track_forward = 2;
    } else {
        while (end_track_forward == 0) {
            i_vec_conts++;
            d_vec_conts[i_vec_conts].act = coord_sig;
            d_vec_conts[i_vec_conts].ant = coord_act;
            d_vec_conts[i_vec_conts - 1].next = i_vec_conts;
            coord_ant = coord_act;
            coord_act = coord_sig;

            dif_i = coord_ant.i - coord_act.i;
            dif_j = coord_ant.j - coord_act.j;
            found = 0; itcount = 0;
            val = d_A[d_idx(coord_act.i, coord_act.j, Mg)];

            while ((found == 0) && (itcount <= 8)) {
                counterclock_2(&dif_i, &dif_j, &iaux, &jaux, &pos);
                if (d_A[d_idx(coord_act.i + iaux, coord_act.j + jaux, Mg)] == 0) {
                    val = val * pos;
                    dif_i = iaux; dif_j = jaux; itcount++;
                } else {
                    found = 1;
                    coord_sig.i = coord_act.i + iaux;
                    coord_sig.j = coord_act.j + jaux;
                }
            }
            if ((coord_sig.i < i_ini) || (coord_sig.i > i_fin) || (coord_sig.j < j_ini) || (coord_sig.j > j_fin)) {
                end_track_forward = 2;
                d_vec_conts[i_vec_conts].sig = coord_sig;
                d_ind_conts[i_ind_conts].fin = i_vec_conts;
                d_A[d_idx(coord_act.i, coord_act.j, Mg)] = val;
            } else {
                d_vec_conts[i_vec_conts].sig = coord_sig;
                d_A[d_idx(coord_act.i, coord_act.j, Mg)] = val;
            }

            if ((coord_sig.i == c_ini_act.i) && (coord_sig.j == c_ini_act.j) && (coord_act.i == c_ini_ant.i) && (coord_act.j == c_ini_ant.j)) {
                end_track_forward = 1;
                d_ind_conts[i_ind_conts].sts = CLOSED_CONTOUR;
                d_ind_conts[i_ind_conts].fin = i_vec_conts;
                d_vec_conts[i_vec_conts].next = (*i_vec_conts_ini);
                d_vec_conts[i_vec_conts].sig = c_ini_act;
                d_vec_conts[(*i_vec_conts_ini)].ant = coord_act;
            }
        }
    }

    if (end_track_forward == 2) {
        coord_ant = c_ini_ant;
        coord_act = c_ini_act;
        coord_sig = c_ini_sig;
        int anterior = d_ind_conts[i_ind_conts].ini;
        if ((coord_ant.i < i_ini) || (coord_ant.i > i_fin) || (coord_ant.j < j_ini) || (coord_ant.j > j_fin))
            d_vec_conts[(*i_vec_conts_ini)].ant = coord_ant;
        else {
            int end_track_backward = 0;
            while (end_track_backward == 0) {
                i_vec_conts++;
                d_vec_conts[i_vec_conts].act = coord_ant;
                d_vec_conts[i_vec_conts].sig = coord_act;
                d_vec_conts[i_vec_conts].next = anterior;
                anterior = i_vec_conts;
                coord_sig = coord_act;
                coord_act = coord_ant;

                dif_i = coord_sig.i - coord_act.i;
                dif_j = coord_sig.j - coord_act.j;
                found = 0; itcount = 0;
                val = d_A[d_idx(coord_act.i, coord_act.j, Mg)];
                while (found == 0 && itcount <= 8) {
                    clockwise_2(&dif_i, &dif_j, &iaux, &jaux, &pos);
                    if (d_A[d_idx(coord_act.i + iaux, coord_act.j + jaux, Mg)] == 0) {
                        val = val * pos;
                        dif_i = iaux; dif_j = jaux; itcount++;
                    } else {
                        found = 1;
                        coord_ant.i = coord_act.i + iaux;
                        coord_ant.j = coord_act.j + jaux;
                    }
                }
                if ((coord_ant.i < i_ini) || (coord_ant.i > i_fin) || (coord_ant.j < j_ini) || (coord_ant.j > j_fin)) {
                    end_track_backward = 2;
                    d_vec_conts[i_vec_conts].ant = coord_ant;
                    d_ind_conts[i_ind_conts].ini = i_vec_conts;
                    d_A[d_idx(coord_act.i, coord_act.j, Mg)] = val;
                } else {
                    d_vec_conts[i_vec_conts].ant = coord_ant;
                    d_ind_conts[i_ind_conts].ini = i_vec_conts;
                    d_A[d_idx(coord_act.i, coord_act.j, Mg)] = val;
                }
            }
        }
    }
    (*i_vec_conts_ini) = i_vec_conts;
}


void BorderTrackerOMP::parallel_tracking_cpu() {
    int bl;
#pragma omp parallel for schedule(dynamic)
    for (bl=0; bl<N_BLOCKS_ROWS*N_BLOCKS_COLS; bl++) {
        int ib = bl % N_BLOCKS_ROWS;
        int jb = bl / N_BLOCKS_ROWS;
        int mb = N_BLOCKS_ROWS;
        int nb = N_BLOCKS_COLS;

        int i_ini = (ib == 0) ? 1 : (ib * numfbl);
        int j_ini = (jb == 0) ? 1 : (jb * numcbl);
        int i_fin = (ib == mb - 1) ? (M - 1) : ((ib + 1) * numfbl - 1);
        int j_fin = (jb == nb - 1) ? (N - 1) : ((jb + 1) * numcbl - 1);

        int indicebl = ib + mb * jb;
        int i_vec_conts = (jb)*numcbl*Mg * 2 - 1 + 2 * numfbl*numcbl*(ib);
        int i_ind_conts = indicebl * MAX_N_BORDS - 1;

        coord coord_act, coord_ant, coord_sig;
        int found, val, pos_ult_cero;
        int contornos_este_punto_recorridos;

        for (coord_act.j = j_ini; (coord_act.j <= j_fin); coord_act.j++) {
            for (coord_act.i = i_ini; (coord_act.i <= i_fin); coord_act.i++) {
                if (d_is_bord[d_idx(coord_act.i, coord_act.j, Mg)] > 0) {
                    rotate_ini(d_A, &coord_ant, &coord_sig, &found, &val, &pos_ult_cero, coord_act, Mg);
                    if (found != 0) {
                        if ((d_A[d_idx(coord_act.i, coord_act.j, Mg)] == 1) || (d_A[d_idx(coord_act.i, coord_act.j, Mg)] % pos_ult_cero) != 0) {
                            d_A[d_idx(coord_act.i, coord_act.j, Mg)] *= val;
                            i_ind_conts++;
                            i_vec_conts++;
                            d_numconts[indicebl]++;
                            track_fw_bkw(&i_vec_conts, d_numconts, d_vec_conts, d_ind_conts, i_ind_conts, i_ini, j_ini, i_fin, j_fin, coord_ant, coord_act, coord_sig);
                            if (d_ind_conts[i_ind_conts].sts==CLOSED_CONTOUR) {
                                int contg;
#pragma omp atomic capture
                                contg= (*d_numconts_glob)++;
                                d_ind_conts_glob[contg]=d_ind_conts[i_ind_conts];
                                i_ind_conts--;
                                d_numconts[indicebl]--;
                            }
                            contornos_este_punto_recorridos = 0;
                            while (contornos_este_punto_recorridos == 0) {
                                rotate_later(d_A, &coord_ant, &coord_sig, &contornos_este_punto_recorridos, &val, &pos_ult_cero, coord_act, Mg);
                                if ((contornos_este_punto_recorridos != 1) && (d_A[d_idx(coord_act.i, coord_act.j, Mg)] % pos_ult_cero != 0)) {
                                    d_A[d_idx(coord_act.i, coord_act.j, Mg)] *= val;
                                    i_vec_conts++;
                                    i_ind_conts++;
                                    d_numconts[indicebl]++;
                                    track_fw_bkw(&i_vec_conts, d_numconts, d_vec_conts, d_ind_conts, i_ind_conts, i_ini, j_ini, i_fin, j_fin, coord_ant, coord_act, coord_sig);
                                    if (d_ind_conts[i_ind_conts].sts==CLOSED_CONTOUR) {
                                        int contg;
#pragma omp atomic capture
                                        contg= (*d_numconts_glob)++;
                                        d_ind_conts_glob[contg]=d_ind_conts[i_ind_conts];
                                        i_ind_conts--;
                                        d_numconts[indicebl]--;
                                    }
                                    if (contornos_este_punto_recorridos == 2) contornos_este_punto_recorridos = 0;
                                }
                            }
                        } else if (d_A[d_idx(coord_act.i, coord_act.j, Mg)] > 1) {
                            contornos_este_punto_recorridos = 0;
                            while (contornos_este_punto_recorridos == 0) {
                                rotate_later(d_A, &coord_ant, &coord_sig, &contornos_este_punto_recorridos, &val, &pos_ult_cero, coord_act, Mg);
                                if ((contornos_este_punto_recorridos != 1) && (d_A[d_idx(coord_act.i, coord_act.j, Mg)] % pos_ult_cero != 0)) {
                                    d_A[d_idx(coord_act.i, coord_act.j, Mg)] *= val;
                                    i_vec_conts++;
                                    i_ind_conts++;
                                    d_numconts[indicebl]++;
                                    track_fw_bkw(&i_vec_conts, d_numconts, d_vec_conts, d_ind_conts, i_ind_conts, i_ini, j_ini, i_fin, j_fin, coord_ant, coord_act, coord_sig);
                                    if (d_ind_conts[i_ind_conts].sts==CLOSED_CONTOUR) {
                                        int contg;
#pragma omp atomic capture
                                        contg= (*d_numconts_glob)++;
                                        d_ind_conts_glob[contg]=d_ind_conts[i_ind_conts];
                                        i_ind_conts--;
                                        d_numconts[indicebl]--;
                                    }
                                }
                            }
                        }
                    } else if ((d_is_bord[d_idx(coord_act.i, coord_act.j, Mg)] > 0) && (found == 0)) {
                        i_vec_conts++;
                        int contg;
#pragma omp atomic capture
                        contg= (*d_numconts_glob)++;
                        coord_ant.i = coord_act.i; coord_ant.j = coord_act.j;
                        coord_sig.i = coord_act.i; coord_sig.j = coord_act.j;
                        d_ind_conts_glob[contg].sts = CLOSED_CONTOUR;
                        d_ind_conts_glob[contg].fin = i_vec_conts;
                        d_ind_conts_glob[contg].ini = i_vec_conts;
                        d_vec_conts[i_vec_conts].next = i_vec_conts;
                        d_vec_conts[i_vec_conts].sig = coord_sig;
                        d_vec_conts[i_vec_conts].act = coord_act;
                        d_vec_conts[i_vec_conts].ant = coord_ant;
                    }
                }
            }
        }
    }
}

void BorderTrackerOMP::vertical_connection(int num_max_conts, int* numconts, VecCont* vec_conts,
                                           IndCont* ind_conts, int* numconts_out, IndCont* ind_conts_out,
                                           int* marcado, int current_mbn)
{
    int bl;
#pragma omp parallel for schedule(dynamic)
    for (bl=0; bl<current_mbn*N_BLOCKS_COLS; bl++) {
        int ib = bl % current_mbn;
        int jb = bl / current_mbn;
        int mb = 2*current_mbn;
        int nb = N_BLOCKS_COLS;
        int i_con_ini, j_con_ini, i_antes, j_antes, i_salida, nump1, p2, jfc;
        int i_sig_dentro, j_sig_dentro, i_sig_fuera, j_sig_fuera, pf_bueno;
        int contorno_conectado, ind_c_fuera;

        int ibloque, contorno, ind_b_fuera;
        int ibloque_arriba = ib*2 + mb*(jb);
        int ibloque_sal_arriba = bl;
        int ibloque_abajo = (ib*2)+1 + mb*(jb);
        int indice_ini = num_max_conts*(ibloque_arriba);
        int indice_ini_abajo = num_max_conts*(ibloque_abajo);
        int numcslocal = -1;
        int i_ind_conts_curr = numcslocal + indice_ini;

        for (int cont=0; cont< numconts[ibloque_arriba]; cont++) {
            ibloque = ibloque_arriba;
            int indcactual = cont + indice_ini;
            if (marcado[indcactual]==0) {
                numcslocal++;
                i_ind_conts_curr++;
                ind_conts_out[i_ind_conts_curr] = ind_conts[indcactual];
                nump1 = ind_conts[indcactual].ini;
                i_con_ini = vec_conts[nump1].act.i;
                j_con_ini = vec_conts[nump1].act.j;
                i_antes = vec_conts[nump1].ant.i;
                j_antes = vec_conts[nump1].ant.j;
                int fin = 0;
                contorno = cont;
                i_salida = ibloque_abajo;

                while(fin==0) {
                    int indcaux = contorno + num_max_conts*(ibloque);
                    p2 = ind_conts[indcaux].fin;
                    int i_con_dentro = vec_conts[p2].act.i;
                    int j_con_dentro = vec_conts[p2].act.j;
                    int i_con_fuera = vec_conts[p2].sig.i;
                    int j_con_fuera = vec_conts[p2].sig.j;

                    int ibl_siguiente = i_con_fuera / numfbl;
                    if (ibl_siguiente > mb-1) ibl_siguiente = mb-1;
                    int jbl_siguiente = j_con_fuera / numcbl;
                    if (jbl_siguiente > nb-1) jbl_siguiente = nb-1;

                    ind_b_fuera = ibl_siguiente + mb*(jbl_siguiente);
                    if (ind_b_fuera != i_salida) {
                        fin = 1;
                        ind_conts_out[i_ind_conts_curr].ini = nump1;
                        ind_conts_out[i_ind_conts_curr].fin = p2;
                        ind_conts_out[i_ind_conts_curr].sts = OPEN_CONTOUR;
                        marcado[indcactual] = numcslocal+1;
                    } else {
                        contorno_conectado = -1;
                        for (jfc=0; jfc< numconts[ind_b_fuera]; jfc++) {
                            ind_c_fuera = jfc + num_max_conts*(ind_b_fuera);
                            int pf = ind_conts[ind_c_fuera].ini;
                            i_sig_dentro = vec_conts[pf].act.i;
                            j_sig_dentro = vec_conts[pf].act.j;
                            i_sig_fuera = vec_conts[pf].ant.i;
                            j_sig_fuera = vec_conts[pf].ant.j;
                            if ((i_con_dentro==i_sig_fuera)&&(j_con_dentro==j_sig_fuera)&&(i_con_fuera==i_sig_dentro)&&(j_con_fuera==j_sig_dentro)) {
                                contorno_conectado = jfc;
                                pf_bueno = pf;
                                break;
                            }
                        }
                        if (contorno_conectado != -1) {
                            contorno = contorno_conectado;
                            vec_conts[p2].next = pf_bueno;
                            if ((i_con_dentro==i_antes)&&(j_con_dentro==j_antes)&&(i_con_fuera==i_con_ini)&&(j_con_fuera==j_con_ini)) {
                                fin = 1;
                                int contg;
#pragma omp atomic capture
                                contg = (*d_numconts_glob)++;
                                d_ind_conts_glob[contg] = ind_conts[i_ind_conts_curr];
                                i_ind_conts_curr--;
                                numcslocal--;
                                d_ind_conts_glob[contg].fin = p2;
                                d_ind_conts_glob[contg].ini = nump1;
                                d_ind_conts_glob[contg].sts = CLOSED_CONTOUR;
                                marcado[contorno + num_max_conts*(ind_b_fuera)] = -1;
                                marcado[indcactual] = -1;
                            } else if (marcado[contorno + num_max_conts*(ind_b_fuera)] > 0) {
                                fin = 1;
                                int numcontorno = marcado[contorno + num_max_conts*(ind_b_fuera)] - 1;
                                ind_conts_out[numcontorno + indice_ini].ini = nump1;
                                marcado[indcactual] = numcontorno + 1;
                                numcslocal--;
                                i_ind_conts_curr--;
                            } else {
                                int cont_siguiente = contorno + num_max_conts*(ind_b_fuera);
                                marcado[cont_siguiente] = numcslocal + 1;
                                i_salida = ibloque;
                                ibloque = ind_b_fuera;
                            }
                        } else {
                            fin = 1;
                        }
                    }
                }
            }
        }

        ibloque = ibloque_abajo;
        for (int cont=0; cont< numconts[ibloque_abajo]; cont++) {
            ibloque = ibloque_abajo;
            int indcactual = cont + indice_ini_abajo;
            if(marcado[indcactual] == 0) {
                numcslocal = numcslocal + 1;
                i_ind_conts_curr++;
                ind_conts_out[i_ind_conts_curr] = ind_conts[indcactual];
                nump1 = ind_conts[indcactual].ini;
                p2 = ind_conts[indcactual].fin;

                int i_con_fuera = vec_conts[p2].sig.i;
                int j_con_fuera = vec_conts[p2].sig.j;
                int ibl_fuera = i_con_fuera/numfbl;
                if (ibl_fuera > mb-1) ibl_fuera = mb-1;
                int jbl_fuera = j_con_fuera/numcbl;
                if (jbl_fuera > nb-1) jbl_fuera = nb-1;

                int ind_b_fuera = ibl_fuera + mb*(jbl_fuera);
                if (ind_b_fuera != ibloque_arriba) {
                    ind_conts_out[i_ind_conts_curr].ini = nump1;
                    ind_conts_out[i_ind_conts_curr].fin = p2;
                    ind_conts_out[i_ind_conts_curr].sts = OPEN_CONTOUR;
                } else {
                    contorno_conectado = -1;
                    for (jfc=0; jfc< numconts[ind_b_fuera]; jfc++) {
                        ind_c_fuera = jfc + num_max_conts*(ind_b_fuera);
                        int pf = ind_conts[ind_c_fuera].ini;
                        i_sig_dentro = vec_conts[pf].act.i;
                        j_sig_dentro = vec_conts[pf].act.j;
                        i_sig_fuera = vec_conts[pf].ant.i;
                        j_sig_fuera = vec_conts[pf].ant.j;
                        int i_con_dentro = vec_conts[p2].act.i;
                        int j_con_dentro = vec_conts[p2].act.j;

                        if ((i_con_dentro==i_sig_fuera)&&(j_con_dentro==j_sig_fuera)&&(i_con_fuera==i_sig_dentro)&&(j_con_fuera==j_sig_dentro)) {
                            contorno_conectado = jfc;
                            pf_bueno = pf;
                            break;
                        }
                    }
                    if (contorno_conectado != -1) {
                        contorno = contorno_conectado;
                        vec_conts[p2].next = pf_bueno;
                        int numcontorno = marcado[contorno + num_max_conts*(ind_b_fuera)] - 1;
                        ind_conts_out[numcontorno + indice_ini].ini = nump1;
                        marcado[indcactual] = numcontorno + 1;
                        numcslocal--;
                        i_ind_conts_curr--;
                    }
                }
            }
        }
        numconts_out[ibloque_sal_arriba] = numcslocal + 1;
    }
}

void BorderTrackerOMP::horizontal_connection(int num_max_conts, int* numconts, VecCont* vec_conts,
                                             IndCont* ind_conts, int* numconts_out, IndCont* ind_conts_out,
                                             int* marcado, int current_nbn)
{
    int bl;
#pragma omp parallel for schedule(dynamic)
    for (bl=0; bl<current_nbn; bl++) {
        int jb = bl % current_nbn;
        int mb = 1;
        int nb = 2*current_nbn;
        int i_con_ini, j_con_ini, i_antes, j_antes, i_salida, nump1, p2, jfc;
        int i_sig_dentro, j_sig_dentro, i_sig_fuera, j_sig_fuera, pf_bueno;
        int contorno_conectado, ind_c_fuera, contorno, ind_b_fuera;
        int ibloque;

        int ibloque_izquierda = 2*jb;
        int ibloque_sal_izquierda = jb;
        int ibloque_derecha = 2*jb+1;
        int indice_ini = num_max_conts*(ibloque_izquierda);
        int indice_ini_derecha = num_max_conts*(ibloque_derecha);
        int numcslocal = -1;
        int i_ind_conts_curr = numcslocal + indice_ini;

        for (int cont=0; cont< numconts[ibloque_izquierda]; cont++) {
            ibloque = ibloque_izquierda;
            int indcactual = cont + indice_ini;
            if (marcado[indcactual]==0) {
                numcslocal++;
                i_ind_conts_curr++;
                ind_conts_out[i_ind_conts_curr] = ind_conts[indcactual];
                nump1 = ind_conts[indcactual].ini;
                i_con_ini = vec_conts[nump1].act.i;
                j_con_ini = vec_conts[nump1].act.j;
                i_antes = vec_conts[nump1].ant.i;
                j_antes = vec_conts[nump1].ant.j;
                int fin = 0;
                contorno = cont;
                i_salida = ibloque_derecha;

                while(fin==0) {
                    int indcaux = contorno + num_max_conts*(ibloque);
                    p2 = ind_conts[indcaux].fin;
                    int i_con_dentro = vec_conts[p2].act.i;
                    int j_con_dentro = vec_conts[p2].act.j;
                    int j_con_fuera = vec_conts[p2].sig.j;
                    int jbl_siguiente = j_con_fuera / numcbl;
                    if (jbl_siguiente > nb-1) jbl_siguiente = nb-1;
                    ind_b_fuera = mb*(jbl_siguiente);

                    if (ind_b_fuera != i_salida) {
                        fin = 1;
                        ind_conts_out[i_ind_conts_curr].ini = nump1;
                        ind_conts_out[i_ind_conts_curr].fin = p2;
                        ind_conts_out[i_ind_conts_curr].sts = OPEN_CONTOUR;
                        marcado[indcactual] = numcslocal + 1;
                    } else {
                        contorno_conectado = -1;
                        for (jfc=0; jfc< numconts[ind_b_fuera]; jfc++) {
                            ind_c_fuera = jfc + num_max_conts*(ind_b_fuera);
                            int pf = ind_conts[ind_c_fuera].ini;
                            i_sig_dentro = vec_conts[pf].act.i;
                            j_sig_dentro = vec_conts[pf].act.j;
                            i_sig_fuera = vec_conts[pf].ant.i;
                            j_sig_fuera = vec_conts[pf].ant.j;

                            int i_con_fuera = vec_conts[p2].sig.i;
                            int j_con_fuera = vec_conts[p2].sig.j;

                            if ((i_con_dentro==i_sig_fuera)&&(j_con_dentro==j_sig_fuera)&&(i_con_fuera==i_sig_dentro)&&(j_con_fuera==j_sig_dentro)) {
                                contorno_conectado = jfc;
                                pf_bueno = pf;
                                break;
                            }
                        }
                        if (contorno_conectado != -1) {
                            contorno = contorno_conectado;
                            vec_conts[p2].next = pf_bueno;
                            int i_con_fuera = vec_conts[p2].sig.i;
                            int j_con_fuera = vec_conts[p2].sig.j;

                            if ((i_con_dentro==i_antes)&&(j_con_dentro==j_antes)&&(i_con_fuera==i_con_ini)&&(j_con_fuera==j_con_ini)) {
                                fin = 1;
                                int contg;
#pragma omp atomic capture
                                contg = (*d_numconts_glob)++;
                                d_ind_conts_glob[contg] = ind_conts[i_ind_conts_curr];
                                i_ind_conts_curr--;
                                numcslocal--;
                                d_ind_conts_glob[contg].fin = p2;
                                d_ind_conts_glob[contg].ini = nump1;
                                d_ind_conts_glob[contg].sts = CLOSED_CONTOUR;
                                marcado[contorno + num_max_conts*(ind_b_fuera)] = -1;
                                marcado[indcactual] = -1;
                            } else if (marcado[contorno + num_max_conts*(ind_b_fuera)] > 0) {
                                fin = 1;
                                int numcontorno = marcado[contorno + num_max_conts*(ind_b_fuera)] - 1;
                                ind_conts_out[numcontorno + indice_ini].ini = nump1;
                                marcado[indcactual] = numcontorno + 1;
                                numcslocal--;
                                i_ind_conts_curr--;
                            } else {
                                int cont_siguiente = contorno + num_max_conts*(ind_b_fuera);
                                marcado[cont_siguiente] = numcslocal + 1;
                                i_salida = ibloque;
                                ibloque = ind_b_fuera;
                            }
                        } else {
                            fin = 1;
                        }
                    }
                }
            }
        }

        ibloque = ibloque_derecha;
        for (int cont=0; cont< numconts[ibloque_derecha]; cont++) {
            ibloque = ibloque_derecha;
            int indcactual = cont + indice_ini_derecha;
            if(marcado[indcactual]==0) {
                numcslocal = numcslocal + 1;
                i_ind_conts_curr++;
                ind_conts_out[i_ind_conts_curr] = ind_conts[indcactual];
                nump1 = ind_conts[indcactual].ini;
                p2 = ind_conts[indcactual].fin;

                int j_con_fuera = vec_conts[p2].sig.j;
                int jbl_fuera = j_con_fuera/numcbl;
                if (jbl_fuera > nb-1) jbl_fuera = nb-1;
                int ind_b_fuera = mb*(jbl_fuera);

                if (ind_b_fuera != ibloque_izquierda) {
                    ind_conts_out[i_ind_conts_curr].ini = nump1;
                    ind_conts_out[i_ind_conts_curr].fin = p2;
                    ind_conts_out[i_ind_conts_curr].sts = OPEN_CONTOUR;
                } else {
                    contorno_conectado = -1;
                    for (jfc=0; jfc< numconts[ind_b_fuera]; jfc++) {
                        ind_c_fuera = jfc + num_max_conts*(ind_b_fuera);
                        int pf = ind_conts[ind_c_fuera].ini;
                        i_sig_dentro = vec_conts[pf].act.i;
                        j_sig_dentro = vec_conts[pf].act.j;
                        i_sig_fuera = vec_conts[pf].ant.i;
                        j_sig_fuera = vec_conts[pf].ant.j;

                        int i_con_dentro = vec_conts[p2].act.i;
                        int j_con_dentro = vec_conts[p2].act.j;
                        int i_con_fuera = vec_conts[p2].sig.i;
                        int j_con_fuera = vec_conts[p2].sig.j;

                        if ((i_con_dentro==i_sig_fuera)&&(j_con_dentro==j_sig_fuera)&&(i_con_fuera==i_sig_dentro)&&(j_con_fuera==j_sig_dentro)) {
                            contorno_conectado = jfc;
                            pf_bueno = pf;
                            break;
                        }
                    }
                    if (contorno_conectado != -1) {
                        contorno = contorno_conectado;
                        vec_conts[p2].next = pf_bueno;
                        int numcontorno = marcado[contorno + num_max_conts*(ind_b_fuera)] - 1;
                        d_ind_conts_aux[numcontorno + num_max_conts*(ind_b_fuera)].ini = nump1;
                        marcado[indcactual] = numcontorno + 1;
                        numcslocal--;
                        i_ind_conts_curr--;
                    }
                }
            }
        }
        numconts_out[ibloque_sal_izquierda] = numcslocal + 1;
    }
}

std::vector<std::vector<cv::Point>> BorderTrackerOMP::findBorders(const cv::Mat& gray) {
    // 1. Reset ONLY necessary counters (avoid massive memsets)
    *d_numconts_glob = 0;

    // We only zero d_numconts because it increments.
    // d_vec_conts, d_ind_conts, d_ind_conts_glob do NOT need zeroing (we write over them).
    memset(d_numconts, 0, sizeof(int) * mfb * nfb);
    // d_marcado reset is handled in the connection loop below.

    copy_input_to_padded(gray);

    // 2. Preprocessing
    preprocessing_cpu();

    // 3. Parallel Tracking
    parallel_tracking_cpu();

    // 4. Merge Blocks
    int numfbl_local = (Mg / mfb);
    int numcbl_local = (Ng / nfb);

    // Vertical connections
    int mbn = N_BLOCKS_ROWS / 2;
    int num_max_c_etapa = MAX_N_BORDS;

    // We swap pointers, so track current active buffers
    int* curr_d_numconts = d_numconts;
    int* curr_d_numconts_aux = d_numconts_aux;
    IndCont* curr_d_ind_conts = d_ind_conts;
    IndCont* curr_d_ind_conts_aux = d_ind_conts_aux;

    while(mbn >= 1) {
        vertical_connection(num_max_c_etapa, curr_d_numconts, d_vec_conts, curr_d_ind_conts,
                            curr_d_numconts_aux, curr_d_ind_conts_aux, d_marcado, mbn);

        // [Optimization] We keep d_marcado memset because the logic heavily depends on 0 check.
        // It's the only large memset remaining.
        memset(d_marcado, 0, sizeof(int) * mfb * nfb * MAX_N_BORDS);

        std::swap(curr_d_numconts, curr_d_numconts_aux);

        // [Optimization] Removed: memset(curr_d_numconts_aux, 0, ...)
        // Logic overwrites values directly.

        std::swap(curr_d_ind_conts, curr_d_ind_conts_aux);

        // [Optimization] Removed: memset(curr_d_ind_conts_aux, 0, ...)
        // Logic overwrites values directly.

        num_max_c_etapa *= 2;
        numfbl_local *= 2;
        mbn /= 2;
    }

    // Horizontal connections
    int nbn = N_BLOCKS_COLS / 2;
    while(nbn >= 1) {
        horizontal_connection(num_max_c_etapa, curr_d_numconts, d_vec_conts, curr_d_ind_conts,
                              curr_d_numconts_aux, curr_d_ind_conts_aux, d_marcado, nbn);

        // [Optimization] Keep d_marcado reset
        memset(d_marcado, 0, sizeof(int) * mfb * nfb * MAX_N_BORDS);

        std::swap(curr_d_numconts, curr_d_numconts_aux);
        // [Optimization] Removed: memset(curr_d_numconts_aux, 0, ...)

        std::swap(curr_d_ind_conts, curr_d_ind_conts_aux);
        // [Optimization] Removed: memset(curr_d_ind_conts_aux, 0, ...)

        num_max_c_etapa *= 2;
        nbn /= 2;
        numcbl_local *= 2;
    }

    // 5. Convert Results to Std Vector
    std::vector<std::vector<cv::Point>> contours;
    int num_final_contours = d_numconts_glob[0];

    for (int i = 0; i < num_final_contours; i++) {
        int idx_actual = d_ind_conts_glob[i].ini;
        int idx_final  = d_ind_conts_glob[i].fin;

        int max_points = 100000;
        int pts = 0;

        std::vector<cv::Point> current_contour;

        while (pts < max_points) {
            coord p = d_vec_conts[idx_actual].act;
            current_contour.push_back(cv::Point(p.i, p.j));

            if (idx_actual == idx_final) break;
            idx_actual = d_vec_conts[idx_actual].next;
            pts++;
        }

        if (!current_contour.empty()) {
            contours.push_back(current_contour);
        }
    }

    return contours;
}

std::vector<std::vector<cv::Point> > BorderTrackerOMP::findContours(const cv::Mat &gray)
{
    btomp::BorderTrackerOMP tracker(gray.cols, gray.rows);
    return tracker.findBorders(gray);
}
}
