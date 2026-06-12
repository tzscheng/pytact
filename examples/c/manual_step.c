#include "tact.h"

#include <stdio.h>
#include <string.h>

static void eye(double *m, int n)
{
    memset(m, 0, (size_t)n * (size_t)n * sizeof(double));
    for (int i = 0; i < n; ++i) m[n*i + i] = 1.0;
}

int main(void)
{
    int nb = 1;
    int parent[1] = {-1};
    int jtype[1] = {1};
    double X[36], I6[36], Ti[16];
    double ff[1] = {0.0};
    double sk[1] = {0.0};
    double floss[1] = {0.0};
    double armature[1] = {0.0};
    double jnt_lo[1] = {0.0};
    double jnt_hi[1] = {0.0};
    double g[3] = {0.0, 0.0, 0.0};

    eye(X, 6);
    eye(I6, 6);
    eye(Ti, 4);

    tact_t *h = tact_create(
        nb, parent, jtype,
        X, I6, Ti, ff, sk, floss, armature, jnt_lo, jnt_hi,
        g, 0.001, 2,
        0, 0,
        NULL, NULL, NULL, NULL, NULL, NULL, NULL,
        0.2, 0.001, 1e-8, 0.05, 20, 1e-8);
    if (!h) {
        fprintf(stderr, "tact_create failed\n");
        return 1;
    }

    int nq = tact_get_nq(h);
    int lam_size = tact_get_lam_size(h);
    double q[1] = {0.0};
    double qd[1] = {0.0};
    double tau[1] = {1.0};
    double lam_in[26] = {0.0};
    double lam_out[26] = {0.0};

    if (nq != 1 || lam_size != 26) {
        fprintf(stderr, "unexpected dimensions: nq=%d lam_size=%d\n", nq, lam_size);
        tact_destroy(h);
        return 1;
    }

    tact_step_lcp(h, q, qd, tau, NULL, NULL, NULL, NULL, lam_in, lam_out);

    printf("nb=%d nq=%d lam_size=%d dt=%.6f\n",
           tact_get_nb(h), nq, lam_size, tact_get_dt(h));
    printf("q_next=%.9f qd_next=%.9f\n",
           tact_get_q_next(h)[0], tact_get_qd_next(h)[0]);

    tact_destroy(h);
    return 0;
}
