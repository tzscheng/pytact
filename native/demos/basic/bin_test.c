#include "tact.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NQ 64
#define MAX_Y  256

static void usage(const char *argv0)
{
    fprintf(stderr, "usage: %s model.bin\n", argv0);
}

static void cleanup(tact_t *model, double *ctx, double *ctx_next)
{
    free(ctx);
    free(ctx_next);
    tact_destroy(model);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        usage(argv[0]);
        return 2;
    }

    tact_t *model = NULL;
    int rc = tact_create_from_bin(argv[1], &model);
    if (rc != 0) {
        fprintf(stderr, "tact_create_from_bin failed: %d\n", rc);
        return 1;
    }

    int nb = tact_nb(model);
    int nq = tact_nq(model);
    int n_shape = tact_n_shape(model);
    int n_pair = tact_n_pair(model);
    int y_size = tact_y_size(model);
    int ctx_size = tact_ctx_size(model);
    double dt = tact_dt(model);

    if (nq > MAX_NQ || y_size > MAX_Y) {
        fprintf(stderr, "model too large for this demo: nq=%d/%d y=%d/%d\n", nq, MAX_NQ, y_size, MAX_Y);
        cleanup(model, NULL, NULL);
        return 1;
    }

    double q[MAX_NQ] = {0};
    double qd[MAX_NQ] = {0};
    double y[MAX_Y] = {0};
    double tau[MAX_NQ] = {0};
    double q_ref[MAX_NQ] = {0}, qd_ref[MAX_NQ] = {0};
    double kp[MAX_NQ] = {0}, kd[MAX_NQ] = {0};

    size_t ctx_count = (size_t)(ctx_size > 0 ? ctx_size : 1);
    double *ctx = (double*)calloc(ctx_count, sizeof(double));
    double *ctx_next = (double*)calloc(ctx_count, sizeof(double));
    if (!ctx || !ctx_next) {
        fprintf(stderr, "allocation failed\n");
        cleanup(model, ctx, ctx_next);
        return 1;
    }

    if (nq > 0) {
        memcpy(q, tact_q0(model), (size_t)nq * sizeof(double));
        memcpy(qd, tact_qd0(model), (size_t)nq * sizeof(double));
    }
    printf("nb=%d nq=%d n_shape=%d n_pair=%d dt=%.6f\n",
           nb, nq, n_shape, n_pair, dt);

    int ran_steps = 0;
    while (1) {
        rc = tact_step_lcp(model, q, qd, tau, kp, kd, q_ref, qd_ref, ctx, ctx_next);
        if (rc != 0) {
            fprintf(stderr, "tact_step_lcp failed at step %d: %d\n", ran_steps, rc);
            cleanup(model, ctx, ctx_next);
            return 1;
        }

        if (nq > 0) {
            memcpy(q, tact_q_next(model), (size_t)nq * sizeof(double));
            memcpy(qd, tact_qd_next(model), (size_t)nq * sizeof(double));
        }
        if (y_size > 0) {
            memcpy(y, tact_y(model), (size_t)y_size * sizeof(double));
        }

        double *tmp_ctx = ctx;
        ctx = ctx_next;
        ctx_next = tmp_ctx;
        ++ran_steps;

        if (ran_steps % 8 == 0) {
            rc = tact_render(model, q);
            if (rc == -1) break;
            if (rc < 0) {
                fprintf(stderr, "tact_render failed: %d\n", rc);
                break;
            }
        }
    }

    /*
    printf("nb=%d nq=%d n_shape=%d n_pair=%d dt=%.6f steps=%d\n",
           nb, nq, n_shape, n_pair, dt, ran_steps);
    printf("frames=%d root_id=%d missing_id=%d\n",
           tact_frame_count(model), tact_frame_id(model, "root"),
           tact_frame_id(model, "__missing__"));
    if (nq > 0) {
        printf("q0=%.9f qd0=%.9f\n", q[0], qd[0]);
	}*/

    //printf("q:");
    //for (int i = 0; i < nq; ++i) printf(" %.17g", q[i]);
    //printf("\nqd:");
    //for (int i = 0; i < nq; ++i) printf(" %.17g", qd[i]);
    //printf("\ny:");
    //for (int i = 0; i < y_size; ++i) printf(" %.17g", y[i]);
    //printf("\n");

    cleanup(model, ctx, ctx_next);
    return 0;
}
