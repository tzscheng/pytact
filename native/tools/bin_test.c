#include "tact.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0)
{
    fprintf(stderr, "usage: %s model.bin [steps] [--headless] [--pd] [--steps N]\n", argv0);
}

int main(int argc, char **argv)
{
    int headless = 0;
    int use_pd = 0;
    int steps = 1000;
    int saw_steps = 0;

    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--headless") == 0) {
            headless = 1;
            if (!saw_steps) steps = 1;
        } else if (strcmp(argv[i], "--pd") == 0) {
            use_pd = 1;
            headless = 1;
            if (!saw_steps) steps = 1;
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
            saw_steps = 1;
        } else if (!saw_steps && argv[i][0] != '-') {
            steps = atoi(argv[i]);
            saw_steps = 1;
        } else {
            usage(argv[0]);
            return 2;
        }
        if (steps < 0) {
            fprintf(stderr, "steps must be non-negative\n");
            return 2;
        }
    }

    tact_t *model = NULL;
    int rc = tact_load(argv[1], &model);
    if (rc != 0) {
        fprintf(stderr, "tact_load failed: %d\n", rc);
        return 1;
    }

    tact_info_t info;
    rc = tact_info(model, &info);
    if (rc != 0) {
        fprintf(stderr, "tact_info failed: %d\n", rc);
        tact_destroy(model);
        return 1;
    }

    size_t nq_alloc = (size_t)(info.nq > 0 ? info.nq : 1);
    size_t y_alloc = (size_t)(info.y_size > 0 ? info.y_size : 1);
    double *q = (double*)malloc(nq_alloc * sizeof(double));
    double *qd = (double*)malloc(nq_alloc * sizeof(double));
    double *q_next = (double*)malloc(nq_alloc * sizeof(double));
    double *qd_next = (double*)malloc(nq_alloc * sizeof(double));
    double *y = (double*)calloc(y_alloc, sizeof(double));
    double *tau = (double*)calloc(nq_alloc, sizeof(double));
    double *q_ref = NULL, *qd_ref = NULL, *kp = NULL, *kd = NULL;
    tact_ctx_t *ctx = NULL;
    tact_ctx_t *ctx_next = NULL;
    if (!q || !qd || !q_next || !qd_next || !y || !tau ||
        tact_create_ctx(model, &ctx) != 0 ||
        tact_create_ctx(model, &ctx_next) != 0) {
        fprintf(stderr, "allocation failed\n");
        free(q); free(qd); free(q_next); free(qd_next); free(y); free(tau);
        tact_destroy_ctx(ctx);
        tact_destroy_ctx(ctx_next);
        tact_destroy(model);
        return 1;
    }
    if (info.nq > 0) {
        memcpy(q, tact_q0(model), (size_t)info.nq * sizeof(double));
        memcpy(qd, tact_qd0(model), (size_t)info.nq * sizeof(double));
    }
    if (use_pd) {
        q_ref = (double*)calloc(nq_alloc, sizeof(double));
        qd_ref = (double*)calloc(nq_alloc, sizeof(double));
        kp = (double*)malloc(nq_alloc * sizeof(double));
        kd = (double*)malloc(nq_alloc * sizeof(double));
        if (!q_ref || !qd_ref || !kp || !kd) {
            fprintf(stderr, "allocation failed\n");
            free(q); free(qd); free(q_next); free(qd_next); free(y); free(tau);
            free(q_ref); free(qd_ref); free(kp); free(kd);
            tact_destroy_ctx(ctx);
            tact_destroy_ctx(ctx_next);
            tact_destroy(model);
            return 1;
        }
        for (int i = 0; i < info.nq; ++i) {
            kp[i] = 10.0;
            kd[i] = 0.1;
        }
    }

    if (!headless) {
        printf("nb=%d nq=%d n_shape=%d n_pair=%d dt=%.6f\n",
               info.nb, info.nq, info.n_shape, info.n_pair, info.dt);
    }

    int ran_steps = 0;
    for (int step = 0; step < steps; ++step) {
        if (use_pd) {
            rc = tact_step_pd(model, q, qd, tau, q_ref, qd_ref, kp, kd,
                              ctx, q_next, qd_next, y, ctx_next);
        } else {
            rc = tact_step(model, q, qd, tau, ctx, q_next, qd_next, y, ctx_next);
        }
        if (rc != 0) {
            fprintf(stderr, "tact_step failed at step %d: %d\n", step, rc);
            free(q); free(qd); free(q_next); free(qd_next); free(y); free(tau);
            free(q_ref); free(qd_ref); free(kp); free(kd);
            tact_destroy_ctx(ctx);
            tact_destroy_ctx(ctx_next);
            tact_destroy(model);
            return 1;
        }

        double *tmp = q; q = q_next; q_next = tmp;
        tmp = qd; qd = qd_next; qd_next = tmp;
        tact_ctx_t *tmp_ctx = ctx; ctx = ctx_next; ctx_next = tmp_ctx;
        ++ran_steps;

        if (!headless) {
            rc = tact_render(model, q);
            if (rc == -1) break;
            if (rc < 0) {
                fprintf(stderr, "tact_render failed: %d\n", rc);
                break;
            }
        }
    }

    printf("nb=%d nq=%d n_shape=%d n_pair=%d dt=%.6f steps=%d\n",
           info.nb, info.nq, info.n_shape, info.n_pair, info.dt, ran_steps);
    printf("frames=%d root_id=%d missing_id=%d\n",
           tact_frame_count(model), tact_frame_id(model, "root"),
           tact_frame_id(model, "__missing__"));
    if (info.nq > 0) {
        printf("q0=%.9f qd0=%.9f\n", q[0], qd[0]);
    }
    printf("q:");
    for (int i = 0; i < info.nq; ++i) printf(" %.17g", q[i]);
    printf("\nqd:");
    for (int i = 0; i < info.nq; ++i) printf(" %.17g", qd[i]);
    printf("\ny:");
    for (int i = 0; i < info.y_size; ++i) printf(" %.17g", y[i]);
    printf("\n");

    free(q);
    free(qd);
    free(q_next);
    free(qd_next);
    free(y);
    free(tau);
    free(q_ref);
    free(qd_ref);
    free(kp);
    free(kd);
    tact_destroy_ctx(ctx);
    tact_destroy_ctx(ctx_next);
    tact_destroy(model);
    return 0;
}
