#include "tact.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    int use_pd = 0;
    int steps = 1;
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.tactbin [--pd] [--steps N]\n", argv[0]);
        return 2;
    }
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--pd") == 0) {
            use_pd = 1;
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
            if (steps < 0) {
                fprintf(stderr, "--steps must be non-negative\n");
                return 2;
            }
        } else {
            fprintf(stderr, "usage: %s model.tactbin [--pd] [--steps N]\n", argv[0]);
            return 2;
        }
    }

    tact_model_t *m = NULL;
    tact_state_t *s = NULL;
    int rc = tact_load_model(argv[1], &m);
    if (rc != 0) {
        fprintf(stderr, "tact_load_model failed: %d\n", rc);
        return 1;
    }

    tact_model_info_t info;
    tact_model_info(m, &info);
    rc = tact_create_state(m, &s);
    if (rc != 0) {
        fprintf(stderr, "tact_create_state failed: %d\n", rc);
        tact_destroy_model(m);
        return 1;
    }

    double *tau = (double*)calloc((size_t)info.nq, sizeof(double));
    double *q_ref = NULL, *qd_ref = NULL, *kp = NULL, *kd = NULL;
    if (use_pd) {
        q_ref = (double*)calloc((size_t)info.nq, sizeof(double));
        qd_ref = (double*)calloc((size_t)info.nq, sizeof(double));
        kp = (double*)malloc((size_t)info.nq * sizeof(double));
        kd = (double*)malloc((size_t)info.nq * sizeof(double));
        for (int i = 0; i < info.nq; ++i) {
            kp[i] = 10.0;
            kd[i] = 0.1;
        }
    }
    if (!tau || (use_pd && (!q_ref || !qd_ref || !kp || !kd))) {
        free(tau); free(q_ref); free(qd_ref); free(kp); free(kd);
        tact_destroy_state(s);
        tact_destroy_model(m);
        return 1;
    }

    for (int step = 0; step < steps; ++step) {
        if (use_pd) rc = tact_step_pd(m, s, tau, q_ref, qd_ref, kp, kd);
        else        rc = tact_step(m, s, tau);
        if (rc != 0) {
            fprintf(stderr, "tact_step failed at step %d: %d\n", step, rc);
            free(tau); free(q_ref); free(qd_ref); free(kp); free(kd);
            tact_destroy_state(s);
            tact_destroy_model(m);
            return 1;
        }
    }

    printf("nb=%d nq=%d n_shape=%d n_pair=%d dt=%.6f steps=%d\n",
           info.nb, info.nq, info.n_shape, info.n_pair, info.dt, steps);
    if (info.nq > 0) {
        printf("q0=%.9f qd0=%.9f\n", tact_q(s)[0], tact_qd(s)[0]);
    }
    printf("q:");
    for (int i = 0; i < info.nq; ++i) printf(" %.17g", tact_q(s)[i]);
    printf("\n");
    printf("qd:");
    for (int i = 0; i < info.nq; ++i) printf(" %.17g", tact_qd(s)[i]);
    printf("\n");
    printf("y:");
    for (int i = 0; i < info.y_size; ++i) printf(" %.17g", tact_y(s)[i]);
    printf("\n");

    free(tau);
    free(q_ref);
    free(qd_ref);
    free(kp);
    free(kd);
    tact_destroy_state(s);
    tact_destroy_model(m);
    return 0;
}
