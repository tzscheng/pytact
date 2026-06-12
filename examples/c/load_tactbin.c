#include "tact.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s model.tactbin\n", argv[0]);
        return 2;
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
    if (!tau) {
        tact_destroy_state(s);
        tact_destroy_model(m);
        return 1;
    }

    rc = tact_step(m, s, tau);
    if (rc != 0) {
        fprintf(stderr, "tact_step failed: %d\n", rc);
        free(tau);
        tact_destroy_state(s);
        tact_destroy_model(m);
        return 1;
    }

    printf("nb=%d nq=%d n_shape=%d n_pair=%d dt=%.6f\n",
           info.nb, info.nq, info.n_shape, info.n_pair, info.dt);
    if (info.nq > 0) {
        printf("q0=%.9f qd0=%.9f\n", tact_q(s)[0], tact_qd(s)[0]);
    }

    free(tau);
    tact_destroy_state(s);
    tact_destroy_model(m);
    return 0;
}
