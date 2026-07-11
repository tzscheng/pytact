#include "tact.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NQ 64
#define MAX_Y  256

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s model.bin\n", argv[0]);
        return 2;
    }

    tact_t *model = NULL;
    int rc = tact_create_from_bin(argv[1], &model);
    if (rc != 0) {
        fprintf(stderr, "tact_create_from_bin failed: %d\n", rc);
        return 1;
    }

    if (model->nq > MAX_NQ || model->y_size > MAX_Y) {
        fprintf(stderr, "model too large for this demo: nq=%d/%d y=%d/%d\n", model->nq, MAX_NQ, model->y_size, MAX_Y);
        tact_destroy(model);
        return 1;
    }

    double q[MAX_NQ] = {0};
    double qd[MAX_NQ] = {0};
    double y[MAX_Y] = {0};
    double tau[MAX_NQ] = {0};
    double q_ref[MAX_NQ] = {0}, qd_ref[MAX_NQ] = {0};
    double kp[MAX_NQ] = {0}, kd[MAX_NQ] = {0};

    if (model->nq > 0) {
        memcpy(q, model->q0, (size_t)model->nq * sizeof(double));
        memcpy(qd, model->qd0, (size_t)model->nq * sizeof(double));
    }

    printf("nb=%d nq=%d n_shape=%d n_pair=%d dt=%.6f\n", model->nb, model->nq, model->n_shape, model->n_pair, model->dt);
    printf("frames=%d root_id=%d missing_id=%d\n", tact_frame_count(model), tact_frame_id(model, "root"), tact_frame_id(model, "__missing__"));
    long ran_steps = 0;

    while (1) {
        /* ctx = model->ctx_next: continue from the engine's own last warm-start */
        rc = tact_step_lcp(model, q, qd, tau, kp, kd, q_ref, qd_ref, model->ctx_next);
        if (rc != 0) {
            fprintf(stderr, "tact_step_lcp failed at step %ld: %d\n", ran_steps, rc);
            tact_destroy(model);
            return 1;
        }

        if (model->nq > 0) {
            memcpy(q, model->q_next, (size_t)model->nq * sizeof(double));
            memcpy(qd, model->qd_next, (size_t)model->nq * sizeof(double));
        }

        if (model->y_size > 0) {
            memcpy(y, model->y, (size_t)model->y_size * sizeof(double));
        }
        ++ran_steps;

        if (ran_steps % 8 == 0) {
            rc = tact_render(model, q);
            if (rc == -1) break;  /* window closed / ESC */
            if (rc < 0) {
                fprintf(stderr, "tact_render failed: %d\n", rc);
                break;
            }
        }
    }

    tact_destroy(model);
    return 0;
}
