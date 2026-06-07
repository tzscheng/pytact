#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <turbojpeg.h>
#include "mujoco.h"
#include "GLFW/glfw3.h"

mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data

long n_step = 0;
double cmd[256];

int render = 0;
int eq_active0[16];

// dual-actuator support — per-step PD gains (2026-06-07; mirrors the tact-side
// YAML k: removal: gains are control-policy inputs, not model constants).
// Convention: every actuated joint has one motor actuator (biastype=NONE) for
// feedforward torque AND one position actuator (biastype=AFFINE) for PD. The
// caller passes (tau, q_ref, qd_ref, kp, kd) per step; each step STATELESSLY
// writes the position actuators' gainprm/biasprm from kp/kd (q_ref+kp present)
// or zeros them (PD off). The XML's kp/kv values are structural placeholders —
// the runtime never reads them (the former nominal save/restore toggle is gone).
//
// XML must satisfy: n_motor == n_position, and the i-th motor pairs with the i-th
// position actuator (by transmission joint id). Mismatch is reported at init and PD
// support degrades gracefully (q_ref is ignored, only motor torques flow).
int   motor_idx[256];               // ctrl-space indices of motor actuators
int   position_idx[256];            // ctrl-space indices of position actuators
int   n_motor = 0;
int   n_position = 0;
bool  pd_supported = false;         // set true at init if XML has matched motor/position pairs

mjvCamera cam;                      // abstract camera
mjvPerturb pert;                    // perturbation 
mjvOption v_opt;                      // visualization option
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context
GLFWwindow* window;

bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

//int mode[256];


void update_control_input(const mjModel *m, mjData *d) {
    for(int i = 0; i < m->nu; i++) d->ctrl[i] = cmd[i];
}

// classify actuators into motor (biastype=NONE) and position (biastype=AFFINE).
// pd_supported is set only when XML has exactly matching motor↔position pairs in joint id.
static void classify_actuators(){
    n_motor = n_position = 0;
    for(int i = 0; i < m->nu; i++){
        if(m->actuator_biastype[i] == mjBIAS_NONE){
            if(n_motor < 256) motor_idx[n_motor++] = i;
        } else if(m->actuator_biastype[i] == mjBIAS_AFFINE){
            if(n_position < 256) position_idx[n_position++] = i;
        }
    }
    // pairing validation: same count AND each motor[i]'s transmission joint matches position[i]'s
    pd_supported = (n_motor > 0 && n_motor == n_position);
    for(int i = 0; pd_supported && i < n_motor; i++){
        if(m->actuator_trntype[motor_idx[i]]    != m->actuator_trntype[position_idx[i]] ||
           m->actuator_trnid[motor_idx[i]*2]    != m->actuator_trnid[position_idx[i]*2]){
            pd_supported = false;
        }
    }
    printf("[mjenv] actuators: %d motor + %d position, PD support: %s\n",
           n_motor, n_position, pd_supported ? "YES" : "NO (q_ref will be ignored)");
}

// stateless per-step PD gain application: q_ref && kp present → write the caller's
// gains (gainprm[0]=kp, biasprm=[0, -kp, -kd]; kd NULL → P-only), else zero (PD
// off). Writes a few doubles per position actuator every step — negligible next to
// mj_step, and removes the toggle state (pd_active_now / nominal save-restore):
// the gainprm content is a pure function of this step's arguments.
static void apply_pd_gains(double* q_ref, double* kp, double* kd){
    if(!pd_supported) return;
    bool on = (q_ref != NULL && kp != NULL);
    for(int k = 0; k < n_position; k++){
        int i = position_idx[k];
        memset(m->actuator_gainprm + i*mjNGAIN, 0, sizeof(double)*mjNGAIN);
        memset(m->actuator_biasprm + i*mjNBIAS, 0, sizeof(double)*mjNBIAS);
        if(on){
            m->actuator_gainprm[i*mjNGAIN + 0] = kp[k];
            m->actuator_biasprm[i*mjNBIAS + 1] = -kp[k];
            if(kd) m->actuator_biasprm[i*mjNBIAS + 2] = -kd[k];
        }
    }
}


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    if (act == GLFW_PRESS && key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods){
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS); //|| (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_RELEASE);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS); //|| (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_RELEASE);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS); //|| (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_RELEASE);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);

    if (button_middle && act == GLFW_PRESS){
	pert.active = 0;
	int width, height;
	glfwGetWindowSize(window, &width, &height);

    	mjtNum selpnt[3];

	//3.0.0
	int selgeom, selflex, selskin;
	int selbody = mjv_select(m, d, &v_opt, (double)width/(double)height, lastx/(double)width, ((double)height-lasty)/(double)height, &scn, selpnt, &selgeom, &selflex, &selskin);
	
	//2.3.7
	//int selgeom, selskin;
	//int selbody = mjv_select(m, d, &v_opt, (double)width/(double)height, lastx/(double)width, ((double)height-lasty)/(double)height, &scn, selpnt, &selgeom, &selskin);


	if (selbody < 0) {
	    pert.select = 0;
	    pert.skinselect = -1;
	    pert.flexselect = -1;
	    return;
	}
    
	pert.select = selbody;
	pert.skinselect = selskin;
	pert.flexselect = selflex;
	
	mjtNum tmp[3];
	mju_sub3(tmp, selpnt, d->xpos+3*pert.select);
	mju_mulMatTVec(pert.localpos, d->xmat+9*pert.select, tmp, 3, 3);
    }

    else if (act == GLFW_PRESS and pert.select > 0) {
	if (!pert.active)  mjv_initPerturb(m, d, &scn, &pert);
	if (button_right) pert.active = mjPERT_TRANSLATE;
	else if(button_left) pert.active = mjPERT_ROTATE;
    }

    else if (act == GLFW_RELEASE) pert.active = 0;
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos){
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right ) action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left ) action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else action = mjMOUSE_ZOOM;


    if (pert.active) mjv_movePerturb(m, d, action, dx/(float)height, dy/(float)height, &scn, &pert);
    else mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset) {
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}


extern "C" void init(char* xml, int _render=0) {
    mjcb_control = update_control_input;
    render = _render;
    
    // load and compile model
    char error[1000] = "Could not load binary model";
    m = mj_loadXML(xml, 0, error, 1000);
    
    if(!m) {
	mju_error_s("Load model error: %s", error);
	exit(0); //return -1;
    }

    //printf("nbody: %d  ngeom: %d\n", m->nbody, m->ngeom);
    d = mj_makeData(m);    
    pert.skinselect = -1;

    //store initial state of eq_active
    for(int i=0; i < 16; i++) eq_active0[i] = d->eq_active[i];

    //classify actuators (motor/position pairs); zero the position-actuator gains for
    //a safe start — they are runtime-written every step from the caller's kp/kd
    //(the XML values are structural placeholders, never active).
    classify_actuators();
    apply_pd_gains(NULL, NULL, NULL);
    
    //manipulating floating body's 6d pose: doesn't work
    /*int bodyid = mj_name2id(m, mjOBJ_BODY, "model");
    if( bodyid>=0 && m->body_jntnum[bodyid]==1 && m->jnt_type[m->body_jntadr[bodyid]]==mjJNT_FREE ){
	int qposadr = m->jnt_qposadr[m->body_jntadr[bodyid]];
	int qveladr = m->jnt_dofadr[m->body_jntadr[bodyid]];
	d->qpos[qposadr] = 0.5;
	d->qvel[qveladr] = 0.0;
	}*/
    
    if(render > 0){
	if(!glfwInit()) mju_error("Could not initialize GLFW");

	//anti aliasing level
	glfwWindowHint(GLFW_SAMPLES, 8);
	
	// create window, make OpenGL context current, request v-sync
	window = glfwCreateWindow(1200, 800, "mujoco", NULL, NULL);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	
	// initialize visualization data structures
	mjv_defaultCamera(&cam);
	//mjv_defaultPerturb(&pert);
	mjv_defaultOption(&v_opt);
	mjv_defaultScene(&scn);
	mjr_defaultContext(&con);
		
	// create scene and context
	mjv_makeScene(m, &scn, 2000);
	mjr_makeContext(m, &con, mjFONTSCALE_150);

	// install GLFW mouse and keyboard callbacks
	glfwSetKeyCallback(window, keyboard);
	glfwSetCursorPosCallback(window, mouse_move);
	glfwSetMouseButtonCallback(window, mouse_button);
	glfwSetScrollCallback(window, scroll);

	cam.lookat[0] = 0.0;
	cam.lookat[1] = 0.0;
	cam.lookat[2] = 0.5;
	cam.distance = 5;
	cam.azimuth = -130;
	cam.elevation = -27.45;
	
	//if (render == 1) {cam.lookat[2] = -0.35; cam.distance = 6.5; cam.azimuth = -130; cam.elevation = -27.45;} //pal
	//if (view == 1) {cam.lookat[2] = -0.35; cam.distance = 2.0; cam.azimuth = -130; cam.elevation = -27.45;} //pal
	//else if (view == 2){ cam.lookat[2] = 0.5; cam.distance = 3; cam.azimuth = -130; cam.elevation = -27.45; } //leg
	//else if (view == 3){ cam.lookat[2] = 0.12; cam.distance = 0.6; cam.azimuth = -150; cam.elevation = -18; } //hand
	//else if (view == 4) {cam.lookat[2] = 0.55; cam.distance = 1.8; cam.azimuth = -130; cam.elevation = -27.45;} //rb5
	//else if (view == 5) {cam.lookat[2] = 0.45; cam.distance = 3.0; cam.azimuth = -130; cam.elevation = -27.45;} //rb10
	//else if (view == 6){ cam.lookat[2] = 0.5; cam.distance = 5; cam.azimuth = -130; cam.elevation = -27.45; } //leg
	//else if (view == 7){ cam.lookat[2] = 0.0; cam.distance = 0.8; cam.azimuth = -150; cam.elevation = -18; } //hand
	//else if (view == 8){ cam.lookat[1] = -0.1; cam.lookat[2] = -0.1; cam.distance = 0.9; cam.azimuth = -150; cam.elevation = -18; } //ug4
    }

    //random piece picing...
    m->nuser_geom = 1; 
    
    //int n_out = m->nsensordata;

    //----------------------------------------
    mju_zero(d->xfrc_applied, 6*m->nbody);
    if(render > 0) mjv_applyPerturbForce(m, d, &pert);
    mj_step(m, d);
    //------------------------------------------
        
    //return n_out;
}


// Unified step — same control-input set as tact's Env.step. tau: per-motor
// feedforward torque (length n_motor), q_ref/qd_ref: per-position targets
// (NULL → that term off), kp/kd: per-step PD gains (length n_position; gains
// are control inputs, not model constants — the XML kp/kv are placeholders).
// PD active iff q_ref AND kp (the Python wrapper raises on q_ref-without-kp /
// qd_ref-without-kd, mirroring tact's Model.step; here a missing gain just
// leaves its term off). When PD is not supported by the loaded XML, tau is the
// legacy full-ctrl vector (m->nu) — still a generalized force in tact's
// convention. (The former step/step_internal split — a relic of the legacy
// step(u, y) compat body — was merged 2026-06-07; single caller, and the
// folding no longer needs a tau_eff side buffer.)
extern "C" int step(double* tau, double* q_ref, double* qd_ref, double* kp, double* kd, double* y){
    if(pd_supported){
        // write this step's gains into the position actuators (stateless)
        apply_pd_gains(q_ref, kp, kd);
        bool pd_on = (q_ref != NULL && kp != NULL);

        // motor actuators receive tau (feedforward torque); when the servo runs
        // with a velocity target, Kd·qd_ref is folded in so the position
        // actuator's intrinsic -Kd·qvel term yields full PD with both targets:
        //   τ_total = Kp·(q_ref-q) - Kd·qvel + (τ + Kd·qd_ref)
        //           = Kp·(q_ref-q) + Kd·(qd_ref-qvel) + τ
        for(int i = 0; i < n_motor; i++){
            double ff = tau[i];
            if(pd_on && qd_ref && kd) ff += kd[i] * qd_ref[i];
            cmd[motor_idx[i]] = ff;
        }
        // position actuators receive q_ref when active; harmless value otherwise
        for(int i = 0; i < n_position; i++) cmd[position_idx[i]] = pd_on ? q_ref[i] : 0.0;
    } else {
        // legacy XML (no motor↔position pairs, e.g. torque-only rb5m/rb10/
        // anymal_b_torque): m->nu single-channel actuators, tau is the full
        // ctrl vector; q_ref/kp/kd are ignored.
        for(int i = 0; i < m->nu; i++) cmd[i] = tau[i];
    }

    mju_zero(d->xfrc_applied, 6*m->nbody);
    if(render > 0) mjv_applyPerturbForce(m, d, &pert);
    mj_step(m, d);
    
    int idx = 0;		
    for(int i = 0; i < m->nsensor; i++){
	int adr  = m->sensor_adr[i];
	int dim  = m->sensor_dim[i];
	int type = m->sensor_type[i];

	for(int j = 0; j < dim; j++) {
	    y[idx] = d->sensordata[adr+j];
	    //printf("%f\n", d->sensordata[adr+j]);
	    idx++;
	}
    }

    if(render > 0 && n_step%render == 0){
	//d->geom_xpos[1*3+0] = 0.0001*n_step;
	//printf("%f %f %f\n", d->geom_xpos[1*3+0], d->geom_xpos[1*3+1], d->geom_xpos[1*3+2]);

	mjrRect viewport = {0, 0, 0, 0};  // get framebuffer viewport
	glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
	mjv_updateScene(m, d, &v_opt, &pert, &cam, mjCAT_ALL, &scn); // update scene and render
	mjr_render(viewport, &scn, &con);
	glfwSwapBuffers(window); // swap OpenGL buffers (blocking call due to v-sync)
	glfwPollEvents(); // process pending GUI events, call GLFW callbacks
	if (glfwWindowShouldClose(window)) return -1;
    }

    n_step++;
    return 0;
}

// Gym-style reset: restore initial state and write the post-reset observation into y
// (without advancing physics by one mj_step). mj_forward + mj_sensor recomputes derived
// quantities and the sensor data array based on qpos0/qvel0; the per-sensor copy mirrors
// the loop inside step() so y has the same layout as a normal step output.
extern "C" void reset(double* y){
    mj_resetData(m, d);
    mj_forward(m, d);   // mj_forward internally runs mj_sensorPos/Vel/Acc, filling d->sensordata

    for(int i = 0; i < 16; i++){
	d->eq_active[i] = eq_active0[i];
    }

    int idx = 0;
    for(int i = 0; i < m->nsensor; i++){
	int adr = m->sensor_adr[i];
	int dim = m->sensor_dim[i];
	for(int j = 0; j < dim; j++) y[idx++] = d->sensordata[adr+j];
    }
}


extern "C" void finish(){
    ;
    
    /*if(render > 0){
	mjv_freeScene(&scn);
	mjr_freeContext(&con);
    }
    
    mj_deleteData(d);
    mj_deleteModel(m);*/
}


// NOTE: unlock()/is_locked() (XML weld staging — the mujoco twin of tact's
// removed `lock:` mechanism) were deleted 2026-06-06, sim-trick reduction.
// The corresponding <equality><weld> blocks were removed from the project
// XMLs at the same time.


extern "C" void push(char* msg){
    
    //do nothing
    ; //printf("msg:%s\n", msg);
}

extern "C" double raycast(double pos_x, double pos_y, double pos_z, double dir_x, double dir_y, double dir_z){
    double pos[3] = {pos_x, pos_y, pos_z};
    double dir[3] = {dir_x, dir_y, dir_z};

    int geomid;
    double dist = mj_ray(m, d, pos, dir, NULL, 1, -1, &geomid, NULL);
    return dist;
}
    

// body id lookup for sensor attachment — CEnv resolves the YAML sensor spec's
// `body` name against the loaded XML once at init (plan (a): one YAML drives
// every backend's sensors). -1 = not in this model (CEnv warns and skips).
extern "C" int body_id(const char* name){
    return mj_name2id(m, mjOBJ_BODY, name);
}

// raycast_frame — the mjenv twin of tact_raycast_frame: n rays from a
// body-attached sensor frame. T_off = the frame-in-body transform (row-major
// 4x4 — tact Model.ftran of the YAML lidar frame, so the SAME spec positions
// the sensor on both backends). dirs = unit ray directions in the frame's
// REGISTERED coordinates (Python _ray_grid_dirs output verbatim — single
// source; the -90° optical roll is already folded in). bid < 0 → frame is
// world-attached (YAML body 'root'; T_off IS the world pose). Visibility:
// geom groups 1..5 (robot geoms live in group 0, mirroring tact's
// `raycast: false` robot shapes — height_scan stays terrain-only, group 1).
// t_out[k] = forward range along dirs[k] (meters; dirs unit), -1 = no hit.
extern "C" void raycast_frame(int bid, double* T_off, double* dirs, int n, double* t_out){
    mjtByte group[6] = {0, 1, 1, 1, 1, 1};
    int geomid;
    double Rw[9], ow[3];
    if(bid < 0){
        for(int r = 0; r < 3; r++){
            for(int c = 0; c < 3; c++) Rw[3*r+c] = T_off[4*r+c];
            ow[r] = T_off[4*r+3];
        }
    } else {
        double* xm = d->xmat + 9*bid;   // body world rotation (row-major)
        double* xp = d->xpos + 3*bid;
        for(int r = 0; r < 3; r++){     // Rw = xm @ R_off ; ow = xp + xm @ p_off
            for(int c = 0; c < 3; c++)
                Rw[3*r+c] = xm[3*r+0]*T_off[0+c] + xm[3*r+1]*T_off[4+c] + xm[3*r+2]*T_off[8+c];
            ow[r] = xp[r] + xm[3*r+0]*T_off[3] + xm[3*r+1]*T_off[7] + xm[3*r+2]*T_off[11];
        }
    }
    for(int k = 0; k < n; k++){
        double dx = dirs[3*k], dy = dirs[3*k+1], dz = dirs[3*k+2];
        double vec[3] = {Rw[0]*dx + Rw[1]*dy + Rw[2]*dz,
                         Rw[3]*dx + Rw[4]*dy + Rw[5]*dz,
                         Rw[6]*dx + Rw[7]*dy + Rw[8]*dz};
        double dist = mj_ray(m, d, ow, vec, group, 1, -1, &geomid, NULL);
        t_out[k] = (dist < 0) ? -1.0 : dist;
    }
}

// height_scan — the FULL terrain-scan contract in C, mirroring tact's
// Env.height_scan (the Python-side CEnv._height_scan is a thin alloc+dict
// shim). offsets = G (x, y) pairs in the gravity-aligned base-yaw frame;
// each scan point casts a vertical ray from z_top down against group-1 geoms
// only (terrain/floor convention — robot geoms live in group 0, mirroring
// tact's `raycast: false` robot shapes), plus one under-base reference ray.
// h_out[i] = terrain top RELATIVE to the terrain under base_xy (dflt where
// the ray misses); valid_out[i] ∈ {0,1}; *base_valid_out / *ref_out describe
// the reference ray (ref 0 when the base ray misses). Absolute world-z never
// crosses the interface — only the relative quantities leave this function.
// (Replaced the per-point get_z export 2026-06-07; widened to the full
// contract the same day.)
extern "C" void height_scan(double* base_xy, double yaw, double* offsets, int G,
                            double z_top, double dflt,
                            double* h_out, int* valid_out,
                            int* base_valid_out, double* ref_out){
    double ray[3] = {0, 0, -1};
    // terrain = group-1 only (robot group-0 excluded); full mjNGROUP-length
    // mask — mj_ray reads 6 entries (the old 2-length array under-allocated)
    mjtByte group[6] = {0, 1, 0, 0, 0, 0};
    int geomid;
    double cy = cos(yaw), sy = sin(yaw);

    // under-base reference ray
    double pos[3] = {base_xy[0], base_xy[1], z_top};
    double dist = mj_ray(m, d, pos, ray, group, 1, -1, &geomid, NULL);
    *base_valid_out = (dist >= 0);
    *ref_out = (dist >= 0) ? z_top - dist : 0.0;

    for(int i = 0; i < G; i++){
        double ox = offsets[2*i], oy = offsets[2*i+1];
        double p[3] = {base_xy[0] + cy*ox - sy*oy,
                       base_xy[1] + sy*ox + cy*oy, z_top};
        dist = mj_ray(m, d, p, ray, group, 1, -1, &geomid, NULL);
        valid_out[i] = (dist >= 0);
        h_out[i] = (dist >= 0) ? (z_top - dist) - *ref_out : dflt;
    }
}

extern "C" int get_rgb_image(const char* frame, unsigned char* out_buf){
    const int W = 640, H = 480;

    static int rgb_init = 0;
    static tjhandle tj_handle = NULL;
    static unsigned char* jpeg_buf = NULL;
    static unsigned long max_jpeg_len = 0;
    static unsigned char rgb_buf[640*480*3];
    static unsigned char tmp_row[640*3];
    static mjvCamera rgb_cam;
    static mjvScene rgb_scn;

    if(render <= 0) return 0;  //no GL context

    if(!rgb_init){
        tj_handle = tjInitCompress();
        max_jpeg_len = tjBufSize(W, H, TJSAMP_420);
        jpeg_buf = tjAlloc(max_jpeg_len);
        mjv_defaultCamera(&rgb_cam);
        mjv_defaultScene(&rgb_scn);
        mjv_makeScene(m, &rgb_scn, 2000);
        rgb_init = 1;
    }

    //resolve frame: site name first (tact convention), then <camera> name
    int sid   = mj_name2id(m, mjOBJ_SITE,   frame);
    int camid = (sid < 0) ? mj_name2id(m, mjOBJ_CAMERA, frame) : -1;
    if(sid < 0 && camid < 0) return 0;

    if(camid >= 0){
	rgb_cam.type = mjCAMERA_FIXED;
	rgb_cam.fixedcamid = camid;
    }
    else rgb_cam.type = mjCAMERA_USER;

    //offscreen rendering
    mjr_setBuffer(mjFB_OFFSCREEN, &con);
    mjrRect viewport = {0, 0, W, H};

    //user-camera: build view from site pose BEFORE mjv_updateScene
    //(updateScene runs mjv_cameraInModel internally and validates frustum_near)
    //convention: site +X = look direction, site +Z = up
    if(sid >= 0){
	mjtNum* spos = d->site_xpos + 3*sid;
	mjtNum* smat = d->site_xmat + 9*sid;  //row-major 3x3
	float aspect = (float)W/(float)H;
	float fovy_deg = m->vis.global.fovy > 0 ? m->vis.global.fovy : 45.0f;
	float fovy = fovy_deg * (float)M_PI / 180.0f;
	//znear/zfar are fractions of model extent (MuJoCo convention)
	float znear = (float)(m->stat.extent * m->vis.map.znear);
	float zfar  = (float)(m->stat.extent * m->vis.map.zfar);
	float top = znear * tanf(fovy*0.5f);

	for(int eye = 0; eye < 2; eye++){
	    rgb_scn.camera[eye].pos[0] = (float)spos[0];
	    rgb_scn.camera[eye].pos[1] = (float)spos[1];
	    rgb_scn.camera[eye].pos[2] = (float)spos[2];
	    //forward = +x column of site rotation (look direction)
	    rgb_scn.camera[eye].forward[0] = (float)smat[0];
	    rgb_scn.camera[eye].forward[1] = (float)smat[3];
	    rgb_scn.camera[eye].forward[2] = (float)smat[6];
	    //up = +z column
	    rgb_scn.camera[eye].up[0] = (float)smat[2];
	    rgb_scn.camera[eye].up[1] = (float)smat[5];
	    rgb_scn.camera[eye].up[2] = (float)smat[8];

	    rgb_scn.camera[eye].frustum_near   = znear;
	    rgb_scn.camera[eye].frustum_far    = zfar;
	    rgb_scn.camera[eye].frustum_bottom = -top;
	    rgb_scn.camera[eye].frustum_top    =  top;
	    rgb_scn.camera[eye].frustum_center = 0.0f;
	    rgb_scn.camera[eye].frustum_width  = 2.0f * top * aspect;
	}
    }

    mjv_updateScene(m, d, &v_opt, NULL, &rgb_cam, mjCAT_ALL, &rgb_scn);

    mjr_render(viewport, &rgb_scn, &con);
    mjr_readPixels(rgb_buf, NULL, viewport, &con);

    //flip vertical (GL bottom-up -> top-down)
    for(int y = 0; y < H/2; y++){
	unsigned char* a = rgb_buf + y*W*3;
	unsigned char* b = rgb_buf + (H-1-y)*W*3;
	memcpy(tmp_row, a, W*3);
	memcpy(a, b, W*3);
	memcpy(b, tmp_row, W*3);
    }

    unsigned long jpeg_len = max_jpeg_len;
    int rc = tjCompress2(tj_handle, rgb_buf, W, 0, H, TJPF_RGB,
			 &jpeg_buf, &jpeg_len, TJSAMP_420, 75, TJFLAG_NOREALLOC);

    //restore window buffer so step()'s window render keeps working
    mjr_setBuffer(mjFB_WINDOW, &con);

    if(rc != 0) return 0;
    memcpy(out_buf, jpeg_buf, jpeg_len);
    return (int)jpeg_len;
}

/*extern "C" int get_num_output(){
    return m->nsensordata;
}

extern "C" int get_num_input(){
    return m->nu;
    }*/


/*
extern "C" int scan3(double x, double y, double z, double roll, double pitch, double yaw, double *data){
    //double* pos = d->site_xpos+site_idx*3;
    //double* R = d->site_xmat+site_idx*9;
    //rpy2rotation(rpy, R);
    
    double pos[3] = {x, y, z};
    double rpy[3] = {roll, pitch, yaw};
    
    double R[9];
    rpy2rotation(rpy, R);

    double Ry[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double Rz[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    double tmp1[9];
    double tmp2[9];
    double ray[3];

    int n_cp = 0; //num. contact point
    int geomid;

    //ray collision with geom group-0 only
    mjtByte ray_collision_group[2] = {1, 0};
    
    for(int i = 0; i < 21; i++){
	double h = (-15.0 + 1.5*i)*M_PI/180.0;
	Ry[0] = cos(h);
	Ry[2] = sin(h);
	Ry[6] = -sin(h);
	Ry[8] = cos(h);
	mju_mulMatMat(tmp1, R, Ry, 3, 3, 3);
	
	for(int j = 0; j < 21; j++){
	    double w =  (-15.0 + 1.5*j)*M_PI/180.0;
	    Rz[0] = cos(w);
	    Rz[1] = -sin(w);
	    Rz[3] = sin(w);
	    Rz[4] = cos(w);
	    
	    mju_mulMatMat(tmp2, tmp1, Rz, 3, 3, 3);
	    
	    ray[0] = tmp2[0];
	    ray[1] = tmp2[3];
	    ray[2] = tmp2[6];

	    //double dist = mj_ray(m, d, pos, ray, ray_collision_group, 1, -1, &geomid);
	    double dist = mj_ray(m, d, pos, ray, NULL, 1, -1, &geomid);
	    if(dist < 0) continue;
			     
	    for(int k = 0; k < 3; k++) data[3*n_cp+k] = pos[k] + ray[k]*dist;
	    n_cp++;	    
	}
    }

    return n_cp;
}


extern "C" int pose6(int *code, double* pos, double* rpy){
    double R[9]; //rotation matrix;

    //random piece picking...
    int off = 34;
    int num_obj = 7;

    
    for(int i = 0; i < num_obj; i++){
	int idx = off + i;

	memcpy(pos+3*i, d->geom_xpos+idx*3, 3*sizeof(double));
	memcpy(R,   d->geom_xmat+idx*9, 9*sizeof(double));

	//mju_mat2Quat(quat, R);
	rotation2rpy(R, rpy+3*i);
	
	int type, _code;
	double size[3];

	memcpy(&type, m->geom_type+idx, sizeof(int));
	memcpy(size, m->geom_size+3*idx, 3*sizeof(double));

	//double user_id;
	//memcpy(&user_id, m->geom_user+idx, sizeof(mjtNum));
	//if(user_id > 0)_code = (int) user_id;
	
	if (type == mjGEOM_SPHERE) _code = 1000 + (int)(size[0]*2000);
	else if (type == mjGEOM_CYLINDER) _code = 1000000 + (int)(size[0]*2000000) + (int)(size[1]*2000);
	else if (type == mjGEOM_BOX) _code = 1000000000 + (int)(size[0]*2000000000) + (int)(size[1]*2000000) + (int)(size[2]*2000);
	code[i] = _code;

	//printf("_code=%d\n", _code);
    }
    
    return num_obj;
}


extern "C" void scan3(int site_idx, double *data){
    double* pos = d->site_xpos+site_idx*3;
    double* R = d->site_xmat+site_idx*9;
    
    double rpy[3];
    rotation2rpy(R, rpy);
    rpy[0] = 0;
    rpy[1] = 0;
    rpy2rotation(rpy, R);
    
    int idx = 0;
    int geomid;

    double Ry[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double Rz[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    double tmp1[9];
    double tmp2[9];
    double ray[3];
    
    //ray collision with geom group-0 only
    mjtByte ray_collision_group[2] = {1, 0};
    
    for(int i = 0; i < 21; i++){
	double h = (60.0 + 3.0*i)*M_PI/180.0;
	Ry[0] = cos(h);
	Ry[2] = sin(h);
	Ry[6] = -sin(h);
	Ry[8] = cos(h);
	mju_mulMatMat(tmp1, R, Ry, 3, 3, 3);
	
	for(int j = 0; j < 21; j++){
	    double w =  (-30.0 + 3.0*j)*M_PI/180.0;
	    Rz[0] = cos(w);
	    Rz[1] = -sin(w);
	    Rz[3] = sin(w);
	    Rz[4] = cos(w);
	    
	    mju_mulMatMat(tmp2, tmp1, Rz, 3, 3, 3);

	    //printf("%f %f %f   %f %f %f   %f %f %f\n", tmp2[0], tmp2[1], tmp2[2], tmp2[3], tmp2[4], tmp2[5], tmp2[6], tmp2[7], tmp2[8]);
	    
	    ray[0] = tmp2[0];
	    ray[1] = tmp2[3];
	    ray[2] = tmp2[6];

	    //printf("%f %f %f\n", ray[0], ray[1], ray[2]);

	    //double dist = mj_ray(m, d, pos, ray, ray_collision_group, 1, -1, &geomid);
	    double dist = mj_ray(m, d, pos, ray, NULL, 1, -1, &geomid);

	    //if(dist > 100.0) dist = 100.0;
	    //data[idx] = dist;
	    //idx++;

	    double cp[3] = {0, 0, 0}; //contact point
	    if(dist > 0){
		cp[0] = pos[0] + ray[0]*dist;
		cp[1] = pos[1] + ray[1]*dist;
		cp[2] = pos[2] + ray[2]*dist;
	    }
	    
	    for(int k = 0; k < 3; k++){
		data[idx] = cp[k];
		idx++;
	    }
	}
    }
    }
*/


/*extern "C" void scan2(int site_idx, double* data){
    double pos[3]; //cartesian position
    double R[9]; //rotation matrix;
    double rpy[3];
    double y_shift = 0.2;
    
    memcpy(pos, d->site_xpos+site_idx*3, 3*sizeof(double));
    memcpy(R,   d->site_xmat+site_idx*9, 9*sizeof(double));
    
    rotation2rpy(R, rpy);
    rpy[0] = 0;
    rpy[1] = 0;
    rpy2rotation(rpy, R);

    double p[2][3];
    //double u[3] = {R[0], R[3], R[6]}; //x-axis unit vector
    double v[3] = {R[1], R[4], R[7]}; //y-axis unit vector

    for(int i=0; i<3; i++) {
	p[0][i] = pos[i] - y_shift*v[i];
	p[1][i] = pos[i] + y_shift*v[i];
    }


    //printf("%f %f %f   %f %f %f\n", p[0][0], p[0][1], p[0][2], p[1][0], p[1][1], p[1][2]);
    //printf("%f %f %f   %f %f %f\n", u[0], u[1], u[2], v[0], v[1], v[2]);
    
    double Ry[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double tmp[9];
    double ray[3];
    int geomid;
    int idx = 0;

    //ray collision with geom group-0 only
    mjtByte ray_collision_group[2] = {1, 0};

    
    for(int i = 0; i < 2; i++){
	for(int j = 0; j < 16; j++){
	    double h = (45.0 + 3.0*j)*M_PI/180.0;

	    Ry[0] = cos(h);
	    Ry[2] = sin(h);
	    Ry[6] = -sin(h);
	    Ry[8] = cos(h);

	    mju_mulMatMat(tmp, R, Ry, 3, 3, 3);

	    ray[0] = tmp[0];
 	    ray[1] = tmp[3];
	    ray[2] = tmp[6];
	
	    //double dist = mj_ray(m, d, p[i], ray, NULL, 1, -1, &geomid);
	    double dist = mj_ray(m, d, p[i], ray, ray_collision_group, 1, -1, &geomid);
	    
	    data[idx] = dist;
	    idx++;
	}
    }
}


extern "C" void scan4(int site_idx, double* data){
    double pos[3]; //cartesian position
    double R[9]; //rotation matrix;
    double rpy[3];
    double y_shift = 0.2;
    
    memcpy(pos, d->site_xpos+site_idx*3, 3*sizeof(double));
    memcpy(R,   d->site_xmat+site_idx*9, 9*sizeof(double));
    
    rotation2rpy(R, rpy);
    rpy[0] = 0;
    rpy[1] = 0;
    rpy2rotation(rpy, R);

    double p[2][3];
    double v[3] = {R[1], R[4], R[7]}; //y-axis unit vector

    for(int i=0; i<3; i++) {
	p[0][i] = pos[i] - y_shift*v[i];
	p[1][i] = pos[i] + y_shift*v[i];
    }

    double ray[3] = {0, 0, -1};
    int geomid;
    int idx = 0;

    //ray collision with geom group-0 only
    mjtByte ray_collision_group[2] = {1, 0};

    
    for(int i = 0; i < 2; i++){
	for(int j = 0; j < 16; j++){

	    p[i][0] += 0.02;
	    double dist = mj_ray(m, d, p[i], ray, ray_collision_group, 1, -1, &geomid);
	    
	    data[idx] = dist;
	    idx++;
	}
    }
}


extern "C" void scan6(int site_idx, double *data){
    int idx = 0;
    double pos[3]; //cartesian position
    double R[9]; //rotation matrix;
    double rpy[3];
    
    memcpy(pos, d->site_xpos+site_idx*3, 3*sizeof(double));
    memcpy(R,   d->site_xmat+site_idx*9, 9*sizeof(double));
    
    rotation2rpy(R, rpy);
    rpy[0] = 0.0;
    rpy[1] = 0.0;
    rpy2rotation(rpy, R);

    double ray[3];
    int geomid;
        
    ray[0] = R[0];
    ray[1] = R[3];
    ray[2] = R[6];

    mjtByte group[2] = {1, 0};

    for(int i=0; i<2; i++){
	double dist = mj_ray(m, d, pos, ray, group, 1, -1, &geomid);
	if(dist == -1) dist = 100.0;
	data[idx] = dist;
	pos[2] -= 0.05;
	idx++;
    }
}


extern "C" double height(int site_idx){
    double pos[3]; //cartesian position
    memcpy(pos, d->site_xpos+site_idx*3, 3*sizeof(double));

    double ray[3] = {0, 0, -1};
    int geomid;
    mjtByte group[2] = {1, 0};

    double dist = mj_ray(m, d, pos, ray, group, 1, -1, &geomid);
    if(dist == -1) dist = 100.0;
    return dist;
    }*/



/*extern "C" int map_type(){
    return -1;
    }*/


/*extern "C" int pose7(int idx, double* pos, double* quat){
    double R[9]; //rotation matrix;
    
    memcpy(pos, d->geom_xpos+idx*3, 3*sizeof(double));
    memcpy(R,   d->geom_xmat+idx*9, 9*sizeof(double));
    mju_mat2Quat(quat, R);
    
    int type, code;
    double size[3];
    memcpy(&type, m->geom_type+idx, sizeof(double));
    memcpy(size, m->geom_size+3*idx, 3*sizeof(double));
    
    if      (type == mjGEOM_SPHERE) code = 1000 + (int)(size[0]*2000);
    else if (type == mjGEOM_CYLINDER) code = 1000000 + (int)(size[0]*2000000) + (int)(size[1]*2000);
    else if (type == mjGEOM_BOX) code = 1000000000 + (int)(size[0]*2000000000) + (int)(size[1]*2000000) + (int)(size[2]*2000);

    return code;
    }*/
