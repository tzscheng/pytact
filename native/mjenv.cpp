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

// dual-actuator support (option 2: runtime gainprm toggle for PD enable/disable).
// Convention: every actuated joint has one motor actuator (biastype=NONE) for feedforward
// torque AND one position actuator (biastype=AFFINE) for implicit-style PD. The agent
// publishes (u, q_ref); when q_ref is present, position actuators run at nominal gain;
// when absent, they're zeroed (effectively disabled).
//
// XML must satisfy: n_motor == n_position, and the i-th motor pairs with the i-th
// position actuator (by transmission joint id). Mismatch is reported at init and PD
// support degrades gracefully (q_ref is ignored, only motor torques flow).
int   motor_idx[256];               // ctrl-space indices of motor actuators
int   position_idx[256];            // ctrl-space indices of position actuators
int   n_motor = 0;
int   n_position = 0;
double pos_gain_nominal[256 * mjNGAIN];   // saved nominal gainprm of position actuators
double pos_bias_nominal[256 * mjNBIAS];   // saved nominal biasprm of position actuators
// pd_active_now reflects the *current state* of gainprm. XML loads position actuators
// at their nominal gains, so we start with true to keep state and gainprm consistent.
// init() then calls set_pd_active(false) which forces zeroing for a safe start.
bool  pd_active_now = true;
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

// classify actuators into motor (biastype=NONE) and position (biastype=AFFINE), save
// nominal gain/bias of position actuators so we can later toggle to zero and back.
// pd_supported is set only when XML has exactly matching motor↔position pairs in joint id.
static void classify_actuators(){
    n_motor = n_position = 0;
    for(int i = 0; i < m->nu; i++){
        if(m->actuator_biastype[i] == mjBIAS_NONE){
            if(n_motor < 256) motor_idx[n_motor++] = i;
        } else if(m->actuator_biastype[i] == mjBIAS_AFFINE){
            if(n_position < 256){
                position_idx[n_position] = i;
                memcpy(pos_gain_nominal + n_position*mjNGAIN, m->actuator_gainprm + i*mjNGAIN, sizeof(double)*mjNGAIN);
                memcpy(pos_bias_nominal + n_position*mjNBIAS, m->actuator_biasprm + i*mjNBIAS, sizeof(double)*mjNBIAS);
                n_position++;
            }
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

// toggle position-actuator gains: active=true restores nominal, false zeros them out.
// Idempotent (skips work if already in requested state). Only touches if pd_supported.
static void set_pd_active(bool active){
    if(!pd_supported || pd_active_now == active) return;
    for(int k = 0; k < n_position; k++){
        int i = position_idx[k];
        if(active){
            memcpy(m->actuator_gainprm + i*mjNGAIN, pos_gain_nominal + k*mjNGAIN, sizeof(double)*mjNGAIN);
            memcpy(m->actuator_biasprm + i*mjNBIAS, pos_bias_nominal + k*mjNBIAS, sizeof(double)*mjNBIAS);
        } else {
            memset(m->actuator_gainprm + i*mjNGAIN, 0, sizeof(double)*mjNGAIN);
            memset(m->actuator_biasprm + i*mjNBIAS, 0, sizeof(double)*mjNBIAS);
        }
    }
    pd_active_now = active;
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

    //classify actuators (motor/position pairs); start in PD-disabled mode for backward
    //compat with legacy step(u, y) callers — agent publishing q_ref will activate PD.
    classify_actuators();
    set_pd_active(false);
    
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


// shared body for step(). tau: per-motor torque (length n_motor), q_ref: per-position
// target (length n_position, may be NULL → PD disabled). When PD is not supported by
// the loaded XML, tau is the legacy full-ctrl vector (m->nu) — still a generalized
// force in tact's convention.
static int step_internal(double* tau, double* q_ref, double* y){
    if(pd_supported){
        // toggle position actuator gains based on q_ref presence (option 2: runtime gainprm)
        set_pd_active(q_ref != NULL);

        // motor actuators always receive tau (feedforward torque)
        for(int i = 0; i < n_motor; i++)    cmd[motor_idx[i]]    = tau[i];
        // position actuators receive q_ref when active; harmless value otherwise
        if(q_ref) for(int i = 0; i < n_position; i++) cmd[position_idx[i]] = q_ref[i];
        else      for(int i = 0; i < n_position; i++) cmd[position_idx[i]] = 0.0;
    } else {
        // legacy XML: m->nu single-channel actuators, tau is the full ctrl vector
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

// Unified step. tau: per-motor feed-forward torque (always generalized force in tact's
// convention; the old ambiguous "u" name is gone now that implicit PD is its own input).
// q_ref/qd_ref are optional (NULL → PD off, bit-identical to legacy behavior). When
// both targets are supplied and pd_supported, Kd·qd_ref is folded into the motor channel
// so the position actuator's intrinsic -Kd·qvel term yields full PD with both targets:
//   τ_total = Kp·(q_ref - q) - Kd·qvel + (τ + Kd·qd_ref) = Kp·(q_ref - q) + Kd·(qd_ref - qvel) + τ
// Kd[i] comes from the paired position actuator's biasprm[2] (MuJoCo convention:
// bias = -Kp·qpos - Kv·qvel, so Kv = -biasprm[2]).
extern "C" int step(double* tau, double* q_ref, double* qd_ref, double* y){
    if(pd_supported && q_ref && qd_ref){
        static double tau_eff[256];
        for(int k = 0; k < n_motor; k++){
            double kv = -pos_bias_nominal[k*mjNBIAS + 2];
            tau_eff[k] = tau[k] + kv * qd_ref[k];
        }
        return step_internal(tau_eff, q_ref, y);
    }
    return step_internal(tau, q_ref, y);
}


// Gym-style reset: restore initial state and write the post-reset observation into y
// (without advancing physics by one mj_step). mj_forward + mj_sensor recomputes derived
// quantities and the sensor data array based on qpos0/qvel0; the per-sensor copy mirrors
// the loop inside step_internal so y has the same layout as a normal step output.
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


extern "C" void unlock(){

    //2.3.7
    //m->eq_active[idx] = 0;

    //3.0.0
    for(int i = 0; i < 16; i++){
	//if(d->eq_active[i] != 0)
	d->eq_active[i] = 0;
    }
}


extern "C" int is_locked(){
    for(int i = 0; i < 16; i++){
	if(d->eq_active[i] != 0) return 1;
    }

    return 0;
}


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
    

extern "C" double get_z(double x, double y){
    double pos[3]; //cartesian position
    pos[0] = x;
    pos[1] = y;
    pos[2] = 10.0;
    
    double ray[3] = {0, 0, -1};
    int geomid;

    //exclude group-0 and include group-1
    mjtByte group[2] = {0, 1};

    double dist = mj_ray(m, d, pos, ray, group, 1, -1, &geomid, NULL);

    //printf("x:%lf  y:%lf  geomid: %d  dist: %lf\n", x, y, geomid, dist);
    
    double z = pos[2] - dist;
    return z;
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
