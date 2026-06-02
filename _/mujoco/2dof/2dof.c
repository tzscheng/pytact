#include <stdio.h>
#include <math.h>
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

double q1, q2, u1, u2, v1, v2, e1, e2, g1, g2;
double kp=200, kd=10;

double q1_d = 1, q2_d = 1;  //desired joint position (radian) M_PI => pi
long cnt;

int main(void) {
    // load XML model
    m = mj_loadXML("./2dof.xml", NULL, NULL, 0);
    d = mj_makeData(m);

    // init GLFW
    if (!glfwInit()) mju_error("Could not initialize GLFW");
  
    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // run main loop, target real-time simulation and 60 fps rendering
    while (!glfwWindowShouldClose(window)) {
	mjtNum simstart = d->time;
	
	while (d->time - simstart < 1.0/60.0){
	    mj_step(m, d);

	    //get joint position
	    q1 = d->sensordata[0];
	    q2 = d->sensordata[1];

	    //get joint velocity
	    v1 = d->sensordata[2];
	    v2 = d->sensordata[3];

	    //get gravity induced joint torque
	    g1 = 7.3575*sin(q1) + 2.4525*cos(q1)*sin(q2) + 2.4525*cos(q2)*sin(q1);
	    g2 = 2.4525*cos(q1)*sin(q2) + 2.4525*cos(q2)*sin(q1);

	    //get position error
	    e1 = q1_d - q1;
	    e2 = q2_d - q2;

	    //calculate joint torque (joint PD control with gravity compensation)
	    u1 = kp*e1 - kd*v1 - g1;
	    u2 = kp*e2 - kd*v2 - g2;

	    //set joint torque
	    d->ctrl[0] = u1;
	    d->ctrl[1] = u2;

	    //print joint position error
	    printf("%ld:  %6.3f %6.3f\n", cnt, e1, e2);
	    cnt++;
	}
	
	// get framebuffer viewport
	mjrRect viewport = {0, 0, 0, 0};
	glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

	// update scene and render
	mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
	mjr_render(viewport, &scn, &con);

	// swap OpenGL buffers (blocking call due to v-sync)
	glfwSwapBuffers(window);
	
	// process pending GUI events, call GLFW callbacks
	glfwPollEvents();
    }

    //free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data
    mj_deleteData(d);
    mj_deleteModel(m);

    return 0;
}
