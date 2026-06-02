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

double q1, q2, q3, u1, u2, u3, v1, v2, v3, e1, e2, e3, g1, g2, g3;
double kp=100, kd=6;
double q1_d = 0.5, q2_d = 0.2, q3_d = 0.5;
long cnt;

int main(void) {
    // load XML model
    m = mj_loadXML("./3dof.xml", NULL, NULL, 0);
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


    // adjust camera angle
    cam.lookat[2] = -0.35;
    cam.distance = 2.0;
    cam.azimuth = -130;
    cam.elevation = -27.45;
    

    // run main loop, target real-time simulation and 60 fps rendering
    while (!glfwWindowShouldClose(window)) {
	mjtNum simstart = d->time;
	
	while (d->time - simstart < 1.0/60.0){
	    mj_step(m, d);
	    
	    q1 = d->sensordata[0];
	    q2 = d->sensordata[1];
	    q3 = d->sensordata[2];

	    v1 = d->sensordata[3];
	    v2 = d->sensordata[4];
	    v3 = d->sensordata[5];

	    g1 = -7.3575*cos(q2)*sin(q1)-2.4525*cos(q1)*sin(q3)-2.4525*cos(q2)*cos(q3)*sin(q1);
	    g2 = -2.4525*cos(q1)*sin(q2)-9.81*cos(q1)*(0.5*sin(q2)+0.25*cos(q3)*sin(q2));
	    g3 = -9.81*cos(q2)*(0.25*cos(q1)*sin(q3)+0.25*cos(q2)*cos(q3)*sin(q1))-2.4525*cos(q3)*sin(q1)*sin(q2)*sin(q2);
	    
	    e1 = q1_d - q1;
	    e2 = q2_d - q2;
	    e3 = q3_d - q3;
	    
	    u1 = kp*e1 - kd*v1 - g1;
	    u2 = kp*e2 - kd*v2 - g2;
	    u3 = kp*e3 - kd*v3 - g3;

	    if(cnt > 3000){
		d->ctrl[0] = u1;
		d->ctrl[1] = u2;
		d->ctrl[2] = u3;
	    }
	    
	    printf("%ld:  %6.3f %6.3f %6.3f\n", cnt, e1, e2, e3);
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
