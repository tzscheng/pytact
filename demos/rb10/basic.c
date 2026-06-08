#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <arpa/inet.h>
#include <math.h>

typedef union{
    struct{
        char    header[4];
        // 0
        float   time;                   // time [sec]
        float   jnt_ref[6];             // joint reference [deg]
        float   jnt_ang[6];             // joint encoder value [deg]
        float   cur[6];                 // joint current value [mA]
        // 19
        float   tcp_ref[6];             // calculated tool center point from reference [mm, deg]
        float   tcp_pos[6];             // calculated tool center point from encoder [mm, deg]
        // 31
        float   analog_in[4];           // analog input value of control box [V]
        float   analog_out[4];          // analog output value of control box [V]
        int     digital_in[16];         // digital input value of control box [0 or 1]
        int     digital_out[16];        // digital input value of control box [0 or 1]
        // 71
        float   temperature_mc[6];      // board temperature of each joint [celcius]
        // 77
        int     task_pc;                // (ignore)
        int     task_repeat;            // (ignore)
        int     task_run_id;            // (ignore)
        int     task_run_num;           // (ignore)
        float   task_run_time;          // (ignore)
        int     task_state;             // (ignore)
        // 83
        float   default_speed;          // overriding speed [0~1]
        int     robot_state;            // state of robot motion [1:idle  2:paused or stopped by accident  3: moving]
        int     power_state;            // power state
        // 86
        float   tcp_target[6];          // (ignore)
        int     jnt_info[6];            // joint information (look mSTAT)
        // 98
        int     collision_detect_onoff; // collision detect onoff [0:off  1:on]
        int     is_freedrive_mode;      // current freedrive status [0:off  1:on]
        int     program_mode;           // current program mode [0:real mode  1:simulation mode]
        // 101
        int     init_state_info;        // status information of robot initialization process
        int     init_error;             // error code of robot initialization process
        // 103
        float   tfb_analog_in[2];       // analog input value of tool flange board [V]
        int     tfb_digital_in[2];      // digital input value of tool flange board [0 or 1]
        int     tfb_digital_out[2];     // digital output value of tool flange board [0 or 1]
        float   tfb_voltage_out;        // reference voltage of tool flange board [0, 12, 24]
        // 110
        int     op_stat_collision_occur;
        int     op_stat_sos_flag;
        int     op_stat_self_collision;
        int     op_stat_soft_estop_occur;
        int     op_stat_ems_flag;
        // 115
        int     digital_in_config[2];
        // 117
        int     inbox_trap_flag[2];
        int     inbox_check_mode[2];
    }sdata;
}systemSTAT;

systemSTAT systemStat_1;

int fd0;//rb command
int fd1;//rb data_receive
char buf[4096];


void servo_on(void){
    sprintf(buf, "mc jall init");
    int ret = write(fd0, buf, strlen(buf));
    if (ret <=0){perror("error: "); exit(0);}    

    usleep(10000000);
    printf("io: servo on\n");

    sprintf(buf, "pgmode real");
    ret = write(fd0, buf, strlen(buf));
    if (ret <=0){perror("error: "); exit(0);}    
    
    usleep(1000000);
    printf("io: real mode\n");
}


void get_state(double* y){
    usleep(100000);
    sprintf(buf, "reqdata");

    int ret = write(fd1, buf, strlen(buf));
    if (ret <=0){ perror("error: "); exit(0); }    

    //receive joint position info.
    ssize_t len = read(fd1, buf, sizeof(buf));
    if (len < 0) {perror("read error:"); exit(0); }

    //if(buf[0] == '$')
    memcpy(&systemStat_1, buf, sizeof(systemSTAT));
    
    y[0] = systemStat_1.sdata.jnt_ang[0]*(M_PI/180);
    y[1] = systemStat_1.sdata.jnt_ang[1]*(M_PI/180);
    y[2] = systemStat_1.sdata.jnt_ang[2]*(M_PI/180);
    y[3] = systemStat_1.sdata.jnt_ang[3]*(M_PI/180);
    y[4] = systemStat_1.sdata.jnt_ang[4]*(M_PI/180);
    y[5] = systemStat_1.sdata.jnt_ang[5]*(M_PI/180);

    printf("joint angle: %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f\n", systemStat_1.sdata.jnt_ang[0], systemStat_1.sdata.jnt_ang[1], systemStat_1.sdata.jnt_ang[2], systemStat_1.sdata.jnt_ang[3], systemStat_1.sdata.jnt_ang[4], systemStat_1.sdata.jnt_ang[5]);
    printf("tcp pos: %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f\n", systemStat_1.sdata.tcp_pos[0], systemStat_1.sdata.tcp_pos[1], systemStat_1.sdata.tcp_pos[2], systemStat_1.sdata.tcp_pos[3], systemStat_1.sdata.tcp_pos[4], systemStat_1.sdata.tcp_pos[5]);
}


void set_joint_pos(double* v){    
    sprintf(buf, "jointall 0.05, 0.05, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf", v[0]*(180/M_PI), v[1]*(180/M_PI), v[2]*(180/M_PI), v[3]*(180/M_PI), v[4]*(180/M_PI), v[5]*(180/M_PI));
    printf("io: %s\n", buf);
    int ret = write(fd0, buf, strlen(buf));
    if (ret <=0) {perror("error: "); exit(0);}
}

void set_joint_pos2(double* v){    
    sprintf(buf, "jointall 0.05, 0.05, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf", v[0], v[1], v[2], v[3], v[4], v[5]);
    printf("io: %s\n", buf);
    int ret = write(fd0, buf, strlen(buf));
    if (ret <=0) {perror("error: "); exit(0);}
}

void set_task_pos(double* v){    
    sprintf(buf, "movetcp 0.02, 0.02, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf", v[0]*1000, v[1]*1000, v[2]*1000, v[3]*(180/M_PI), v[4]*(180/M_PI), v[5]*(180/M_PI));
    printf("io: %s\n", buf);
    int ret = write(fd0, buf, strlen(buf));
    if (ret <=0) {perror("error: "); exit(0);}
}

void set_task_pos2(double* v){    
    sprintf(buf, "movetcp 0.02, 0.02, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf", v[0], v[1], v[2], v[3], v[4], v[5]);
    printf("io: %s\n", buf);
    int ret = write(fd0, buf, strlen(buf));
    if (ret <=0) {perror("error: "); exit(0);}
}

void set_joint_path(int num, double* v){
    int i;
    int len = sprintf(buf, "move_jb_clear()\n");
    for(i=0; i < num; i++) len += sprintf(buf+len, "move_jb_add(jnt[%.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf])\n", v[6*i+0]*(180/M_PI), v[6*i+1]*(180/M_PI), v[6*i+2]*(180/M_PI), v[6*i+3]*(180/M_PI), v[6*i+4]*(180/M_PI), v[6*i+5]*(180/M_PI));
    sprintf(buf+len, "move_jb_run(10, 10)");
    printf("io: %s\n", buf);
    int ret = write(fd0, buf, strlen(buf));
    if (ret <=0) {perror("error: "); exit(0);}

}

void set_task_path(int num, double* v){
    int i;
    int len = sprintf(buf, "move_pb_clear()\n");
    //for(i=0; i < num; i++) len += sprintf(buf+len, "move_pb_add(pnt[%.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf], 50, 0, 0.5)\n", v[6*i+0]*1000, v[6*i+1]*1000, v[6*i+2]*1000, v[6*i+3]*(180/M_PI), v[6*i+4]*(180/M_PI), v[6*i+5]*(180/M_PI));
    for(i=0; i < num; i++) len += sprintf(buf+len, "move_pb_add(pnt[%.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf], 100, 1, 10)\n", v[6*i+0]*1000, v[6*i+1]*1000, v[6*i+2]*1000, v[6*i+3]*(180/M_PI), v[6*i+4]*(180/M_PI), v[6*i+5]*(180/M_PI));
    sprintf(buf+len, "move_pb_run(100, 0)");
    printf("io: %s\n", buf);
    int ret = write(fd0, buf, strlen(buf));
    if (ret <=0) {perror("error: "); exit(0);}
}


int main(void) {
    struct sockaddr_in server_addr0;
    struct sockaddr_in server_addr1;

    printf("start\n");
    
    //memset(&server_addr0, 0, sizeof(server_addr0));
    server_addr0.sin_family = AF_INET;
    //server_addr0.sin_addr.s_addr = inet_addr("192.168.0.8");
    server_addr0.sin_addr.s_addr = inet_addr("192.168.7.151");
    server_addr0.sin_port = htons((unsigned short)5000);
    
    if ((fd0 = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0){
	perror("socket error1: ");
	return 0;
    }
    
    if (connect(fd0, (struct sockaddr *)&server_addr0, sizeof(server_addr0)) < 0) {
	perror("connect error1: ");
	return 0;
    }

    server_addr1.sin_family = AF_INET;
    //server_addr1.sin_addr.s_addr = inet_addr("192.168.0.8");
    server_addr1.sin_addr.s_addr = inet_addr("192.168.7.151");
    server_addr1.sin_port = htons((unsigned short)5001);
    
    if ((fd1 = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0){
	perror("socket error: ");
	return 0;
    }
    
    if (connect(fd1, (struct sockaddr *)&server_addr1, sizeof(server_addr1)) < 0) {
	perror("connect error2: ");
	return 0;
    }
    
    printf("connected!\n");

    char op[64];
    double v[64];
    double y[6];
    
    while(1){
	printf("input: ");
	fgets(buf, sizeof(buf), stdin);

	int ret = sscanf(buf, "%s %lf %lf %lf %lf %lf %lf", op, v, v+1, v+2, v+3, v+4, v+5);
	if (ret <= 0 ) {printf("wrong msg\n"); exit(-1); }

	if(strcmp(op, "servo_on")==0 && ret == 1){
	    servo_on();
	    printf("SERVO_ON\n");
	    
	}

	else if(strcmp(op, "shutdown")==0 && ret == 1){
	    sprintf(buf, "shutdown");
	    int ret = write(fd0, buf, strlen(buf));
	    if (ret <=0){perror("error: "); exit(0);}    
	    usleep(10000000);
	    printf("io: shutdown\n");
	}
	
	else if(strcmp(op, "state")==0 && ret == 1){
	    get_state(y);
	    //printf("get joint pos: %lf %lf %lf %lf %lf %lf\n", y[0], y[1], y[2], y[3], y[4], y[5]);
	}

	else if(strcmp(op, "set_joint")==0 && ret == 7){
	    set_joint_pos(v);
	    printf("set joint pos: %lf %lf %lf %lf %lf %lf\n", v[0], v[1], v[2], v[3], v[4], v[5]);
	}
	
	else if(strcmp(op, "home")==0){
	    v[0] = 0.0;
	    v[1] = -0.54;
	    v[2] = 2.12;
	    v[3] = 0.0;
	    v[4] = M_PI/2;
	    v[5] = 0.0;
	    set_joint_pos(v);
	}

	else if(strcmp(op, "zero")==0){
	    v[0] = 0.0;
	    v[1] = 0.0;
	    v[2] = 0.0;
	    v[3] = 0.0;
	    v[4] = 0.0;
	    v[5] = 0.0;
	    set_joint_pos(v);
	}
	
	else if(strcmp(op, "task1")==0){
	    v[0] = 0.35;
	    v[1] = -0.16;
	    v[2] = 0.50;
	    v[3] = M_PI/2;
	    v[4] = 0.0;
	    v[5] = M_PI/2;	    
	    set_task_pos(v);
	}
      
	else if(strcmp(op, "task2")==0){
	    v[0] = 0.40;
	    v[1] = -0.16;
	    v[2] = 0.50;
	    v[3] = M_PI/2;
	    v[4] = 0.0;
	    v[5] = M_PI/2;
	    set_task_pos(v);
	}

	else if(strcmp(op, "path1")==0){
	    v[0] = 0.50;
	    v[1] = -0.16;
	    v[2] = 0.50;
	    v[3] = M_PI/2;
	    v[4] = 0.0;
	    v[5] = M_PI/2;
	    
	    v[6] = 0.50;
	    v[7] = -0.16;
	    v[8] = 0.40;
	    v[9] = M_PI/2;
	    v[10] = 0.0;
	    v[11] = M_PI/2;

	    set_task_path(2, v);
	}

	else if(strcmp(op, "path2")==0){
	    v[0] = 0.5;
	    v[1] = -0.54;
	    v[2] = 2.12;
	    v[3] = 0.0;
	    v[4] = M_PI/2;
	    v[5] = 0.0;

	    v[6] = 0.0;
	    v[7] = -0.54;
	    v[8] = 2.12;
	    v[9] = 0.0;
	    v[10] = M_PI/2;
	    v[11] = 0.0;
	    
	    v[12] = 0.5;
	    v[13] = -0.54;
	    v[14] = 2.12;
	    v[15] = 0.0;
	    v[16] = M_PI/2;
	    v[17] = 0.0;

	    v[18] = 0.0;
	    v[19] = -0.54;
	    v[20] = 2.12;
	    v[21] = 0.0;
	    v[22] = M_PI/2;
	    v[23] = 0.0;

	    
	    set_joint_path(4, v);
	}
	
	else if (strcmp(op, "quit")==0) { break; }
	else { printf("wrong op\n"); }
    }

    return 0;
}



/*void set_joint_pos(double* v){    
    sprintf(buf, "move_j(jnt[%.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf], 10, 10)", v[0]*(180/M_PI), v[1]*(180/M_PI), v[2]*(180/M_PI), v[3]*(180/M_PI), v[4]*(180/M_PI), v[5]*(180/M_PI));    
    printf("io: %s\n", buf);
    int ret = write(fd0, buf, strlen(buf));
    if (ret <=0) {perror("error: "); exit(0);}
}


void set_task_pos(double* v){    
    sprintf(buf, "move_l(pnt[%.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf], 50, 20)", v[0]*1000, v[1]*1000, v[2]*1000, v[3]*(180/M_PI), v[4]*(180/M_PI), v[5]*(180/M_PI));    
    printf("io: %s\n", buf);
    int ret = write(fd0, buf, strlen(buf));
    if (ret <=0) {perror("error: "); exit(0);}
    }*/


