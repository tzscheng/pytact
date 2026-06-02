#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
// tact.h is intentionally NOT included here — its #define SPHERE 102 collides with
// Chrono's `enum class ChGeometry::Type { ... SPHERE, ... }`. This file only depends
// on Chrono and libm anyway; the legacy pose6() helper that used tact's
// rotation_to_eulerxyz is kept commented out below.

#include "chrono/physics/ChSystemNSC.h"
#include "chrono/physics/ChBody.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/physics/ChLinkMotorRotationTorque.h"
#include "chrono/physics/ChLinkMotorLinearPosition.h"
#include "chrono/physics/ChLinkMotorLinearForce.h"
#include "chrono/physics/ChLinkMate.h"
#include "chrono_irrlicht/ChVisualSystemIrrlicht.h"

#define TOOLFACE_X -0.27155
#define TOOLFACE_Y 1.497

using namespace chrono;
using namespace chrono::irrlicht;


int render = 0;
int tool = 0;
int env = 0;
int num_obj = 0;

long cnt = 0;
int objtype[32];
bool suction_on = false;

double z[32]; //net force/torque
double tau[32]; //control input force/torque (unused at file scope — kept for legacy reference; step param shadows it)
double f[32]; //joint friction
double q[32]; //joint position
double d[32]; //joint speed
double h[32]; //joint reaction torque

ChSystemNSC sys;
auto vis = chrono_types::make_shared<ChVisualSystemIrrlicht>();

std::shared_ptr<ChBody> body[64];
std::shared_ptr<ChMarker> mark[32];
std::shared_ptr<ChContactMaterialNSC> mat[32];
std::shared_ptr<ChLinkMateFix> fix[32];
std::shared_ptr<ChLinkMotorRotation> rev[32];
std::shared_ptr<ChLinkMotorLinear> lin[32];
std::shared_ptr<ChFunctionSetpoint> msp[32];
//std::shared_ptr<ChBoxShape> vbox[32];


double random_pos_x() {return 0.70 + 0.01*(double)(rand()%40) - 0.20;}
double random_pos_y() {return -0.35 + 0.01*(double)(rand()%40) - 0.20;}
double random_pos_z() {return 0.20 + 0.01*(double)(rand()%20) - 0.10;}
    

void build_model(){
    double _r, _h, _m, _d, _x, _y, _z;

    //robot material
    mat[0] = chrono_types::make_shared<ChContactMaterialNSC>();
    mat[0]->SetFriction(0.8);
    mat[0]->SetRollingFriction(0.005);
    mat[0]->SetSpinningFriction(0.00005);

    //object material
    mat[1] = chrono_types::make_shared<ChContactMaterialNSC>();
    //mat[1]->SetRestitution(0.3);
    mat[1]->SetFriction(0.5);
    mat[1]->SetRollingFriction(0.005);
    mat[1]->SetSpinningFriction(0.00005);
    //mat[1]->SetCohesion(0.01);
	
    _r = 0.12; _h = 0.05; _m = 2.0; _d = _m/(CH_PI*_r*_r*_h);
    body[0] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, true, mat[0]);
    body[0]->SetPos(ChVector3d(0, 0.025, 0));
    body[0]->SetFixed(true);
    sys.Add(body[0]);

    _r = 0.08; _h = 0.25; _m = 2.0; _d = _m/(CH_PI*_r*_r*_h);
    body[1] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, true, mat[0]);
    body[1]->SetPos(ChVector3d(0, 0.197, 0));
    sys.Add(body[1]);

    _r = 0.08; _h = 0.16; _m = 1.0; _d = _m/(CH_PI*_r*_r*_h);
    body[2] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, true, mat[0]);
    body[2]->SetPos(ChVector3d(-0.1875, 0.197, 0));
    body[2]->SetRot(QuatFromAngleZ(CH_PI_2));
    sys.Add(body[2]);

    _r = 0.06; _h = 0.45; _m = 2.0; _d = _m/(CH_PI*_r*_r*_h);
    body[3] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, true, mat[0]);
    body[3]->SetPos(ChVector3d(-0.1875, 0.52, 0));
    sys.Add(body[3]);

    _r = 0.05; _h = 0.14; _m = 1.0; _d = _m/(CH_PI*_r*_r*_h);
    body[4] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, true, mat[0]);
    body[4]->SetPos(ChVector3d(-0.1875, 0.8097, 0));
    body[4]->SetRot(QuatFromAngleZ(CH_PI_2));
    sys.Add(body[4]);

    _r = 0.05; _h = 0.13; _m = 1.0; _d = _m/(CH_PI*_r*_r*_h);
    body[5] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, true, mat[0]);
    body[5]->SetPos(ChVector3d(-0.0391, 0.8097, 0));
    body[5]->SetRot(QuatFromAngleZ(CH_PI_2));
    sys.Add(body[5]);

    _r = 0.05; _h = 0.45; _m = 1.0; _d = _m/(CH_PI*_r*_r*_h);
    body[6] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, true, mat[0]);
    body[6]->SetPos(ChVector3d(-0.0391, 1.095, 0));
    sys.Add(body[6]);

    _r = 0.05; _h = 0.12; _m = 1.0; _d = _m/(CH_PI*_r*_r*_h);
    body[7] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, true, mat[0]);
    body[7]->SetPos(ChVector3d(-0.0391, 1.37985, 0));
    body[7]->SetRot(QuatFromAngleZ(CH_PI_2));
    sys.Add(body[7]);

    _r = 0.04; _h = 0.12; _m = 1.0; _d = _m/(CH_PI*_r*_r*_h);
    body[8] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, true, mat[0]);
    body[8]->SetPos(ChVector3d(-0.15625, 1.37985, 0));
    sys.Add(body[8]);

    _r = 0.04; _h = 0.14; _m = 1.0; _d = _m/(CH_PI*_r*_r*_h);
    body[9] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, true, mat[0]);
    body[9]->SetPos(ChVector3d(-0.15625, 1.497, 0));
    body[9]->SetRot(QuatFromAngleZ(CH_PI_2));
    sys.Add(body[9]);

    //END FACE: (0,-0.27155, 1.497)
    _r = 0.04; _h = 0.04; _m = 0.2; _d = _m/(CH_PI*_r*_r*_h);
    body[10] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, true, mat[0]);
    body[10]->SetPos(ChVector3d(-0.25155, 1.497, 0));
    body[10]->SetRot(QuatFromAngleZ(CH_PI_2));
    sys.Add(body[10]);
    

    fix[0] = chrono_types::make_shared<ChLinkMateFix>();
    fix[0]->Initialize(body[2], body[3]);
    sys.Add(fix[0]);

    fix[1] = chrono_types::make_shared<ChLinkMateFix>();
    fix[1]->Initialize(body[3], body[4]);
    sys.Add(fix[1]);

    fix[2] = chrono_types::make_shared<ChLinkMateFix>();
    fix[2]->Initialize(body[5], body[6]);
    sys.Add(fix[2]);

    fix[3] = chrono_types::make_shared<ChLinkMateFix>();
    fix[3]->Initialize(body[6], body[7]);
    sys.Add(fix[3]);

    
    rev[0] = chrono_types::make_shared<ChLinkMotorRotationTorque>();
    rev[0]->Initialize(body[1], body[0], ChFrame<>(ChVector3d(0, 0, 0), QuatFromAngleX(-CH_PI_2)));
    msp[0] = chrono_types::make_shared<ChFunctionSetpoint>();
    rev[0]->SetMotorFunction(msp[0]);
    sys.Add(rev[0]);
    
    rev[1] = chrono_types::make_shared<ChLinkMotorRotationTorque>();
    rev[1]->Initialize(body[1], body[2], ChFrame<>(ChVector3d(0, 0.1970, 0), QuatFromAngleY(-CH_PI_2)));
    msp[1] = chrono_types::make_shared<ChFunctionSetpoint>();
    rev[1]->SetMotorFunction(msp[1]);
    sys.Add(rev[1]);
    
    rev[2] = chrono_types::make_shared<ChLinkMotorRotationTorque>();
    rev[2]->Initialize(body[4], body[5], ChFrame<>(ChVector3d(0, 0.8097, 0), QuatFromAngleY(-CH_PI_2)));
    msp[2] = chrono_types::make_shared<ChFunctionSetpoint>();
    rev[2]->SetMotorFunction(msp[2]);
    sys.Add(rev[2]);

    rev[3] = chrono_types::make_shared<ChLinkMotorRotationTorque>();
    rev[3]->Initialize(body[7], body[8], ChFrame<>(ChVector3d(0, 1.37985, 0), QuatFromAngleY(-CH_PI_2)));
    msp[3] = chrono_types::make_shared<ChFunctionSetpoint>();
    rev[3]->SetMotorFunction(msp[3]);
    sys.Add(rev[3]);

    rev[4] = chrono_types::make_shared<ChLinkMotorRotationTorque>();
    rev[4]->Initialize(body[9], body[8], ChFrame<>(ChVector3d(-0.15625, 1.37985, 0), QuatFromAngleX(-CH_PI_2)));
    msp[4] = chrono_types::make_shared<ChFunctionSetpoint>();
    rev[4]->SetMotorFunction(msp[4]);
    sys.Add(rev[4]);

    rev[5] = chrono_types::make_shared<ChLinkMotorRotationTorque>();
    rev[5]->Initialize(body[9], body[10], ChFrame<>(ChVector3d(-0.15625, 1.497, 0), QuatFromAngleY(-CH_PI_2)));
    msp[5] = chrono_types::make_shared<ChFunctionSetpoint>();
    rev[5]->SetMotorFunction(msp[5]);
    sys.Add(rev[5]);


    //dummy mass due to unstability
    if (tool == 0){
	_x = 0.06; _y = 0.06; _z = 0.06;  _m = 0.2; _d = _m/(_x*_y*_z);
	body[11] = chrono_types::make_shared<ChBodyEasyBox>(_y, _z, _x, _d, true, false, mat[0]);
	body[11]->SetPos(ChVector3d(TOOLFACE_X-0.11, TOOLFACE_Y, 0));
	body[11]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/green1.png"));
	sys.Add(body[11]);

	fix[4] = chrono_types::make_shared<ChLinkMateFix>();
	fix[4]->Initialize(body[10], body[11]);
	sys.Add(fix[4]);
    }
    
    else if(tool == 1){
	_r = 0.04; _h = 0.22; _m = 0.2; _d = _m/(CH_PI*_r*_r*_h);
	body[11] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, false, mat[0]);
	body[11]->SetPos(ChVector3d(TOOLFACE_X-0.11, TOOLFACE_Y, 0));
	body[11]->SetRot(QuatFromAngleZ(CH_PI_2));
	body[11]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/gray1.png"));
	body[11]->EnableCollision(false);
	sys.Add(body[11]);

	_x = 0.12; _y = 0.02; _z = 0.02; _m = 0.02; _d = _m/(_x*_y*_z);
	body[12] = chrono_types::make_shared<ChBodyEasyBox>(_x, _y, _z, _d, true, true, mat[0]);
	body[12]->SetPos(ChVector3d(TOOLFACE_X-0.36, TOOLFACE_Y, 0.01));
	body[12]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/blue.png"));
	body[12]->GetCollisionModel()->SetDefaultSuggestedEnvelope(0.001);
	body[12]->GetCollisionModel()->SetDefaultSuggestedMargin(0.0005);
	sys.Add(body[12]);

	_x = 0.12; _y = 0.02; _z = 0.02; _m = 0.02; _d = _m/(_x*_y*_z);
	body[13] = chrono_types::make_shared<ChBodyEasyBox>(_x, _y, _z, _d, true, true, mat[0]);
	body[13]->SetPos(ChVector3d(TOOLFACE_X-0.36, TOOLFACE_Y, -0.01));
	body[13]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/gray1.png"));
	body[13]->GetCollisionModel()->SetDefaultSuggestedEnvelope(0.001);
	body[13]->GetCollisionModel()->SetDefaultSuggestedMargin(0.0005);
	sys.Add(body[13]);

	_r = 0.01; _h = 0.40; _m = 0.005; _d = _m/(CH_PI*_r*_r*_h);
	body[14] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, false, mat[0]);
	body[14]->SetPos(ChVector3d(TOOLFACE_X-0.05, TOOLFACE_Y, 0));
	body[14]->SetRot(QuatFromAngleZ(CH_PI_2));
	body[14]->EnableCollision(false);
	sys.Add(body[14]);

	_r = 0.015; _h = 0.04; _m = 0.01; _d = _m/(CH_PI*_r*_r*_h);
	body[15] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, false, mat[0]);
	body[15]->SetPos(ChVector3d(TOOLFACE_X-0.28, TOOLFACE_Y, 0));
	body[15]->SetRot(QuatFromAngleZ(CH_PI_2));
	body[15]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/green1.png"));
	body[15]->EnableCollision(false);
	sys.Add(body[15]);

	
	lin[0] = chrono_types::make_shared<ChLinkMotorLinearForce>();
	lin[0]->Initialize(body[12], body[11], ChFrame<>(ChVector3d(-0.36, TOOLFACE_Y, -0.35), QuatFromAngleZ(CH_PI_2)));
	msp[6] = chrono_types::make_shared<ChFunctionSetpoint>();
	lin[0]->SetMotorFunction(msp[6]);
	sys.Add(lin[0]);
	
	lin[1] = chrono_types::make_shared<ChLinkMotorLinearPosition>();
	lin[1]->Initialize(body[13], body[11], ChFrame<>(ChVector3d(-0.36, TOOLFACE_Y, -0.35), QuatFromAngleZ(CH_PI_2)));
	msp[7] = chrono_types::make_shared<ChFunctionSetpoint>();
	lin[1]->SetMotorFunction(msp[7]);
	sys.Add(lin[1]);

	lin[2] = chrono_types::make_shared<ChLinkMotorLinearForce>();
	lin[2]->Initialize(body[14], body[11], ChFrame<>(ChVector3d(-0.35, TOOLFACE_Y, 0), QuatFromAngleY(-CH_PI_2)));
	msp[8] = chrono_types::make_shared<ChFunctionSetpoint>();
	lin[2]->SetMotorFunction(msp[8]);
	sys.Add(lin[2]);
	    
	fix[4] = chrono_types::make_shared<ChLinkMateFix>();
	fix[4]->Initialize(body[10], body[11]);
	sys.Add(fix[4]);
	
	fix[5] = chrono_types::make_shared<ChLinkMateFix>();
	fix[5]->Initialize(body[14], body[15]);
	sys.Add(fix[5]);
    }
    

    if(env == 1){
	num_obj = 0;
	
	_x = 0.08; _y = 0.08; _z = 0.08;  _m = 0.1; _d = _m/(_x*_y*_z);
	body[32] = chrono_types::make_shared<ChBodyEasyBox>(_y, _z, _x, _d, true, true, mat[1]);
	body[32]->SetPos(ChVector3d(random_pos_y(), random_pos_z(), random_pos_x()));
	body[32]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/pink.png"));
	sys.Add(body[32]);
	objtype[0] = 1080080080;
	num_obj += 1;
	
	_x = 0.05; _y = 0.05; _z = 0.15;  _m = 0.1; _d = _m/(_x*_y*_z);
	body[33] = chrono_types::make_shared<ChBodyEasyBox>(_y, _z, _x, _d, true, true, mat[1]);
	body[33]->SetPos(ChVector3d(random_pos_y(), random_pos_z(), random_pos_x()));
	body[33]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/blue.png"));
	sys.Add(body[33]);
	objtype[1] = 1050050150;
	num_obj += 1;
	
	_x = 0.03; _y = 0.06; _z = 0.12;  _m = 0.1; _d = _m/(_x*_y*_z);
	body[34] = chrono_types::make_shared<ChBodyEasyBox>(_y, _z, _x, _d, true, true, mat[1]);
	body[34]->SetPos(ChVector3d(random_pos_y(), random_pos_z(), random_pos_x()));
	body[34]->SetRot(QuatFromAngleX(CH_PI_2));
	body[34]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/green1.png"));
	sys.Add(body[34]);
	objtype[2] = 1030060120;
	num_obj += 1;
	
	_r = 0.03; _h = 0.12; _m = 0.1; _d = _m/(CH_PI*_r*_r*_h);
	body[35] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, true, mat[1]);
	body[35]->SetPos(ChVector3d(random_pos_y(), random_pos_z(), random_pos_x()));
	sys.Add(body[35]);
	objtype[3] = 1060120;
	num_obj += 1;
	
	_r = 0.03; _m = 0.1; _d = _m/(4.0*CH_PI*_r*_r*_r/3.0);
	body[36] = chrono_types::make_shared<ChBodyEasySphere>(_r, _d, true, true, mat[1]);
	body[36]->SetPos(ChVector3d(random_pos_y(), random_pos_z(), random_pos_x()));
	body[36]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/red1.png"));
	sys.Add(body[36]);
	objtype[4] = 1060;
	num_obj += 1;
	
	_r = 0.04; _h = 0.20; _m = 0.1; _d = _m/(CH_PI*_r*_r*_h);
	body[37] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, true, mat[1]);
	body[37]->SetPos(ChVector3d(random_pos_y(), random_pos_z(), random_pos_x()));
	body[37]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/redwhite.png"));
	sys.Add(body[37]);
	objtype[5] = 1080200;
	num_obj += 1;
	
	_x = 0.04; _y = 0.08; _z = 0.16;  _m = 0.1; _d = _m/(_x*_y*_z);
	body[38] = chrono_types::make_shared<ChBodyEasyBox>(_y, _z, _x, _d, true, true, mat[1]);
	body[38]->SetPos(ChVector3d(random_pos_y(), random_pos_z(), random_pos_x()));
	body[38]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/cubetexture_wood.png"));
	sys.Add(body[38]);
	objtype[6] = 1040080160;
	num_obj += 1;
	
	_r = 0.04; _m = 0.1; _d = _m/(4.0*CH_PI*_r*_r*_r/3.0);
	body[39] = chrono_types::make_shared<ChBodyEasySphere>(_r, _d, true, true, mat[1]);
	body[39]->SetPos(ChVector3d(random_pos_y(), random_pos_z(), random_pos_x()));
	sys.Add(body[39]);
	objtype[7] = 1080;
	num_obj += 1;
	
	_r = 0.02; _h = 0.16; _m = 0.1; _d = _m/(CH_PI*_r*_r*_h);
	body[40] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, true, mat[1]);
	body[40]->SetPos(ChVector3d(random_pos_y(), random_pos_z(), random_pos_x()));
	body[40]->SetRot(QuatFromAngleX(CH_PI_2));
	sys.Add(body[40]);
	objtype[8] = 1040160;
	num_obj += 1;
	
	_x = 0.03; _y = 0.06; _z = 0.10;  _m = 0.1; _d = _m/(_x*_y*_z);
	body[41] = chrono_types::make_shared<ChBodyEasyBox>(_y, _z, _x, _d, true, true, mat[1]);
	body[41]->SetPos(ChVector3d(random_pos_y(), random_pos_z(), random_pos_x()));
	body[41]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/rock.png"));
	sys.Add(body[41]);
	objtype[9] = 1030060100;
	num_obj += 1;
	
	/*_r = 0.03; _h = 0.10; _m = 0.1; _d = _m/(CH_C_PI*_r*_r*_h);
	body[42] = chrono_types::make_shared<ChBodyEasyCylinder>(ChAxis((0, 0, 1)), _r, _h, _d, true, true, mat[1]);
	body[42]->SetPos(ChVector3d(random_pos_y(), random_pos_z(), random_pos_x()));
	body[42]->SetRot(QuatFromAngleX(CH_PI_2));
	body[42]->SetFixed(false);
	body[42]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/blue.png"));
	sys.Add(body[42]);
	objtype[10] = 1060100;
	num_obj += 1;
	
	_x = 0.01; _y = 0.10; _z = 0.10;  _m = 0.1; _d = _m/(_x*_y*_z);
	body[43] = chrono_types::make_shared<ChBodyEasyBox>(_y, _z, _x, _d, true, true, mat[1]);
	body[43]->SetPos(ChVector<>(random_pos_y(), random_pos_z(), random_pos_x()));
	//body[43]->SetRot(QuatFromAngleZ(CH_PI_2));
	body[43]->SetFixed(false);
	body[43]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/red1.png"));
	sys.Add(body[43]);
	objtype[11] = 1010100100;
	num_obj += 1;*/
	

	//bin1
	body[48] = chrono_types::make_shared<ChBodyEasyBox>(0.60, 0.02, 0.40, 1, true, true, mat[1]);
	body[48]->SetPos(ChVector3d(-0.35, -0.13, 0.70));
	body[48]->SetFixed(true);
	body[48]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/yellow.png"));
	sys.Add(body[48]);

	body[49] = chrono_types::make_shared<ChBodyEasyBox>(0.60, 0.14, 0.02, 1, true, true, mat[1]);
	body[49]->SetPos(ChVector3d(-0.35, -0.05, 0.89));
	body[49]->SetFixed(true);
	body[49]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/yellow.png"));
	sys.Add(body[49]);

	body[50] = chrono_types::make_shared<ChBodyEasyBox>(0.02, 0.14, 0.35, 1, true, true, mat[1]);
	body[50]->SetPos(ChVector3d(-0.64, -0.05, 0.70));
	body[50]->SetFixed(true);
	body[50]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/yellow.png"));
	sys.Add(body[50]);

	body[51] = chrono_types::make_shared<ChBodyEasyBox>(0.60, 0.14, 0.02, 1, true, true, mat[1]);
	body[51]->SetPos(ChVector3d(-0.35, -0.05, 0.51));
	body[51]->SetFixed(true);
	body[51]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/yellow.png"));
	sys.Add(body[51]);

	body[52] = chrono_types::make_shared<ChBodyEasyBox>(0.02, 0.14, 0.35, 1, true, true, mat[1]);
	body[52]->SetPos(ChVector3d(-0.06, -0.05, 0.70));
	body[52]->SetFixed(true);
	body[52]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/yellow.png"));
	sys.Add(body[52]);


	//bin2
	body[53] = chrono_types::make_shared<ChBodyEasyBox>(0.60, 0.02, 0.40, 1, true, true, mat[1]);
	body[53]->SetPos(ChVector3d(0.35, -0.13, 0.70));
	body[53]->SetFixed(true);
	body[53]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/yellow.png"));
	sys.Add(body[53]);

	body[54] = chrono_types::make_shared<ChBodyEasyBox>(0.60, 0.14, 0.02, 1, true, true, mat[1]);
	body[54]->SetPos(ChVector3d(0.35, -0.05, 0.89));
	body[54]->SetFixed(true);
	body[54]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/yellow.png"));
	sys.Add(body[54]);

	body[55] = chrono_types::make_shared<ChBodyEasyBox>(0.02, 0.14, 0.35, 1, true, true, mat[1]);
	body[55]->SetPos(ChVector3d(0.64, -0.05, 0.70));
	body[55]->SetFixed(true);
	body[55]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/yellow.png"));
	sys.Add(body[55]);

	body[56] = chrono_types::make_shared<ChBodyEasyBox>(0.60, 0.14, 0.02, 1, true, true, mat[1]);
	body[56]->SetPos(ChVector3d(0.35, -0.05, 0.51));
	body[56]->SetFixed(true);
	body[56]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/yellow.png"));
	sys.Add(body[56]);

	body[57] = chrono_types::make_shared<ChBodyEasyBox>(0.02, 0.14, 0.35, 1, true, true, mat[1]);
	body[57]->SetPos(ChVector3d(0.06, -0.05, 0.70));
	body[57]->SetFixed(true);
	body[57]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/yellow.png"));
	sys.Add(body[57]);

	
	//table
	body[58] = chrono_types::make_shared<ChBodyEasyBox>(2.0, 0.02, 2.0, 1000, true, true, mat[1]);
	body[58]->SetPos(ChVector3d(0, -0.15, 0));
	body[58]->SetFixed(true);
	body[58]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/blue.png"));
	sys.Add(body[58]);
	
	body[59] = chrono_types::make_shared<ChBodyEasyBox>(0.06, 0.50, 0.06, 1000, true, false, mat[1]);
	body[59]->SetPos(ChVector3d(0.95, -0.40, 0.95));
	body[59]->SetFixed(true);
	sys.Add(body[59]);

	body[60] = chrono_types::make_shared<ChBodyEasyBox>(0.06, 0.50, 0.06, 1000, true, false, mat[1]);
	body[60]->SetPos(ChVector3d(-0.95, -0.40, 0.95));
	body[60]->SetFixed(true);
	sys.Add(body[60]);

	body[61] = chrono_types::make_shared<ChBodyEasyBox>(0.06, 0.50, 0.06, 1000, true, false, mat[1]);
	body[61]->SetPos(ChVector3d(0.95, -0.40, -0.95));
	body[61]->SetFixed(true);
	sys.Add(body[61]);
	
	body[62] = chrono_types::make_shared<ChBodyEasyBox>(0.06, 0.50, 0.06, 1000, true, false, mat[1]);
	body[62]->SetPos(ChVector3d(-0.95, -0.40, -0.95));
	body[62]->SetFixed(true);
	sys.Add(body[62]);

    
	//robot base
	body[63] = chrono_types::make_shared<ChBodyEasyBox>(0.30, 0.14, 0.30, 1, true, true, mat[1]);
	body[63]->SetPos(ChVector3d(0.00, -0.07, 0.00));
	body[63]->SetFixed(true);
	body[63]->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/blue.png"));
	sys.Add(body[63]);
    }
}	


extern "C" void init(int _render=0, int _env=0, int _tool=0) {
    // cnt-based guard is unreliable: cnt only advances in step(), so the initial
    // ctypes init() AND the reset()-driven init() both run with cnt==0. Without
    // a separate latch, reset()'s zero-arg init() clobbers render/env/tool back
    // to 0 and re-runs vis SetWindowSize/Initialize on an already-live device.
    static bool initialized = false;
    if(!initialized) {
	render = _render;
	env = _env;
	tool = _tool;
    }

    srand((unsigned int)time(NULL));
    SetChronoDataPath("/home/ubuntu/fgx/chrono/data/");
    //SetChronoDataPath("/home/binpicking/fgx/chrono/data/");

    build_model();


    if(render > 0){
	if(!initialized){
	    vis->AttachSystem(&sys);
	    vis->SetWindowSize(1200, 800);
	    vis->SetWindowTitle("chrono");
	    vis->Initialize();
	    vis->AddLogo();
	    //vis->AddSkyBox();
	    vis->AddLight(ChVector3d(100, 30, 100), 200, ChColor(0.7f, 0.7f, 0.7f));
	    //vis->AddCamera(ChVector3d(0.8, 1.33, 1.33), ChVector3d(0, 0.45, 0));

	    if (env == 0) vis->AddCamera(ChVector3d(0.6, 1.6, 1.0), ChVector3d(0, 0.8, 0));
	    else if (env == 1) vis->AddCamera(ChVector3d(0.8, 1.1, 1.33), ChVector3d(0, 0.1, 0));
	}
	vis->BindAll();   // (re)bind bodies after each build_model — initial + reset
    }
    initialized = true;
    
    sys.SetSolverType(ChSolver::Type::PSOR);
    //system.SetSolverType(ChSolver::Type::BARZILAIBORWEIN);
    sys.SetTimestepperType(ChTimestepper::Type::EULER_IMPLICIT_LINEARIZED);
    //system.SetTimestepperType(ChTimestepper::Type::NEWMARK);

    sys.GetSolver()->AsIterative()->SetMaxIterations(100);

    sys.SetCollisionSystemType(ChCollisionSystem::Type::BULLET);
    //sys.SetCollisionSystemType(ChCollisionSystem::Type::MULTICORE);
}


extern "C" int step(double* tau, double* q_ref, double* qd_ref, double* y){
    (void)q_ref; (void)qd_ref;   // chrono backend has no implicit PD yet — targets ignored
    //double vis_damping[9] = {2.0, 2.0, 2.0, 0.5, 0.5, 0.5,    0.4, 0.4, 0.02};
    double vis_damping[9] = {5.0, 5.0, 5.0, 1.5, 1.5, 0.4,    0.4, 0.4, 0.02};

    double t = sys.GetChTime();

    for(int i = 0; i < 6; i++){
	q[i] = rev[i]->GetMotorAngle();
	d[i] = rev[i]->GetMotorAngleDt();
	h[i] = rev[i]->GetMotorTorque();
    }


    for(int i = 0; i < 6; i++){
	f[i] = -vis_damping[i]*d[i];
	z[i] = tau[i] + f[i];
	msp[i]->SetSetpoint(z[i], t);
    }


    //int idx = 0;
    //for(int i=0; i < 6; i++) {y[idx] = q[i]; idx++;}
    //for(int i=0; i < 6; i++) {y[idx] = d[i]; idx++;}

    for(int i=0; i < 6; i++){
	y[i] = q[i];
	y[i+6] = d[i];
    }
    
    
    if(tool == 1){
	q[6] = lin[0]->GetMotorPos();
	q[7] = lin[2]->GetMotorPos();
	
	d[6] = lin[0]->GetMotorPosDt();
	d[7] = lin[2]->GetMotorPosDt();	

	msp[6]->SetSetpoint(tau[6] - vis_damping[6]*d[6], t);
	msp[7]->SetSetpoint(-1*q[6], t);
	msp[8]->SetSetpoint(tau[7] - vis_damping[7]*d[7], t);

	y[12] = q[6];
	y[13] = q[7];

	y[14] = d[6];
	y[15] = d[7]; 

	//suction pad feedback
	if(suction_on) y[16] = 1.0;
	else y[16] = 0.0;

	y[17] = 0;
	y[18] = 0;
    }

    
    sys.DoStepDynamics(0.001);
    cnt++;
	
    if(render > 0 && cnt%render == 0){
	vis->Run();
	vis->BeginScene();
	
	if(env == 0) chrono::irrlicht::tools::drawGrid(vis.get(), 0.2, 0.2, 10, 10, ChCoordsys<>(ChVector3d(0, 0, 0), QuatFromAngleX(CH_PI_2)), ChColor(1.0f, 1.0f, 1.0f), true);
	else if(env == 1) chrono::irrlicht::tools::drawGrid(vis.get(), 0.2, 0.2, 10, 10, ChCoordsys<>(ChVector3d(0, -0.135, 0), QuatFromAngleX(CH_PI_2)), ChColor(1.0f, 1.0f, 1.0f), true);

	vis->Render();
	vis->EndScene();
    }
    return 0;
}


// Gym-style reset: rebuild the system and write the post-reset observation into y
// (without calling step()/DoStepDynamics, which would advance physics by one dt and
// break reproducibility of the sim). Layout mirrors step()'s y exactly.
extern "C" void reset(double* y){
    sys.Clear();
    init();

    for(int i = 0; i < 6; i++){
	q[i] = rev[i]->GetMotorAngle();
	d[i] = rev[i]->GetMotorAngleDt();
	y[i]   = q[i];
	y[i+6] = d[i];
    }

    if(tool == 1){
	q[6] = lin[0]->GetMotorPos();
	q[7] = lin[2]->GetMotorPos();
	d[6] = lin[0]->GetMotorPosDt();
	d[7] = lin[2]->GetMotorPosDt();
	y[12] = q[6];
	y[13] = q[7];
	y[14] = d[6];
	y[15] = d[7];
	y[16] = suction_on ? 1.0 : 0.0;
	y[17] = 0;
	y[18] = 0;
    }
}


// pose6() is unused by current rb10.py and depends on tact.h's rotation_to_eulerxyz
// (now removed). Mirroring mjenv.cpp, we keep the body commented out — if/when this
// is needed again, inline a 3x3-rotation → XYZ-euler conversion here.
/*
extern "C" int pose6(int *code, double* pos, double* rpy){
    int off = 32;

    for(int i=0; i < num_obj; i++){

	int idx = off + i;

	ChVector3d p = body[idx]->GetPos();
	ChQuaternion<> quaternion = body[idx]->GetRot();
	ChMatrix33<> R(quaternion);

	double _R[9];
	_R[0] = R(2,2); _R[1] = R(2,0); _R[2] = R(2,1); _R[3] = R(0,2); _R[4] = R(0,0); _R[5] = R(0,1); _R[6] = R(1,2); _R[7] = R(1,0); _R[8] = R(1,1);

	rotation_to_eulerxyz(_R, rpy+3*i);

	pos[3*i+0] = p[2];
	pos[3*i+1] = p[0];
	pos[3*i+2] = p[1];

	code[i] = objtype[i];
    }

    return num_obj;
}
*/



extern "C" void push(char* msg){
    char buf[1024];

    memset(buf,'\0',sizeof(buf));
    memcpy(buf, msg, strlen(msg));
    char *token = strtok(buf, " ");

    //suction on 
    if(strcmp(token, "tc8") == 0) {
	int num = rand()%100;
	if(num < 10) {
	    printf("suction failed\n");
	    return;
	}
	
	suction_on = true;
	token = strtok(NULL, " ");
	int obj_id = atoi(token);
	int idx = obj_id + 31; // <--------------- obj_id starts from 1 not 0
	fix[6] = chrono_types::make_shared<ChLinkMateFix>();
	fix[6]->Initialize(body[15], body[idx]);
	sys.Add(fix[6]);
    }

    //suction off
    else if(strcmp(token, "tc9") == 0) {
	if (suction_on == true){
	    sys.Remove(fix[6]);
	    suction_on = false;
	}
    }
}


extern "C" void finish(){
    ;
}


extern "C" int get_num_output(){
    int n_y = 12;
    if(tool == 1) n_y = 19;
    return n_y;
}

extern "C" int get_num_input(){
    return 6;
}
