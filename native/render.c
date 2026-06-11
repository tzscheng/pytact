#define GL_GLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLFW/glfw3.h>
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
//#include <arpa/inet.h>
//#include <sys/socket.h>
#include <turbojpeg.h>
#include <zstd.h>

// Shared shape-asset slot storage (mesh_path, hf_* grids) — defined in shape.c.
#include "shape.h"

// Rendering/encoding dependencies are intentionally not link-time dependencies
// of libtact.so. Headless physics users should be able to load libtact.so on a
// machine without GL/GLFW/jpeg/zstd runtime libraries. Render entrypoints dlopen
// their runtime libraries on demand and resolve every used symbol explicitly.
static int load_shared(const char* const* names, char* err, size_t err_sz){
    for(int i = 0; names[i]; i++){
        void* h = dlopen(names[i], RTLD_LAZY | RTLD_GLOBAL);
        if(h) return 1;
    }
    snprintf(err, err_sz, "missing shared library: %s", names[0]);
    return 0;
}

#define TACT_GLFW_PROCS(X) \
    X(int,         glfwInit,                       (void)) \
    X(void,        glfwTerminate,                  (void)) \
    X(void,        glfwWindowHint,                 (int hint, int value)) \
    X(GLFWwindow*, glfwCreateWindow,               (int width, int height, const char *title, GLFWmonitor *monitor, GLFWwindow *share)) \
    X(void,        glfwMakeContextCurrent,         (GLFWwindow *window)) \
    X(GLFWglproc,  glfwGetProcAddress,             (const char *procname)) \
    X(void,        glfwSwapInterval,               (int interval)) \
    X(GLFWmousebuttonfun, glfwSetMouseButtonCallback,     (GLFWwindow *window, GLFWmousebuttonfun callback)) \
    X(GLFWcursorposfun,   glfwSetCursorPosCallback,       (GLFWwindow *window, GLFWcursorposfun callback)) \
    X(GLFWscrollfun,      glfwSetScrollCallback,          (GLFWwindow *window, GLFWscrollfun callback)) \
    X(GLFWframebuffersizefun, glfwSetFramebufferSizeCallback, (GLFWwindow *window, GLFWframebuffersizefun callback)) \
    X(GLFWkeyfun,         glfwSetKeyCallback,             (GLFWwindow *window, GLFWkeyfun callback)) \
    X(void,        glfwGetCursorPos,               (GLFWwindow *window, double *xpos, double *ypos)) \
    X(int,         glfwGetWindowAttrib,            (GLFWwindow *window, int attrib)) \
    X(void,        glfwGetFramebufferSize,         (GLFWwindow *window, int *width, int *height)) \
    X(void,        glfwSetWindowTitle,             (GLFWwindow *window, const char *title)) \
    X(void,        glfwSwapBuffers,                (GLFWwindow *window)) \
    X(void,        glfwPollEvents,                 (void))

#define TACT_EGL_PROCS(X) \
    X(EGLDisplay, eglGetDisplay,              (EGLNativeDisplayType display_id)) \
    X(EGLBoolean, eglInitialize,              (EGLDisplay dpy, EGLint *major, EGLint *minor)) \
    X(__eglMustCastToProperFunctionPointerType, eglGetProcAddress, (const char *procname)) \
    X(EGLBoolean, eglChooseConfig,            (EGLDisplay dpy, const EGLint *attrib_list, EGLConfig *configs, EGLint config_size, EGLint *num_config)) \
    X(EGLBoolean, eglBindAPI,                 (EGLenum api)) \
    X(EGLContext, eglCreateContext,           (EGLDisplay dpy, EGLConfig config, EGLContext share_context, const EGLint *attrib_list)) \
    X(EGLSurface, eglCreatePbufferSurface,    (EGLDisplay dpy, EGLConfig config, const EGLint *attrib_list)) \
    X(EGLBoolean, eglMakeCurrent,             (EGLDisplay dpy, EGLSurface draw, EGLSurface read, EGLContext ctx)) \
    X(EGLBoolean, eglDestroySurface,          (EGLDisplay dpy, EGLSurface surface))

#define TACT_TJ_PROCS(X) \
    X(tjhandle,       tjInitCompress, (void)) \
    X(unsigned long,  tjBufSize,      (int width, int height, int jpegSubsamp)) \
    X(unsigned char*, tjAlloc,        (int bytes)) \
    X(void,           tjFree,         (unsigned char *buffer)) \
    X(int,            tjCompress2,    (tjhandle handle, const unsigned char *srcBuf, int width, int pitch, int height, int pixelFormat, unsigned char **jpegBuf, unsigned long *jpegSize, int jpegSubsamp, int jpegQual, int flags))

#define TACT_ZSTD_PROCS(X) \
    X(ZSTD_CCtx*,   ZSTD_createCCtx,    (void)) \
    X(size_t,       ZSTD_compressBound, (size_t srcSize)) \
    X(size_t,       ZSTD_compressCCtx,  (ZSTD_CCtx *cctx, void *dst, size_t dstCapacity, const void *src, size_t srcSize, int compressionLevel)) \
    X(unsigned int, ZSTD_isError,       (size_t code)) \
    X(const char*,  ZSTD_getErrorName,  (size_t code))

#define DECL_PROC(ret, name, args) static ret (*p_##name) args;
TACT_GLFW_PROCS(DECL_PROC)
TACT_EGL_PROCS(DECL_PROC)
TACT_TJ_PROCS(DECL_PROC)
TACT_ZSTD_PROCS(DECL_PROC)
#undef DECL_PROC

#define LOAD_PROC(ret, name, args) do { \
    p_##name = (ret (*) args)dlsym(RTLD_DEFAULT, #name); \
    if(!p_##name){ \
        snprintf(err, err_sz, "cannot resolve symbol: %s", #name); \
        return 0; \
    } \
} while(0);

static int render_load_glfw_procs(char* err, size_t err_sz){
    static int ok = 0;
    if(ok) return 1;
    TACT_GLFW_PROCS(LOAD_PROC)
    ok = 1;
    return 1;
}

static int render_load_egl_procs(char* err, size_t err_sz){
    static int ok = 0;
    if(ok) return 1;
    TACT_EGL_PROCS(LOAD_PROC)
    ok = 1;
    return 1;
}

static int render_load_tj_procs(char* err, size_t err_sz){
    static int ok = 0;
    if(ok) return 1;
    TACT_TJ_PROCS(LOAD_PROC)
    ok = 1;
    return 1;
}

static int render_load_zstd_procs(char* err, size_t err_sz){
    static int ok = 0;
    if(ok) return 1;
    TACT_ZSTD_PROCS(LOAD_PROC)
    ok = 1;
    return 1;
}

#undef LOAD_PROC

static int render_load_window_deps(void){
    static int ok = 0, tried = 0;
    static char err[256];
    if(ok) return 1;
    if(tried){ fprintf(stderr, "tact render unavailable: %s\n", err); return 0; }
    tried = 1;
    const char* gl[]   = {"libOpenGL.so.0", "libOpenGL.so", "libGL.so.1", "libGL.so", NULL};
    const char* glfw[] = {"libglfw.so.3", "libglfw.so", NULL};
    if(!load_shared(gl, err, sizeof(err))) return 0;
    if(!load_shared(glfw, err, sizeof(err))) return 0;
    if(!render_load_glfw_procs(err, sizeof(err))){
        fprintf(stderr, "tact render unavailable: %s\n", err);
        return 0;
    }
    ok = 1;
    return 1;
}

static int render_load_egl_deps(void){
    static int gl_ok = 0, egl_ok = 0, tj_ok = 0, zstd_ok = 0;
    static char err[256];
    const char* gl[]   = {"libOpenGL.so.0", "libOpenGL.so", "libGL.so.1", "libGL.so", NULL};
    const char* egl[]  = {"libEGL.so.1", "libEGL.so", NULL};
    const char* tj[]   = {"libturbojpeg.so.0", "libturbojpeg.so", NULL};
    const char* zstd[] = {"libzstd.so.1", "libzstd.so", NULL};

    if(!gl_ok   && !(gl_ok   = load_shared(gl,   err, sizeof(err)))) goto fail;
    if(!egl_ok  && !(egl_ok  = load_shared(egl,  err, sizeof(err)))) goto fail;
    if(!render_load_egl_procs(err, sizeof(err))) goto fail;
    // egl_render currently initializes both encoder contexts on first use
    // (tjInitCompress + ZSTD_createCCtx), even though each frame uses only one
    // opt. Load both here to keep that existing lifecycle unchanged.
    if(!tj_ok   && !(tj_ok   = load_shared(tj,   err, sizeof(err)))) goto fail;
    if(!zstd_ok && !(zstd_ok = load_shared(zstd, err, sizeof(err)))) goto fail;
    if(!render_load_tj_procs(err, sizeof(err))) goto fail;
    if(!render_load_zstd_procs(err, sizeof(err))) goto fail;
    return 1;
fail:
    fprintf(stderr, "tact egl render unavailable: %s\n", err);
    return 0;
}

#define TACT_GL_PROCS(X) \
    X(void,   glActiveTexture,            (GLenum texture)) \
    X(void,   glAttachShader,             (GLuint program, GLuint shader)) \
    X(void,   glBindBuffer,               (GLenum target, GLuint buffer)) \
    X(void,   glBindFramebuffer,          (GLenum target, GLuint framebuffer)) \
    X(void,   glBindRenderbuffer,         (GLenum target, GLuint renderbuffer)) \
    X(void,   glBindTexture,              (GLenum target, GLuint texture)) \
    X(void,   glBindVertexArray,          (GLuint array)) \
    X(void,   glBlendFunc,                (GLenum sfactor, GLenum dfactor)) \
    X(void,   glBlitFramebuffer,          (GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter)) \
    X(void,   glBufferData,               (GLenum target, GLsizeiptr size, const void *data, GLenum usage)) \
    X(GLenum, glCheckFramebufferStatus,   (GLenum target)) \
    X(void,   glClear,                    (GLbitfield mask)) \
    X(void,   glClearColor,               (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)) \
    X(void,   glCompileShader,            (GLuint shader)) \
    X(GLuint, glCreateProgram,            (void)) \
    X(GLuint, glCreateShader,             (GLenum type)) \
    X(void,   glCullFace,                 (GLenum mode)) \
    X(void,   glDeleteBuffers,            (GLsizei n, const GLuint *buffers)) \
    X(void,   glDeleteShader,             (GLuint shader)) \
    X(void,   glDeleteVertexArrays,       (GLsizei n, const GLuint *arrays)) \
    X(void,   glDepthMask,                (GLboolean flag)) \
    X(void,   glDisable,                  (GLenum cap)) \
    X(void,   glDrawArrays,               (GLenum mode, GLint first, GLsizei count)) \
    X(void,   glDrawBuffer,               (GLenum buf)) \
    X(void,   glEnable,                   (GLenum cap)) \
    X(void,   glEnableVertexAttribArray,  (GLuint index)) \
    X(void,   glFramebufferRenderbuffer,  (GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer)) \
    X(void,   glFramebufferTexture2D,     (GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level)) \
    X(void,   glGenBuffers,               (GLsizei n, GLuint *buffers)) \
    X(void,   glGenFramebuffers,          (GLsizei n, GLuint *framebuffers)) \
    X(void,   glGenRenderbuffers,         (GLsizei n, GLuint *renderbuffers)) \
    X(void,   glGenTextures,              (GLsizei n, GLuint *textures)) \
    X(void,   glGenVertexArrays,          (GLsizei n, GLuint *arrays)) \
    X(GLenum, glGetError,                 (void)) \
    X(void,   glGetProgramInfoLog,        (GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog)) \
    X(void,   glGetProgramiv,             (GLuint program, GLenum pname, GLint *params)) \
    X(void,   glGetShaderInfoLog,         (GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog)) \
    X(void,   glGetShaderiv,              (GLuint shader, GLenum pname, GLint *params)) \
    X(GLint,  glGetUniformLocation,       (GLuint program, const GLchar *name)) \
    X(void,   glLinkProgram,              (GLuint program)) \
    X(void,   glPolygonOffset,            (GLfloat factor, GLfloat units)) \
    X(void,   glReadBuffer,               (GLenum src)) \
    X(void,   glReadPixels,               (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, void *pixels)) \
    X(void,   glRenderbufferStorage,      (GLenum target, GLenum internalformat, GLsizei width, GLsizei height)) \
    X(void,   glShaderSource,             (GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length)) \
    X(void,   glTexImage2D,               (GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels)) \
    X(void,   glTexParameterfv,           (GLenum target, GLenum pname, const GLfloat *params)) \
    X(void,   glTexParameteri,            (GLenum target, GLenum pname, GLint param)) \
    X(void,   glUniform1i,                (GLint location, GLint v0)) \
    X(void,   glUniform3f,                (GLint location, GLfloat v0, GLfloat v1, GLfloat v2)) \
    X(void,   glUniform4f,                (GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3)) \
    X(void,   glUniformMatrix3fv,         (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)) \
    X(void,   glUniformMatrix4fv,         (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)) \
    X(void,   glUseProgram,               (GLuint program)) \
    X(void,   glVertexAttribPointer,      (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer)) \
    X(void,   glViewport,                 (GLint x, GLint y, GLsizei width, GLsizei height))

#define DECL_GL_PROC(ret, name, args) static ret (*p_##name) args;
TACT_GL_PROCS(DECL_GL_PROC)
#undef DECL_GL_PROC

static void* render_get_gl_proc(const char* name, int use_glfw){
    void* p = NULL;
    if(use_glfw) p = (void*)p_glfwGetProcAddress(name);
    else         p = (void*)p_eglGetProcAddress(name);
    if(!p) p = dlsym(RTLD_DEFAULT, name);
    return p;
}

static int render_load_gl_procs(int use_glfw){
    static int window_ok = 0;
    static int egl_ok = 0;
    if(use_glfw && window_ok) return 1;
    if(!use_glfw && egl_ok) return 1;

    #define LOAD_GL_PROC(ret, name, args) do { \
        p_##name = (ret (*) args)render_get_gl_proc(#name, use_glfw); \
        if(!p_##name){ \
            fprintf(stderr, "tact render unavailable: cannot resolve OpenGL symbol %s\n", #name); \
            return 0; \
        } \
    } while(0);
    TACT_GL_PROCS(LOAD_GL_PROC)
    #undef LOAD_GL_PROC

    if(use_glfw) window_ok = 1;
    else egl_ok = 1;
    return 1;
}

#define glActiveTexture p_glActiveTexture
#define glAttachShader p_glAttachShader
#define glBindBuffer p_glBindBuffer
#define glBindFramebuffer p_glBindFramebuffer
#define glBindRenderbuffer p_glBindRenderbuffer
#define glBindTexture p_glBindTexture
#define glBindVertexArray p_glBindVertexArray
#define glBlendFunc p_glBlendFunc
#define glBlitFramebuffer p_glBlitFramebuffer
#define glBufferData p_glBufferData
#define glCheckFramebufferStatus p_glCheckFramebufferStatus
#define glClear p_glClear
#define glClearColor p_glClearColor
#define glCompileShader p_glCompileShader
#define glCreateProgram p_glCreateProgram
#define glCreateShader p_glCreateShader
#define glCullFace p_glCullFace
#define glDeleteBuffers p_glDeleteBuffers
#define glDeleteShader p_glDeleteShader
#define glDeleteVertexArrays p_glDeleteVertexArrays
#define glDepthMask p_glDepthMask
#define glDisable p_glDisable
#define glDrawArrays p_glDrawArrays
#define glDrawBuffer p_glDrawBuffer
#define glEnable p_glEnable
#define glEnableVertexAttribArray p_glEnableVertexAttribArray
#define glFramebufferRenderbuffer p_glFramebufferRenderbuffer
#define glFramebufferTexture2D p_glFramebufferTexture2D
#define glGenBuffers p_glGenBuffers
#define glGenFramebuffers p_glGenFramebuffers
#define glGenRenderbuffers p_glGenRenderbuffers
#define glGenTextures p_glGenTextures
#define glGenVertexArrays p_glGenVertexArrays
#define glGetError p_glGetError
#define glGetProgramInfoLog p_glGetProgramInfoLog
#define glGetProgramiv p_glGetProgramiv
#define glGetShaderInfoLog p_glGetShaderInfoLog
#define glGetShaderiv p_glGetShaderiv
#define glGetUniformLocation p_glGetUniformLocation
#define glLinkProgram p_glLinkProgram
#define glPolygonOffset p_glPolygonOffset
#define glReadBuffer p_glReadBuffer
#define glReadPixels p_glReadPixels
#define glRenderbufferStorage p_glRenderbufferStorage
#define glShaderSource p_glShaderSource
#define glTexImage2D p_glTexImage2D
#define glTexParameterfv p_glTexParameterfv
#define glTexParameteri p_glTexParameteri
#define glUniform1i p_glUniform1i
#define glUniform3f p_glUniform3f
#define glUniform4f p_glUniform4f
#define glUniformMatrix3fv p_glUniformMatrix3fv
#define glUniformMatrix4fv p_glUniformMatrix4fv
#define glUseProgram p_glUseProgram
#define glVertexAttribPointer p_glVertexAttribPointer
#define glViewport p_glViewport

#define glfwInit p_glfwInit
#define glfwTerminate p_glfwTerminate
#define glfwWindowHint p_glfwWindowHint
#define glfwCreateWindow p_glfwCreateWindow
#define glfwMakeContextCurrent p_glfwMakeContextCurrent
#define glfwGetProcAddress p_glfwGetProcAddress
#define glfwSwapInterval p_glfwSwapInterval
#define glfwSetMouseButtonCallback p_glfwSetMouseButtonCallback
#define glfwSetCursorPosCallback p_glfwSetCursorPosCallback
#define glfwSetScrollCallback p_glfwSetScrollCallback
#define glfwSetFramebufferSizeCallback p_glfwSetFramebufferSizeCallback
#define glfwSetKeyCallback p_glfwSetKeyCallback
#define glfwGetCursorPos p_glfwGetCursorPos
#define glfwGetWindowAttrib p_glfwGetWindowAttrib
#define glfwGetFramebufferSize p_glfwGetFramebufferSize
#define glfwSetWindowTitle p_glfwSetWindowTitle
#define glfwSwapBuffers p_glfwSwapBuffers
#define glfwPollEvents p_glfwPollEvents

#define eglGetDisplay p_eglGetDisplay
#define eglInitialize p_eglInitialize
#define eglGetProcAddress p_eglGetProcAddress
#define eglChooseConfig p_eglChooseConfig
#define eglBindAPI p_eglBindAPI
#define eglCreateContext p_eglCreateContext
#define eglCreatePbufferSurface p_eglCreatePbufferSurface
#define eglMakeCurrent p_eglMakeCurrent
#define eglDestroySurface p_eglDestroySurface

#define tjInitCompress p_tjInitCompress
#define tjBufSize p_tjBufSize
#define tjAlloc p_tjAlloc
#define tjFree p_tjFree
#define tjCompress2 p_tjCompress2

#define ZSTD_createCCtx p_ZSTD_createCCtx
#define ZSTD_compressBound p_ZSTD_compressBound
#define ZSTD_compressCCtx p_ZSTD_compressCCtx
#define ZSTD_isError p_ZSTD_isError
#define ZSTD_getErrorName p_ZSTD_getErrorName

static EGLDisplay try_init_egl_display(EGLDisplay d, const char* label){
    if(d == EGL_NO_DISPLAY) return EGL_NO_DISPLAY;
    if(eglInitialize(d, NULL, NULL)){
        fprintf(stderr, "egl_render: EGL display = %s\n", label);
        return d;
    }
    return EGL_NO_DISPLAY;
}

static EGLDisplay choose_egl_display(void){
    PFNEGLGETPLATFORMDISPLAYEXTPROC gpd =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    PFNEGLQUERYDEVICESEXTPROC qd =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    const char* (*qds)(EGLDeviceEXT, EGLint) =
        (const char*(*)(EGLDeviceEXT, EGLint))eglGetProcAddress("eglQueryDeviceStringEXT");

    // Explicit iGPU/Mesa path retained from the previous code: force Mesa's EGL
    // ICD before the first successful display init, then choose the first
    // hardware device with a DRM render node.
    if(getenv("TACT_RENDER_IGPU") && qd && gpd){
        setenv("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json", 1);
        EGLDeviceEXT devs[8]; EGLint n_dev = 0;
        if(qd(8, devs, &n_dev)){
            for(int i = 0; i < n_dev; i++){
                const char* node = qds ? qds(devs[i], EGL_DRM_RENDER_NODE_FILE_EXT) : NULL;
                if(!node) continue;
                EGLDisplay d = try_init_egl_display(gpd(EGL_PLATFORM_DEVICE_EXT, devs[i], NULL), "device/iGPU");
                if(d != EGL_NO_DISPLAY){
                    fprintf(stderr, "egl_render: TACT_RENDER_IGPU -> device %d/%d node=%s\n", i, n_dev, node);
                    return d;
                }
            }
        }
        fprintf(stderr, "egl_render: TACT_RENDER_IGPU set but no mesa HW device; using fallback display search\n");
        unsetenv("__EGL_VENDOR_LIBRARY_FILENAMES");
    }

    // Best match for headless machines without DISPLAY: Mesa surfaceless EGL.
    if(gpd){
#ifdef EGL_PLATFORM_SURFACELESS_MESA
        EGLDisplay d = try_init_egl_display(gpd(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL), "surfaceless");
        if(d != EGL_NO_DISPLAY) return d;
#endif
    }

    // Generic device-platform fallback. This may select software llvmpipe on
    // Mesa if no hardware node is available; still useful for correctness tests.
    if(qd && gpd){
        EGLDeviceEXT devs[8]; EGLint n_dev = 0;
        if(qd(8, devs, &n_dev)){
            for(int i = 0; i < n_dev; i++){
                EGLDisplay d = try_init_egl_display(gpd(EGL_PLATFORM_DEVICE_EXT, devs[i], NULL), "device");
                if(d != EGL_NO_DISPLAY) return d;
            }
        }
    }

    // Last: legacy default display, usually requires DISPLAY or vendor-specific
    // default-device support.
    return try_init_egl_display(eglGetDisplay(EGL_DEFAULT_DISPLAY), "default");
}

//---- Stage R.1: in-window mp4 recording (toggled by Shift+R key in win_render) ----
static int rec_request = 0;            //toggled by key_cb; win_render acts on transitions
static int rec_active  = 0;            //actual state in win_render
static int space_request = 0;          //set by key_cb on SPACE press; win_render returns 1 once per press (consumed)
static FILE* rec_pipe  = NULL;         //ffmpeg stdin
static int rec_w = 0, rec_h = 0;       //locked dimensions
static unsigned char* rec_buf = NULL;  //glReadPixels target

static int quit_request = 0;

//---- Light + shadow state (settable via render_set_light()) ----
// Defaults match the historical hardcoded values; YAML `lights:` overrides.
#define SHADOW_MAP_SIZE 2048
static int   g_shadow_enabled = 1;
static float g_light_pos[3]    = {7.0f, 7.0f, 7.0f};
static float g_light_target[3] = {0.0f, 0.0f, 0.0f};
// ortho half-extent of light frustum in world units. Smaller = denser shadow
// texels (sharper edges) but objects outside this radius from origin won't
// cast shadows. ~5 fits dog/h-series; raise for larger scenes.
static float g_light_ortho     = 5.0f;
static const float g_light_znear = 0.1f;   // not user-tunable yet
static const float g_light_zfar  = 30.0f;
#define SHADOW_TEX_UNIT 1   // GL_TEXTURE1 — main pass binds the depth tex here

// Called from Python before each render. No GL state touched here — just
// stashes values; win_render/egl_render reads them per frame.
void render_set_light(float pos[3], float target[3], float ortho, int shadow_enabled){
    g_light_pos[0]   = pos[0];    g_light_pos[1]   = pos[1];    g_light_pos[2]   = pos[2];
    g_light_target[0]= target[0]; g_light_target[1]= target[1]; g_light_target[2]= target[2];
    g_light_ortho    = ortho;
    g_shadow_enabled = shadow_enabled;
}

//#define MAX_PACKET   1400
#define MAX_PACKET 1024*16
#define HEADER_SIZE  16
#define PAYLOAD_SIZE (MAX_PACKET-HEADER_SIZE)

// -------------------- vec3 as float[3] --------------------
static void v3_set(float v[3], float x, float y, float z){
    v[0]=x; v[1]=y; v[2]=z;
}

static void v3_sub(float out[3], const float a[3], const float b[3]){
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

static void v3_add(float out[3], const float a[3], const float b[3]){
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
}

static void v3_scale(float r[3], const float a[3], float s){
    r[0]=a[0]*s; r[1]=a[1]*s; r[2]=a[2]*s;
}

/*static float v3_dot(const float a[3], const float b[3]){
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    }*/

static void v3_cross(float out[3], const float a[3], const float b[3]){
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static void v3_norm(float out[3], const float a[3]){
    float l = sqrtf(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    if(l <= 1e-8f){
	out[0] = out[1] = out[2] = 0.0f;
	return;
    }
    out[0] = a[0]/l;
    out[1] = a[1]/l;
    out[2] = a[2]/l;
}

// -------------------- mat4 as float[16] column-major --------------------
static void m4_identity(float out[16]){
    memset(out, 0, 16*sizeof(float));
    out[0] = out[5] = out[10] = out[15] = 1.0f;
}

static void m4_mul(float out[16], const float a[16], const float b[16]){
    float r[16];
    for(int c = 0; c < 4; c++){
        for(int rr = 0; rr < 4; rr++){
            r[c*4 + rr] =
                a[0*4 + rr]*b[c*4 + 0] +
                a[1*4 + rr]*b[c*4 + 1] +
                a[2*4 + rr]*b[c*4 + 2] +
                a[3*4 + rr]*b[c*4 + 3];
        }
    }
    memcpy(out,r,sizeof(r));
}

static void m4_perspective(float out[16], float fovy_rad, float aspect, float znear, float zfar){
    float f = 1.0f / tanf(fovy_rad*0.5f);
    memset(out,0,16*sizeof(float));
    out[0]  = f/aspect;
    out[5]  = f;
    out[10] = (zfar+znear)/(znear-zfar);
    out[11] = -1.0f;
    out[14] = (2.0f*zfar*znear)/(znear-zfar);
}

static void m4_lookat(float out[16], const float eye[3], const float center[3], const float up[3]){
    float f[3], s[3], u[3], tmp[3];

    v3_sub(tmp, center, eye);
    v3_norm(f, tmp);

    v3_cross(tmp, f, up);
    v3_norm(s, tmp);

    v3_cross(u, s, f);

    m4_identity(out);
    out[0]=s[0]; out[4]=s[1]; out[8] =s[2];
    out[1]=u[0]; out[5]=u[1]; out[9] =u[2];
    out[2]=-f[0];out[6]=-f[1];out[10]=-f[2];

    out[12] = -(s[0]*eye[0] + s[1]*eye[1] + s[2]*eye[2]);
    out[13] = -(u[0]*eye[0] + u[1]*eye[1] + u[2]*eye[2]);
    out[14] =  (f[0]*eye[0] + f[1]*eye[1] + f[2]*eye[2]);
}

// -------------------- mat3 normal matrix --------------------
static void m3_from_m4(float out[9], const float m[16]){
    out[0]=m[0]; out[1]=m[1]; out[2]=m[2];
    out[3]=m[4]; out[4]=m[5]; out[5]=m[6];
    out[6]=m[8]; out[7]=m[9]; out[8]=m[10];
}

static float m3_det(const float A[9]){
    float a=A[0], b=A[3], c=A[6];
    float d=A[1], e=A[4], f=A[7];
    float g=A[2], h=A[5], i=A[8];
    return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
}

static void m3_inv(float out[9], const float A[9]){
    float a=A[0], b=A[3], c=A[6];
    float d=A[1], e=A[4], f=A[7];
    float g=A[2], h=A[5], i=A[8];
    float det = m3_det(A);

    if(fabsf(det)<1e-8f){
        out[0]=1; out[1]=0; out[2]=0;
        out[3]=0; out[4]=1; out[5]=0;
        out[6]=0; out[7]=0; out[8]=1;
        return;
    }
    float invdet=1.0f/det;

    out[0] =  (e*i - f*h)*invdet;
    out[3] = -(b*i - c*h)*invdet;
    out[6] =  (b*f - c*e)*invdet;

    out[1] = -(d*i - f*g)*invdet;
    out[4] =  (a*i - c*g)*invdet;
    out[7] = -(a*f - c*d)*invdet;

    out[2] =  (d*h - e*g)*invdet;
    out[5] = -(a*h - b*g)*invdet;
    out[8] =  (a*e - b*d)*invdet;
}

static void m3_transpose(float out[9], const float A[9]){
    out[0]=A[0]; out[1]=A[3]; out[2]=A[6];
    out[3]=A[1]; out[4]=A[4]; out[5]=A[7];
    out[6]=A[2]; out[7]=A[5]; out[8]=A[8];
}

// -----------------------------
// Orbit + Pan camera
// -----------------------------

typedef struct{
    float target[3];
    float distance;
    float yaw, pitch;
    float worldUp[3];

    float rotateSpeed;
    float panSpeed;
    float zoomSpeed;

    int draggingL;
    int draggingR;
    double lastX, lastY;
} Camera;
static Camera g_cam;


static void cam_eye(const Camera *c, float eye[3]){
    float cp = cosf(c->pitch);
    float sp = sinf(c->pitch);
    float cy = cosf(c->yaw);
    float sy = sinf(c->yaw);

    //for Y-up
    //eye[0] = c->target[0] + c->distance*cp*sy;
    //eye[1] = c->target[1] + c->distance*sp;
    //eye[2] = c->target[2] + c->distance*cp*cy;

    //for Z-up
    eye[0] = c->target[0] + c->distance*cp*cy;
    eye[1] = c->target[1] + c->distance*cp*sy;
    eye[2] = c->target[2] + c->distance*sp;
}

static void cam_onMouseMove(Camera* c, double x, double y){
    double dx = c->lastX - x;
    double dy = c->lastY - y;

    c->lastX = x;
    c->lastY = y;

    if(c->draggingL){
        c->yaw += dx*c->rotateSpeed;
        c->pitch += -dy*c->rotateSpeed;

	float eps = (float)(1.0 * M_PI / 180.0);
	float lo = -(float)M_PI/2.0f + eps;
	float hi =  (float)M_PI/2.0f - eps;
	if(c->pitch < lo) c->pitch = lo;
	if(c->pitch > hi) c->pitch = hi;
    }

    if(c->draggingR){
        float eye[3], f[3], r[3], u[3], pan[3], tmp[3];
	
        cam_eye(c, eye);
        v3_sub(tmp, c->target, eye);
        v3_norm(f, tmp);

        v3_cross(tmp, f, c->worldUp);
        v3_norm(r, tmp);
	
        v3_cross(u, r, f);
        float scale = c->distance * c->panSpeed;

	v3_scale(tmp, r, dx*scale);
	v3_scale(pan, u, -dy*scale);
	
        v3_add(pan, pan, tmp);
        v3_add(c->target, c->target, pan);
    }
}

static void cam_onScroll(Camera *c, double yoffset){
    c->distance *= 1.0f - yoffset * c->zoomSpeed;
    if(c->distance < 0.2f) c->distance = 0.2f;
}

static void mouse_button_cb(GLFWwindow* win, int button, int action, int mods){
    (void)mods;
    double x, y;
    glfwGetCursorPos(win, &x, &y);
    if(action == GLFW_PRESS){
	g_cam.lastX = x;
	g_cam.lastY = y;
	if(button == GLFW_MOUSE_BUTTON_LEFT)  g_cam.draggingL = 1;
	if(button == GLFW_MOUSE_BUTTON_RIGHT) g_cam.draggingR = 1;
    }
    else if(action == GLFW_RELEASE){
	if(button == GLFW_MOUSE_BUTTON_LEFT)  g_cam.draggingL = 0;
	if(button == GLFW_MOUSE_BUTTON_RIGHT) g_cam.draggingR = 0;
    }
}

static void cursor_pos_cb(GLFWwindow* win, double x, double y){
    (void)win;
    cam_onMouseMove(&g_cam, x, y);
}

static void scroll_cb(GLFWwindow* win, double xoff, double yoff){
    (void)win; (void)xoff;
    cam_onScroll(&g_cam, yoff);
}

static void framebuffer_size_cb(GLFWwindow* win, int w, int h){
    (void)win;
    glViewport(0, 0, w, h);
}

static void key_cb(GLFWwindow* win, int key, int scancode, int action, int mods){
    (void)win; (void)scancode;
    //if (key == GLFW_KEY_R && action == GLFW_PRESS) rec_request = !rec_request;
    //else if(key == GLFW_KEY_Q && action == GLFW_PRESS) quit_request = !quit_request;
    if (key == GLFW_KEY_R && action == GLFW_PRESS && (mods & GLFW_MOD_SHIFT)) rec_request = !rec_request;
    else if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) quit_request = !quit_request;
    else if(key == GLFW_KEY_SPACE && action == GLFW_PRESS) space_request = 1;
}

// -------------------- shader (Phong + shadow map) --------------------
static const char* VS =
"#version 330 core\n"
"layout(location=0) in vec3 aPos;\n"
"layout(location=1) in vec3 aNormal;\n"
"uniform mat4 uMVP;\n"
"uniform mat4 uModel;\n"
"uniform mat3 uNormalMat;\n"
"uniform mat4 uLightVP;\n"
"out vec3 vWorldPos;\n"
"out vec3 vWorldN;\n"
"out vec4 vLightSpace;\n"
"void main(){\n"
"  vec4 wp = uModel * vec4(aPos,1.0);\n"
"  vWorldPos = wp.xyz;\n"
"  vWorldN = normalize(uNormalMat * aNormal);\n"
"  vLightSpace = uLightVP * wp;\n"
"  gl_Position = uMVP * vec4(aPos,1.0);\n"
"}\n";

static const char* FS =
"#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 vWorldPos;\n"
"in vec3 vWorldN;\n"
"in vec4 vLightSpace;\n"
"uniform vec4 uColor;\n"
"uniform vec3 uLightPos;\n"
"uniform vec3 uViewPos;\n"
"uniform sampler2DShadow uShadowMap;\n"
"uniform int uShadowEnabled;\n"
"void main(){\n"
"  vec3 N = normalize(vWorldN);\n"
"  vec3 L = normalize(uLightPos - vWorldPos);\n"
"  vec3 V = normalize(uViewPos - vWorldPos);\n"
"  vec3 R = reflect(-L, N);\n"
"  float diff = max(dot(N,L), 0.0);\n"
"  float spec = pow(max(dot(V,R), 0.0), 64.0);\n"
"  float vis = 1.0;\n"
"  if(uShadowEnabled == 1){\n"
"    vec3 p = vLightSpace.xyz / vLightSpace.w;\n"
"    p = p * 0.5 + 0.5;\n"
"    if(p.x>=0.0 && p.x<=1.0 && p.y>=0.0 && p.y<=1.0 && p.z<=1.0){\n"
"      float bias = max(0.0015 * (1.0 - dot(N,L)), 0.0003);\n"
"      p.z -= bias;\n"
"      // 3x3 PCF on top of hardware 2x2 → effectively 6x6 footprint\n"
"      vec2 texel = 1.0 / vec2(textureSize(uShadowMap, 0));\n"
"      float s = 0.0;\n"
"      for(int dy=-1; dy<=1; dy++)\n"
"        for(int dx=-1; dx<=1; dx++)\n"
"          s += texture(uShadowMap, vec3(p.xy + vec2(dx,dy)*texel, p.z));\n"
"      vis = s / 9.0;\n"
"      // Floor the shadow factor: fully-shadowed pixels stay at MIN_VIS\n"
"      // brightness instead of going to 0. Lit areas unaffected. Tune 0..1.\n"
"      const float MIN_VIS = 0.5;\n"
"      vis = mix(MIN_VIS, 1.0, vis);\n"
"    }\n"
"  }\n"
"  vec3 ambient  = 0.12 * uColor.rgb;\n"
"  vec3 diffuse  = 0.85 * diff * uColor.rgb * vis;\n"
"  vec3 specular = 0.35 * spec * vec3(1.0) * vis;\n"
"  FragColor = vec4(ambient + diffuse + specular, uColor.a);\n"
"}\n";

// Depth-only shader for the shadow pass. Only positions matter.
static const char* SHADOW_VS =
"#version 330 core\n"
"layout(location=0) in vec3 aPos;\n"
"uniform mat4 uLightMVP;\n"
"void main(){\n"
"  gl_Position = uLightMVP * vec4(aPos,1.0);\n"
"}\n";

static const char* SHADOW_FS =
"#version 330 core\n"
"void main(){}\n";

static void check_shader(GLuint s, const char* name){
    GLint ok=0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if(!ok){
        char log[4096] = {0};
        GLsizei n = 0;
        glGetShaderInfoLog(s, sizeof(log)-1, &n, log);
        fprintf(stderr,"[%s] compile error (%d bytes):\n%s\n", name, (int)n, log);
    }
}

static void check_program(GLuint p){
    GLint ok=0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if(!ok){
        char log[4096] = {0};
        GLsizei n = 0;
        glGetProgramInfoLog(p, sizeof(log)-1, &n, log);
        fprintf(stderr,"[program] link error (%d bytes):\n%s\n", (int)n, log);
    }
}

static GLuint make_program(void){
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs,1,&VS,NULL);
    glCompileShader(vs);
    check_shader(vs,"vs");

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs,1,&FS,NULL);
    glCompileShader(fs);
    check_shader(fs,"fs");

    GLuint p = glCreateProgram();
    glAttachShader(p,vs);
    glAttachShader(p,fs);
    glLinkProgram(p);
    check_program(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

static GLuint make_shadow_program(void){
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs,1,&SHADOW_VS,NULL);
    glCompileShader(vs);
    check_shader(vs,"shadow_vs");

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs,1,&SHADOW_FS,NULL);
    glCompileShader(fs);
    check_shader(fs,"shadow_fs");

    GLuint p = glCreateProgram();
    glAttachShader(p,vs);
    glAttachShader(p,fs);
    glLinkProgram(p);
    check_program(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

// Creates a depth-only FBO with a SHADOW_MAP_SIZE^2 depth texture.
// Texture is set up for hardware PCF (sampler2DShadow + GL_LINEAR + COMPARE_REF).
// Border color = 1.0 so samples outside the light frustum read "far" → lit.
static void make_shadow_fbo(GLuint* out_fbo, GLuint* out_tex){
    GLuint tex = 0, fbo = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24,
                 SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 0,
                 GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float border[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if(status != GL_FRAMEBUFFER_COMPLETE){
        fprintf(stderr, "shadow FBO incomplete: 0x%x\n", status);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    *out_fbo = fbo;
    *out_tex = tex;
}

// Builds the view-projection matrix used to render the depth pass from the
// light's perspective. Orthographic projection sized for typical robot scenes.
static void compute_light_vp(float out_LVP[16]){
    float Lview[16], Lproj[16];
    float up[3] = {0.0f, 0.0f, 1.0f};
    m4_lookat(Lview, g_light_pos, g_light_target, up);
    // Orthographic: glOrtho(-h, h, -h, h, znear, zfar), column-major
    float h = g_light_ortho;
    float zn = g_light_znear, zf = g_light_zfar;
    memset(Lproj, 0, sizeof(Lproj));
    Lproj[0]  = 1.0f / h;
    Lproj[5]  = 1.0f / h;
    Lproj[10] = -2.0f / (zf - zn);
    Lproj[14] = -(zf + zn) / (zf - zn);
    Lproj[15] = 1.0f;
    m4_mul(out_LVP, Lproj, Lview);
}

// -------------------- mesh (pos+normal interleaved) --------------------
typedef struct {
    GLuint vao, vbo;
    int vertex_count;
} Mesh;

static Mesh mesh_from_pos_nrm(const float* data6, int vertex_count){
    Mesh m = {0};
    m.vertex_count = vertex_count;

    glGenVertexArrays(1,&m.vao);
    glGenBuffers(1,&m.vbo);
    
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, (size_t)vertex_count*6*sizeof(float), data6, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
    return m;
}

/*static void mesh_destroy(Mesh* m){
    if(m->vbo) glDeleteBuffers(1,&m->vbo);
    if(m->vao) glDeleteVertexArrays(1,&m->vao);
    memset(m,0,sizeof(*m));
    }*/

// -------------------- dynamic arrays --------------------
typedef struct {
    float* data;
    int count, cap;
} FloatArr;

static void fa_init(FloatArr* a){
    a->data=NULL;
    a->count=0;
    a->cap=0;
}

static void fa_free(FloatArr* a){
    free(a->data);
    a->data=NULL;
    a->count=0;
    a->cap=0;
}

static void fa_push6(FloatArr* a, float px,float py,float pz, float nx,float ny,float nz){
    if(a->count + 6 > a->cap){
        a->cap = (a->cap==0)? (6*4096) : (a->cap*2);
        a->data = (float*)realloc(a->data, (size_t)a->cap*sizeof(float));
        if(!a->data){ fprintf(stderr,"OOM\n"); exit(1); }
    }
    a->data[a->count++] = px; a->data[a->count++] = py; a->data[a->count++] = pz;
    a->data[a->count++] = nx; a->data[a->count++] = ny; a->data[a->count++] = nz;
}

static void add_tri_pn(FloatArr* a, const float p0[3], const float n0[3], const float p1[3], const float n1[3], const float p2[3], const float n2[3]){
    fa_push6(a, p0[0], p0[1], p0[2], n0[0], n0[1], n0[2]);
    fa_push6(a, p1[0], p1[1], p1[2], n1[0], n1[1], n1[2]);
    fa_push6(a, p2[0], p2[1], p2[2], n2[0], n2[1], n2[2]);
}

/*static void add_tri_flat(FloatArr* a, const float p0[3], const float p1[3], const float p2[3]){
    float e1[3], e2[3], n[3];
    v3_sub(e1, p1, p0);
    v3_sub(e2, p2, p0);
    v3_cross(n, e1, e2);
    v3_norm(n, n);
    add_tri_pn(a, p0, n, p1, n, p2, n);
    }*/

// -------------------- basic shapes --------------------
static Mesh make_box(float hx, float hy, float hz){
    float x = hx, y = hy, z = hz;
    FloatArr a; fa_init(&a);

    // Helper: push a vertex with normal
    #define V(px, py, pz, nx, ny, nz) fa_push6(&a, (px), (py), (pz), (nx), (ny), (nz))

    // Each face: 2 triangles, CCW when viewed from outside.
    // +X face (normal +1,0,0)
    V(+x,-y,-z, 1,0,0);  V(+x,+y,-z, 1,0,0);  V(+x,+y,+z, 1,0,0);
    V(+x,-y,-z, 1,0,0);  V(+x,+y,+z, 1,0,0);  V(+x,-y,+z, 1,0,0);

    // -X face (normal -1,0,0)
    V(-x,-y,+z,-1,0,0);  V(-x,+y,+z,-1,0,0);  V(-x,+y,-z,-1,0,0);
    V(-x,-y,+z,-1,0,0);  V(-x,+y,-z,-1,0,0);  V(-x,-y,-z,-1,0,0);

    // +Y face (normal 0,+1,0)
    V(-x,+y,-z, 0,1,0);  V(-x,+y,+z, 0,1,0);  V(+x,+y,+z, 0,1,0);
    V(-x,+y,-z, 0,1,0);  V(+x,+y,+z, 0,1,0);  V(+x,+y,-z, 0,1,0);

    // -Y face (normal 0,-1,0)
    V(-x,-y,+z, 0,-1,0); V(-x,-y,-z, 0,-1,0); V(+x,-y,-z, 0,-1,0);
    V(-x,-y,+z, 0,-1,0); V(+x,-y,-z, 0,-1,0); V(+x,-y,+z, 0,-1,0);

    // +Z face (normal 0,0,+1)
    V(-x,-y,+z, 0,0,1);  V(+x,-y,+z, 0,0,1);  V(+x,+y,+z, 0,0,1);
    V(-x,-y,+z, 0,0,1);  V(+x,+y,+z, 0,0,1);  V(-x,+y,+z, 0,0,1);

    // -Z face (normal 0,0,-1)
    V(+x,-y,-z, 0,0,-1); V(-x,-y,-z, 0,0,-1); V(-x,+y,-z, 0,0,-1);
    V(+x,-y,-z, 0,0,-1); V(-x,+y,-z, 0,0,-1); V(+x,+y,-z, 0,0,-1);
    #undef V

    Mesh m = mesh_from_pos_nrm(a.data, a.count/6);
    fa_free(&a);
    return m;
}

static Mesh make_sphere(float r,int stacks,int slices){
    FloatArr a; fa_init(&a);
    for(int i=0;i<stacks;i++){
        float v0=(float)i/stacks, v1=(float)(i+1)/stacks;
        float th0=v0*(float)M_PI, th1=v1*(float)M_PI;
	
        for(int j=0;j<slices;j++){
            float u0=(float)j/slices, u1=(float)(j+1)/slices;
            float ph0=u0*2.0f*(float)M_PI, ph1=u1*2.0f*(float)M_PI;

            float p00[3]={ r*sinf(th0)*cosf(ph0), r*cosf(th0), r*sinf(th0)*sinf(ph0) };
            float p01[3]={ r*sinf(th0)*cosf(ph1), r*cosf(th0), r*sinf(th0)*sinf(ph1) };
            float p10[3]={ r*sinf(th1)*cosf(ph0), r*cosf(th1), r*sinf(th1)*sinf(ph0) };
            float p11[3]={ r*sinf(th1)*cosf(ph1), r*cosf(th1), r*sinf(th1)*sinf(ph1) };

            float n00[3],n01[3],n10[3],n11[3];
            v3_norm(n00,p00); v3_norm(n01,p01); v3_norm(n10,p10); v3_norm(n11,p11);

            add_tri_pn(&a,p00,n00,p10,n10,p11,n11);
            add_tri_pn(&a,p00,n00,p11,n11,p01,n01);
        }
    }
    Mesh m = mesh_from_pos_nrm(a.data, a.count/6);
    fa_free(&a);
    return m;
}

static Mesh make_cylinder(float r, float hh, int slices){
    FloatArr a;
    fa_init(&a);

    float z0 = -hh;
    float z1 =  hh;

    float topC[3] = {0,0,z1};
    float botC[3] = {0,0,z0};

    float nTop[3] = {0,0, 1};
    float nBot[3] = {0,0,-1};

    for(int j=0; j<slices; j++){
        float u0 = (float)j/slices;
        float u1 = (float)(j+1)/slices;

        float ph0 = u0 * 2.0f * (float)M_PI;
        float ph1 = u1 * 2.0f * (float)M_PI;

        // bottom ring
        float b0[3] = { r*cosf(ph0), r*sinf(ph0), z0 };
        float b1[3] = { r*cosf(ph1), r*sinf(ph1), z0 };

        // top ring
        float t0[3] = { r*cosf(ph0), r*sinf(ph0), z1 };
        float t1[3] = { r*cosf(ph1), r*sinf(ph1), z1 };

        // ---- side normals (radial in XY plane)
        float nb0[3] = { b0[0], b0[1], 0 };
        float nb1[3] = { b1[0], b1[1], 0 };
        float nt0[3] = { t0[0], t0[1], 0 };
        float nt1[3] = { t1[0], t1[1], 0 };

        v3_norm(nb0, nb0);
        v3_norm(nb1, nb1);
        v3_norm(nt0, nt0);
        v3_norm(nt1, nt1);

        // side surface (CCW from outside)
        add_tri_pn(&a, b0,nb0, t0,nt0, t1,nt1);
        add_tri_pn(&a, b0,nb0, t1,nt1, b1,nb1);

        // ---- top cap
        add_tri_pn(&a, topC,nTop, t1,nTop, t0,nTop);

        // ---- bottom cap
        add_tri_pn(&a, botC,nBot, b0,nBot, b1,nBot);
    }

    Mesh m = mesh_from_pos_nrm(a.data, a.count/6);
    fa_free(&a);
    return m;
}

static void add_hemisphere(FloatArr* a, float r, int stacks, int slices, float zCenter, int frontHalf){
    // frontHalf=1 : +Z 쪽 반구 (theta 0..pi/2)
    // frontHalf=0 : -Z 쪽 반구 (theta pi/2..pi)
    int half = stacks/2;
    int iStart = frontHalf ? 0 : half;
    int iEnd   = frontHalf ? half : stacks;

    for(int i=iStart; i<iEnd; i++){
        float v0=(float)i/stacks, v1=(float)(i+1)/stacks;
        float th0=v0*(float)M_PI, th1=v1*(float)M_PI;

        for(int j=0; j<slices; j++){
            float u0=(float)j/slices, u1=(float)(j+1)/slices;
            float ph0=u0*2.0f*(float)M_PI, ph1=u1*2.0f*(float)M_PI;

            // Sphere (axis along Z):
            // x = r*sin(theta)*cos(phi)
            // y = r*sin(theta)*sin(phi)
            // z = r*cos(theta)
            float s00[3]={ r*sinf(th0)*cosf(ph0), r*sinf(th0)*sinf(ph0), r*cosf(th0) };
            float s01[3]={ r*sinf(th0)*cosf(ph1), r*sinf(th0)*sinf(ph1), r*cosf(th0) };
            float s10[3]={ r*sinf(th1)*cosf(ph0), r*sinf(th1)*sinf(ph0), r*cosf(th1) };
            float s11[3]={ r*sinf(th1)*cosf(ph1), r*sinf(th1)*sinf(ph1), r*cosf(th1) };

            // Shift hemisphere to zCenter
            float p00[3]={ s00[0], s00[1], s00[2] + zCenter };
            float p01[3]={ s01[0], s01[1], s01[2] + zCenter };
            float p10[3]={ s10[0], s10[1], s10[2] + zCenter };
            float p11[3]={ s11[0], s11[1], s11[2] + zCenter };

            // Normals are sphere normals (from origin of the cap)
            float n00[3],n01[3],n10[3],n11[3];
            v3_norm(n00,s00); v3_norm(n01,s01); v3_norm(n10,s10); v3_norm(n11,s11);

            add_tri_pn(a,p00,n00,p10,n10,p11,n11);
            add_tri_pn(a,p00,n00,p11,n11,p01,n01);
        }
    }
}

static Mesh make_capsule(float r, float hh, int stacks, int slices){
    FloatArr a; fa_init(&a);

    // Cylinder runs along Z now (hh = half cylindrical length)
    float z0 = -hh;
    float z1 =  hh;

    // Side surface
    for(int j=0; j<slices; j++){
        float u0=(float)j/slices, u1=(float)(j+1)/slices;
        float ph0=u0*2.0f*(float)M_PI, ph1=u1*2.0f*(float)M_PI;

        // ring points at z0, z1
        float b0[3]={ r*cosf(ph0), r*sinf(ph0), z0 };
        float b1[3]={ r*cosf(ph1), r*sinf(ph1), z0 };
        float t0[3]={ r*cosf(ph0), r*sinf(ph0), z1 };
        float t1[3]={ r*cosf(ph1), r*sinf(ph1), z1 };

        // normals (radial in XY)
        float nb0[3]={ b0[0], b0[1], 0 };
        float nb1[3]={ b1[0], b1[1], 0 };
        float nt0[3]={ t0[0], t0[1], 0 };
        float nt1[3]={ t1[0], t1[1], 0 };
        v3_norm(nb0,nb0); v3_norm(nb1,nb1); v3_norm(nt0,nt0); v3_norm(nt1,nt1);

        // two triangles, CCW when viewed from outside
        add_tri_pn(&a, b0, nb0, t0, nt0, t1, nt1);
        add_tri_pn(&a, b0, nb0, t1, nt1, b1, nb1);
    }

    // Caps (hemispheres) along +Z and -Z
    add_hemisphere(&a, r, stacks, slices, z1, 1); // +Z cap
    add_hemisphere(&a, r, stacks, slices, z0, 0); // -Z cap

    Mesh m = mesh_from_pos_nrm(a.data, a.count/6);
    fa_free(&a);
    return m;
}

// -------------------- OBJ loader: supports only "v" and "f", faces may be polygons. no normals in file. --------------------
typedef struct {
    float* v;
    int vcount;
    int vcap;
} Verts;

typedef struct {
    int*   idx;
    int icount;
    int icap;
} Inds;

static void verts_push(Verts* a, float x,float y,float z){
    if(a->vcount+3 > a->vcap){
        a->vcap = (a->vcap==0)? (3*4096) : (a->vcap*2);
        a->v = (float*)realloc(a->v, (size_t)a->vcap*sizeof(float));
        if(!a->v){ fprintf(stderr,"OBJ OOM (verts)\n"); exit(1); }
    }
    a->v[a->vcount++] = x; a->v[a->vcount++] = y; a->v[a->vcount++] = z;
}

static void inds_push(Inds* a, int i){
    if(a->icount+1 > a->icap){
        a->icap = (a->icap==0)? 4096 : (a->icap*2);
        a->idx = (int*)realloc(a->idx, (size_t)a->icap*sizeof(int));
        if(!a->idx){ fprintf(stderr,"OBJ OOM (inds)\n"); exit(1); }
    }
    a->idx[a->icount++] = i;
}

// parse face token: "i" or "i/..." or "i//..." etc. return vertex index (0-based), supports negative indices.
static int parse_obj_index(const char* tok, int vtx_count){
    // copy until '/' or end
    char buf[64];
    int k=0;
    while(tok[k] && tok[k] != '/' && k < (int)sizeof(buf)-1){
        buf[k]=tok[k]; k++;
    }
    buf[k]=0;
    int idx = atoi(buf);
    if(idx == 0) return -1; // invalid
    if(idx < 0) idx = vtx_count + idx;      // -1 means last
    else        idx = idx - 1;              // OBJ is 1-based
    return idx;
}

static Mesh load_obj_as_mesh(const char* path){
    FILE* f = fopen(path, "rb");
    if(!f){
        fprintf(stderr, "OBJ: cannot open %s\n", path);
        Mesh empty={0};
        return empty;
    }

    Verts verts={0};
    Inds tris={0};
    char line[4096];
    
    while(fgets(line, sizeof(line), f)){
        // skip leading spaces
        char* s=line;
        while(*s && isspace((unsigned char)*s)) s++;
        if(*s=='#' || *s==0) continue;

        if(s[0]=='v' && isspace((unsigned char)s[1])){
            float x,y,z;
            if(sscanf(s+1, "%f %f %f", &x,&y,&z)==3){
                verts_push(&verts,x,y,z);
            }
        } else if(s[0]=='f' && isspace((unsigned char)s[1])){
            // read all tokens after 'f'
            int face_idx[64];
            int n=0;

            char* p = s+1;
            while(*p){
                while(*p && isspace((unsigned char)*p)) p++;
                if(!*p || *p=='\n' || *p=='\r') break;

                char tok[128];
                int tlen=0;
                while(*p && !isspace((unsigned char)*p) && tlen < (int)sizeof(tok)-1){
                    tok[tlen++] = *p++;
                }
                tok[tlen]=0;

                int vcount = verts.vcount/3;
                int idx = parse_obj_index(tok, vcount);
                if(idx >= 0 && n < (int)(sizeof(face_idx)/sizeof(face_idx[0]))){
                    face_idx[n++] = idx;
                }
            }

            // triangulate polygon fan: (0, i, i+1)
            if(n >= 3){
                for(int i=1;i+1<n;i++){
                    inds_push(&tris, face_idx[0]);
                    inds_push(&tris, face_idx[i]);
                    inds_push(&tris, face_idx[i+1]);
                }
            }
        }
    }
    
    fclose(f);
    int vcount = verts.vcount/3;
    int tcount = tris.icount/3;

    if(vcount == 0 || tcount == 0){
        fprintf(stderr, "OBJ: no vertices or faces in %s\n", path);
        free(verts.v); free(tris.idx);
        Mesh empty={0};
        return empty;
    }

    // compute smooth vertex normals: sum face normals
    float* vN = (float*)calloc((size_t)vcount*3, sizeof(float));
    if(!vN){ fprintf(stderr,"OBJ OOM (normals)\n"); exit(1); }

    for(int ti=0; ti<tcount; ti++){
        int i0 = tris.idx[ti*3+0];
        int i1 = tris.idx[ti*3+1];
        int i2 = tris.idx[ti*3+2];

        float p0[3]={ verts.v[i0*3+0], verts.v[i0*3+1], verts.v[i0*3+2] };
        float p1[3]={ verts.v[i1*3+0], verts.v[i1*3+1], verts.v[i1*3+2] };
        float p2[3]={ verts.v[i2*3+0], verts.v[i2*3+1], verts.v[i2*3+2] };

        float e1[3], e2[3], fn[3];
        v3_sub(e1,p1,p0);
        v3_sub(e2,p2,p0);
        v3_cross(fn,e1,e2);
        // area-weighted normal (no normalize here)
        vN[i0*3+0] += fn[0]; vN[i0*3+1] += fn[1]; vN[i0*3+2] += fn[2];
        vN[i1*3+0] += fn[0]; vN[i1*3+1] += fn[1]; vN[i1*3+2] += fn[2];
        vN[i2*3+0] += fn[0]; vN[i2*3+1] += fn[1]; vN[i2*3+2] += fn[2];
    }

    for(int i=0;i<vcount;i++){
        float n[3]={ vN[i*3+0], vN[i*3+1], vN[i*3+2] };
        v3_norm(n,n);
        vN[i*3+0]=n[0]; vN[i*3+1]=n[1]; vN[i*3+2]=n[2];
    }

    // build interleaved triangle list (pos+normal) for glDrawArrays
    FloatArr out; fa_init(&out);
    out.cap = 6 * (tris.icount); // rough reserve
    out.data = (float*)malloc((size_t)out.cap * sizeof(float));
    if(!out.data){ fprintf(stderr,"OBJ OOM (out)\n"); exit(1); }
    out.count = 0;

    //full smooth
    /*for(int k=0;k<tris.icount;k++){
        int vi = tris.idx[k];
        float px = verts.v[vi*3+0], py=verts.v[vi*3+1], pz=verts.v[vi*3+2];
        float nx = vN[vi*3+0], ny=vN[vi*3+1], nz=vN[vi*3+2];
        fa_push6(&out, px,py,pz, nx,ny,nz);
	}*/


    //flat shading
    for(int ti=0; ti<tcount; ti++){
	int i0 = tris.idx[ti*3+0];
	int i1 = tris.idx[ti*3+1];
	int i2 = tris.idx[ti*3+2];

	float p0[3]={ verts.v[i0*3+0], verts.v[i0*3+1], verts.v[i0*3+2] };
	float p1[3]={ verts.v[i1*3+0], verts.v[i1*3+1], verts.v[i1*3+2] };
	float p2[3]={ verts.v[i2*3+0], verts.v[i2*3+1], verts.v[i2*3+2] };

	float e1[3], e2[3], n[3];
	v3_sub(e1,p1,p0);
	v3_sub(e2,p2,p0);
	v3_cross(n,e1,e2);
	v3_norm(n,n);
	
	// (선택) 카메라쪽으로 뒤집기: winding 섞인 모델 디버그에 도움
	// float triCenter[3]={(p0[0]+p1[0]+p2[0])/3, (p0[1]+p1[1]+p2[1])/3, (p0[2]+p1[2]+p2[2])/3};
	// float toEye[3]={ eye[0]-triCenter[0], eye[1]-triCenter[1], eye[2]-triCenter[2] };
	// if(v3_dot(n,toEye) < 0) { n[0]=-n[0]; n[1]=-n[1]; n[2]=-n[2]; }

	fa_push6(&out, p0[0],p0[1],p0[2], n[0],n[1],n[2]);
	fa_push6(&out, p1[0],p1[1],p1[2], n[0],n[1],n[2]);
	fa_push6(&out, p2[0],p2[1],p2[2], n[0],n[1],n[2]);
    }

    Mesh m = mesh_from_pos_nrm(out.data, out.count/6);
    free(verts.v);
    free(tris.idx);
    free(vN);
    fa_free(&out);
    return m;
}

// Build a triangulated surface mesh for height-field slot `slot`. The grid spans
// local [-sx, sx] × [-sy, sy] in XY with the height along +Z; each cell is two
// flat-shaded triangles (matching load_obj_as_mesh's active shading path).
static Mesh make_hfield(int slot){
    int nr = hf_nrow[slot], nc = hf_ncol[slot];
    const double *H = hf_data[slot];
    if (nr < 2 || nc < 2 || !H){ Mesh empty={0}; return empty; }

    float sx = (float)hf_sx[slot], sy = (float)hf_sy[slot];
    float dx = 2.0f*sx/(nc-1), dy = 2.0f*sy/(nr-1);

    FloatArr a; fa_init(&a);
    for(int i = 0; i+1 < nr; i++){
        float y0 = -sy + i*dy, y1 = -sy + (i+1)*dy;
        for(int j = 0; j+1 < nc; j++){
            float x0 = -sx + j*dx, x1 = -sx + (j+1)*dx;
            float p00[3] = { x0, y0, (float)H[ i   *nc + j  ] };
            float p10[3] = { x1, y0, (float)H[ i   *nc + j+1] };
            float p01[3] = { x0, y1, (float)H[(i+1)*nc + j  ] };
            float p11[3] = { x1, y1, (float)H[(i+1)*nc + j+1] };

            float e1[3], e2[3], n[3];
            // tri A: p00, p10, p11
            v3_sub(e1, p10, p00); v3_sub(e2, p11, p00); v3_cross(n, e1, e2); v3_norm(n, n);
            add_tri_pn(&a, p00, n, p10, n, p11, n);
            // tri B: p00, p11, p01
            v3_sub(e1, p11, p00); v3_sub(e2, p01, p00); v3_cross(n, e1, e2); v3_norm(n, n);
            add_tri_pn(&a, p00, n, p11, n, p01, n);
        }
    }
    Mesh m = mesh_from_pos_nrm(a.data, a.count/6);
    fa_free(&a);
    return m;
}

static void flip_vertical(void* img, int width, int height, int bytes_per_pixel, void* temp_row){
    int stride = width * bytes_per_pixel;

    for (int y = 0; y < height / 2; y++) {
        char* row_top = (char*)img + y * stride;
        char* row_bottom = (char*)img + (height - 1 - y) * stride;

        memcpy(temp_row, row_top, stride);
        memcpy(row_top, row_bottom, stride);
        memcpy(row_bottom, temp_row, stride);
    }
}

// Draw object i's lit mesh. Shared by win_render/egl_render Pass 2. V/P are
// column-major 4x4; objcolor is rgba (4 floats/obj), objpose 16 floats/obj.
static void draw_lit_object(int i, const float* V, const float* P,
                            const float* objcolor, const float* objpose,
                            GLint locModel, GLint locMVP, GLint locNMat, GLint locColor,
                            const Mesh* mesh){
    float VM[16], MVP[16], A3[9], inv3[9], Nmat[9];
    const float* T = objpose + 16*i;
    m4_mul(VM, V, T);
    m4_mul(MVP, P, VM);
    m3_from_m4(A3, T);
    m3_inv(inv3, A3);
    m3_transpose(Nmat, inv3);
    glUniformMatrix4fv(locModel, 1, GL_FALSE, T);
    glUniformMatrix4fv(locMVP,   1, GL_FALSE, MVP);
    glUniformMatrix3fv(locNMat,  1, GL_FALSE, Nmat);
    glUniform4f(locColor, objcolor[4*i+0], objcolor[4*i+1], objcolor[4*i+2], objcolor[4*i+3]);
    glBindVertexArray(mesh[i].vao);
    glDrawArrays(GL_TRIANGLES, 0, mesh[i].vertex_count);
}

// Draw one translucent object with correct intra-mesh ordering. Object-level
// sorting (sort_transparent) only fixes order *between* objects; within a single
// closed mesh its own near-side and far-side triangles still blend in arbitrary
// buffer order, which shows up as sawtooth seams along the silhouette. Drawing
// the mesh twice — far (back) faces first, then near (front) faces — restores
// back-to-front order. Triangles are wound CCW-from-outside (see the make_*
// builders), so GL_FRONT cull leaves the far shell, GL_BACK the near shell.
// Exact for convex shapes (box/sphere/cyl/capsule); still a large improvement
// for arbitrary meshes. Caller has GL_BLEND on, GL_CULL_FACE on, depth-write off.
static void draw_translucent_object(int i, const float* V, const float* P,
                                    const float* objcolor, const float* objpose,
                                    GLint locModel, GLint locMVP, GLint locNMat, GLint locColor,
                                    const Mesh* mesh){
    glCullFace(GL_FRONT);
    draw_lit_object(i, V, P, objcolor, objpose, locModel, locMVP, locNMat, locColor, mesh);
    glCullFace(GL_BACK);
    draw_lit_object(i, V, P, objcolor, objpose, locModel, locMVP, locNMat, locColor, mesh);
}

// Build a transparency draw order into `order` (objects with alpha<1, visible),
// sorted far->near by view-space depth so back-to-front blending is correct.
// Returns the count. `order`/`vz` must hold at least n_obj entries.
static int sort_transparent(int n_obj, const float* objcolor, const float* objpose,
                            const float* V, int* order, float* vz){
    int nt = 0;
    for(int i=0; i < n_obj; i++){
        if(objcolor[4*i+0] < 0)    continue;   // invisible sentinel
        if(objcolor[4*i+3] >= 1.0f) continue;  // opaque: drawn in the first pass
        const float* T = objpose + 16*i;
        // view-space z of the object origin (column-major V: row 2 = idx 2,6,10,14)
        order[nt] = i;
        vz[nt]    = V[2]*T[12] + V[6]*T[13] + V[10]*T[14] + V[14];
        nt++;
    }
    for(int a=1; a<nt; a++){          // insertion sort, ascending vz (most negative = farthest first)
        int oi = order[a]; float oz = vz[a]; int b = a-1;
        while(b>=0 && vz[b] > oz){ order[b+1]=order[b]; vz[b+1]=vz[b]; b--; }
        order[b+1]=oi; vz[b+1]=oz;
    }
    return nt;
}

int win_render(int n_obj, int* obj_type, float* shape, float* objcolor, float* objpose, float* campose){
    static GLFWwindow* window;

    static GLuint program;
    static GLint locMVP, locModel, locNMat, locColor, locLPos, locVPos;
    static GLint locLightVP, locShadowMap, locShadowEnabled;
    static GLuint shadow_program;
    static GLint locShadowLightMVP;
    static GLuint shadow_fbo, shadow_tex;
    #define WIN_MAX_OBJ 64
    #define WIN_N_PADDING 8
    static Mesh mesh[WIN_MAX_OBJ];
    // Per-slot fingerprint so add/delete (which mutates n_obj and shifts the
    // type/shape arrays) gets caught and only changed slots are rebuilt.
    static int   prev_type[WIN_MAX_OBJ];
    static float prev_shape[WIN_MAX_OBJ * WIN_N_PADDING];
    static int   prev_n_obj = 0;
    static long cnt=0;

    int win_width = 1200;
    int win_height = 800;  //initial window size; matches mjenv.cpp default

    if(!render_load_window_deps()) return -2;

    if(cnt == 0) {
	if(!glfwInit()){ fprintf(stderr,"GLFW init failed\n"); exit(0); }
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_SAMPLES, 4);

	window = glfwCreateWindow(win_width, win_height, "test", NULL, NULL);
	if(!window){ fprintf(stderr,"Window create failed\n"); glfwTerminate(); exit(0); }

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);   //명시적 vsync. 안 해도 NVIDIA 드라이버 기본값(SyncToVBlank=1)으로 vsync 걸리지만, mesa/iGPU 시스템 이식성을 위해 명시.
	//glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	if(!render_load_gl_procs(1)) return -2;

	glfwSetMouseButtonCallback(window, mouse_button_cb);
	glfwSetCursorPosCallback(window, cursor_pos_cb);
	glfwSetScrollCallback(window, scroll_cb);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_cb);
	glfwSetKeyCallback(window, key_cb);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_MULTISAMPLE);

	// 2. 렌더링 리소스 준비 (기존 코드 사용)
	program = make_program();
	locMVP   = glGetUniformLocation(program, "uMVP");
	locModel = glGetUniformLocation(program, "uModel");
	locNMat  = glGetUniformLocation(program, "uNormalMat");
	locColor = glGetUniformLocation(program, "uColor");
	locLPos  = glGetUniformLocation(program, "uLightPos");
	locVPos  = glGetUniformLocation(program, "uViewPos");
	locLightVP       = glGetUniformLocation(program, "uLightVP");
	locShadowMap     = glGetUniformLocation(program, "uShadowMap");
	locShadowEnabled = glGetUniformLocation(program, "uShadowEnabled");

	// Shadow program + FBO. Created unconditionally so toggling
	// g_shadow_enabled at runtime works without a re-init.
	shadow_program = make_shadow_program();
	locShadowLightMVP = glGetUniformLocation(shadow_program, "uLightMVP");
	make_shadow_fbo(&shadow_fbo, &shadow_tex);

	for(int i=0; i < WIN_MAX_OBJ; i++) prev_type[i] = -1;

	v3_set(g_cam.target, campose[0], campose[1], campose[2]);
	v3_set(g_cam.worldUp, 0, 0, 1);
	g_cam.distance = campose[3];
	g_cam.yaw = campose[4]*M_PI/180.0f;
	g_cam.pitch = campose[5]*M_PI/180.0f;

	g_cam.rotateSpeed = 0.002f;
	g_cam.panSpeed = 0.0005f;
	g_cam.zoomSpeed = 0.03f;

	g_cam.draggingL = 0;
	g_cam.draggingR = 0;
	g_cam.lastX = 0.0;
	g_cam.lastY = 0.0;
    }

    // Sync mesh table to current (type, shape) per slot. Handles Env.add()
    // (new slot index now has type/shape that didn't exist last frame) and
    // Env.delete() (slot index now holds what used to live further down).
    if (n_obj > WIN_MAX_OBJ) n_obj = WIN_MAX_OBJ;
    for(int i = 0; i < n_obj; i++){
	float *s = &shape[WIN_N_PADDING*i];
	int changed = (obj_type[i] != prev_type[i]);
	if(!changed){
	    for(int k = 0; k < WIN_N_PADDING; k++){
		if(s[k] != prev_shape[i*WIN_N_PADDING + k]){ changed = 1; break; }
	    }
	}
	if(changed){
	    if(mesh[i].vbo) glDeleteBuffers(1, &mesh[i].vbo);
	    if(mesh[i].vao) glDeleteVertexArrays(1, &mesh[i].vao);
	    mesh[i].vao = mesh[i].vbo = 0; mesh[i].vertex_count = 0;

	    if      (obj_type[i] == 101) mesh[i] = make_box(s[0], s[1], s[2]);
	    else if (obj_type[i] == 102) mesh[i] = make_sphere(s[0], 26, 40);
	    else if (obj_type[i] == 103) mesh[i] = make_cylinder(s[0], s[1], 44);
	    else if (obj_type[i] == 104) mesh[i] = make_capsule(s[0], s[1], 26, 44);
	    else if (obj_type[i] == 100){
		int idx = (int)s[0];
		if (mesh_path[idx][0] == '\0') {
		    fprintf(stderr, "win_render: no path registered for mesh slot %d\n", idx);
		    exit(0);
		}
		mesh[i] = load_obj_as_mesh(mesh_path[idx]);
	    }
	    else if (obj_type[i] == 105) mesh[i] = make_hfield((int)s[0]);

	    prev_type[i] = obj_type[i];
	    for(int k = 0; k < WIN_N_PADDING; k++) prev_shape[i*WIN_N_PADDING + k] = s[k];
	}
    }
    // Free GPU resources from slots that no longer exist (n_obj decreased).
    for(int i = n_obj; i < prev_n_obj; i++){
	if(mesh[i].vbo) glDeleteBuffers(1, &mesh[i].vbo);
	if(mesh[i].vao) glDeleteVertexArrays(1, &mesh[i].vao);
	mesh[i].vao = mesh[i].vbo = 0; mesh[i].vertex_count = 0;
	prev_type[i] = -1;
    }
    prev_n_obj = n_obj;

    if(glfwGetWindowAttrib(window, GLFW_ICONIFIED)) return 0;
    if(!glfwGetWindowAttrib(window, GLFW_VISIBLE)) return 0;

    float LightVP[16], LMVP[16];
    float *T;
    compute_light_vp(LightVP);  // recomputed each frame so render_set_light() changes apply immediately

    //---- Pass 1: depth-only render from the light, into the shadow FBO ----
    if(g_shadow_enabled){
	glUseProgram(shadow_program);
	glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo);
	glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
	glClear(GL_DEPTH_BUFFER_BIT);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1.5f, 4.0f);
	for(int i=0; i < n_obj; i++){
	    if(objcolor[4*i+0] < 0) continue;  // skip invisible
	    T = objpose+16*i;
	    m4_mul(LMVP, LightVP, T);
	    glUniformMatrix4fv(locShadowLightMVP, 1, GL_FALSE, LMVP);
	    glBindVertexArray(mesh[i].vao);
	    glDrawArrays(GL_TRIANGLES, 0, mesh[i].vertex_count);
	}
	glDisable(GL_POLYGON_OFFSET_FILL);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    //---- Pass 2: main lit render into the window framebuffer ----
    //adjust framebuffer size according to the window size
    glfwGetFramebufferSize(window, &win_width, &win_height);
    glViewport(0, 0, win_width, win_height);

    //glClearColor(0.06f, 0.07f, 0.09f, 1.0f);
    glClearColor(0.9f, 0.9f, 0.9f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(program);

    float aspect;
    aspect = (float)win_width/(float)win_height;

    float P[16], V[16];
    m4_perspective(P, 45.0f*(float)M_PI/180.0f, aspect, 0.1f, 200.0f);

    float eye[3];
    cam_eye(&g_cam, eye);
    m4_lookat(V, eye, g_cam.target, g_cam.worldUp);

    //light setting (kept in sync with g_light_pos used by the shadow camera)
    glUniform3f(locLPos, g_light_pos[0], g_light_pos[1], g_light_pos[2]);
    glUniform3f(locVPos, V[12], V[13], V[14]);
    glUniformMatrix4fv(locLightVP, 1, GL_FALSE, LightVP);
    glUniform1i(locShadowEnabled, g_shadow_enabled);
    glActiveTexture(GL_TEXTURE0 + SHADOW_TEX_UNIT);
    glBindTexture(GL_TEXTURE_2D, shadow_tex);
    glUniform1i(locShadowMap, SHADOW_TEX_UNIT);

    // Transparency: draw opaque objects first (depth writes on), then alpha<1
    // objects back-to-front with depth writes off so they blend over the scene.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    for(int i=0; i < n_obj; i++) {              // Pass 2a: opaque
	if(objcolor[4*i+0] < 0)    continue;    // invisible
	if(objcolor[4*i+3] < 1.0f) continue;    // defer translucent
	draw_lit_object(i, V, P, objcolor, objpose, locModel, locMVP, locNMat, locColor, mesh);
    }

    int torder[WIN_MAX_OBJ]; float tvz[WIN_MAX_OBJ];   // Pass 2b: translucent, far->near
    int nt = sort_transparent(n_obj, objcolor, objpose, V, torder, tvz);
    if(nt){
	glDepthMask(GL_FALSE);
	glEnable(GL_CULL_FACE);   // two-pass far/near cull per object (intra-mesh order)
	for(int k=0; k<nt; k++)
	    draw_translucent_object(torder[k], V, P, objcolor, objpose, locModel, locMVP, locNMat, locColor, mesh);
	glDisable(GL_CULL_FACE);
	glDepthMask(GL_TRUE);
    }

    //---- Stage R.1: handle 'R' toggle, capture frame to ffmpeg pipe ----
    if(rec_request && !rec_active){
	rec_w = win_width;
	rec_h = win_height;
	rec_buf = (unsigned char*)malloc((size_t)rec_w*rec_h*3);

	char filename[256];
	time_t t = time(NULL);
	struct tm tm;
	localtime_r(&t, &tm);
	strftime(filename, sizeof(filename), "/dev/shm/rec_%Y%m%d_%H%M%S.mp4", &tm);

	char cmd[1024];
	snprintf(cmd, sizeof(cmd),
		 "ffmpeg -loglevel error -y -f rawvideo -pixel_format rgb24 "
		 "-video_size %dx%d -framerate 50 -i - "
		 "-vf vflip -c:v libx264 -pix_fmt yuv420p %s",
		 rec_w, rec_h, filename);
	rec_pipe = popen(cmd, "w");
	if(!rec_pipe){
	    fprintf(stderr, "[REC] popen failed; recording aborted\n");
	    free(rec_buf); rec_buf = NULL;
	    rec_request = 0;
	} else {
	    rec_active = 1;
	    glfwSetWindowTitle(window, "test [REC]");
	    fprintf(stderr, "[REC] start: %s (%dx%d, 50fps)\n", filename, rec_w, rec_h);
	}
    }

    if(!rec_request && rec_active){
	pclose(rec_pipe);
	rec_pipe = NULL;
	free(rec_buf); rec_buf = NULL;
	rec_active = 0;
	glfwSetWindowTitle(window, "test");
	fprintf(stderr, "[REC] stop\n");
    }

    if(rec_active){
	if(win_width != rec_w || win_height != rec_h){
	    static int warned = 0;
	    if(!warned){
		fprintf(stderr, "[REC] window resized during recording; capturing locked %dx%d region from bottom-left\n", rec_w, rec_h);
		warned = 1;
	    }
	}
	glReadPixels(0, 0, rec_w, rec_h, GL_RGB, GL_UNSIGNED_BYTE, rec_buf);
	size_t n = (size_t)rec_w*rec_h*3;
	if(fwrite(rec_buf, 1, n, rec_pipe) != n){
	    fprintf(stderr, "[REC] write failed; will stop next frame\n");
	    rec_request = 0;
	}
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
    cnt++;

    if (quit_request) return -1;            // ESC → caller exits
    if (space_request) { space_request = 0; return 1; }   // SPACE → one-shot signal to caller
    return 0;
}

int egl_render(int n_obj, int* obj_type, float* shape, float* objcolor, float* objpose, float* campose, char* out_buf, int opt, int req_width, int req_height, float fovy_deg){
    static EGLDisplay egl_dpy;
    static EGLContext egl_ctx;
    static EGLSurface egl_surf = EGL_NO_SURFACE;
    static EGLConfig  egl_cfg;   // kept to recreate the pbuffer on a resolution grow

    static GLuint program;
    static GLint locMVP, locModel, locNMat, locColor, locLPos, locVPos;
    static GLint locLightVP, locShadowMap, locShadowEnabled;
    static GLuint shadow_program;
    static GLint locShadowLightMVP;
    static GLuint shadow_fbo, shadow_tex;
    // Single-sample FBO + renderbuffers used to resolve the 4x MSAA EGL
    // pbuffer before glReadPixels. The pbuffer is multisample (driver gives
    // GL_SAMPLES=4) but glReadPixels on a multisample default framebuffer is
    // not well-defined; on Mesa/NVIDIA Linux it reads sample 0 → aliased
    // output. The fix is an explicit glBlitFramebuffer resolve into this
    // single-sample target, then read from it.
    static GLuint resolve_fbo, resolve_color_rb, resolve_depth_rb;
    #define EGL_MAX_OBJ 64
    #define EGL_N_PADDING 8
    static Mesh mesh[EGL_MAX_OBJ];
    // Per-slot fingerprint so Env.add/Env.delete (which mutates n_obj and shifts
    // type/shape arrays) get reflected here too — same pattern as win_render.
    static int   prev_type[EGL_MAX_OBJ];
    static float prev_shape[EGL_MAX_OBJ * EGL_N_PADDING];
    static int   prev_n_obj = 0;
    static long cnt=0;
    
    static tjhandle tj_handle;
    static unsigned char *jpeg_buf = NULL;
    static unsigned long  jpeg_cap = 0;
    //static int sock;

    // Size-dependent CPU scratch, (re)allocated grow-only by the capacity block
    // below. temp_row holds one image row (width*4 bytes) for the vertical flip;
    // raw_buf the glReadPixels output (width*height*4 — fits RGB ubyte and float32
    // depth alike); zstd_buf the depth compression target, sized to
    // ZSTD_compressBound(width*height*4) since the compressed worst case can
    // exceed the raw size.
    static unsigned char* temp_row = NULL;
    static char* zstd_buf = NULL;
    static size_t zstd_cap = 0;
    static char* raw_buf  = NULL;
    static ZSTD_CCtx* cctx;

    if(!render_load_egl_deps()) return -2;

    // Output size requested per call (from the YAML camera `resolution`); <=0
    // falls back to the legacy 640x480 so callers that don't pass a size are
    // unchanged.
    if (req_width  <= 0) req_width  = 640;
    if (req_height <= 0) req_height = 480;
    int egl_width  = req_width;
    int egl_height = req_height;

    // Vertical field of view (degrees) from the YAML camera `vfov`; <=0 -> legacy 45.
    if (fovy_deg <= 0.0f) fovy_deg = 45.0f;

    // Allocated capacity for every size-dependent GL + CPU resource. Grow-only:
    // a request larger than the current capacity reallocates; smaller requests
    // reuse the existing buffers and just viewport/read the actual requested
    // size. So a mix of camera resolutions stops reallocating after the first
    // frame at the largest one (no per-frame thrash when cameras differ).
    static int cap_w = 0, cap_h = 0;
    int need_grow = (egl_width > cap_w || egl_height > cap_h);
    if (need_grow) {
        cap_w = egl_width  > cap_w ? egl_width  : cap_w;
        cap_h = egl_height > cap_h ? egl_height : cap_h;
    }

    
    if(cnt == 0) {
	// 1. EGL 초기화. choose_egl_display() tries iGPU/Mesa override,
	// surfaceless EGL, generic device platform, then the legacy default display.
	egl_dpy = choose_egl_display();
        if (egl_dpy == EGL_NO_DISPLAY) {
            fprintf(stderr, "egl_render: cannot initialize any EGL display\n");
            return -2;
        }

	//Set configuration
	EGLint config_attribs[] = {
	    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
	    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
	    EGL_BLUE_SIZE, 8,
	    EGL_GREEN_SIZE, 8,
	    EGL_RED_SIZE, 8,
	    EGL_DEPTH_SIZE, 24,
	    EGL_SAMPLE_BUFFERS, 1,
	    EGL_SAMPLES, 4,
	    EGL_NONE
	};
	EGLint num_configs;
	if(!eglChooseConfig(egl_dpy, config_attribs, &egl_cfg, 1, &num_configs) || num_configs <= 0){
            fprintf(stderr, "egl_render: eglChooseConfig failed\n");
            return -2;
        }

	//OpenGL API bind
	if(!eglBindAPI(EGL_OPENGL_API)){
            fprintf(stderr, "egl_render: eglBindAPI(EGL_OPENGL_API) failed\n");
            return -2;
        }

	//Generate context (the pbuffer surface + makeCurrent live in the
	// resolution-capacity block below, which also handles later grows)
	EGLint context_attribs[] = {
	    EGL_CONTEXT_MAJOR_VERSION, 3,
	    EGL_CONTEXT_MINOR_VERSION, 3,
	    EGL_NONE
	};
	egl_ctx = eglCreateContext(egl_dpy, egl_cfg, EGL_NO_CONTEXT, context_attribs);
        if (egl_ctx == EGL_NO_CONTEXT) {
            fprintf(stderr, "egl_render: eglCreateContext failed\n");
            return -2;
        }
    }

    // (Re)create the pbuffer surface whenever the requested resolution grows the
    // capacity. The pbuffer just needs to be at least as large as the viewport;
    // we make it cap_w x cap_h and viewport/read the actual egl_width x egl_height
    // (<= cap) below. Runs before glewInit on the first call (GL needs a current
    // surface) and again on any later grow (cnt>0, surface destroyed + remade).
    if (need_grow) {
	if (egl_surf != EGL_NO_SURFACE) eglDestroySurface(egl_dpy, egl_surf);
	EGLint pbuffer_attribs[] = {
	    EGL_WIDTH, cap_w,
	    EGL_HEIGHT, cap_h,
	    EGL_NONE
	};
	egl_surf = eglCreatePbufferSurface(egl_dpy, egl_cfg, pbuffer_attribs);
        if (egl_surf == EGL_NO_SURFACE) {
            fprintf(stderr, "egl_render: eglCreatePbufferSurface failed\n");
            return -2;
        }
	if(!eglMakeCurrent(egl_dpy, egl_surf, egl_surf, egl_ctx)){
            fprintf(stderr, "egl_render: eglMakeCurrent failed\n");
            return -2;
        }
	if(!render_load_gl_procs(0)) return -2;
    }

    if(cnt == 0) {
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_MULTISAMPLE);

	program = make_program();
	locMVP   = glGetUniformLocation(program, "uMVP");
	locModel = glGetUniformLocation(program, "uModel");
	locNMat  = glGetUniformLocation(program, "uNormalMat");
	locColor = glGetUniformLocation(program, "uColor");
	locLPos  = glGetUniformLocation(program, "uLightPos");
	locVPos  = glGetUniformLocation(program, "uViewPos");
	locLightVP       = glGetUniformLocation(program, "uLightVP");
	locShadowMap     = glGetUniformLocation(program, "uShadowMap");
	locShadowEnabled = glGetUniformLocation(program, "uShadowEnabled");

	// Shadow program + FBO. Created unconditionally so toggling
	// g_shadow_enabled at runtime works without a re-init.
	shadow_program = make_shadow_program();
	locShadowLightMVP = glGetUniformLocation(shadow_program, "uLightMVP");
	make_shadow_fbo(&shadow_fbo, &shadow_tex);

	// Resolve target for MSAA pbuffer → single-sample glReadPixels. Only the
	// names are reserved here; bind + storage + attachment (sized to capacity)
	// happen in the grow block below so they track resolution changes.
	glGenFramebuffers(1, &resolve_fbo);
	glGenRenderbuffers(1, &resolve_color_rb);
	glGenRenderbuffers(1, &resolve_depth_rb);

	for(int i=0; i < EGL_MAX_OBJ; i++) prev_type[i] = -1;

	//jpeg compressor handle (jpeg_buf sized to capacity in the grow block below)
	tj_handle = tjInitCompress();

	//zstd context
	cctx = ZSTD_createCCtx();
	
	//UDP streaming socket
	//sock = socket(AF_INET, SOCK_DGRAM, 0);

	//GLint depthBits = -1;
	//glGetIntegerv(GL_DEPTH_BITS, &depthBits);
	//GLenum err = glGetError();
	//printf("Depth bits: %d, err=0x%x\n", depthBits, err);
    }

    // Size the resolve renderbuffers + CPU scratch + JPEG buffer to the current
    // capacity. Runs on the first call (after the FBO/renderbuffers were generated
    // above) and on every later resolution grow. Grow-only, so steady state does
    // no allocation.
    if (need_grow) {
	glBindFramebuffer(GL_FRAMEBUFFER, resolve_fbo);
	glBindRenderbuffer(GL_RENDERBUFFER, resolve_color_rb);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB8, cap_w, cap_h);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, resolve_color_rb);
	glBindRenderbuffer(GL_RENDERBUFFER, resolve_depth_rb);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, cap_w, cap_h);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, resolve_depth_rb);
	{
	    GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	    if(st != GL_FRAMEBUFFER_COMPLETE) fprintf(stderr, "resolve FBO incomplete: 0x%x\n", st);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	free(raw_buf);  raw_buf  = malloc((size_t)cap_w * cap_h * 4);
	free(zstd_buf); zstd_cap = ZSTD_compressBound((size_t)cap_w * cap_h * 4);
	zstd_buf = malloc(zstd_cap);
	free(temp_row); temp_row = malloc((size_t)cap_w * 4);

	if (jpeg_buf) tjFree(jpeg_buf);
	jpeg_cap = tjBufSize(cap_w, cap_h, TJSAMP_420);
	jpeg_buf = tjAlloc(jpeg_cap);
    }

    // Sync mesh table to current (type, shape) per slot. Handles Env.add()
    // and Env.delete() by detecting per-slot fingerprint changes — same
    // approach as win_render. Without this, dynamically added shapes would
    // not have GPU resources and slots reindexed by delete() would draw the
    // wrong mesh.
    if (n_obj > EGL_MAX_OBJ) n_obj = EGL_MAX_OBJ;
    for(int i = 0; i < n_obj; i++){
	float *s = &shape[EGL_N_PADDING*i];
	int changed = (obj_type[i] != prev_type[i]);
	if(!changed){
	    for(int k = 0; k < EGL_N_PADDING; k++){
		if(s[k] != prev_shape[i*EGL_N_PADDING + k]){ changed = 1; break; }
	    }
	}
	if(changed){
	    if(mesh[i].vbo) glDeleteBuffers(1, &mesh[i].vbo);
	    if(mesh[i].vao) glDeleteVertexArrays(1, &mesh[i].vao);
	    mesh[i].vao = mesh[i].vbo = 0; mesh[i].vertex_count = 0;

	    if      (obj_type[i] == 101) mesh[i] = make_box(s[0], s[1], s[2]);
	    else if (obj_type[i] == 102) mesh[i] = make_sphere(s[0], 26, 40);
	    else if (obj_type[i] == 103) mesh[i] = make_cylinder(s[0], s[1], 44);
	    else if (obj_type[i] == 104) mesh[i] = make_capsule(s[0], s[1], 26, 44);
	    else if (obj_type[i] == 100){
		int idx = (int)s[0];
		if (mesh_path[idx][0] == '\0') {
		    fprintf(stderr, "egl_render: no path registered for mesh slot %d\n", idx);
		    exit(0);
		}
		mesh[i] = load_obj_as_mesh(mesh_path[idx]);
	    }
	    else if (obj_type[i] == 105) mesh[i] = make_hfield((int)s[0]);

	    prev_type[i] = obj_type[i];
	    for(int k = 0; k < EGL_N_PADDING; k++) prev_shape[i*EGL_N_PADDING + k] = s[k];
	}
    }
    for(int i = n_obj; i < prev_n_obj; i++){
	if(mesh[i].vbo) glDeleteBuffers(1, &mesh[i].vbo);
	if(mesh[i].vao) glDeleteVertexArrays(1, &mesh[i].vao);
	mesh[i].vao = mesh[i].vbo = 0; mesh[i].vertex_count = 0;
	prev_type[i] = -1;
    }
    prev_n_obj = n_obj;

    float LightVP[16], LMVP[16];
    float *T;
    compute_light_vp(LightVP);  // recomputed each frame so render_set_light() changes apply immediately

    //---- Pass 1: depth-only render from the light into the shadow FBO ----
    if(g_shadow_enabled){
	glUseProgram(shadow_program);
	glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo);
	glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
	glClear(GL_DEPTH_BUFFER_BIT);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1.5f, 4.0f);
	for(int i=0; i < n_obj; i++){
	    T = objpose+16*i;
	    m4_mul(LMVP, LightVP, T);
	    glUniformMatrix4fv(locShadowLightMVP, 1, GL_FALSE, LMVP);
	    glBindVertexArray(mesh[i].vao);
	    glDrawArrays(GL_TRIANGLES, 0, mesh[i].vertex_count);
	}
	glDisable(GL_POLYGON_OFFSET_FILL);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    //---- Pass 2: main lit render into the EGL pbuffer ----
    //in case of egl render, use fixed viewport size
    glViewport(0, 0, egl_width, egl_height);

    //glClearColor(0.06f, 0.07f, 0.09f, 1.0f);
    glClearColor(0.9f, 0.9f, 0.9f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(program);

    float aspect;
    aspect = (float)egl_width/(float)egl_height;

    float P[16], V[16];
    const float znear = 0.1f, zfar = 200.0f;   // also used by the opt==2 depth linearization
    m4_perspective(P, fovy_deg*(float)M_PI/180.0f, aspect, znear, zfar);
    memcpy(V, campose, sizeof(V));

    //light setting (kept in sync with g_light_pos used by the shadow camera)
    glUniform3f(locLPos, g_light_pos[0], g_light_pos[1], g_light_pos[2]);
    glUniform3f(locVPos, V[12], V[13], V[14]);
    glUniformMatrix4fv(locLightVP, 1, GL_FALSE, LightVP);
    glUniform1i(locShadowEnabled, g_shadow_enabled);
    glActiveTexture(GL_TEXTURE0 + SHADOW_TEX_UNIT);
    glBindTexture(GL_TEXTURE_2D, shadow_tex);
    glUniform1i(locShadowMap, SHADOW_TEX_UNIT);

    // Transparency: opaque first, then alpha<1 back-to-front with depth writes
    // off. (Matches win_render; egl keeps drawing rgb[0]<0 shapes as before —
    // only translucent ones are deferred to the second pass.)
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    for(int i=0; i < n_obj; i++) {              // Pass 2a: opaque
	if(objcolor[4*i+3] < 1.0f) continue;    // defer translucent
	draw_lit_object(i, V, P, objcolor, objpose, locModel, locMVP, locNMat, locColor, mesh);
    }

    int torder[EGL_MAX_OBJ]; float tvz[EGL_MAX_OBJ];   // Pass 2b: translucent, far->near
    int nt = sort_transparent(n_obj, objcolor, objpose, V, torder, tvz);
    if(nt){
	glDepthMask(GL_FALSE);
	glEnable(GL_CULL_FACE);   // two-pass far/near cull per object (intra-mesh order)
	for(int k=0; k<nt; k++)
	    draw_translucent_object(torder[k], V, P, objcolor, objpose, locModel, locMVP, locNMat, locColor, mesh);
	glDisable(GL_CULL_FACE);
	glDepthMask(GL_TRUE);
    }

    int imglen = 0;

    // Resolve the multisample EGL pbuffer into the single-sample resolve_fbo
    // before reading. Color uses LINEAR (bilinear MSAA resolve); depth must
    // use NEAREST (linear filter on depth is undefined). Read pixels then
    // sources from resolve_fbo.
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, resolve_fbo);
    if (opt == 1) {
	glBlitFramebuffer(0, 0, egl_width, egl_height,
	                  0, 0, egl_width, egl_height,
	                  GL_COLOR_BUFFER_BIT, GL_LINEAR);
    } else if (opt == 2) {
	glBlitFramebuffer(0, 0, egl_width, egl_height,
	                  0, 0, egl_width, egl_height,
	                  GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    }
    glBindFramebuffer(GL_READ_FRAMEBUFFER, resolve_fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    //rgb 8bit 3channel
    if (opt == 1){
	glReadPixels(0, 0, egl_width, egl_height, GL_RGB, GL_UNSIGNED_BYTE, raw_buf);
	flip_vertical(raw_buf, egl_width, egl_height, 3, temp_row);

	unsigned long jpeg_len = 0;
	tjCompress2(tj_handle, (unsigned char*)raw_buf, egl_width, 0, egl_height, TJPF_RGB, &jpeg_buf, &jpeg_len, TJSAMP_420, 75, TJFLAG_NOREALLOC); //80, TJFLAG_FASTDCT);

	imglen = (int)jpeg_len;
	memcpy(out_buf, jpeg_buf, imglen);
    }

    //depth: linear eye-space distance (meters), float32, 1 channel, zstd-compressed.
    else if (opt == 2) {
	// Read the window-space depth at full precision (GL_FLOAT over the DEPTH24
	// resolve target) and convert each pixel to linear eye-space distance. Raw
	// window depth is non-linear (precision crammed near the near plane) and not
	// metric, so it's near-useless for a depth sensor; linearizing here makes the
	// payload directly consumable in meters. Pixels with no geometry read 1.0 ->
	// zfar. znear/zfar match the projection matrix above (coupled via the consts).
	float* depth = (float*)raw_buf;
	glReadPixels(0, 0, egl_width, egl_height, GL_DEPTH_COMPONENT, GL_FLOAT, depth);

	int npx = egl_width * egl_height;
	for (int i = 0; i < npx; i++) {
	    float zndc = depth[i] * 2.0f - 1.0f;                  // [0,1] -> NDC [-1,1]
	    depth[i] = (2.0f * znear * zfar) / (zfar + znear - zndc * (zfar - znear));
	}
	flip_vertical(raw_buf, egl_width, egl_height, sizeof(float), temp_row);

	size_t raw_size = (size_t)egl_width * egl_height * sizeof(float);
	size_t zstd_len = ZSTD_compressCCtx(cctx, zstd_buf, zstd_cap, raw_buf, raw_size, 3);

	if (ZSTD_isError(zstd_len)){
	    fprintf(stderr, "egl_render depth: zstd error: %s\n", ZSTD_getErrorName(zstd_len));
	    return 0;
	}

	imglen = (int)zstd_len;
	memcpy(out_buf, zstd_buf, imglen);
    }

    cnt++;
    return imglen;
}

/*

//char filename[64];
//sprintf(filename, "rgb-%ld", cnt);
//save_ppm(filename, egl_width, egl_height, rgb);
//save_pgm("depth.pgm", egl_width, egl_height, depth);


// PPM (RAW-RGB) 저장 함수
static void save_ppm(const char* filename, int w, int h, unsigned char* data) {
    FILE* f = fopen(filename, "wb");
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    // OpenGL은 하단부터 저장되므로 위아래를 뒤집어서 기록
    for (int y = h - 1; y >= 0; y--) {
        fwrite(&data[y * w * 3], 1, w * 3, f);
    }
    fclose(f);
}

// PGM (RAW-Depth) 저장 함수
static void save_pgm(const char* filename, int w, int h, float* data) {
    FILE* f = fopen(filename, "wb");
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    unsigned char* grey = (unsigned char*)malloc(w * h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            // Depth 값을 0~255 범위로 변환 (뒤집어서 저장)
            float z = data[(h - 1 - y) * w + x];
            grey[y * w + x] = (unsigned char)(z * 255.0f);
        }
    }
    fwrite(grey, 1, w * h, f);
    free(grey);
    fclose(f);
}
*/

/*void rgbd_finish(){
    eglDestroySurface(egl_dpy, egl_surf);
    eglDestroyContext(egl_dpy, egl_ctx);
    eglTerminate(egl_dpy);

    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
    }
*/

/*
static void print_gl_info(void){
    const GLubyte* vendor   = glGetString(GL_VENDOR);
    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* version  = glGetString(GL_VERSION);
    const GLubyte* sl       = glGetString(GL_SHADING_LANGUAGE_VERSION);
    fprintf(stdout, "GL_VENDOR    : %s\n", vendor);
    fprintf(stdout, "GL_RENDERER  : %s\n", renderer);
    fprintf(stdout, "GL_VERSION   : %s\n", version);
    fprintf(stdout, "GLSL_VERSION : %s\n", sl);
}
*/


/*typedef struct {
    uint32_t frame_id;
    uint16_t packet_id;
    uint16_t packet_count;
    uint32_t payload_size;
    uint32_t reserved;
} PacketHeader;

static void send_frame(int sock, char* data, int data_len, uint32_t frame_id, int type, int port){
    struct sockaddr_in srv_addr;
    srv_addr.sin_family = AF_INET;
    srv_addr.sin_port = htons(port);
    inet_pton(AF_INET, (char*)"127.0.0.1", &srv_addr.sin_addr);
    
    int addr_len = sizeof(srv_addr);
    int packet_count = (data_len + PAYLOAD_SIZE-1)/PAYLOAD_SIZE;

    for(int i = 0; i < packet_count; i++){
        uint8_t buffer[MAX_PACKET];
        PacketHeader header;

        header.frame_id = frame_id;
        header.packet_id = i;
        header.packet_count = packet_count;
        header.payload_size = data_len;
        header.reserved = type;
        memcpy(buffer, &header, HEADER_SIZE);

        int offset = i*PAYLOAD_SIZE;

	int size;
	if (data_len - offset >= PAYLOAD_SIZE) size = PAYLOAD_SIZE;
	else size = data_len - offset;
	
        memcpy(buffer + HEADER_SIZE, data + offset, size);
        sendto(sock, buffer, size + HEADER_SIZE, 0, (struct sockaddr*)&srv_addr, addr_len);
    }
}
*/
