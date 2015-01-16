cdef extern from "math.h":
    float cosf(float theta)
    float sinf(float theta)
    float atan2f(float y, float x)

def atan2c_longitude(float azr, float gr, float klatr, float glatr):
    cdef float re = 6371.1
    return atan2f(sinf(azr)*sinf(gr/re)*cosf(klatr),
                 cosf(gr/re)-sinf(klatr)*sinf(glatr))

def calc_turb_cython(float csnr, float cpr, float cswv, float czh, float crng,   
                     float eps, float num, float tot):
    cdef float csw
    csw = (csnr**0.6667) * cpr * cswv * czh * crng
    return num + csw * eps**2, tot + csw

def calc_cswv_cython(float dummy):
    cdef float cswv
    cswv = 0.0
    if dummy >= 0 and dummy < 4:
        cswv = 1.0
    elif dummy >= 4 and dummy < 16:
        cswv = 1.0 - (1.0 / 12.0) * (dummy - 4.0)
    return cswv

def atan2c(float y, float x):
    return atan2f(y, x)


