/// \file Color.h
/// \author Andrei Gheata

#ifndef VECGEOM_BENCHMARKING_COLOR_H_
#define VECGEOM_BENCHMARKING_COLOR_H_

#include "base/inc/CopCore/include/CopCore/Global.h"
#include <VecGeom/base/Global.h>

namespace vecgeom {
union Color_t {
  unsigned int fColor; // color representation as unsigned integer
  struct {
    unsigned char alpha;
    unsigned char blue;
    unsigned char green;
    unsigned char red;
  } fComp;

  VECCORE_ATT_HOST_DEVICE
  Color_t() : fColor(0) {}
  VECCORE_ATT_HOST_DEVICE
  Color_t(unsigned int col) { fColor = col; }
  VECCORE_ATT_HOST_DEVICE
  Color_t(unsigned char r, unsigned char g, unsigned char b, unsigned char a)
  {
    fComp.red   = r;
    fComp.green = g;
    fComp.blue  = b;
    fComp.alpha = a;
  }
  VECCORE_ATT_HOST_DEVICE
  Color_t(float r, float g, float b, float a)
  {
    fComp.red   = 255 * r;
    fComp.green = 255 * g;
    fComp.blue  = b;
    fComp.alpha = 255 * a;
  }

  VECCORE_ATT_HOST_DEVICE
  Color_t &operator+=(Color_t const &other)
  {
    if ((fComp.alpha == 0) && (other.fComp.alpha == 0)) return *this;
    float alpha = 1 - (1 - other.Alpha()) * (1 - Alpha()); // cannot be 0
    float red   = (other.Red() * other.Alpha() + Red() * Alpha() * (1 - other.Alpha())) / alpha;
    float green = (other.Green() * other.Alpha() + Green() * Alpha() * (1 - other.Alpha())) / alpha;
    float blue  = (other.Blue() * other.Alpha() + Blue() * Alpha() * (1 - other.Alpha())) / alpha;
    fComp.red   = 255 * red;
    fComp.green = 255 * green;
    fComp.blue  = 255 * blue;
    fComp.alpha = 255 * alpha;
    return *this;
  }

  VECCORE_ATT_HOST_DEVICE
  Color_t &operator*=(float val)
  {
    using vecCore::math::Max;
    using vecCore::math::Min;
    float alpha = val * Alpha();
    alpha       = Max(alpha, 0.0f);
    alpha       = Min(alpha, 1.0f);
    fComp.alpha = 255 * alpha;
    return *this;
  }

  VECCORE_ATT_HOST_DEVICE
  Color_t &operator/=(float val)
  {
    using vecCore::math::Max;
    using vecCore::math::Min;
    float alpha = Alpha() / val;
    alpha       = Max(alpha, 0.0f);
    alpha       = Min(alpha, 1.0f);
    fComp.alpha = 255 * alpha;
    return *this;
  }

  VECCORE_ATT_HOST_DEVICE
  float Red() const { return 1. / 255 * fComp.red; }
  VECCORE_ATT_HOST_DEVICE
  float Green() const { return 1. / 255 * fComp.green; }
  VECCORE_ATT_HOST_DEVICE
  float Blue() const { return 1. / 255 * fComp.blue; }
  VECCORE_ATT_HOST_DEVICE
  float Alpha() const { return 1. / 255 * fComp.alpha; }
  int GetColor() const { return fColor; }

  VECCORE_ATT_HOST_DEVICE
  void MultiplyLightChannel(float fact)
  {
    float hue, light, satur;
    GetHLS(hue, light, satur);
    SetHLS(hue, fact * light, satur);
  }

  VECCORE_ATT_HOST_DEVICE
  void GetHLS(float &hue, float &light, float &satur) const
  {
    float rnorm, gnorm, bnorm, msum, mdiff;

    float minval = Min(Red(), Green(), Blue());
    float maxval = Max(Red(), Green(), Blue());

    rnorm = gnorm = bnorm = 0;
    mdiff                 = maxval - minval;
    msum                  = maxval + minval;
    light                 = 0.5 * msum;
    if (maxval != minval) {
      rnorm = (maxval - Red()) / mdiff;
      gnorm = (maxval - Green()) / mdiff;
      bnorm = (maxval - Blue()) / mdiff;
    } else {
      satur = hue = 0;
      return;
    }

    if (light < 0.5)
      satur = mdiff / msum;
    else
      satur = mdiff / (2.0 - msum);

    if (Red() == maxval)
      hue = 60.0 * (6.0 + bnorm - gnorm);
    else if (Green() == maxval)
      hue = 60.0 * (2.0 + rnorm - bnorm);
    else
      hue = 60.0 * (4.0 + gnorm - rnorm);

    if (hue > 360) hue = hue - 360;
  }

  VECCORE_ATT_HOST_DEVICE
  void SetHLS(float hue, float light, float satur)
  {
    float rh, rl, rs, rm1, rm2;
    rh = rl = rs = 0;
    if (hue > 0) {
      rh = hue;
      if (rh > 360) rh = 360;
    }
    if (light > 0) {
      rl = light;
      if (rl > 1) rl = 1;
    }
    if (satur > 0) {
      rs = satur;
      if (rs > 1) rs = 1;
    }

    if (rl <= 0.5)
      rm2 = rl * (1.0 + rs);
    else
      rm2 = rl + rs - rl * rs;
    rm1 = 2.0 * rl - rm2;

    if (!rs) {
      fComp.red   = 255 * rl;
      fComp.green = 255 * rl;
      fComp.blue  = 255 * rl;
      return;
    }

    auto HLStoRGB1 = [](float rn1, float rn2, float huei) {
      float hue = huei;
      if (hue > 360) hue = hue - 360;
      if (hue < 0) hue = hue + 360;
      if (hue < 60) return rn1 + (rn2 - rn1) * hue / 60;
      if (hue < 180) return rn2;
      if (hue < 240) return rn1 + (rn2 - rn1) * (240 - hue) / 60;
      return rn1;
    };

    fComp.red   = 255 * HLStoRGB1(rm1, rm2, rh + 120);
    fComp.green = 255 * HLStoRGB1(rm1, rm2, rh);
    fComp.blue  = 255 * HLStoRGB1(rm1, rm2, rh - 120);
  }
};

VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
Color_t operator+(Color_t const &left, Color_t const &right)
{
  Color_t color(left);
  color += right;
  return color;
}
} // End namespace vecgeom

#endif // VECGEOM_BENCHMARKING_COLOR_H_
