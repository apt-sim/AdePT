#include "Raytracer.h"

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/navigation/NavStatePath.h>
#include <VecGeom/base/Stopwatch.h>

void getMaterialStruct(MyMediumProp *volume_container, std::vector<vecgeom::LogicalVolume *> logicalvolumes,
                       bool on_gpu)
{

  int i = 0;

  for (auto lvol : logicalvolumes) {
    // lvol->Print();
    if (!strcmp(lvol->GetName(), "World")) {
      volume_container[i].material            = kRTtransparent;
      volume_container[i].fObjColor           = 0x0000FF80;
      volume_container[i].refr_index          = 1.;
      volume_container[i].transparency_per_cm = 1.;
    }

    else if (!strncmp(lvol->GetName(), "SphVol", 6)) {
      volume_container[i].material            = kRTtransparent;
      volume_container[i].fObjColor           = 0x0000FF80;
      volume_container[i].refr_index          = 1.1;
      volume_container[i].transparency_per_cm = 0.82;
    }

    else if (!strncmp(lvol->GetName(), "BoxVol", 6)) {
      volume_container[i].material  = kRTspecular;
      volume_container[i].fObjColor = 0xFF000080;
    }

    else if (!strcmp(lvol->GetName(), "M12_10")    || !strcmp(lvol->GetName(), "M12_100x2") ||
             !strcmp(lvol->GetName(), "M12_100x4") || !strcmp(lvol->GetName(), "M12_12")    ||
             !strcmp(lvol->GetName(), "M12_120x2") || !strcmp(lvol->GetName(), "M12_120x4") ||
             !strcmp(lvol->GetName(), "M12_2")     || !strcmp(lvol->GetName(), "M12_20x2")  ||
             !strcmp(lvol->GetName(), "M12_20x4")  || !strcmp(lvol->GetName(), "M12_4")     ||
             !strcmp(lvol->GetName(), "M12_40x2")  || !strcmp(lvol->GetName(), "M12_40x4")  ||
             !strcmp(lvol->GetName(), "M12_6")     || !strcmp(lvol->GetName(), "M12_60x2")  ||
             !strcmp(lvol->GetName(), "M12_60x4")  || !strcmp(lvol->GetName(), "M12_8")     ||
             !strcmp(lvol->GetName(), "M12_80x2")  || !strcmp(lvol->GetName(), "M12_80x4")  ||
             !strcmp(lvol->GetName(), "M14_10")    || !strcmp(lvol->GetName(), "M14_100x2") ||
             !strcmp(lvol->GetName(), "M14_100x4") || !strcmp(lvol->GetName(), "M14_12")    ||
             !strcmp(lvol->GetName(), "M14_120x2") || !strcmp(lvol->GetName(), "M14_120x4") ||
             !strcmp(lvol->GetName(), "M13_2")     || !strcmp(lvol->GetName(), "M13_4")     ||
             !strcmp(lvol->GetName(), "M13_6")     || !strcmp(lvol->GetName(), "M13_8")     ||
             !strcmp(lvol->GetName(), "M16_6")     || !strcmp(lvol->GetName(), "M16_60x2")  ||
             !strcmp(lvol->GetName(), "M14_2")     || !strcmp(lvol->GetName(), "M14_20x2")  ||
             !strcmp(lvol->GetName(), "M14_20x4")  || !strcmp(lvol->GetName(), "M14_4")     ||
             !strcmp(lvol->GetName(), "M14_40x2")  || !strcmp(lvol->GetName(), "M14_40x4")  ||
             !strcmp(lvol->GetName(), "M14_6")     || !strcmp(lvol->GetName(), "M14_60x2")  ||
             !strcmp(lvol->GetName(), "M14_60x4")  || !strcmp(lvol->GetName(), "M14_8")     ||
             !strcmp(lvol->GetName(), "M14_80x2")  || !strcmp(lvol->GetName(), "M14_80x4")  ||
             !strcmp(lvol->GetName(), "M16_12")    || !strcmp(lvol->GetName(), "M16_120x2") ||
             !strcmp(lvol->GetName(), "M16_10")    || !strcmp(lvol->GetName(), "M16_100x2") ||
             !strcmp(lvol->GetName(), "M16_2")     || !strcmp(lvol->GetName(), "M16_20x2")  ||
             !strcmp(lvol->GetName(), "M16_4")     || !strcmp(lvol->GetName(), "M16_40x2")  ||
             !strcmp(lvol->GetName(), "M16_8")     || !strcmp(lvol->GetName(), "M16_80x2")  ||
             !strcmp(lvol->GetName(), "M17_2")     || !strcmp(lvol->GetName(), "M17_4")     ||
             !strcmp(lvol->GetName(), "M18_12")    || !strcmp(lvol->GetName(), "M18_120x2") ||
             !strcmp(lvol->GetName(), "M18_10")    || !strcmp(lvol->GetName(), "M18_100x2") ||
             !strcmp(lvol->GetName(), "M18_2")     || !strcmp(lvol->GetName(), "M18_20x2")  ||
             !strcmp(lvol->GetName(), "M18_4")     || !strcmp(lvol->GetName(), "M18_40x2")  ||
             !strcmp(lvol->GetName(), "M18_6")     || !strcmp(lvol->GetName(), "M18_60x2")  ||
             !strcmp(lvol->GetName(), "M18_8")     || !strcmp(lvol->GetName(), "M18_80x2")  ||
             !strcmp(lvol->GetName(), "M7_10")     || !strcmp(lvol->GetName(), "M7_12")     ||
             !strcmp(lvol->GetName(), "M7_14")     || !strcmp(lvol->GetName(), "M7_2")      ||
             !strcmp(lvol->GetName(), "M7_4")      || !strcmp(lvol->GetName(), "M7_6")      || 
             !strcmp(lvol->GetName(), "M8_2")      || !strcmp(lvol->GetName(), "M8_4")      || 
             !strcmp(lvol->GetName(), "M8_6")      || !strcmp(lvol->GetName(), "M7_8")      ||
             !strcmp(lvol->GetName(), "M8_8")      || !strcmp(lvol->GetName(), "M9_10")     ||
             !strcmp(lvol->GetName(), "M9_12")     || !strcmp(lvol->GetName(), "M9_14")     ||
             !strcmp(lvol->GetName(), "M9_2")      || !strcmp(lvol->GetName(), "M9_4")      || 
             !strcmp(lvol->GetName(), "M9_6")      || !strcmp(lvol->GetName(), "M9_8")) {
      volume_container[i].material            = kRTtransparent;
      volume_container[i].fObjColor           = 0x0000FF80;
      volume_container[i].refr_index          = 1.5;
      volume_container[i].transparency_per_cm = 0.7;
    } else {
      volume_container[i].material            = kRTtransparent;
      volume_container[i].fObjColor           = 0x0000FF80;
      volume_container[i].refr_index          = 1.;
      volume_container[i].transparency_per_cm = 1.;
    }

    if (!on_gpu) lvol->SetBasketManagerPtr(&volume_container[i]);

    i++;
  }
}