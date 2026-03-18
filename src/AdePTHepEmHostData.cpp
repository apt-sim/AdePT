// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/core/AdePTHepEmHostData.hh>

#include <G4HepEmConfig.hh>
#include <G4HepEmData.hh>
#include <G4HepEmElectronInit.hh>
#include <G4HepEmGammaInit.hh>
#include <G4HepEmMaterialInit.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmParameters.hh>
#include <G4ios.hh>

namespace AsyncAdePT {

/// @brief Release the tables owned by `G4HepEmData` and then delete the outer object.
/// @details
/// This is the cleanup path for the fully owned `fData` member. It is separate
/// from the class destructor because `std::unique_ptr` needs a deleter for the
/// deep cleanup before the outer `G4HepEmData` allocation itself can be deleted.
void HepEmHostData::DataDeleter::operator()(G4HepEmData *data) const
{
  if (data == nullptr) return;
  FreeG4HepEmData(data);
  delete data;
}

/// @brief Rebuild a complete host-side G4HepEm view from the supplied G4HepEm config.
HepEmHostData::HepEmHostData(G4HepEmConfig *hepEmConfig)
    : fData(new G4HepEmData), fParameters(hepEmConfig->GetG4HepEmParameters())
{
  // Rebuild the HepEm tables from the G4HepEm-owned config so the host-side
  // transport preparation has a complete, self-contained view of the data.
  InitG4HepEmData(fData.get());
  InitMaterialAndCoupleData(fData.get(), fParameters);

  // Build all EM species
  InitElectronData(fData.get(), fParameters, true);
  InitElectronData(fData.get(), fParameters, false);
  InitGammaData(fData.get(), fParameters);

  G4HepEmMatCutData *cutData = fData->fTheMatCutData;
  G4cout << "fNumG4MatCuts = " << cutData->fNumG4MatCuts << ", fNumMatCutData = " << cutData->fNumMatCutData << G4endl;
}

/// @brief Release the GPU mirror of the borrowed HepEm parameters.
/// @details
/// The host-side `G4HepEmParameters` remain owned by the `G4HepEmConfig`.
/// AdePT only owns the device allocation created by `CopyG4HepEmParametersToGPU`,
/// so this destructor releases just that GPU-side mirror. The owned `G4HepEmData`
/// cleanup is handled independently by `DataDeleter`.
HepEmHostData::~HepEmHostData()
{
  FreeG4HepEmParametersOnGPU(fParameters);
}

} // namespace AsyncAdePT
