// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/transport/g4hepem/AdePTG4HepEmState.hh>

#include <G4HepEmConfig.hh>
#include <G4HepEmData.hh>
#include <G4HepEmElectronInit.hh>
#include <G4HepEmGammaInit.hh>
#include <G4HepEmMaterialInit.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmParameters.hh>
#include <G4ios.hh>

#include <algorithm>
#include <stdexcept>

namespace AsyncAdePT {

/// @brief Release the tables owned by `G4HepEmData` and then delete the outer object.
/// @details
/// This is the cleanup path for the fully owned `fData` member. It is separate
/// from the class destructor because `std::unique_ptr` needs a deleter for the
/// deep cleanup before the outer `G4HepEmData` allocation itself can be deleted.
void AdePTG4HepEmState::DataDeleter::operator()(G4HepEmData *data) const
{
  if (data == nullptr) return;
  FreeG4HepEmData(data);
  delete data;
}

/// @brief Release the copied HepEm parameters and then delete the outer object.
/// @details
/// The copied `G4HepEmParameters` block is fully owned by
/// `AdePTG4HepEmState`. This deep
/// cleanup therefore releases both the host-side per-region array and the
/// device-side mirror created during transport upload before deleting the outer
/// `G4HepEmParameters` allocation itself.
void AdePTG4HepEmState::ParametersDeleter::operator()(G4HepEmParameters *parameters) const
{
  if (parameters == nullptr) return;
  FreeG4HepEmParameters(parameters);
  delete parameters;
}

/// @brief Rebuild a complete AdePT-owned set of host-side G4HepEm inputs from the supplied config.
/// @details
/// `AdePTG4HepEmState` owns two different G4HepEm objects:
/// - a deep copy of the `G4HepEmParameters` stored in the supplied `G4HepEmConfig`
/// - a freshly rebuilt `G4HepEmData` derived from that copied parameter block
///
/// We must copy `G4HepEmParameters` because the original object remains owned by
/// the worker-local `G4HepEmConfig`, while the shared AdePT transport can outlive
/// the worker that first created it. `G4HepEmData` is rebuilt here directly, so
/// it is already fully owned by AdePT and does not need a second copy step.
AdePTG4HepEmState::AdePTG4HepEmState(G4HepEmConfig *hepEmConfig)
    : fData(new G4HepEmData), fParameters(new G4HepEmParameters)
{
  if (hepEmConfig == nullptr) {
    throw std::runtime_error("AdePTG4HepEmState requires a non-null G4HepEmConfig.");
  }

  G4HepEmParameters *sourceParameters = hepEmConfig->GetG4HepEmParameters();
  if (sourceParameters == nullptr) {
    throw std::runtime_error("AdePTG4HepEmState requires initialized G4HepEmParameters in the supplied config.");
  }

  // Deep-copy the G4HepEmParameters so the shared transport does not keep a
  // pointer into a worker-owned G4HepEmConfig.
  *fParameters                      = *sourceParameters;
  fParameters->fParametersPerRegion = nullptr;
#ifdef G4HepEm_CUDA_BUILD
  fParameters->fParametersPerRegion_gpu = nullptr;
#endif
  if (sourceParameters->fNumRegions > 0) {
    if (sourceParameters->fParametersPerRegion == nullptr) {
      throw std::runtime_error("AdePTG4HepEmState requires initialized per-region G4HepEmParameters.");
    }
    fParameters->fParametersPerRegion = new G4HepEmRegionParmeters[sourceParameters->fNumRegions];
    std::copy_n(sourceParameters->fParametersPerRegion, sourceParameters->fNumRegions,
                fParameters->fParametersPerRegion);
  }

  // Rebuild the G4HepEmData tables from the copied G4HepEmParameters so the
  // transport owns a complete, self-contained set of host-side inputs.
  InitG4HepEmData(fData.get());
  InitMaterialAndCoupleData(fData.get(), fParameters.get());

  // Build all EM species
  InitElectronData(fData.get(), fParameters.get(), true);
  InitElectronData(fData.get(), fParameters.get(), false);
  InitGammaData(fData.get(), fParameters.get());

  G4HepEmMatCutData *cutData = fData->fTheMatCutData;
  G4cout << "fNumG4MatCuts = " << cutData->fNumG4MatCuts << ", fNumMatCutData = " << cutData->fNumMatCutData << G4endl;
}

AdePTG4HepEmState::~AdePTG4HepEmState() = default;

} // namespace AsyncAdePT
