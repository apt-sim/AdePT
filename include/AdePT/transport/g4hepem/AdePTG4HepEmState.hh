// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_G4_HEPEM_STATE_HH
#define ADEPT_G4_HEPEM_STATE_HH

#include <memory>

struct G4HepEmConfig;
struct G4HepEmData;
struct G4HepEmParameters;

namespace AsyncAdePT {

/// @brief Owns the prepared host-side G4HepEm inputs used by transport.
/// @details
/// The Geant4 integration side prepares one of these objects before the shared
/// transport is created. This wrapper owns both:
/// - the rebuilt `G4HepEmData`
/// - a deep copy of the `G4HepEmParameters` taken from the provided config
///
/// Cleanup is intentionally split:
/// - `DataDeleter` performs the deep cleanup of the owned `G4HepEmData`
///   and then deletes the outer `G4HepEmData` allocation.
/// - `ParametersDeleter` performs the deep cleanup of the owned
///   `G4HepEmParameters`, including the GPU mirror created by AdePT, and then
///   deletes the outer `G4HepEmParameters` allocation.
class AdePTG4HepEmState {
public:
  /// @brief Build the AdePT-owned `G4HepEmData` and `G4HepEmParameters` copies from the supplied config.
  explicit AdePTG4HepEmState(G4HepEmConfig *hepEmConfig);

  /// @brief Destroy the owned `G4HepEmData` and `G4HepEmParameters` copies.
  ~AdePTG4HepEmState();

  AdePTG4HepEmState(const AdePTG4HepEmState &)            = delete;
  AdePTG4HepEmState &operator=(const AdePTG4HepEmState &) = delete;

  AdePTG4HepEmState(AdePTG4HepEmState &&) noexcept            = default;
  AdePTG4HepEmState &operator=(AdePTG4HepEmState &&) noexcept = default;

  /// @brief Access the owned host-side HepEm data tables.
  G4HepEmData *GetData() const { return fData.get(); }

  /// @brief Access the owned HepEm parameter copy.
  G4HepEmParameters *GetParameters() const { return fParameters.get(); }

private:
  /// @brief Deletes the outer `G4HepEmData` object after first freeing all tables it owns.
  /// @details
  /// `FreeG4HepEmData` releases both the host-side tables and any device-side
  /// mirrors embedded in the `G4HepEmData` object, but it does not delete the outer
  /// `G4HepEmData` allocation itself. This deleter performs both steps for the
  /// owned `fData` member. It does not touch the separately owned
  /// `G4HepEmParameters` copy stored in `fParameters`.
  struct DataDeleter {
    void operator()(G4HepEmData *data) const;
  };

  /// @brief Deletes the outer `G4HepEmParameters` object after first freeing
  /// all host/device allocations it owns.
  /// @details
  /// The copied parameter block owns its `fParametersPerRegion` host array and
  /// the GPU mirror pointed to by `fParametersPerRegion_gpu` after transport
  /// upload. `FreeG4HepEmParameters` releases those nested allocations, while
  /// this deleter also deletes the outer `G4HepEmParameters` allocation.
  struct ParametersDeleter {
    void operator()(G4HepEmParameters *parameters) const;
  };

  /// Owned `G4HepEmData` rebuilt for AdePT.
  std::unique_ptr<G4HepEmData, DataDeleter> fData;

  /// Owned deep copy of `G4HepEmParameters` used to build and upload transport data.
  std::unique_ptr<G4HepEmParameters, ParametersDeleter> fParameters;
};

} // namespace AsyncAdePT

#endif
