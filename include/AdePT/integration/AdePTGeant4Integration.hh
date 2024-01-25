// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

///   The AdePT-Geant4 integration layer
///   - Initialization and checking of VecGeom geometry
///   - G4Hit reconstruction from GPU hits
///   - Processing of reconstructed hits using the G4 Sensitive Detector

#ifndef ADEPTGEANT4_INTEGRATION_H
#define ADEPTGEANT4_INTEGRATION_H

#include <unordered_map>

#include <G4HepEmState.hh>

#include <AdePT/integration/CommonStruct.h>
#include <AdePT/integration/HostScoring.h>

#include <G4VPhysicalVolume.hh>
#include <G4LogicalVolume.hh>
#include <G4VPhysicalVolume.hh>
#include <G4NavigationHistory.hh>
#include <G4Step.hh>

#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/volumes/LogicalVolume.h>


class AdePTGeant4Integration {
    public: 
        AdePTGeant4Integration() = default;
        ~AdePTGeant4Integration() = default;

        /// @brief Initializes VecGeom geometry
        /// @details Currently VecGeom geometry is initialized by loading it from a GDML file, 
        /// however ideally this function will call the G4 to VecGeom geometry converter
        static void CreateVecGeomWorld();

        /// @brief This function compares G4 and VecGeom geometries and reports any differences
        static void CheckGeometry(G4HepEmState *hepEmState);

        /// @brief Fills the auxiliary data needed for AdePT
        void InitVolAuxData(adeptint::VolAuxData *volAuxData, G4HepEmState *hepEmState);

        /// @brief Initializes the mapping of VecGeom to G4 volumes for sensitive volumes and their parents
        void InitScoringData(adeptint::VolAuxData *volAuxData);

        void ProcessGPUHits(HostScoring &aScoring, HostScoring::Stats &aStats);

        /// @brief Reconstruct G4TouchableHistory from a VecGeom Navigation index
        void FillG4NavigationHistory(unsigned int aNavIndex, G4NavigationHistory *aG4NavigationHistory);

        void FillG4Step(GPUHit *aGPUHit, 
                                  G4Step *aG4Step, 
                                  G4TouchableHandle &aPreG4TouchableHandle,
                                  G4TouchableHandle &aPostG4TouchableHandle);

        // Get G4 Event ID
        // Get G4 Thread ID

    private:
        std::unordered_map<size_t, const G4VPhysicalVolume *>
            fglobal_vecgeom_to_g4_map;  ///< Maps Vecgeom PV IDs to G4 PV IDs
};

#endif

