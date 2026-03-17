// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
//

#include "DetectorConstruction.hh"
#include "ActionInitialisation.hh"

#include "G4RunManagerFactory.hh"
#include "G4Types.hh"
#include "G4UImanager.hh"
#include "FTFP_BERT_HepEm.hh"
#include "FTFP_BERT_AdePT.hh"
#include "G4HadronicProcessStore.hh"
#include "G4EmParameters.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include <cstdlib>
#include <sstream>

int main(int argc, char **argv)
{
  // Macro name from arguments
  G4String batchMacroName;
  G4String outputDirectory = "";
  G4String outputFilename  = "";
  bool doAccumulatedEvents = false; // whether the edep is accumulated across events in the validation csv file
  bool writeTruthROOT      = false;
  G4bool useAdePT          = true;
  G4bool allSensitive      = false; // If set, ignores the sensitive detector flags in the GDML and marks all volumes as
                                    // sensitive. Useful for validation of geometries with no SD info
  G4String helpMsg("Usage: " + G4String(argv[0]) +
                   " [option(s)] \n No additional arguments triggers an interactive mode "
                   "executing vis.mac macro. \n Options:\n\t-h\t\tdisplay this help "
                   "message\n\t-m MACRO\ttriggers a batch mode executing MACRO\n\t--truth_root\twrite ROOT truth "
                   "histograms when ROOT support is compiled in\n\t--supports_truth_root\tprint 1 if ROOT truth "
                   "output is supported, otherwise 0\n");
  for (G4int i = 1; i < argc; ++i) {
    G4String argument(argv[i]);
    if (argument == "-h" || argument == "--help") {
      G4cout << helpMsg << G4endl;
      return 0;
    } else if (argument == "--supports_truth_root") {
#ifdef ADEPT_INTEGRATIONTEST_HAS_ROOT
      G4cout << "1" << G4endl;
#else
      G4cout << "0" << G4endl;
#endif
      return 0;
    } else if (argument == "-m") {
      batchMacroName = G4String(argv[i + 1]);
      ++i;
    } else if (argument == "--output_dir") {
      outputDirectory = G4String(argv[i + 1]);
      ++i;
    } else if (argument == "--output_file") {
      outputFilename = G4String(argv[i + 1]);
      ++i;
    } else if (argument == "--accumulated_events") {
      doAccumulatedEvents = true;
    } else if (argument == "--truth_root") {
      writeTruthROOT = true;
    } else if (argument == "--noadept") {
      useAdePT = false;
    } else if (argument == "--allsensitive") {
      allSensitive = true;
    } else {
      G4Exception("main", "Unknown argument", FatalErrorInArgument,
                  ("Unknown argument passed to " + G4String(argv[0]) + " : " + argument + "\n" + helpMsg).c_str());
    }
  }

#ifndef ADEPT_INTEGRATIONTEST_HAS_ROOT
  if (writeTruthROOT) {
    G4Exception("main", "TruthROOTUnavailable", FatalErrorInArgument,
                "integrationTest was built without ROOT support, so --truth_root is unavailable");
  }
#endif

  // Initialization of default Run manager
  std::unique_ptr<G4RunManager> runManager(G4RunManagerFactory::CreateRunManager());

  // Detector geometry:
  auto detector = new DetectorConstruction(allSensitive);
  runManager->SetUserInitialization(detector);

  // Physics list
  //
  G4VUserPhysicsList *physicsList;

  if (useAdePT) {
    physicsList = new FTFP_BERT_AdePT();
  } else {
    physicsList = new FTFP_BERT_HepEm();
  }

  runManager->SetUserInitialization(physicsList);

  // reduce verbosity of physics lists
  G4EmParameters::Instance()->SetVerbose(0);
  G4HadronicProcessStore::Instance()->SetVerbose(0);

  //-------------------------------
  // UserAction classes
  //-------------------------------
  runManager->SetUserInitialization(
      new ActionInitialisation(outputDirectory, outputFilename, doAccumulatedEvents, writeTruthROOT));

  G4UImanager *UImanager = G4UImanager::GetUIpointer();
  G4String command       = "/control/execute ";
  G4int err              = UImanager->ApplyCommand(command + batchMacroName);

  runManager.reset();

#ifdef ADEPT_INTEGRATIONTEST_HAS_ROOT
  if (writeTruthROOT) {
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(err);
  }
#endif

  // Free the store: user actions, physics_list and detector_description are
  //                 owned and deleted by the run manager, so they should not
  //                 be deleted in the main() program !

  return err;
}
