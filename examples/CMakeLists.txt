# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

# - Common data for testing
configure_file(data/cms2018_sd.gdml ${PROJECT_BINARY_DIR}/cms2018_sd.gdml)
configure_file(data/lhcb2024_downstream_calos.gdml ${PROJECT_BINARY_DIR}/lhcb2024_downstream_calos.gdml)
configure_file(data/ppttbar.hepmc3 ${PROJECT_BINARY_DIR}/ppttbar.hepmc3)

# - Subprojects
add_subdirectory(Example1)
add_subdirectory(IntegrationBenchmark)
#add_subdirectory(AsyncExample)
