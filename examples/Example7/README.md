<!--
SPDX-FileCopyrightText: 2021 CERN
SPDX-License-Identifier: CC-BY-4.0
-->
# AdePT Example 7
Demonstrate data (geometry+physics) export from a Geant4 application
and import of that data into an AdePT application. GDML and G4HepEM
are used as the geometry/physics data.

This decoupling isn't strictly necessary as the geometry/data can be
created directly and shared in in a Geant4+AdePT application (see Example6).
However, there are several reasons to decouple the components that this
example explores:

- Don't depend directly on Geant4 for pure GPU test applications (reduce linking/deps)
- Allow sharing of common data between AdePT prototypes/examples
- Reduce re-calculation of physics data on each application start
  - For simple applications this isn't significant but can be substantial for more
    complex geometries/physics models

## Programs and running
