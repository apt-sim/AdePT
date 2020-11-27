#ifndef TRACKH
#define TRACKH

enum TrackStatus {alive, dead};

struct track {
    int index{0};
    int pdg{0};
    double energy{10};
    double pos[3]{0};
    double dir[3]{1};
    int mother_index{0};
    TrackStatus status{alive}; 
    int current_process{0};
    float interaction_length{FLT_MAX};
    float energy_loss{0}; // primitive version of scoring 
    int number_of_secondaries{0}; // primitive version of scoring
  };

#endif