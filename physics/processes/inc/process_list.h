#ifndef PROCESSLISTH
#define PROCESSLISTH

#include "process.h"

class process_list {
    public:
        __device__ process_list() {}
        __device__ process_list(process **l, int n) {list = l; list_size = n; }
        __device__ virtual bool GetPhysicsInteractionLength(int particle_index, adept::BlockData<track> *block, curandState_t *states) const;
        process **list;
        int list_size;
};

__device__ bool process_list::GetPhysicsInteractionLength(int particle_index, adept::BlockData<track> *block, curandState_t *states) const {
        
    bool physics_wins = false;
    float current_length = FLT_MAX;

    for (int i = 0; i < list_size; i++) {
        float temp = list[i]->GetPhysicsInteractionLength(particle_index, block, states);
        if (temp < current_length) {
                (*block)[particle_index].interaction_length = temp;
                (*block)[particle_index].current_process = i;
                physics_wins = true;
                current_length = temp;
            }
    }
  return physics_wins; 
}

#endif