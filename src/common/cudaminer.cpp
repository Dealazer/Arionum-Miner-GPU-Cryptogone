//
// Created by guli on 31/01/18.
//

#include <iostream>
#include "../../include/cudaminer.h"

using namespace std;

CudaMiner::CudaMiner(int *id) : Miner(id) {

}

void CudaMiner::mine() {
    cout << "Miner" << *id << endl;
}