//
// Created by guli on 31/01/18.
//

#include <iostream>
#include "../../include/openclminer.h"

using namespace std;

OpenClMiner::OpenClMiner(int *id) : Miner(id) {

}

void OpenClMiner::mine() {
    cout << "Miner" << *id << endl;
}