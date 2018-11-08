#pragma once

#include "mining_services.h"

class Stats;

class AroNonceProviderTestMode : public AroNonceProvider {
public:
    AroNonceProviderTestMode(Stats & stats) : stats(stats) {};
    const std::string& salt(BLOCK_TYPE bt) const override;
    bool update() override;
    BLOCK_TYPE currentBlockType() const override { return blockType; };
    BlockDesc currentBlockDesc() const override {
        BlockDesc bd;
        bd.type = blockType;
        return bd;
    };

protected:
    void generateNoncesImpl(std::size_t count, Nonces &nonces) override;

private:
    BLOCK_TYPE blockType;
    Stats & stats;
};

class AroResultsProcessorTestMode : public IAroResultsProcessor {
public:
    virtual bool processResult(const Input& i) override;
};


