#include "../../include/test_services.h"
#include "../../include/testMode.h"

const std::string& AroNonceProviderTestMode::salt(BLOCK_TYPE bt) const {
    static const std::string
        SALT_CPU = "0KVwsNr6yT42uDX9",
        SALT_GPU = "cifE2rK4nvmbVgQu";
    return (bt == BLOCK_CPU) ? SALT_CPU : SALT_GPU;
}

bool AroNonceProviderTestMode::update() {
    updateTestMode(stats);
    blockType = testModeBlockType();
    return true;
}

void AroNonceProviderTestMode::generateNoncesImpl(std::size_t count, Nonces &nonces) {
    // $base = $this->publicKey."-".$nonce."-".$this->block."-".$this->difficulty;
    const std::string REF_NONCE_GPU = "swGetfIyLrh8XYHcL7cM5kEElAJx3XkSrgTGveDN2w";
    const std::string REF_BASE_GPU =
        /* pubkey */ std::string("PZ8Tyr4Nx8MHsRAGMpZmZ6TWY63dXWSCy7AEg3h9oYjeR74yj73q3gPxbxq9R3nxSSUV4KKgu1sQZu9Qj9v2q2HhT5H3LTHwW7HzAA28SjWFdzkNoovBMncD") + std::string("-") +
        /* nonce  */ REF_NONCE_GPU + std::string("-") +
        /* block  */ std::string("6327pZD7RSArjnD9wiVM6eUKkNck4Q5uCErh5M2H4MK2PQhgPmFTSmnYHANEVxHB82aVv6FZvKdmyUKkCoAhCXDy") + std::string("-") +
        /* diff   */ std::string("10614838");

    const std::string REF_NONCE_CPU = "YYEETiqrzrmgIApJlA3WKfuYPdSQI4F3U04GirBhA";
    const std::string REF_BASE_CPU =
        /* pubkey */ std::string("PZ8Tyr4Nx8MHsRAGMpZmZ6TWY63dXWSCy7AEg3h9oYjeR74yj73q3gPxbxq9R3nxSSUV4KKgu1sQZu9Qj9v2q2HhT5H3LTHwW7HzAA28SjWFdzkNoovBMncD") + std::string("-") +
        /* nonce  */ REF_NONCE_CPU + std::string("-") +
        /* block  */ std::string("KwkMnGF1qJeFh9nwZPTf3x86TmVF1RJaCPwfpKePVsAimJKTzA8H2ndx3FaRu7K54Md36yTcYKLaQtQNzRX4tAg") + std::string("-") +
        /* diff   */ std::string("30792058");

    bool isGPU = blockType == BLOCK_GPU;
    std::string REF_NONCE = isGPU ? REF_NONCE_GPU : REF_NONCE_CPU;
    std::string REF_BASE = isGPU ? REF_BASE_GPU : REF_BASE_CPU;

    for (uint32_t j = 0; j < count; j++) {
        nonces.nonces.push_back(REF_NONCE);
        nonces.bases.push_back(REF_BASE);
    }
}

bool AroResultsProcessorTestMode::processResult(const Input& i) {
    auto REF_DURATION = (i.blockDesc.type == BLOCK_GPU) ? 
        mpz_class("491522547412523425129", 10) : mpz_class("1054924814964225626", 10);
    return mpz_cmp(i.result.mpz_result.get_mpz_t(), REF_DURATION.get_mpz_t()) == 0;
}
