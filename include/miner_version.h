#pragma once

const std::string MINER_VERSION = "1.5.0";

inline std::string getVersionStr() {
	return std::string("Guli Arionum GPU Miner (modified by Cryptogone), version ") + MINER_VERSION;
}