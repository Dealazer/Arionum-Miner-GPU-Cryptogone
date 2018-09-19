#pragma once

const std::string MINER_VERSION = "1.5.1";

inline std::string getVersionStr() {
	return std::string("Arionum GPU Miner version ") + MINER_VERSION;
}
