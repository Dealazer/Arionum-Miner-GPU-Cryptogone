#pragma once

const std::string MINER_VERSION = "1.6.0-alpha";

inline std::string getVersionStr() {
	return std::string("Arionum GPU Miner version ") + MINER_VERSION;
}
