#pragma once

const std::string MINER_VERSION = "1.1.0";

inline std::string getVersionStr() {
	return std::string("Guli's gpu miner (Windows port by Cryptogone), version ") + MINER_VERSION;
}