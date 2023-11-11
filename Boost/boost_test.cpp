

#include <string>
#include <iostream>
#include <boost/iostreams/device/mapped_file.hpp>
#include <cuda_runtime.h>
//it works?
using std::string; using std::cout;


int readcsv(const string& file) {
	boost::iostreams::mapped_file_source csv_file(file);
	if (!csv_file.is_open()) {
		//throw exception 
		return -1;
	}
	const char* data = csv_file.data();
	const char* end = data + csv_file.size();
	for (; data != end; data++) {
		cout << *data;
	}
	return 0;
}
/*
int main() {
	//readcsv("temp.csv");

	return 0;
}
*/