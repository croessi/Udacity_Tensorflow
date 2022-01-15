#ifndef READLABELS_H_
#define READLABELS_H_

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <algorithm>

using namespace std;

map<int, string> ReadClasses2Labels(string &filename);
map<int, string> ReadClasses2LabelsSSDV1(string &filename);

#endif