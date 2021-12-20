#ifndef HANDWIRITING_H
#define HANDWIRITING_H


#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>

using namespace std;

struct ImHeaderStruct
{
  u_int32_t magic_number;
  u_int32_t numImages;
  u_int32_t numRows;
  u_int32_t numColumns;
};

vector<vector<int8_t>> ImageReader(std::string filename)
{
  std::fstream s(filename); //(filename, s.binary | s.trunc | s.in);
  if (!s.is_open())
  {
    cout << "failed to open " << filename << '\n';
  }
  else
  {
    // read header as 32 bit integers

    ImHeaderStruct ImHeader;
    s.read((char *)&ImHeader, sizeof(ImHeaderStruct));

    ImHeader.magic_number = __builtin_bswap32(ImHeader.magic_number);
    ImHeader.numColumns = __builtin_bswap32(ImHeader.numColumns);
    ImHeader.numImages = __builtin_bswap32(ImHeader.numImages);
    ImHeader.numRows = __builtin_bswap32(ImHeader.numRows);

    //allocate memory as vector of vectors
    //vector<vector<int8_t>> ret(ImHeader.numImages);
    vector<vector<int8_t>> ret;
    for (int c = 0; c < ImHeader.numImages; c++)
    {
      //alocate memory for image
      vector<int8_t> inIm(ImHeader.numRows * ImHeader.numColumns);
      s.read((char *)inIm.data(), sizeof(inIm));

      ret.emplace_back(move(inIm));
    }
    return ret;
  }
  return vector<vector<int8_t>>();
}

#endif