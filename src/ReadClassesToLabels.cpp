#include "ReadClassesToLabels.h"

map<int, string> ReadClasses2Labels(string filename)
{
    map<int, string> ret;

    ifstream myfile(filename); //(filename, s.binary | s.trunc | s.in);
    if (!myfile.is_open())
    {
        std::cout << "failed to open " << filename << '\n';
    }
    else
    {

        string line;
        while (getline(myfile, line))
        {
            istringstream sline(line);
            std::string key;
            std::string name1;
            int id;
            sline >> key >> id;
            if (key == "id:")
            {
                getline(myfile, line);
                istringstream sline1(line);
                sline1 >> key >> name1;

                name1.erase(remove(name1.begin(), name1.end(), '\"'), name1.end());

                ret[id] = name1;
            }
        }
    }
    return ret;
}