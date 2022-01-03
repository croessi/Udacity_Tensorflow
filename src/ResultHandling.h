#ifndef MQTTHANDLING_H
#define MQTTHANDLING_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <iomanip>

#include <mqtt/client.h>
#include <sstream>
#include <string>

#include "TensorProcessor.h"
#include "Statistics.h"

class ResultHandlerClass
{
private:
    mqtt::client _cli;

public:
    ResultHandlerClass(string ServerIP) : _cli(mqtt::client("tcp://" + ServerIP + ":1883", "DetectorPi")){};
    void ResultHandling(DetectionResultClass &SessionOutput, float display_threshold, float boxwidth_threshold);
};

#endif