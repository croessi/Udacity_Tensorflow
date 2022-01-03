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
    const string _StateTopic;
    const string _AttributeTopic;
    const string _StateTopicStatistics;
    const string _AttributeTopicStatistics;

public:
    ResultHandlerClass(string ServerIP) : _cli(mqtt::client("tcp://" + ServerIP + ":1883", "DetectorPi")),
                                          _StateTopic("homeassistant/sensor/DetectorPi/state"),
                                          _AttributeTopic("homeassistant/sensor/DetectorPi/attributes"),
                                          _StateTopicStatistics("homeassistant/sensor/DetectorPiStatistics/state"),
                                          _AttributeTopicStatistics("homeassistant/sensor/DetectorPiStatistics/attributes")
    {
        cout << "Not connected to Homeassistant MQTT Server -> try to connect...";
        std::string user = "mqtt";
        std::string password = "double45double";
        mqtt::connect_options connOpts;
        connOpts.set_user_name(user);
        connOpts.set_password(password);
        connOpts.set_automatic_reconnect(true);

        cout << "Waiting for the connection to MQTT..." << endl;
        mqtt::connect_response conntok = _cli.connect(connOpts);

        while (!_cli.is_connected())
        {
            waitKey(10);
        }
        cout << "  ...OK" << endl;

        mqtt::message_ptr msg;

        const string DiscoveryTopic = "homeassistant/sensor/DetectorPi/config";
        const string DiscoveryConfig = "{\"name\": \"DetectorPi\", \"state_topic\" : \"" + _StateTopic + "\"" +
                                       ",\"json_attributes_topic\":\"" + _AttributeTopic + "\"" +
                                       ",\"unit_of_measurement\": \"ms\"}";

        msg = mqtt::message::create(DiscoveryTopic, DiscoveryConfig);
        _cli.publish(msg);

        const string DiscoveryTopicStatistics = "homeassistant/sensor/DetectorPiStatistics/config";
        const string DiscoveryConfigStatistics = "{\"name\": \"DetectorPiStatistics\", \"state_topic\" : \"" + _StateTopicStatistics + "\"" +
                                                 ",\"json_attributes_topic\":\"" + _AttributeTopicStatistics + "\"}";

        msg = mqtt::message::create(DiscoveryTopicStatistics, DiscoveryConfigStatistics);
        _cli.publish(msg);
    };

    void ResultHandling(DetectionResultClass &SessionOutput, float display_threshold, float boxwidth_threshold);
};

#endif