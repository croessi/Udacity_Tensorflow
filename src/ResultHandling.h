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
    const bool _sendDetectorFrame;
    const string _MQTTuser;
    const string _MQTTpassword;
    const string _InstanceName;

public:
    ResultHandlerClass(string ServerIP, bool sendDetectorFrame, const string MQTTuser, const string MQTTpassword, const string InstanceName) : _cli(mqtt::client("tcp://" + ServerIP + ":1883", "DetectorPi")),
                                                                                                                                               _StateTopic("homeassistant/sensor/" + InstanceName + "/state"),
                                                                                                                                               _AttributeTopic("homeassistant/sensor/" + InstanceName + "/attributes"),
                                                                                                                                               _StateTopicStatistics("homeassistant/sensor/" + InstanceName + "Statistics/state"),
                                                                                                                                               _AttributeTopicStatistics("homeassistant/sensor/" + InstanceName + "Statistics/attributes"),
                                                                                                                                               _sendDetectorFrame(sendDetectorFrame),
                                                                                                                                               _MQTTuser(MQTTuser),
                                                                                                                                               _MQTTpassword(MQTTpassword),
                                                                                                                                               _InstanceName(InstanceName)

    {
        cout << "Not connected to Homeassistant MQTT Server -> try to connect...\n";

        mqtt::connect_options connOpts;
        connOpts.set_user_name(_MQTTuser);
        connOpts.set_password(_MQTTpassword);
        connOpts.set_automatic_reconnect(true);
        connOpts.set_will_message(mqtt::message::create(_StateTopic, "offline"));

        while (!_cli.is_connected())
        {
            try
            {
                mqtt::connect_response conntok = _cli.connect(connOpts);
            }
            catch (...)
            {
                cout << "MQTT Server not connected - will try again!" << endl;
                waitKey(10000);
            }
        }

        cout << "  MQTT Server connected" << endl;

        mqtt::message_ptr msg;

        const string DiscoveryTopic = "homeassistant/sensor/" + _InstanceName + "/config";
        const string DiscoveryConfig = "{\"name\": \"" + _InstanceName + "\", \"state_topic\" : \"" + _StateTopic + "\"" +
                                       ",\"json_attributes_topic\":\"" + _AttributeTopic + "\"" +
                                       ",\"unit_of_measurement\": \"ms\"}";

        msg = mqtt::message::create(DiscoveryTopic, DiscoveryConfig);
        _cli.publish(msg);

        const string DiscoveryTopicStatistics = "homeassistant/sensor/" + _InstanceName + "Statistics/config";
        const string DiscoveryConfigStatistics = "{\"name\": \"" + _InstanceName + "Statistics\", \"state_topic\" : \"" + _StateTopicStatistics + "\"" +
                                                 ",\"json_attributes_topic\":\"" + _AttributeTopicStatistics + "\"}";

        msg = mqtt::message::create(DiscoveryTopicStatistics, DiscoveryConfigStatistics);
        _cli.publish(msg);
    };

    void ResultHandling(DetectionResultClass &SessionOutput, float display_threshold, float boxwidth_threshold);
};

#endif