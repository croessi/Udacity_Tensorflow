#include "ResultHandling.h"

void ResultHandlerClass::ResultHandling(DetectionResultClass &SessionOutput, float display_threshold, float boxwidth_threshold)
{
    char buffer[100];
    //stringstream RawOutput;
    stringstream NiceOutput;
    NiceOutput << "{";

    StatisticsClass Statistics;
    int NumObjects = 0;

    for (Detection_t d : SessionOutput.GetDetections())
    {
        float boxwidth = (d.BoxBottomRigth.x - d.BoxTopLeft.x) / (float)SessionOutput.GetImage().size[1];
        float boxheight = (d.BoxBottomRigth.y - d.BoxTopLeft.y) / (float)SessionOutput.GetImage().size[0];

        float boxcenterX = d.BoxTopLeft.x / (float)SessionOutput.GetImage().size[1] + boxwidth * 0.5;

        float boxcenterY = d.BoxTopLeft.y / (float)SessionOutput.GetImage().size[0] + boxheight * 0.5;

        //{"Timer1":{"Arm": <status>, "Time": <time>}, "Timer2":{"Arm": <status>, "Time": <time>}}

        if (d.score > display_threshold && boxwidth < boxwidth_threshold)
        {
            NumObjects++;
            //RawOutput.setf(ios::fixed);
            //RawOutput << "Score: " << (int)(d.score * 100) << "% " << d.ClassName << " at: " << setprecision(2) << boxcenterX << ":" << boxcenterY << "\n";

            NiceOutput.setf(ios::fixed);
            if (Statistics.Stat[d.ClassName] > 0)
            {
                NiceOutput << "\"" << d.ClassName << "_" << Statistics.Stat[d.ClassName] << "\":\"" << (int)(d.score * 100) << "% "
                           << " at:" << setprecision(2) << boxcenterX << ":" << boxcenterY << "\",";
            }
            else
            {
                NiceOutput << "\"" << d.ClassName << "\":\"" << (int)(d.score * 100) << "% "
                           << " at:" << setprecision(2) << boxcenterX << ":" << boxcenterY << "\",";
            }

            rectangle(SessionOutput.GetImage(), d.BoxTopLeft, d.BoxBottomRigth, Scalar(0, 255, 0), 1, 8, 0);

            Statistics.Stat[d.ClassName]++;

            snprintf(buffer, 100, "%s %d%%", d.ClassName.c_str(), (int)(d.score * 100));

            putText(SessionOutput.GetImage(),
                    buffer,
                    Point2d(d.BoxTopLeft.x, d.BoxBottomRigth.y),
                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                    0.5,
                    cv::Scalar(255, 255, 255),
                    1, cv::LINE_AA, false);
        }
    }
    //remove last ,
    NiceOutput.seekp(-1, NiceOutput.cur);
    NiceOutput << "}";
    //check if we are connected to MQTT server

    //create statitics message
    cout << "Detected Objects:\n";
    stringstream StatisticsOutput;
    StatisticsOutput << "{";
    for (auto const &x : Statistics.Stat)
    {
        StatisticsOutput << "\"" << x.first << "\":" << x.second << ",";
        if (x.second > 0)
            std::cout << x.first << ':' << x.second << std::endl;
    }
    cout << "--------------------\n";

    //remove last ,
    StatisticsOutput.seekp(-1, NiceOutput.cur);
    StatisticsOutput << "}";

    if (_cli.is_connected())
    {
        //cout << "Publishing MQTT Messages!\n";

        mqtt::message_ptr msg;

        //send available as state
        msg = mqtt::message::create(_StateTopic, to_string(SessionOutput.runtime));
        _cli.publish(msg);

        msg = mqtt::message::create(_AttributeTopic, NiceOutput.str());
        _cli.publish(msg);

        //Send statistics
        msg = mqtt::message::create(_StateTopicStatistics, to_string(NumObjects));
        _cli.publish(msg);

        msg = mqtt::message::create(_AttributeTopicStatistics, StatisticsOutput.str());
        _cli.publish(msg);
    }
    else
    {
        _cli.reconnect();
        cout << "\n\nNot connected to MQTT Server - try to reconnect!!!!!!!!!!!!!!!!!!!!!\n\n";
    }
}