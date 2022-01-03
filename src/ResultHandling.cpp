#include "ResultHandling.h"

void ResultHandlerClass::ResultHandling(DetectionResultClass &SessionOutput, float display_threshold, float boxwidth_threshold)
{
    char buffer[100];
    stringstream RawOutput;
    stringstream NiceOutput;

    StatisticsClass Statistics;

    for (Detection_t d : SessionOutput.GetDetections())
    {
        float boxwidth = (d.BoxBottomRigth.x - d.BoxTopLeft.x) / (float)SessionOutput.GetImage().size[1];
        float boxheight = (d.BoxBottomRigth.y - d.BoxTopLeft.y) / (float)SessionOutput.GetImage().size[0];

        float boxcenterX = d.BoxTopLeft.x / (float)SessionOutput.GetImage().size[1] + boxwidth * 0.5;

        float boxcenterY = d.BoxTopLeft.y / (float)SessionOutput.GetImage().size[0] + boxheight * 0.5;

        //{"Timer1":{"Arm": <status>, "Time": <time>}, "Timer2":{"Arm": <status>, "Time": <time>}}

        if (d.score > display_threshold && boxwidth < boxwidth_threshold)
        {
            RawOutput.setf(ios::fixed);
            RawOutput << "Score: " << (int)(d.score * 100) << "% " << d.ClassName << " at: " << setprecision(2) << boxcenterX << ":" << boxcenterY << "\n";

            NiceOutput << "{\"" << d.ClassName << "\"}:{\"Score:\"" << d.score << "\"},";

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

    //check if we are connected to MQTT server
    if (!_cli.is_connected())
    {
        cout << "Not connected to Homeassistant MQTT Server -> try to connect...";
        _cli.connect();
        if (_cli.is_connected())
            cout << "done \n";
    }

    if (_cli.is_connected())
    {
        const string topic = "DetectorPi_Raw";

        mqtt::message_ptr msg;
        //truncate to max 255 signs for MQTT
        if (RawOutput.str().size() > 250)
            msg = mqtt::message::create(topic, RawOutput.str().substr(0, 250));
        else
            msg = mqtt::message::create(topic, RawOutput.str());

        _cli.publish(msg);
        cout << "Message: " << RawOutput.str() << endl;

        for (auto const &x : Statistics.Stat)
        {
            if (x.second > 0)
                std::cout << x.first << ':' << x.second << std::endl;
        }
    }
}