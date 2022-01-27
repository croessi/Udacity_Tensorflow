#include "ResultHandling.h"

void ResultHandlerClass::ResultHandling(DetectionResultClass &SessionOutput, float display_threshold, float boxwidth_threshold, float overlap_threshold)
{
    char buffer[100];
    //stringstream RawOutput;
    stringstream NiceOutput;
    NiceOutput << "{";

    StatisticsClass Statistics;
    int NumObjects = 0;

    //Loop to filter boxes with too much overlap
    vector<Detection_t> detections = SessionOutput.GetDetections();

    for (int i = 0; i < detections.size() - 1; i++)
    {
        if (detections[i].overlap != -2)
        {
            for (int h = i + 1; h < detections.size(); h++)
            {
                if (detections[i].detclass == detections[h].detclass && detections[h].overlap == -1) // && detections[h].detclass != -2)
                {
                    int XA1 = detections[i].BoxTopLeft.x;
                    int XA2 = detections[i].BoxBottomRigth.x;
                    int YA1 = detections[i].BoxTopLeft.y;
                    int YA2 = detections[i].BoxBottomRigth.y;
                    int XB1 = detections[h].BoxTopLeft.x;
                    int XB2 = detections[h].BoxBottomRigth.x;
                    int YB1 = detections[h].BoxTopLeft.y;
                    int YB2 = detections[h].BoxBottomRigth.y;
                    int overlap_area = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1));

                    float size_A = (XA2 - XA1) * (YA2 - YA1);
                    float size_B = (XB2 - XB1) * (YB2 - YB1);

                    float overlap_A = overlap_area / size_A;
                    float overlap_B = overlap_area / size_B;

                    if (overlap_A && size_A != 0 && overlap_A > detections[i].overlap)
                        detections[i].overlap = overlap_A;
                    if (overlap_B && size_B != 0 && overlap_B > detections[h].overlap)
                        detections[h].overlap = overlap_B;

                    if (overlap_A > overlap_threshold || overlap_B > overlap_threshold)
                    {
                        if (detections[i].score > detections[h].score)
                            detections[h].overlap = -2;
                        else
                            detections[i].overlap = -2;
                    }
                }
            }
        }
    }

    for (Detection_t d : detections)
    {
        float boxwidth = (d.BoxBottomRigth.x - d.BoxTopLeft.x) / (float)SessionOutput.GetImage().size[1];
        float boxheight = (d.BoxBottomRigth.y - d.BoxTopLeft.y) / (float)SessionOutput.GetImage().size[0];

        float boxcenterX = d.BoxTopLeft.x / (float)SessionOutput.GetImage().size[1] + boxwidth * 0.5;
        float boxcenterY = d.BoxTopLeft.y / (float)SessionOutput.GetImage().size[0] + boxheight * 0.5;

        //{"Timer1":{"Arm": <status>, "Time": <time>}, "Timer2":{"Arm": <status>, "Time": <time>}}

        if (d.score > display_threshold && boxwidth < boxwidth_threshold && d.overlap != -2)
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

            snprintf(buffer, 100, "%s %d%% Overlap %d%%", d.ClassName.c_str(), (int)(d.score * 100), (int)(d.overlap * 100));

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