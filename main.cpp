#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <sl/Camera.hpp>

using namespace sl;

float COLOR_THRESHHOLD = 50;
int AREA_THRESHHOLD = 400;
float SAFE_DISTANCE = 150;

void find_objects(cv::Mat source, cv::Mat view);
cv::Mat slMat2cvMat(Mat& input);
void main_zed();

int main(int argc, char **argv) {
    main_zed();
    return 0;
}

void main_zed()
{
    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION_HD720;
    init_params.camera_fps = 5;

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != SUCCESS)
    {
        std::cout << "Can't open the camera!" << std::endl;
        exit(-1);
    }

	sl::Mat depth_zed(zed.getResolution(), MAT_TYPE_32F_C1);
	cv::Mat depth_ocv = slMat2cvMat(depth_zed);

	sl::Mat depth_view_zed(zed.getResolution(), MAT_TYPE_8U_C4);
	cv::Mat depth_view_ocv = slMat2cvMat(depth_view_zed);

	cv::Mat depth_view_converted;

	while (true) {
        // Grab an image
        if (zed.grab() == SUCCESS) {
			zed.retrieveMeasure(depth_zed, MEASURE_DEPTH);
            zed.retrieveImage(depth_view_zed, VIEW_DEPTH);
            std::cout << "Got image" << std::endl;

            cv::cvtColor(depth_view_ocv, depth_view_converted, CV_BGRA2BGR);
            find_objects(depth_ocv, depth_view_converted);
        }
    }

    // Close the camera
    zed.close();
}

void find_objects(cv::Mat source, cv::Mat view)
{
    cv::resize(source, source, cv::Size(160, 90));
    cv::resize(view, view, cv::Size(160, 90));

    source.setTo(0, source != source);
    source.setTo(0, source == std::numeric_limits<float>::infinity());
    source.setTo(0, source == -std::numeric_limits<float>::infinity());

    cv::imshow("original", view);

    int width = source.size().width,
        height = source.size().height;
    cv::Mat filled(height + 2,
                   width + 2,
                   CV_8UC1, cv::Scalar(0));
    cv::Mat filling(height + 2,
                    width + 2,
                    CV_8UC1, cv::Scalar(0));
    cv::Mat difference(height + 2,
                       width + 2,
                       CV_8UC1, cv::Scalar(0));
    cv::Mat invert_difference(height + 2,
                              width + 2,
                              CV_8UC1, cv::Scalar(0));
    cv::Mat result;
    view.copyTo(result);
    
    cv::Scalar boundary_color(255, 0, 0);

    int segment_count = 1;
    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            // std::cout << x << " " << y << std::endl;

            cv::Point point_in_seg(x + 1, y + 1),
                      point_in_source(x, y);
            float distance_at_point = source.at<float>(point_in_source);
            if (filling.at<unsigned char>(point_in_seg) == 0)
            {
                cv::floodFill(view, // image
                              filling, // mask
                              point_in_source, // seed point
                              cv::Scalar(0), // newVal
                              NULL, // rect
                              cv::Scalar(COLOR_THRESHHOLD, COLOR_THRESHHOLD, COLOR_THRESHHOLD), // loDiff
                              cv::Scalar(COLOR_THRESHHOLD, COLOR_THRESHHOLD, COLOR_THRESHHOLD), // upDiff
                              4 | ((1) << 8) | cv::FLOODFILL_MASK_ONLY | cv::FLOODFILL_FIXED_RANGE); // flags
                cv::bitwise_xor(filling, filled, difference);
                invert_difference = cv::Mat::ones(difference.size(), difference.type()) - difference;
                filling.copyTo(filled);

                
                int area_size = (int)cv::sum(difference)[0];

                if (area_size > AREA_THRESHHOLD)
                {
                    double color = cv::mean(view, difference)[0];
                    double distance = cv::mean(source, difference)[0];
                    if (color > SAFE_DISTANCE)
                    {
                        char text[255];
                        cv::Rect boundary = cv::boundingRect(difference);

                        sprintf(text, "%.0f", distance);
                        cv::rectangle(result, boundary, boundary_color, 1);
                        cv::putText(result, text, cv::Point(boundary.x, boundary.y+30), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, boundary_color);
                    }
                }

                segment_count++;
            }
        }
    std::cout << "show image" << std::endl;


    cv::imshow("result", result);
    if (cv::waitKey(200) == 0) exit(0);
}

cv::Mat slMat2cvMat(Mat& input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
        case MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
        case MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
        case MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
        case MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
        case MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
        case MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
        case MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
        case MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
        default: break;
    }

    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
}
