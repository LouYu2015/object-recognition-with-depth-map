#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <sl/Camera.hpp>

using namespace sl;

struct Obj_Info { // Information of detected object
    double distance; // Average distance measured by ZED camera
    cv::Rect boundary; // Rectangle boundary of the object(measured in resized image)
};

const float DISTANCE_DIFFERENCE_THRESHHOLD = 300; // The distance difference tolerance of a single object
const int AREA_THRESHHOLD = 400; // Minimum number of pixel that can be consider as an object(measued in resized image)
const float SAFE_DISTANCE = 2000; // Maximum distance that will be reported
const int RESIZE_WIDTH = 160; // Width of resized image
const int RESIZE_HEIGHT = 90; // Height of resized image

// Find objects that are too close to the robot
//
// Parameters
//   source: depth map measured by Zed camera
//   view: depth map converted to RGB
//
// Return: List of object informations
std::vector<Obj_Info> find_objects(cv::Mat source, cv::Mat view);

// Convert Zed camera matrix to OpenCV matrix
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
    init_params.camera_fps = 10;

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != SUCCESS)
    {
        std::cout << "Can't open the camera!" << std::endl;
        exit(-1);
    }

    // Initialize images
	sl::Mat depth_zed(zed.getResolution(), MAT_TYPE_32F_C1); // Depth map(float)
	cv::Mat depth_ocv = slMat2cvMat(depth_zed);

	sl::Mat depth_view_zed(zed.getResolution(), MAT_TYPE_8U_C4); // Depth map in RGB
	cv::Mat depth_view_ocv = slMat2cvMat(depth_view_zed);

	cv::Mat depth_view_converted;

	while (true) {
        // Grab an image
        if (zed.grab() == SUCCESS) {
            // Retrive image
			zed.retrieveMeasure(depth_zed, MEASURE_DEPTH);
            zed.retrieveImage(depth_view_zed, VIEW_DEPTH);
            std::cout << "Got image" << std::endl;

            // Remove alpha channel
            cv::cvtColor(depth_view_ocv, depth_view_converted, CV_BGRA2BGR);
            
            // Find objects
            std::vector<Obj_Info> result = find_objects(depth_ocv, depth_view_converted);

            // Show result
            for (int i = 0; i < result.size(); i++)
            {
                std::cout << "Position: (" << result[i].boundary.x << ", " << result[i].boundary.y << ")"<< std::endl;
                std::cout << "Distance: " << result[i].distance << std::endl;
                std::cout << std::endl;
            }
        }
    }

    // Close the camera
    zed.close();
}

std::vector<Obj_Info> find_objects(cv::Mat source, cv::Mat view)
{
    // Resize image
    cv::resize(source, source, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT));
    cv::resize(view, view, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT));

    // Deal with special values
    source.setTo(SAFE_DISTANCE*10, source != source); // NaN
    source.setTo(SAFE_DISTANCE*10, source == std::numeric_limits<float>::infinity()); // Infinity
    source.setTo(SAFE_DISTANCE*10, source == -std::numeric_limits<float>::infinity()); // Negative infinity

    // Storage for result
    std::vector<Obj_Info> object_infos;

    // Show original view
    cv::imshow("original", view);

    int width = source.size().width, // Image size
        height = source.size().height;
    cv::Mat filled(height + 2,
                   width + 2,
                   CV_8UC1, cv::Scalar(0)); // Mask of area filled
    cv::Mat filling(height + 2,
                    width + 2,
                    CV_8UC1, cv::Scalar(0)); // Mask of area filled including the area in this iteration
    cv::Mat difference(height + 2,
                       width + 2,
                       CV_8UC1, cv::Scalar(0)); // Mask of new area
    cv::Mat invert_difference(height + 2,
                              width + 2,
                              CV_8UC1, cv::Scalar(0)); // Mask of unchanged area
    cv::Mat result; // Image for result
    view.copyTo(result);
    
    cv::Scalar boundary_color(255, 0, 0);

    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            // Iterate through points
            cv::Point point_in_seg(x + 1, y + 1),
                      point_in_source(x, y);
            
            // If this area is not marked
            if (filling.at<unsigned char>(point_in_seg) == 0)
            {
                // Mark adjecent pixels
                cv::floodFill(source, // image
                              filling, // mask
                              point_in_source, // seed point
                              cv::Scalar(0), // newVal
                              NULL, // rect
                              cv::Scalar(DISTANCE_DIFFERENCE_THRESHHOLD), // loDiff
                              cv::Scalar(DISTANCE_DIFFERENCE_THRESHHOLD), // upDiff
                              4 | ((1) << 8) | cv::FLOODFILL_MASK_ONLY | cv::FLOODFILL_FIXED_RANGE); // flags
                
                // Get the area filled in this iteration
                cv::bitwise_xor(filling, filled, difference);
                invert_difference = cv::Mat::ones(difference.size(), difference.type()) - difference;
                
                // Update mark
                filling.copyTo(filled);

                // Calculate area size
                int area_size = (int)cv::sum(difference)[0];

                if (area_size > AREA_THRESHHOLD)
                {
                    // Measure average distance
                    double distance = cv::mean(source, difference)[0];
                    if (distance < SAFE_DISTANCE)
                    {
                        // Draw information on image
                        char text[255];
                        cv::Rect boundary = cv::boundingRect(difference);

                        sprintf(text, "%.0f", distance);
                        cv::rectangle(result, boundary, boundary_color, 1);
                        cv::putText(result, text, cv::Point(boundary.x, boundary.y+30), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, boundary_color);
                        
                        // Store information in array
                        object_infos.push_back((Obj_Info){distance, boundary});
                    }
                }
            }
        }
    // Show image with information
    std::cout << "show image" << std::endl;

    cv::imshow("result", result);
    if (cv::waitKey(70) == 0) exit(0);

    // Return
    return object_infos;
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
