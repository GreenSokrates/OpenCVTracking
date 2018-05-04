#include <cstring>
#include <ctime>
#include <iostream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tldDataset.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace cv;

// Convert to string
#define SSTR(x)                                                                \
    static_cast<std::ostringstream &>((std::ostringstream() << std::dec << x)) \
        .str()

inline cv::Ptr<cv::Tracker> createTrackerByName(cv::String name) {
    cv::Ptr<cv::Tracker> tracker;

    if (name == "KCF")
        tracker = cv::TrackerKCF::create();
    else if (name == "TLD")
        tracker = cv::TrackerTLD::create();
    else if (name == "BOOSTING")
        tracker = cv::TrackerBoosting::create();
    else if (name == "MEDIAN_FLOW")
        tracker = cv::TrackerMedianFlow::create();
    else if (name == "MIL")
        tracker = cv::TrackerMIL::create();
    else if (name == "GOTURN")
        tracker = cv::TrackerGOTURN::create();
    else if (name == "MOSSE")
        tracker = cv::TrackerMOSSE::create();
    else if (name == "CSRT")
        tracker = cv::TrackerCSRT::create();
    else
        CV_Error(cv::Error::StsBadArg, "Invalid tracking algorithm name\n");

    return tracker;
}

int main() {
    // Set default tracking Algorithm
    std::string trackingAlg = "CSRT"; // "MEDIAN_FLOW" ;"KCF";

    // Create Multitracker
    MultiTracker trackers;

    // Container for Tracked Objects
    vector<Rect2d> objects;

    // Get Video
    // VideoCapture src("../../media/VideoStream.avi");
    VideoCapture src(0);
    if (!src.isOpened()) {
        std::cout << "No Video!" << std::endl;
        return 1;
    }

    Mat frame;

    // get bounding box
    src >> frame;
    vector<Rect> ROIs;
    selectROIs("tracker", frame, ROIs);

    // Quit if no ROI
    if (ROIs.size() < 1)
        return 0;

    // Initialize Trackers
    std::vector<Ptr<Tracker>> algorithms;
    for (size_t i = 0; i < ROIs.size(); i++) {
        algorithms.push_back(createTrackerByName(trackingAlg));
        objects.push_back(ROIs[i]);
    }

    trackers.add(algorithms, frame, objects);

    // Tracking aslong as there are frames
    while (src.read(frame)) {
        // Start timer
        double timer = (double)getTickCount();

        bool ok = trackers.update(frame);

        if (ok) {
            // draw the tracked object
            for (unsigned i = 0; i < trackers.getObjects().size(); i++) {
                rectangle(frame, trackers.getObjects()[i], Scalar(255, 0, 0), 2,
                          1);
            }
        } else {
            // Tracking failure detected.
            putText(frame, "Tracking failure detected", Point(100, 80),
                    FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        }

        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);

        // Display tracker type on frame
        putText(frame, trackingAlg + " Tracker", Point(100, 20),
                FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

        // Display FPS on frame
        putText(frame, "FPS : " + SSTR(int(fps)), Point(100, 50),
                FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

        imshow("Tracker", frame);

        int k = waitKey(33);
        if (k == 27) {
            break;
        }
    }
}
