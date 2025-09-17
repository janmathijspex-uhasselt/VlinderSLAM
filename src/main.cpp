#include <iostream>
#include <thread>

#include <stella_vslam/system.h>
#include <stella_vslam/config.h>
#include <stella_vslam/publish/map_publisher.h>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <DeviceFactory.h>

#include <pangolin_viewer/viewer.h>
#include <stella_vslam/util/yaml.h>

#include "UDPSender.h"
#include "UDPMessage.h"

using namespace std;

void viewer_handler(pangolin_viewer::viewer &viewer) {
    viewer.run();
}

void imu_handler(stella_vslam::system &slam, Device &cap, std::atomic<bool>& running) {
    double timestamp, prev_timestamp = 0.0, accel_reset_timestamp = 0.0;
    Eigen::Vector3f accel, gyro, dtheta;

    auto map_publisher = slam.get_map_publisher();

    while (running)
    {
        cap.captureIMU(accel, gyro, timestamp);

        // Calculate time difference
        double dt = (timestamp - prev_timestamp)/1000;

        // Calculate gyro values
        dtheta = -(gyro * dt);

        cv::Mat rvec, R_cv;
        cv::eigen2cv(dtheta, rvec);

        cv::Rodrigues(rvec, R_cv);

        Eigen::Matrix3d R;
        cv::cv2eigen(R_cv, R);

        // Apply IMU values to pose of camera
        auto current = map_publisher->get_current_cam_pose();

        Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
        transformation.block<3,3>(0,0) = R;
        // transformation.block<3,1>(0,3) = t.cast<double>();

        // std::cout << "transformation:" << std::endl << transformation << std::endl;
        // std::cout << "transformation.block<3,1>(0,3): " << transformation.block<3,1>(0,3).transpose() << std::endl;

        current = transformation * current;

        map_publisher->set_current_cam_pose(current);

        // Reset timestamp
        prev_timestamp = timestamp;

        // std::cout << "Gyro: " << gyro.transpose() << std::endl;
        // std::cout << "Accel: " << accel.transpose() << std::endl;
    }
}

int main()
{
    // Init slam
    const std::string config_path = "assets/stella_config_rs_d455.yml";
    const std::string vocab_path = "assets/orb_vocab.fbow";
    const std::string map_path = "assets/map.msg";

    auto config = std::make_shared<stella_vslam::config>(config_path);

    auto slam = std::make_shared<stella_vslam::system>(config, vocab_path);

    // Viewer
    auto viewer = std::make_shared<pangolin_viewer::viewer>(
        stella_vslam::util::yaml_optional_ref(config->yaml_node_, "PangolinViewer"),
        slam,
        slam->get_frame_publisher(),
        slam->get_map_publisher());

    std::thread viewer_thread(viewer_handler, std::ref(*viewer));

    // Init realsense or other camera
    // ---------------- Realsense ----------------

    DeviceFactory factory;
    auto cap = factory.createFirstDevice("RealSense2", "");
    if (!cap) {
        std::cerr << "Kon geen RealSense2 device aanmaken!" << std::endl;
        return -1;
    }

    // ---------------- Realsense ----------------

    // ---------------- Other cam ----------------
    /*
    // cv::VideoCapture cap_cv(0);
    cv::VideoCapture cap_cv(4);
    cap_cv.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap_cv.set(cv::CAP_PROP_FRAME_HEIGHT, 800);
    // cap_cv.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    if (!cap_cv.isOpened()) {
        std::cerr << "Kan camera niet openen!" << std::endl;
        return -1;
    }
    */
    // ---------------- Other cam ----------------

    // Start slam
    cv::Mat img, depth;
    double timestamp, timestamp_imu;

    std::atomic<bool> running(true);
    std::thread imu_thread(imu_handler, std::ref(*slam), std::ref(*cap), std::ref(running));

    slam->startup(false);
    slam->load_map_database(map_path);
    slam->disable_mapping_module();

    int skip_count = 0;

    UDPMessage msg;
    UDPSender sender("localhost", 6969);

    // TODO send intrinsic of proj

    while (true)
    {

        // ---------------- Realsense ----------------

        cap->captureImages(img, depth, timestamp);
        // cap->captureImages(img, timestamp);
        // bool img_received = cap->tryCaptureImages(img, depth, timestamp);

        // ---------------- Realsense ----------------


        // ---------------- Other cam ----------------
        /*
        cap_cv >> img;
        timestamp = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();

        slam.feed_monocular_frame(img, timestamp);
        */
        // ---------------- Other cam ----------------

        // slam->feed_monocular_frame(img, timestamp);
        slam->feed_RGBD_frame(img, depth, timestamp);

        // if (skip_count > 10) {
        //     slam->feed_RGBD_frame(img, depth, timestamp);
        //     skip_count = 0;
        // }
        // skip_count++;


        // Show camera img
        cv::imshow("img", img);
        auto key = cv::pollKey();
        if (key == 'q') break;        
    }

    // slam.shutdown();
    slam->shutdown();

    viewer->request_terminate();
    viewer_thread.join();

    running = false;
    imu_thread.join();

    cap->stop();

    return 0;
}
