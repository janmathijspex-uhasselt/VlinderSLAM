#include <iostream>
#include <thread>
#include <filesystem>

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
#include "PoseIO.h"

using namespace std;

// ---------------------------- SE3 twist functions -----------------------------

std::vector<double> compute_SE3_twist(const Eigen::Matrix4d &T_prev, const Eigen::Matrix4d &T_curr) {
    std::vector<double> xi(6, 0.0);

    // Relatieve transformatie
    Eigen::Matrix4d T_rel = T_prev.inverse() * T_curr;
    Eigen::Matrix3d R = T_rel.block<3,3>(0,0);
    Eigen::Vector3d t = T_rel.block<3,1>(0,3);

    // Translatie
    xi[0] = t(0);
    xi[1] = t(1);
    xi[2] = t(2);

    // Rotatie vector (axis-angle)
    double cos_theta = (R.trace() - 1.0) / 2.0;
    cos_theta = std::clamp(cos_theta, -1.0, 1.0); // clamp voor numerieke stabiliteit
    double theta = std::acos(cos_theta);

    Eigen::Vector3d w;
    if (theta < 1e-12) {
        w.setZero();
    } else {
        w << R(2,1) - R(1,2),
            R(0,2) - R(2,0),
            R(1,0) - R(0,1);
        w *= 0.5 / std::sin(theta) * theta;
    }

    xi[3] = w(0);
    xi[4] = w(1);
    xi[5] = w(2);

    return xi;
}

bool is_within_threshold(const std::vector<double> &xi, const std::vector<double> &xi_threshold) {
    if (xi.size() != xi_threshold.size()) {
        std::cerr << "Error: vector sizes do not match!\n";
        return false;
    }

    for (size_t i = 0; i < xi.size(); ++i) {
        if (std::abs(xi[i]) > xi_threshold[i]) {
            return false; // een waarde overschrijdt de threshold
        }
    }
    return true; // alle waarden binnen threshold
}

// ---------------------------- SE3 twist functions -----------------------------



// ------------------------------ Helper functions ------------------------------

void convert_mat_cv2unity(Eigen::Matrix4d &pose) {
    // Conversion matrices
    Eigen::Matrix4d cv2unity = Eigen::Matrix4d::Identity();
    cv2unity(1,1) = -1.0;

    pose = cv2unity * pose * cv2unity;
}

void save_traj_TUM(const std::string& path, std::vector<Eigen::Matrix4d> cam_poses) {
    std::cout << "[save_traj] Saving trajectory to " + path << std::endl;

    std::ofstream ofs(path, std::ios::out);
    if (!ofs.is_open()) {
        spdlog::critical("cannot create a file at {}", path);
        throw std::runtime_error("cannot create a file at " + path);
    }

    int index = 0;
    for (auto cam_pose_wc : cam_poses) {
        const Eigen::Matrix3d& rot_wc = cam_pose_wc.block<3, 3>(0, 0);
        const Eigen::Vector3d& trans_wc = cam_pose_wc.block<3, 1>(0, 3);
        const Eigen::Quaterniond quat_wc = Eigen::Quaterniond(rot_wc);
        ofs << std::setprecision(15)
            << index << " " // << timestamps.at(frm_id) << " "
            << std::setprecision(9)
            << trans_wc(0) << " " << trans_wc(1) << " " << trans_wc(2) << " "
            << quat_wc.x() << " " << quat_wc.y() << " " << quat_wc.z() << " " << quat_wc.w() << std::endl;

        index++;
    }

    std::cout << "[save_traj] Trajectory saved!" << std::endl;
}

// ------------------------------ Helper functions ------------------------------



// ------------------------------ Thread functions ------------------------------

void slam_handler(stella_vslam::system &slam, std::atomic<bool>& running, std::filesystem::path &cam_calib_path, std::vector<Eigen::Matrix4d>& tum_poses) {
    // Init realsense or other camera
    // ---------------- Realsense ----------------
    DeviceFactory factory;
    auto cap = factory.createFirstDevice("RealSense2", cam_calib_path);
    if (!cap) {
        std::cerr << "Kon geen RealSense2 device aanmaken!" << std::endl;
        exit(-1);
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

    // ------------------ Video ------------------
    /*
    cv::VideoCapture cap("assets/vid_A105_2909.avi");

    if (!cap.isOpened()) {
        std::cerr << "Kan video niet openen!" << std::endl;
        return -1;
    }
    */
    // ------------------ Video ------------------

    // Init vars
    cv::Mat img, depth;
    double timestamp;

    while (running)
    {
        // Limit fps
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // ---------------- Realsense ----------------

        cap->captureImages(img, depth, timestamp);
        // cap->captureImages(img, timestamp);

        // ---------------- Realsense ----------------


        // ---------------- Other cam ----------------
        /*
        cap_cv >> img;
        timestamp = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();

        slam.feed_monocular_frame(img, timestamp);
        */
        // ---------------- Other cam ----------------

        // ------------------ Video ------------------
        /*
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        timestamp = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
        if (!cap.read(img)) {
            std::cout << "Einde van video" << std::endl;
            break;
        }
        */
        // ------------------ Video ------------------

        auto cam2slam_ptr = slam.feed_monocular_frame(img, timestamp);
        // slam->feed_RGBD_frame(img, depth, timestamp);

        if (cam2slam_ptr == nullptr) continue;

        tum_poses.push_back((*cam2slam_ptr).inverse());
    }

    cap->stop();
    slam.shutdown();
}

void data_sender_handler(stella_vslam::system &slam, std::atomic<bool>& running, std::filesystem::path &cam_calib_path, std::filesystem::path &proj_calib_path, std::filesystem::path &proj_extrinsic_calib_path) {
    cv::Mat cam2proj_cv;
    poseio::loadPose(proj_extrinsic_calib_path, cam2proj_cv);

    Eigen::Matrix4d cam2proj;
    cv::cv2eigen(cam2proj_cv, cam2proj);

    std::cout << "cam2proj:" << std::endl << cam2proj << std::endl;

    // Read proj intrinsics
    CameraCalibration proj_calib, cam_calib;
    proj_calib.loadCalibration(proj_calib_path);
    cam_calib.loadCalibration(cam_calib_path);

    // Init UDP things and send proj intrinsics
    UDPSender sender("127.0.0.1", 5005);
    UDPMessage msg;

    msg.set_flag('i');
    msg.set_intrinsic_data(proj_calib, true);
    sender.send_data(msg);

    // std::this_thread::sleep_for(std::chrono::seconds(1));

    // msg.set_intrinsic_data(cam_calib);
    // sender.send_data(msg);

    // std::this_thread::sleep_for(std::chrono::seconds(1));

    // msg.set_flag('e');
    // msg.set_pose_data(cam2proj);
    // sender.send_data(msg);

    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Get map_publisher to access camera pose
    auto map_publisher = slam.get_map_publisher();

    msg.set_flag('p');
    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        auto slam2cam = map_publisher->get_current_cam_pose();

        auto slam2proj = cam2proj * slam2cam;
        Eigen::Matrix4d pose = slam2proj.inverse();

        convert_mat_cv2unity(pose);

        msg.set_pose_data(pose);
        sender.send_data(msg);
    }
}

void data_sender_with_SE3_handler(stella_vslam::system &slam, std::atomic<bool>& running, std::filesystem::path &cam_calib_path, std::filesystem::path &proj_calib_path, std::filesystem::path &proj_extrinsic_calib_path) {
    cv::Mat cam2proj_cv;
    poseio::loadPose(proj_extrinsic_calib_path, cam2proj_cv);

    Eigen::Matrix4d cam2proj;
    cv::cv2eigen(cam2proj_cv, cam2proj);

    std::cout << "cam2proj:" << std::endl << cam2proj << std::endl;

    // Read proj intrinsics
    CameraCalibration proj_calib, cam_calib;
    proj_calib.loadCalibration(proj_calib_path);
    cam_calib.loadCalibration(cam_calib_path);

    // Init UDP things and send proj intrinsics
    UDPSender sender("127.0.0.1", 5005);
    UDPMessage msg;

    msg.set_flag('i');
    msg.set_intrinsic_data(proj_calib, true);
    sender.send_data(msg);

    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Get map_publisher to access camera pose
    auto map_publisher = slam.get_map_publisher();

    // Start collecting poses and send data
    std::vector<double> xi_threshold;
    std::vector<double> xi_threshold_moving_to_static = {0.01, 0.01, 0.01, 0.005, 0.005, 0.005};
    std::vector<double> xi_threshold_static_to_moving = {0.05, 0.05, 0.05, 0.02, 0.02, 0.02};
    xi_threshold = xi_threshold_moving_to_static;

    Eigen::Matrix4d reference_cam_pose = Eigen::Matrix4d::Identity();
    bool set_reference = true;
    int count = 0;

    msg.set_flag('p');
    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));

        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity(); // if movement is to big keep pose at identity matrix
        pose(2,3) = 100;
        auto slam2cam = map_publisher->get_current_cam_pose();

        // TODO check amount of movement
        auto xi = compute_SE3_twist(reference_cam_pose, slam2cam);

        if (is_within_threshold(xi, xi_threshold)) {
            auto slam2proj = cam2proj * slam2cam;
            pose = slam2proj.inverse();

            convert_mat_cv2unity(pose);

            xi_threshold = xi_threshold_static_to_moving;
            if (set_reference) {
                reference_cam_pose = slam2cam;
                set_reference = false;
            }
        }
        else {
            xi_threshold = xi_threshold_moving_to_static;
            set_reference = true;

            if (count >= 30) {
                reference_cam_pose = slam2cam;
                count = 0;
            }
        }

        count++;

        msg.set_pose_data(pose);
        sender.send_data(msg);
    }
}

// ------------------------------ Thread functions ------------------------------



// ------------------------------------ Main ------------------------------------

int main(int argc, char**argv)
{
    if (argc < 7) {
        std::cerr << "Missing required parameter:" << std::endl;
        std::cerr << "Usage: ./Stella_VSLAM_Tracking vocab_path slam_config slam_map cam_calib proj_calib proj_extrinsic_calib" << std::endl;
        exit(-1);
    }

    std::filesystem::path vocab_path = argv[1];
    std::filesystem::path config_path = argv[2];
    std::filesystem::path path_to_slam_map = std::filesystem::path(argv[3]);
    std::filesystem::path cam_calib_path = std::filesystem::path(argv[4]);
    std::filesystem::path proj_calib_path = std::filesystem::path(argv[5]);
    std::filesystem::path proj_extrinsic_calib_path = std::filesystem::path(argv[6]);

    auto config = std::make_shared<stella_vslam::config>(config_path);

    auto slam = std::make_shared<stella_vslam::system>(config, vocab_path);

    std::vector<Eigen::Matrix4d> tum_poses;

    // Start SLAM
    spdlog::set_level(spdlog::level::off);
    slam->startup(false);
    slam->load_map_database(path_to_slam_map);
    slam->disable_mapping_module();

    // Viewer
    auto viewer = std::make_shared<pangolin_viewer::viewer>(
        stella_vslam::util::yaml_optional_ref(config->yaml_node_, "PangolinViewer"),
        slam,
        slam->get_frame_publisher(),
        slam->get_map_publisher());

    // IMU and data sender thread
    std::atomic<bool> running(true);

    std::thread slam_thread(slam_handler, std::ref(*slam), std::ref(running), std::ref(cam_calib_path), std::ref(tum_poses));

    std::thread data_sender_thread(data_sender_handler, std::ref(*slam), std::ref(running), std::ref(cam_calib_path), std::ref(proj_calib_path), std::ref(proj_extrinsic_calib_path));
    // std::thread data_sender_with_SE3_thread(data_sender_with_SE3_handler, std::ref(*slam), std::ref(running), std::ref(cam_calib_path), std::ref(proj_calib_path), std::ref(proj_extrinsic_calib_path));

    viewer->run();

    save_traj_TUM("assets/traj_TUM.txt", tum_poses);

    running = false;

    slam_thread.join();

    data_sender_thread.join();
    // data_sender_with_SE3_thread.join();

    return 0;
}

// ------------------------------------ Main ------------------------------------
