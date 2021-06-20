/*
 * @Description: kitti数据读取与发布
 * @Author: Jiang Jingwen
 * @Date: 2021-06-16 13:50:45
 */
// common
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <eigen3/Eigen/Dense>
// image
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
// point cloud
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
// odometry
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

std::vector<float> read_lidar_data(const std::string lidar_data_path) { 
    // https://blog.csdn.net/alex1997222/article/details/78976154
    std::ifstream lidar_data_file(lidar_data_path, std::ifstream::in | std::ifstream::binary);
    // seekp 函数用于已经打开要进行输出的文件，而 seekg 函数则用于已经打开要进行输入的文件。可以将 "p" 理解为 "put"，将 "g" 理解为 "get"
    // 第一个实参是一个long类型的整数，表示文件中的偏移量。
    // 第二个实参称为模式标志，它指定从哪里计算偏移量
    // ios::beg	从文件头开始计算偏移量
    // ios::end	从文件末尾开始计算偏移量
    // ios::cur	从当前位置开始计算偏移量
    lidar_data_file.seekg(0, std::ios::end);  //追溯到流的尾部
    // tellp 用于返回写入位置，tellg 则用于返回读取位置
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float); //获取流的长度
    lidar_data_file.seekg(0, std::ios::beg);  //回到流的头部

    std::vector<float> lidar_data_buffer(num_elements); 
    // 函数原型istream& read (char* s, streamsize n);  //用来暂存内容的数组(必须是char*型), 以及流的长度
    lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements * sizeof(float));
    return lidar_data_buffer;
}



int main(int argc, char** argv){
    ros::init(argc, argv, "kitti_helper");
    ros::NodeHandle n("~");

    // 读取launch文件中的参数：数据文件夹位置，是否输出rosbag以及存放位置和延迟（频率）
    std::string dataset_folder, sequence_number, output_bag_file;
    n.getParam("dataset_folder", dataset_folder);
    n.getParam("sequence_number", sequence_number);
    std::cout << "Reading sequence " << sequence_number << " from " << dataset_folder << '\n';
    bool to_bag;
    n.getParam("to_bag", to_bag);
    if (to_bag)
        n.getParam("output_bag_file", output_bag_file);
    int publish_delay;
    n.getParam("publish_delay", publish_delay);
    publish_delay = publish_delay <= 0 ? 1 : publish_delay;

    // 定义点云发布者
    ros::Publisher pub_laser_cloud = n.advertise<sensor_msgs::PointCloud2>("/velodyne_points", 2);

    // 定义图片发布者
    image_transport::ImageTransport it(n);
    image_transport::Publisher pub_image_left = it.advertise("/image_left", 2);
    image_transport::Publisher pub_image_right = it.advertise("/image_right", 2);
    
    // 定义里程计真值发布者
    ros::Publisher pubOdomGT = n.advertise<nav_msgs::Odometry> ("/odometry_gt", 5);
    nav_msgs::Odometry odomGT;
    odomGT.header.frame_id = "/camera_init";
    odomGT.child_frame_id = "/ground_truth";

    // 定义路径真值发布者
    ros::Publisher pubPathGT = n.advertise<nav_msgs::Path> ("/path_gt", 5);
    nav_msgs::Path pathGT;
    pathGT.header.frame_id = "/camera_init";

    // 时间信息
    std::string timestamp_path = "sequences/" + sequence_number + "/times.txt";
    std::ifstream timestamp_file(dataset_folder + timestamp_path, std::ifstream::in);

    // 真值信息
    std::string ground_truth_path = "poses/" + sequence_number + ".txt";
    std::ifstream ground_truth_file(dataset_folder + ground_truth_path, std::ifstream::in);

    rosbag::Bag bag_out;
    if (to_bag)
        bag_out.open(output_bag_file, rosbag::bagmode::Write);

    // kitti的训练集真值pose的坐标系和点云的坐标系不相同, 真值z向前，x向右，y向下
    Eigen::Matrix3d R_transform;
    R_transform << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    Eigen::Quaterniond q_transform(R_transform);

    std::string line;
    std::size_t line_num = 0;
    
    // 定义发布频率
    ros::Rate r(10.0 / publish_delay);

    while (std::getline(timestamp_file, line) && ros::ok()) 
    {   
        // 读取图片数据
        float timestamp = stof(line);
        std::stringstream left_image_path, right_image_path;
        left_image_path << dataset_folder << "sequences/" + sequence_number + "/image_0/" << std::setfill('0') << std::setw(6) << line_num << ".png";
        cv::Mat left_image = cv::imread(left_image_path.str(), CV_LOAD_IMAGE_GRAYSCALE);
        right_image_path << dataset_folder << "sequences/" + sequence_number + "/image_1/" << std::setfill('0') << std::setw(6) << line_num << ".png";
        cv::Mat right_image = cv::imread(left_image_path.str(), CV_LOAD_IMAGE_GRAYSCALE);
        
        // 读取真值数据
        std::getline(ground_truth_file, line);
        std::stringstream pose_stream(line);
        std::string s;
        Eigen::Matrix<double, 3, 4> gt_pose;
        for (std::size_t i = 0; i < 3; ++i)
        {
            for (std::size_t j = 0; j < 4; ++j)
            {
                std::getline(pose_stream, s, ' ');
                gt_pose(i, j) = stof(s);
            }
        }

        // 旋转, 其实更准确的应该使用KITTI提供的左相机到Lidar的标定参数进行变换
        Eigen::Quaterniond q_w_i(gt_pose.topLeftCorner<3, 3>());
        Eigen::Quaterniond q = q_transform * q_w_i;
        q.normalize();
        // 平移
        Eigen::Vector3d t = q_transform * gt_pose.topRightCorner<3, 1>();

        odomGT.header.stamp = ros::Time().fromSec(timestamp);
        odomGT.pose.pose.orientation.x = q.x();
        odomGT.pose.pose.orientation.y = q.y();
        odomGT.pose.pose.orientation.z = q.z();
        odomGT.pose.pose.orientation.w = q.w();
        odomGT.pose.pose.position.x = t(0);
        odomGT.pose.pose.position.y = t(1);
        odomGT.pose.pose.position.z = t(2);
        pubOdomGT.publish(odomGT);
        /* 
        PoseStamped.msg
            #定义有时空基准的位姿
            #文件位置：geometry_msgs/PoseStamped.msg

            Header header
            Pose pose
        */
        geometry_msgs::PoseStamped poseGT;
        poseGT.header = odomGT.header;
        poseGT.pose = odomGT.pose.pose;
        pathGT.header.stamp = odomGT.header.stamp;
        pathGT.poses.push_back(poseGT);
        pubPathGT.publish(pathGT);

        // 读取点云数据，转换为ROS格式并发布
        std::stringstream lidar_data_path;
        lidar_data_path << dataset_folder << "sequences/" + sequence_number + "/velodyne/" 
                        << std::setfill('0') << std::setw(6) << line_num << ".bin";
        std::vector<float> lidar_data = read_lidar_data(lidar_data_path.str());
        std::cout << "totally " << lidar_data.size() / 4.0 << " points in this lidar frame \n";

        std::vector<Eigen::Vector3d> lidar_points; // lidar的x y z 坐标
        std::vector<float> lidar_intensities; // lidar的强度
        pcl::PointCloud<pcl::PointXYZI> laser_cloud; // 含强度点云信息

        for (std::size_t i = 0; i < lidar_data.size(); i += 4)
        {
            lidar_points.emplace_back(lidar_data[i], lidar_data[i+1], lidar_data[i+2]);
            lidar_intensities.push_back(lidar_data[i+3]);

            pcl::PointXYZI point;
            point.x = lidar_data[i];
            point.y = lidar_data[i + 1];
            point.z = lidar_data[i + 2];
            point.intensity = lidar_data[i + 3];
            laser_cloud.push_back(point);
        }

        sensor_msgs::PointCloud2 laser_cloud_msg;
        pcl::toROSMsg(laser_cloud, laser_cloud_msg);
        laser_cloud_msg.header.stamp = ros::Time().fromSec(timestamp);
        laser_cloud_msg.header.frame_id = "/camera_init";
        pub_laser_cloud.publish(laser_cloud_msg);

        // 将图像数据转换为ROS格式并发布
        // ROS header 该消息的encoding以及 OpenCV的Mat格式的图像 https://blog.csdn.net/bigdog_1027/article/details/79090571
        sensor_msgs::ImagePtr image_left_msg = cv_bridge::CvImage(laser_cloud_msg.header, "mono8", left_image).toImageMsg();
        sensor_msgs::ImagePtr image_right_msg = cv_bridge::CvImage(laser_cloud_msg.header, "mono8", right_image).toImageMsg();
        pub_image_left.publish(image_left_msg);
        pub_image_right.publish(image_right_msg);

        // 将数据写出ROSBag包中
        if (to_bag)
        {
            bag_out.write("/image_left", ros::Time::now(), image_left_msg);
            bag_out.write("/image_right", ros::Time::now(), image_right_msg);
            bag_out.write("/velodyne_points", ros::Time::now(), laser_cloud_msg);
            bag_out.write("/path_gt", ros::Time::now(), pathGT);
            bag_out.write("/odometry_gt", ros::Time::now(), odomGT);
        }

        line_num ++;
        r.sleep();
    }

    bag_out.close();
    std::cout << "Done \n";

    return 0;
}




