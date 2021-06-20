/*
 * @Description: 基于前述的4种feature进行帧与帧的点云特征配准，即简单的Lidar Odometry
 * @Author: Jiang Jingwen
 * @Date: 2021-06-16 13:50:45
 */

// common
#include <cmath>
#include <mutex>
#include <queue>
#include <eigen3/Eigen/Dense>
// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
// ROS
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"

#define DISTORTION 0

int corner_correspondence = 0, plane_correspondence = 0;

//扫描周期
constexpr double SCAN_PERIOD = 0.1;
//后面要进行距离比较的参数
constexpr double DISTANCE_SQ_THRESHOLD = 25;
//找点进行匹配优化时的线数距离(13线-10线>2.5就break介样用)
constexpr double NEARBY_SCAN = 2.5;

//多少Frame向mapping发送数据，实际由于主函数效果，是4帧一发
int skipFrameNum = 5;
//目的是在订阅发布，时间戳，互斥锁初始化后输出一下Initialization finished
bool systemInited = false;

//时间戳
double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;

//关于上一帧的KD树
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

//pcl保存形式的输入，角点，降采样角点，面点，降采样面点，上一帧角点，上一帧面点，全部点
pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

//存储上一帧的特征点数量
int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// 点云特征匹配时的优化变量
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};

// 下面的2个分别是优化变量para_q和para_t的映射：表示的是两个world坐标系下的位姿P之间的增量，例如△P = P0.inverse() * P1
Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);  //  Map相当于引用
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);


//定义ros格式的订阅内容
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;

//互斥锁，让订阅信息按次序进行
std::mutex mBuf;

// undistort lidar point，将该帧点云的点都变换的第一个采样点的位姿下
void TransformToStart(PointType const *const pi, PointType *const po)
{
    // interpolation ratio
    double s;
    if (DISTORTION) 
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD; // 用s求解某个点在本次scan中在的比例
    else 
        s = 1.0; // kitti点云已经去除了畸变，所以不再考虑运动补偿
    //s = 1;
    //再根据比例求解变换矩阵的变换比例，再求出推理位姿
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr); // 四元数插值
    Eigen::Vector3d t_point_last = s * t_last_curr;   // 平移量插值
    Eigen::Vector3d point(pi->x, pi->y, pi->z);       // 采样点
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;  // 将当前点变换的第一个采样点的位姿下,去除畸变

    //输出一下
    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

void TransformToEnd(PointType const *const pi, PointType *const po)
{
    pcl::PointXYZI un_point_tmp;
    TransformToStart(pi, &un_point_tmp); 

    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
    Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);

    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();

    //Remove distortion time info
    po->intensity = int(pi->intensity);
}

//订阅信息并且锁死，保证不乱序
void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;

    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);

    printf("Mapping %d Hz \n", 10 / skipFrameNum);


    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2> ("/laser_cloud_sharp", 100, laserCloudSharpHandler);

    ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);

    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);

    ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);

    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);

    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);

    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);

    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);
    
    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);

    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    //定义路径，用于保存帧的位置，发布于pubLaserPath
    nav_msgs::Path laserPath;

    //用于计算处理的帧数，每有skipFrameNum个帧处理了，就向mapping发数据
    int frameCount = 0;

    //设置一下ros频率
    ros::Rate rate(100);

    while (ros::ok())
    {
        //到达这里启动数据节点与ros::spin不同，到达ros::spin程序不再向下运行，只按频率进行节点，这里会继续向下
        ros::spinOnce();
        //如果订阅的东西应有尽有
        if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
            !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
            !fullPointsBuf.empty())
            {
                // 获取时间戳
                timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
                timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
                timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
                timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
                timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();
                //时间不同就不同步报错
                if (timeCornerPointsSharp != timeLaserCloudFullRes ||
                    timeCornerPointsLessSharp != timeLaserCloudFullRes ||
                    timeSurfPointsFlat != timeLaserCloudFullRes ||
                    timeSurfPointsLessFlat != timeLaserCloudFullRes)
                {
                    printf("unsync messeage!");
                    ROS_BREAK();
                }
                
                // 从ROS格式转换为PCL格式
                mBuf.lock();
                cornerPointsSharp->clear();
                pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
                cornerSharpBuf.pop();

                cornerPointsLessSharp->clear();
                pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
                cornerLessSharpBuf.pop();

                surfPointsFlat->clear();
                pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
                surfFlatBuf.pop();

                surfPointsLessFlat->clear();
                pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
                surfLessFlatBuf.pop();

                laserCloudFullRes->clear();
                pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
                fullPointsBuf.pop();
                mBuf.unlock();

                TicToc t_whole;
                // initializing
                if (!systemInited)
                {
                    systemInited = true;
                    std::cout << "Initialization finished \n";
                }
                else 
                {
                    //记录点数
                    int cornerPointsSharpNum = cornerPointsSharp->points.size();
                    int surfPointsFlatNum = surfPointsFlat->points.size();

                    TicToc t_opt;

                    // 点到线以及点到面的ICP，迭代2次
                    for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)
                    {
                        //匹配数量
                        corner_correspondence = 0;
                        plane_correspondence = 0;

                        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                        //为Eigen的表示实现四元数局部参数
		                //输入顺序为[w，x，y，z]
                        ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();

                        ceres::Problem::Options problem_options; // 问题选项

                        ceres::Problem problem(problem_options);
                        problem.AddParameterBlock(para_q, 4, q_parameterization);
                        problem.AddParameterBlock(para_t,3);

                        pcl::PointXYZI pointSel;
                        std::vector<int> pointSearchInd;
                        std::vector<float> pointSearchSqDis;

                        TicToc t_data;
                        // find correspondence for corner features
                        // 基于最近邻（只找2个最近邻点）原理建立corner特征点之间关联
                        for (int i = 0; i < cornerPointsSharpNum; ++i) 
                        {
                            // 将当前帧的corner_sharp特征点O_cur，从当前帧的Lidar坐标系下变换到上一帧的Lidar坐标系下（记为点O，注意与前面的点O_cur不同），
                            // 以利于寻找corner特征点的correspondence
                            TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);
                            // 使用KD-tree求解相对上一帧里点云，pointSel和他们的距离，返回一个最近点的点云线数pointSearchInd和距离pointSearchSqDis
			                // 可以看https://zhuanlan.zhihu.com/p/112246942
                            // kdtree中的点云是上一帧的corner_less_sharp，所以这是在上一帧的corner_less_sharp中寻找当前帧corner_sharp特征点O的最近邻点（记为A）
                            kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                            // closestPointInd 是离pointSel最近点的索引
                            // minPointInd2 是最近点以上离pointSel最近点的索引
                            int closestPointInd = -1, minPointInd2 = -1;

                            if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) // 如果最近邻的corner特征点之间距离平方小于阈值，则最近邻点A有效
                            {   
                                // 最近点索引
                                closestPointInd = pointSearchInd[0];
                                // 最近点所在线号
                                int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);
                                // 最短距离之后更新
                                double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
                                // 寻找点O的另外一个最近邻的点（记为点B） in the direction of increasing scan line
                                // laserCloudCornerLast 来自上一帧的corner_less_sharp特征点,由于提取特征时是
                                // 按照scan的顺序提取的，所以laserCloudCornerLast中的点也是按照scanID递增的顺序存放的
                                // 找临近线的点B，该点线数不能小于等于A线数
                                for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                                {
                                    // if in the same scan line, continue
                                    if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)// intensity整数部分存放的是scanID
                                        continue;
                                    
                                    // if not in nearby scans, end the loop
                                    if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                        break;

                                    double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                        (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);
                                    if (pointSqDis < minPointSqDis2)
                                    {
                                        // find nearer point
                                        minPointSqDis2 = pointSqDis; // 第二个最近邻点有效，更新点B
                                        minPointInd2 = j;
                                    }
                                }
                                
                                // 即特征点O的两个最近邻点A和B都有效
                                if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
                                {
                                    Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                       cornerPointsSharp->points[i].y,
                                                       cornerPointsSharp->points[i].z);
                                    Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                                laserCloudCornerLast->points[closestPointInd].y,
                                                                laserCloudCornerLast->points[closestPointInd].z);
                                    Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                                laserCloudCornerLast->points[minPointInd2].y,
                                                                laserCloudCornerLast->points[minPointInd2].z);

                                    double s;// 运动补偿系数，kitti数据集的点云已经被补偿过，所以s = 1.0
                                    if (DISTORTION)
                                        s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
                                    else
                                        s = 1.0;
                                    
                                    // 用点O，A，B构造点到线的距离的残差项，注意这三个点都是在上一帧的Lidar坐标系下，即，残差 = 点O到直线AB的距离
                                    // 具体到介绍lidarFactor.cpp时再说明该残差的具体计算方法

                                    ceres::CostFunction* cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                    corner_correspondence++;
                                }

                            }

                        }

                        // find correspondence for plane features
                        // 下面说的点符号与上述相同
                        // 与上面的建立corner特征点之间的关联类似，寻找平面特征点O的最近邻点ABC（只找3个最近邻点），
                        // 即基于最近邻原理建立surf特征点之间的关联，find correspondence for plane features
                        for (int i = 0; i < surfPointsFlatNum; ++i)
                        {
                            TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
                            kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                            int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                            if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)// 找到的最近邻点A有效
                            {
                                closestPointInd = pointSearchInd[0];
                                // get closest point's scan ID
                                int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                                double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;
                                // search in the direction of increasing scan line
                                for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                                {
                                    // if not in nearby scans, end the loop
                                    if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                        break;
                                    double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                        (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);
                                    
                                    // if in the same or lower scan line
                                    if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                                    {
                                        minPointSqDis2 = pointSqDis;// 找到的第2个最近邻点有效，更新点B，注意如果scanID准确的话，一般点A和点B的scanID相同
                                        minPointInd2 = j;
                                    }

                                    // if in the higher scan line
                                    else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                                    {
                                        minPointSqDis3 = pointSqDis;// 找到的第3个最近邻点有效，更新点C，注意如果scanID准确的话，一般点A和点B的scanID相同,且与点C的scanID不同，与LOAM的paper叙述一致
                                        minPointInd3 = j;
                                    }
                                }

                                if (minPointInd2 >= 0 && minPointInd3 >= 0)// 如果三个最近邻点都有效
                                {

                                    Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                        surfPointsFlat->points[i].y,
                                        surfPointsFlat->points[i].z);
                                    Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                        laserCloudSurfLast->points[closestPointInd].y,
                                        laserCloudSurfLast->points[closestPointInd].z);
                                    Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                        laserCloudSurfLast->points[minPointInd2].y,
                                        laserCloudSurfLast->points[minPointInd2].z);
                                    Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                        laserCloudSurfLast->points[minPointInd3].y,
                                        laserCloudSurfLast->points[minPointInd3].z);

                                    double s;
                                    if (DISTORTION)
                                        s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
                                    else
                                        s = 1.0;
                                    // 用点O，A，B，C构造点到面的距离的残差项，注意这三个点都是在上一帧的Lidar坐标系下，即，残差 = 点O到平面ABC的距离
                                    // 同样的，具体到介绍lidarFactor.cpp时再说明该残差的具体计算方法
                                    ceres::CostFunction* cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                    plane_correspondence++;
                                }
                            }
                        }

                        //printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
                        printf("data association time %f ms \n", t_data.toc());
                        
                        if ((corner_correspondence + plane_correspondence) < 10)
                        {
                            printf("less correspondence! *************************************************\n");
                        }

                        // ceres优化
                        TicToc t_solver;
                        ceres::Solver::Options options; // solver选项
                        options.linear_solver_type = ceres::DENSE_QR;
                        options.max_num_iterations = 4;
                        options.minimizer_progress_to_stdout = false;
                        ceres::Solver::Summary summary;
                        // 基于构建的所有残差项，求解最优的当前帧位姿与上一帧位姿的位姿增量：para_q和para_t
                        ceres::Solve(options, &problem, &summary);
                        printf("solver time %f ms \n", t_solver.toc());
                    }
                    printf("optimization twice time %f \n", t_opt.toc());

                    // 用最新计算出的位姿增量，更新上一帧的位姿，得到当前帧的位姿，注意这里说的位姿都指的是世界坐标系下的位姿
                    t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                    q_w_curr = q_w_curr * q_last_curr;
                }

                TicToc t_pub;
                // publish odometry
                nav_msgs::Odometry laserOdometry;
                laserOdometry.header.frame_id = "/camera_init";
                laserOdometry.child_frame_id = "/laser_odom";
                laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserOdometry.pose.pose.orientation.x = q_w_curr.x();
                laserOdometry.pose.pose.orientation.y = q_w_curr.y();
                laserOdometry.pose.pose.orientation.z = q_w_curr.z();
                laserOdometry.pose.pose.orientation.w = q_w_curr.w();
                laserOdometry.pose.pose.position.x = t_w_curr.x();
                laserOdometry.pose.pose.position.y = t_w_curr.y();
                laserOdometry.pose.pose.position.z = t_w_curr.z();
                pubLaserOdometry.publish(laserOdometry);

                geometry_msgs::PoseStamped laserPose;
                laserPose.header = laserOdometry.header;
                laserPose.pose = laserOdometry.pose.pose;
                laserPath.header.stamp = laserOdometry.header.stamp;
                laserPath.poses.push_back(laserPose);
                laserPath.header.frame_id = "/camera_init";
                pubLaserPath.publish(laserPath);

                // transform corner features and plane features to the scan end point
                if (0)
                {
                    int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
                    for (int i = 0; i < cornerPointsLessSharpNum; i++)
                    {
                        TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
                    }
                    int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
                    for (int i = 0; i < surfPointsLessFlatNum; i++)
                    {
                        TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
                    }
                    int laserCloudFullResNum = laserCloudFullRes->points.size();
                    for (int i = 0; i < laserCloudFullResNum; i++)
                    {
                        TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
                    }
                }


                // 用cornerPointsLessSharp和surfPointsLessFlat 更新 laserCloudCornerLast和laserCloudSurfLast以及相应的kdtree，
                // 为下一次点云特征匹配提供target
                pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
                cornerPointsLessSharp = laserCloudCornerLast;
                laserCloudCornerLast = laserCloudTemp;

                laserCloudTemp = surfPointsLessFlat;
                surfPointsLessFlat = laserCloudSurfLast;
                laserCloudSurfLast = laserCloudTemp;

                laserCloudCornerLastNum = laserCloudCornerLast->points.size();
                laserCloudSurfLastNum = laserCloudSurfLast->points.size();

                // std::cout << "the size of corner last is " << laserCloudCornerLastNum << ", and the size of surf last is " << laserCloudSurfLastNum << '\n';
                
                // 设置kd树输入点云
                kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
                kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

                //满足skipFrameNum帧数则发送数据
                if (frameCount % skipFrameNum == 0)
                {
                    frameCount = 0;

                    sensor_msgs::PointCloud2 laserCloudCornerLast2;
                    pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
                    laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                    laserCloudCornerLast2.header.frame_id = "/camera";
                    pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

                    sensor_msgs::PointCloud2 laserCloudSurfLast2;
                    pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
                    laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                    laserCloudSurfLast2.header.frame_id = "/camera";
                    pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

                    sensor_msgs::PointCloud2 laserCloudFullRes3;
                    pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
                    laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                    laserCloudFullRes3.header.frame_id = "/camera";
                    pubLaserCloudFullRes.publish(laserCloudFullRes3);
                }
                printf("publication time %f ms \n", t_pub.toc());
                printf("whole laserOdometry time %f ms \n \n", t_whole.toc());
                if(t_whole.toc() > 100)
                    ROS_WARN("odometry process over 100ms");

                frameCount++;
            }
            rate.sleep();
    }
    return 0;
}