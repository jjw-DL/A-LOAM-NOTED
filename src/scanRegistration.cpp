/*
 * @Description: lidar 点面特征提取
 * @Author: Jiang Jingwen
 * @Date: 2021-06-16 13:50:45
 */

// common
#include <cmath>
#include <vector>
#include <string>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
// OPENCV
#include <opencv/cv.h>
// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
// ROS
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::atan2;
using std::cos;
using std::sin;

//扫描周期, velodyne频率10Hz，周期0.1s
const double scanPeriod = 0.1;
//弃用前systemDelay帧初始数据
const int systemDelay = 0;
//systemInitCount用于计数过了多少帧
//超过systemDelay后，systemInited为true即初始化完成
int systemInitCount = 0;
bool systemInited = false;
//激光雷达线数初始化为0
int N_SCANS = 0;
//点云曲率, 400000为一帧点云中点的最大数量
float cloudCurvature[400000];
//曲率点对应的序号
int cloudSortInd[400000];
//点是否筛选过标志：0-未筛选过，1-筛选过
int cloudNeighborPicked[400000];
//点分类标号:2-代表曲率很大，1-代表曲率比较大, 0-曲率比较小, -1-代表曲率很小(其中1包含了2, 0包含了1, 0和1构成了点云全部的点)
int cloudLabel[400000];
//两点曲率比较
bool comp (int i,int j) { return (cloudCurvature[i] < cloudCurvature[j]); }
// 设置发布内容，整体点云，角点，降采样角点，面点，降采样面点，剔除点
ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubRemovePoints;
//ros形式的一线扫描
std::vector<ros::Publisher> pubEachScan;
//是否发布每行Scan
bool PUB_EACH_LINE = false;
//根据距离去除过远的点，距离的参数
double MINIMUM_RANGE = 0.1;


//去除过远点 使用template进行兼容
template<typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                             pcl::PointCloud<PointT> &cloud_out, float thres)
{
    // 统一header(时间戳)和size
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;
    //逐点距离比较
    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    //有点被剔除时，size改变
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    //数据行数，默认1为无组织的数据
    cloud_out.height = 1;
    //可以理解成点数
    cloud_out.width = static_cast<uint32_t>(j);
    //点数是否有限
    cloud_out.is_dense = true;
}

//订阅点云句柄
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    //是否剔除前systemDelay帧
    if (!systemInited)
    {
        systemInitCount++;
        if (systemInitCount >= systemDelay) 
        {
            systemInited = true;
        }
        else
            return;
    }

    //registration计时
    TicToc t_whole;
    //计算曲率前的预处理计时
    TicToc t_prepare;
    //记录每个scan有曲率的点的开始和结束索引
    std::vector<int> scanStartInd(N_SCANS, 0);
    std::vector<int> scanEndInd(N_SCANS, 0);

    //命名一个pcl形式的输入点云
    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    //把ros包的点云转化为pcl形式
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    std::vector<int> indices;
    //这个函数目的是去除过远点，第一个参数是输入，第二个参数是输出，第三个参数是列表保存输出的点在输入里的位置
    //输出里的第i个点，是输入里的第indices[i]个点，就是
    //cloud_out.points[i] = cloud_in.points[indices[i]]
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    //引用上文作者写的去除函数
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);

    //该次sweep的点数
    int cloudSize = laserCloudIn.points.size();
    //每次扫描是一条线，看作者的数据集激光x向前，y向左，那么下面就是线一端到另一端
    //atan2的输出为-pi到pi(PS:atan输出为-pi/2到pi/2)
    //计算旋转角时取负号是因为velodyne是顺时针旋转
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y, laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;

    //激光间距收束到1pi~3pi
    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }
    //printf("end Ori %f\n", endOri);

    //过半记录标志
    bool halfPassed = false;
    //记录总点数
    int count = cloudSize;
    PointType point;

    //按线数保存的点云集合
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);

    //循环对每个点进行以下操作
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;

        //求仰角atan输出为-pi/2到pi/2，实际看scanID应该每条线之间差距是2度
        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scanID = 0;

        //根据不同线数使用不同参数对每个点对应的第几根激光线进行判断
        if (N_SCANS == 16)
        {
            scanID = int((angle + 15) / 2 + 0.5);
            //如果判定线在16以上或是负数则忽视该点回到上面for循环
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 32)
        {
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 64)
        {
            if (angle >= -8.83) 
            {
                scanID = int((2 - angle) * 3.0 + 0.5);
            }
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);
            
            // use [0 50]  > 50 remove outlie
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        //只有16,32,64线
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        //printf("angle %f scanID %d \n", angle, scanID);

        float ori = -atan2(point.y, point.x);
        //根据扫描线是否旋转过半选择与起始位置还是终止位置进行差值计算，从而进行补偿
        if (!halfPassed)
        {
            //确保-pi/2 < ori - startOri < 3*pi/2
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }
 
            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
            //确保-3*pi/2 < ori - endOri < pi/2
            else
            {
                ori += 2 * M_PI;
                if (ori < endOri - M_PI * 3 / 2)
                {
                    ori += 2 * M_PI;
                }
                else if (ori > endOri + M_PI / 2)
                {
                    ori -= 2 * M_PI;
                }

            }
        }

        //看看旋转多少了，记录比例relTime
        float relTime = (ori - startOri) / (endOri - startOri);
        //第几根线和本线时间到多少记录在point.intensity
        point.intensity = scanID + scanPeriod * relTime;
        //按线分类保存
        laserCloudScans[scanID].push_back(point);
    }

    cloudSize = count;
    printf("points size %d \n", cloudSize);

    //也就是把所有线保存在laserCloud一个数据集合里，把每条线的第五个和倒数第五个位置反馈给scanStartInd和scanEndInd
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < N_SCANS; i++)
    {
        scanStartInd[i] = laserCloud->size() + 5;
        *laserCloud += laserCloudScans[i];
        scanEndInd[i] = laserCloud->size() - 6;
    }

    //预处理部分终于完成了
    printf("prepare time %f \n", t_prepare.toc());

    //十点求曲率，自然是在一条线上的十个点
    for (int i = 5; i < cloudSize - 5; i++)
    {
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
        //曲率，序号，是否筛过标志位，曲率分类
        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        cloudSortInd[i] = i;
        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;
    }
    
    //计时用
    TicToc t_pts;
    //角点，降采样角点，面点，降采样面点
    pcl::PointCloud<PointType> cornerPointsSharp;
    pcl::PointCloud<PointType> cornerPointsLessSharp;
    pcl::PointCloud<PointType> surfPointsFlat;
    pcl::PointCloud<PointType> surfPointsLessFlat;

    float t_q_sort = 0;

    for (int i = 0; i < N_SCANS; i++) 
    {
        //点数小于6就退出
        if(scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        //将每个scan的曲率点分成6等份处理,确保周围都有点被选作特征点,或者说每两行都有
        for (int j = 0; j < 6; j++)
        {
            //六等份起点：sp = scanStartInd + (scanEndInd - scanStartInd)*j/6
            //六等份终点：ep = scanStartInd - 1 + (scanEndInd - scanStartInd)*(j+1)/6
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            TicToc t_tmp;
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
            t_q_sort += t_tmp.toc();

            //挑选每个分段的曲率很大和比较大的点
            int largestPickedNum = 0;

            for (int k = ep; k >= sp; k--)
            {
                int ind = cloudSortInd[k];
                //如果筛选标志为0，并且曲率较大
                if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > 0.1)
                {
                    //则曲率大的点记数一下
                    largestPickedNum++;
                    //少于等于两个（但如果有更多的这俩cloudLabel[ind] = 2;就不更新了）
                    if (largestPickedNum <= 2)
                    {
                        cloudLabel[ind] = 2;
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    //保存20个角点
                    else if (largestPickedNum <= 20)
                    {
                        cloudLabel[ind] = 1;
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    //多了就不要了
                    else
                    {
                        break;
                    }
                    //标志位一改
                    cloudNeighborPicked[ind] = 1;
                    //将曲率比较大的点的前后各5个连续距离比较近的点筛选出去，防止特征点聚集，使得特征点在每个方向上尽量分布均匀
                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        //应该是决定简单计算不稳定，直接过
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }
                        //前后几个也
                        cloudNeighborPicked[ind + l] = 1;
                    }

                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }
 
                        //前后几个也
                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            //挑选每个分段的曲率很小比较小的点
            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];
                //如果曲率的确比较小，并且未被筛选出
                if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < 0.1)
                {
                    //-1代表曲率很小的点
                    cloudLabel[ind] = -1; 
                    surfPointsFlat.push_back(laserCloud->points[ind]);
                    
                    //只选最小的四个，剩下的Label==0,就都是曲率比较小的
                    smallestPickedNum++;
                    if (smallestPickedNum >= 4)
                    { 
                        break;
                    }
                    cloudNeighborPicked[ind] = 1;
                }

                //同样防止特征点聚集
                for (int l = 1; l <= 5; l++)
                {
                    float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                    float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                    float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                    if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                    {
                        break;
                    }

                    cloudNeighborPicked[ind + l] = 1;
                }

                for (int l = -1; l >= -5; l--)
                {
                    float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                    float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                    float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                    if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                    {
                        break;
                    }

                    cloudNeighborPicked[ind + l] = 1;
                }
            }
            //将剩余的点（包括之前被排除的点）全部归入平面点中less flat类别中
            for (int k = sp; k <= ep; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }

            //由于less flat点最多，对每个分段less flat的点进行体素栅格滤波
            pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
            pcl::VoxelGrid<PointType> downSizeFilter;
            downSizeFilter.setInputCloud(surfPointsLessFlatScan);
            downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
            downSizeFilter.filter(surfPointsLessFlatScanDS);

            //less flat点汇总
            surfPointsLessFlat += surfPointsLessFlatScanDS;
        }
    }

    // （求出曲率后）降采样和分类的时间
    printf("sort q time %f \n", t_q_sort);
    printf("seperate points time %f \n", t_pts.toc());

    //发布信息准备工作
    sensor_msgs::PointCloud2 laserCloudOutMsg;
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "/camera_init";
    pubLaserCloud.publish(laserCloudOutMsg);

    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "/camera_init";
    pubSurfPointsFlat.publish(surfPointsFlat2);

    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "/camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

    if(PUB_EACH_LINE)
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "/camera_init";
            pubEachScan[i].publish(scanMsg);
        }
    }

    //总时间输出
    printf("scan registration time %f ms *************\n", t_whole.toc());
    if(t_whole.toc() > 100)
        ROS_WARN("scan registration process over 100ms");
}


int main(int argc, char **argv) 
{
    ros::init(argc, argv, "scanRegistration");
    ros::NodeHandle nh;

    //参数，线数
    nh.param<int>("sacn_line", N_SCANS, 16);

    //参数，过远去除 (最后一个参数为默认值)
    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);

    printf("scan line number %d \n", N_SCANS);

     if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }

    //接收激光雷达信号
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

    //发布laserCloud，laserCloud是按线堆栈的全部点云
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

    //发布角点，降采样角点，面点，降采样面点
    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);

    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);

    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);

    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);

    //发布去除点
    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    //发布每行scan
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i < N_SCANS; i++) 
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }
    ros::spin();

    return 0;
}