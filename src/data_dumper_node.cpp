#include <iostream>
#include <fstream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ros/ros.h>

#include "tf/tf.h"
#include "tf/transform_datatypes.h"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/Image.h>
#include <lucrezio_simulation_environments/LogicalImage.h>
#include <gazebo_msgs/GetModelState.h>
#include <gazebo_msgs/GetLinkState.h>

typedef std::vector<lucrezio_simulation_environments::Model> Models;

class ImageDumper{
public:
  ImageDumper(ros::NodeHandle nh_):
    _nh(nh_),
    _logical_image_sub(_nh,"/gazebo/logical_camera_image",1),
    _depth_image_sub(_nh,"/camera/depth/image_raw",1),
    _rgb_image_sub(_nh,"/camera/rgb/image_raw", 1),
    _synchronizer(FilterSyncPolicy(10),_logical_image_sub,_depth_image_sub,_rgb_image_sub){

    _synchronizer.registerCallback(boost::bind(&ImageDumper::filterCallback, this, _1, _2, _3));

    _model_state_client = _nh.serviceClient<gazebo_msgs::GetModelState>("gazebo/get_model_state");

    _seq = 0;

    _out.open("data.txt");

    ROS_INFO("Starting data dumper node!");
  }

  void filterCallback(const lucrezio_simulation_environments::LogicalImage::ConstPtr& logical_image_msg,
                      const sensor_msgs::Image::ConstPtr& depth_image_msg,
                      const sensor_msgs::Image::ConstPtr& rgb_image_msg){

//    ROS_INFO("--------------------------");
//    ROS_INFO("Executing filter callback!");
//    ROS_INFO("--------------------------");
//    std::cerr << std::endl;

    ++_seq;

    //Extract rgb and depth image from ROS messages
    cv_bridge::CvImageConstPtr rgb_cv_ptr,depth_cv_ptr;
    try{
      rgb_cv_ptr = cv_bridge::toCvShare(rgb_image_msg);
      depth_cv_ptr = cv_bridge::toCvShare(depth_image_msg);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat rgb_image = rgb_cv_ptr->image.clone();
    int rgb_rows=rgb_image.rows;
    int rgb_cols=rgb_image.cols;
    std::string rgb_type=type2str(rgb_image.type());
//    ROS_INFO("Got %dx%d %s image",rgb_cols,rgb_rows,rgb_type.c_str());

    //save rgb image
    char rgb_filename[80];
    sprintf(rgb_filename,"rgb_image_%lu.png",_seq);
    cv::imwrite(rgb_filename,rgb_image);

    cv::Mat depth_image = depth_cv_ptr->image.clone();
    int depth_rows=depth_image.rows;
    int depth_cols=depth_image.cols;
    std::string depth_type=type2str(depth_image.type());
//    ROS_INFO("Got %dx%d %s image",depth_cols,depth_rows,depth_type.c_str());

    //save depth image
    cv::Mat temp;
    convert_32FC1_to_16UC1(temp,depth_image);
    char depth_filename[80];
    sprintf(depth_filename,"depth_image_%lu.pgm",_seq);
    cv::imwrite(depth_filename,temp);

    //save camera pose
    gazebo_msgs::GetModelState model_state;
    model_state.request.model_name = "robot";
    tf::StampedTransform robot_pose;
    if(_model_state_client.call(model_state)){
//      ROS_INFO("Received robot model state!");
      tf::poseMsgToTF(model_state.response.pose,robot_pose);
    }else
      ROS_ERROR("Failed to call service gazebo/get_model_state");

    Eigen::Isometry3f rgbd_camera_pose = Eigen::Isometry3f::Identity();
    rgbd_camera_pose.translation() = Eigen::Vector3f(0.0,0.0,0.5);
    rgbd_camera_pose.linear() = Eigen::Quaternionf(0.5,-0.5,0.5,-0.5).toRotationMatrix();
    const Eigen::Isometry3f rgbd_camera_transform=tfTransform2eigen(robot_pose)*rgbd_camera_pose;
    char rgbd_filename[80];
    sprintf(rgbd_filename,"rgbd_pose_%lu.txt",_seq);
    serializeTransform(rgbd_filename,rgbd_camera_transform);

    //save logical camera pose
    tf::StampedTransform logical_camera_pose;
    tf::poseMsgToTF(logical_image_msg->pose,logical_camera_pose);
    const Eigen::Isometry3f logical_camera_transform=tfTransform2eigen(logical_camera_pose);
    char logical_filename[80];
    sprintf(logical_filename,"logical_pose_%lu.txt",_seq);
    serializeTransform(logical_filename,logical_camera_transform);

    //save models
    const Models &models=logical_image_msg->models;
    char models_filename[80];
    sprintf(models_filename,"models_%lu.txt",_seq);
    serializeModels(models_filename,models);

    //write to output file
    _out << _seq << " ";
    _out << rgb_filename << " ";
    _out << depth_filename << " ";
    _out << rgbd_filename << " ";
    _out << logical_filename << " ";
    _out << models_filename << std::endl;

    std::cerr << ".";
  }

protected:
  ros::NodeHandle _nh;

  message_filters::Subscriber<lucrezio_simulation_environments::LogicalImage> _logical_image_sub;
  message_filters::Subscriber<sensor_msgs::Image> _depth_image_sub;
  message_filters::Subscriber<sensor_msgs::Image> _rgb_image_sub;
  typedef message_filters::sync_policies::ApproximateTime<lucrezio_simulation_environments::LogicalImage,
  sensor_msgs::Image,
  sensor_msgs::Image> FilterSyncPolicy;
  message_filters::Synchronizer<FilterSyncPolicy> _synchronizer;

  ros::ServiceClient _model_state_client;

  size_t _seq;
  std::ofstream _out;

private:
  Eigen::Isometry3f tfTransform2eigen(const tf::Transform& p){
    Eigen::Isometry3f iso;
    iso.translation().x()=p.getOrigin().x();
    iso.translation().y()=p.getOrigin().y();
    iso.translation().z()=p.getOrigin().z();
    Eigen::Quaternionf q;
    tf::Quaternion tq = p.getRotation();
    q.x()= tq.x();
    q.y()= tq.y();
    q.z()= tq.z();
    q.w()= tq.w();
    iso.linear()=q.toRotationMatrix();
    return iso;
  }

  std::string type2str(int type) {
    std::string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    switch ( depth ) {
      case CV_8U:  r = "8U"; break;
      case CV_8S:  r = "8S"; break;
      case CV_16U: r = "16U"; break;
      case CV_16S: r = "16S"; break;
      case CV_32S: r = "32S"; break;
      case CV_32F: r = "32F"; break;
      case CV_64F: r = "64F"; break;
      default:     r = "User"; break;
    }
    r += "C";
    r += (chans+'0');
    return r;
  }

  void convert_32FC1_to_16UC1(cv::Mat& dest, const cv::Mat& src, float scale = 1000.0f) {
    assert(src.type() == CV_32FC1 && "convert_32FC1_to_16UC1: source image of different type from 32FC1");
    const float* sptr = (const float*)src.data;
    int size = src.rows * src.cols;
    const float* send = sptr + size;
    dest.create(src.rows, src.cols, CV_16UC1);
    dest.setTo(cv::Scalar(0));
    unsigned short* dptr = (unsigned short*)dest.data;
    while(sptr < send) {
      if(*sptr >= 1e9f) { *dptr = 0; }
      else { *dptr = scale * (*sptr); }
      ++dptr;
      ++sptr;
    }
  }

  void serializeTransform(char* filename, const Eigen::Isometry3f &transform){
    std::ofstream data;
    data.open(filename);

    data << transform.translation().x() << " "
         << transform.translation().y() << " "
         << transform.translation().z() << " ";

    const Eigen::Matrix3f rotation = transform.linear().matrix();
    data << rotation(0,0) << " "
         << rotation(0,1) << " "
         << rotation(0,2) << " "
         << rotation(1,0) << " "
         << rotation(1,1) << " "
         << rotation(1,2) << " "
         << rotation(2,0) << " "
         << rotation(2,1) << " "
         << rotation(2,2) << std::endl;

    data.close();

  }

  void serializeModels(char* filename, const Models &models){

    std::ofstream data;
    data.open(filename);

    int num_models=models.size();
    data << num_models << std::endl;

    for(int i=0; i<num_models; ++i){
      const lucrezio_simulation_environments::Model &model = models[i];
      data << model.type << " ";
      tf::StampedTransform model_pose;
      tf::poseMsgToTF(model.pose,model_pose);
      const Eigen::Isometry3f model_transform=tfTransform2eigen(model_pose);
      data << model_transform.translation().x() << " "
           << model_transform.translation().y() << " "
           << model_transform.translation().z() << " ";

      const Eigen::Matrix3f model_rotation = model_transform.linear().matrix();
      data << model_rotation(0,0) << " "
           << model_rotation(0,1) << " "
           << model_rotation(0,2) << " "
           << model_rotation(1,0) << " "
           << model_rotation(1,1) << " "
           << model_rotation(1,2) << " "
           << model_rotation(2,0) << " "
           << model_rotation(2,1) << " "
           << model_rotation(2,2) << " ";

      data << model.min.x << " "
           << model.min.y << " "
           << model.min.z << " "
           << model.max.x << " "
           << model.max.y << " "
           << model.max.z << std::endl;

    }

    data.close();

  }

};

int main(int argc, char** argv){
  ros::init(argc, argv, "data_dumper");
  ros::NodeHandle nh;
  ImageDumper dumper(nh);

  ros::spin();

  //  ros::Rate loop_rate(1);
  //  while(ros::ok() ){
  //    ros::spinOnce();
  //    loop_rate.sleep();
  //  }

  ROS_INFO("Terminating data_dumper");

  return 0;
}
