#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <sstream>
#include <string>

const int WIDTH = 640, HEIGHT = 480, INIT_FRAME = 60;
double horizontalRatio = 0.0, verticalRatio = 0.0, mouthLength = 0.0;
bool clicked = false;
cv::CascadeClassifier faceDetector;

int main(int argc, char *argv[])
{
  faceDetector.load("resource/haarcascade_frontalface_alt2.xml");
  cv::Ptr<cv::face::Facemark> facemark = cv::face::FacemarkLBF::create();

  facemark->loadModel("resource/lbfmodel2.yaml");

  cv::VideoCapture cam(1);
  cv::Mat frame, gray;
  cv::namedWindow("win1", 1);
  cv::resizeWindow("win1", cv::Size(WIDTH, HEIGHT));

  bool loop_flag = true;
  bool ready = false;

  while (loop_flag && cam.read(frame))
  {
    cv::flip(frame, frame, 1);
    std::vector<cv::Rect> faces;
    cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    faceDetector.detectMultiScale(gray, faces, 1.1, 3, CV_HAAR_SCALE_IMAGE, cv::Size(150, 150));
    std::vector<std::vector<cv::Point2f>> landmarks;
    bool success = facemark->fit(frame, faces, landmarks);
    static bool initialized = false;
    if (success)
    {
      cv::Point2f leftEye, rightEye, nose, topNose, topMouth, bottomMouth, mouth;
      leftEye = (landmarks[0][36] + landmarks[0][39]) / 2;
      rightEye = (landmarks[0][42] + landmarks[0][45]) / 2;
      nose = landmarks[0][30];
      topNose = (leftEye + rightEye) / 2;
      topMouth = (landmarks[0][61] + landmarks[0][62] + landmarks[0][63]) / 3;
      bottomMouth = (landmarks[0][65] + landmarks[0][66] + landmarks[0][67]) / 3;
      mouth = (topMouth + bottomMouth) / 2;

      if (!initialized)
      {
        static int counter = 0;
        cv::circle(frame, cv::Size(WIDTH / 2, HEIGHT / 2), 10, cv::Scalar(0, 0, 255), -1);
        if (ready)
        {
          if (counter++ > INIT_FRAME)
          {
            printf("(%lf, %lf)\n", horizontalRatio, verticalRatio);
            initialized = true;
          }
          cv::putText(frame, "please face the red dot on the screen", cv::Point2d(10, 50), 2, 1.0, cv::Scalar(0, 0, 255));
          cv::Point2f leftVec(leftEye - nose);
          cv::Point2f rightVec(rightEye - nose);
          cv::Point2f upVec(topNose - nose);
          cv::Point2f downVec(topMouth - nose);
          cv::Point2f mouthVec(topMouth - bottomMouth);
          horizontalRatio += sqrt(leftVec.dot(leftVec)) / sqrt(rightVec.dot(rightVec)) / INIT_FRAME;
          verticalRatio += sqrt(upVec.dot(upVec)) / sqrt(downVec.dot(downVec)) / INIT_FRAME;
          mouthLength += sqrt(mouthVec.dot(mouthVec)) / INIT_FRAME;
        }
        else
        {
          cv::putText(frame, "press 'r' if you are ready", cv::Point2d(10, 50), 2, 1.0, cv::Scalar(0, 0, 255));
        }
      }
      else
      {
        cv::circle(frame, leftEye, 5, cv::Scalar(255, 0, 0), -1);
        cv::circle(frame, rightEye, 5, cv::Scalar(0, 255, 0), -1);
        cv::circle(frame, nose, 5, cv::Scalar(0, 0, 255), -1);
        cv::circle(frame, topNose, 5, cv::Scalar(0, 255, 255), -1);
        cv::circle(frame, topMouth, 5, cv::Scalar(255, 255, 0), -1);

        cv::line(frame, leftEye, nose, cv::Scalar(255, 255, 255), 3);
        cv::line(frame, rightEye, nose, cv::Scalar(255, 255, 255), 3);
        cv::line(frame, topNose, nose, cv::Scalar(255, 255, 255), 3);
        cv::line(frame, topMouth, nose, cv::Scalar(255, 255, 255), 3);

        cv::Point2f mouthVec = topMouth - bottomMouth;
        if(sqrt(mouthVec.dot(mouthVec)) > mouthLength * 3)
          clicked = true;
        else
          clicked = false;

        cv::Point3f leftVec(leftEye - nose);
        cv::Point3f rightVec(rightEye - nose);
        double leftLen = sqrt(leftVec.dot(leftVec)), rightLen = sqrt(rightVec.dot(rightVec));
        double horizontalAngle = 0.0, verticalAngle = 0.0;
        if (leftLen > rightLen)
        {
          rightVec.z = sqrt(leftLen * leftLen - rightLen * rightLen);
          cv::Point3f vertVec = leftVec.cross(rightVec);
          horizontalAngle = acos(vertVec.dot(cv::Point3f(0, 0, 1.0)) / sqrt(vertVec.dot(vertVec))) * 180 / M_PI;
        }
        else
        {
          leftVec.z = sqrt(rightLen * rightLen - leftLen * leftLen);
          cv::Point3f vertVec = leftVec.cross(rightVec);
          horizontalAngle = -acos(vertVec.dot(cv::Point3f(0, 0, 1.0)) / sqrt(vertVec.dot(vertVec))) * 180 / M_PI;
        }
        horizontalAngle /= 2;

        cv::Point3f upVec(topNose - nose);
        cv::Point3f downVec(topMouth - nose);
        double upLen = sqrt(upVec.dot(upVec)), downLen = sqrt(downVec.dot(downVec));

        if (upLen > downLen * verticalRatio)
        {
          downVec.z = sqrt(pow((upLen / verticalRatio), 2) - pow(downLen, 2));
          verticalAngle = -asin(downVec.dot(cv::Point3f(0, 0, 1.0)) / sqrt(downVec.dot(downVec))) * 180 / M_PI;
        }
        else
        {
          upVec.z = sqrt(pow(downLen * verticalRatio, 2) - pow(upLen, 2));
          verticalAngle = asin(upVec.dot(cv::Point3f(0, 0, 1.0)) / sqrt(upVec.dot(upVec))) * 180 / M_PI;
        }
        verticalAngle /= 2;
        if(clicked)
          verticalAngle-=3;
          
        std::ostringstream buf;
        if(clicked)
          buf << "  CLICKED";
        else
          buf << "UNCLICKED";
        buf << '(' << horizontalAngle << ", " << verticalAngle << ") ";
        
        cv::putText(frame, buf.str(), cv::Point2d(10, 50), 2, 1.0, cv::Scalar(255, 0, 0));
      }
    }
    cv::imshow("win1", frame);
    int c = cv::waitKey(33);

    switch (c)
    {
    case 27: //ESC
    case 'q':
    case 'Q':
      loop_flag = false;
      break;
    case 'r':
    case 'R':
      ready = true;
      break;
    }
  }
  return 0;
}