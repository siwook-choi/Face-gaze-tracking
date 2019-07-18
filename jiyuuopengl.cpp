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

#define WINDOW_X (1200)
#define WINDOW_Y (800)
#define WINDOW_NAME "test3"

const int WIDTH = 640, HEIGHT = 480, INIT_FRAME = 100;

void init_GL(int argc, char *argv[]);
void init();
void set_callback_functions();

void glut_display();
void glut_keyboard(unsigned char key, int x, int y);
void glut_mouse(int button, int state, int x, int y);
void glut_motion(int x, int y);
void glut_idle();

void draw_pyramid();
void draw_cube();

double g_distance = 20.0;
double horizontalRatio = 0.0, verticalRatio = 0.0;
double horizontalAngle = 0.0, verticalAngle = 0.0;
double mouthLength = 0.0;
cv::Mat frame, gray;
bool g_isLeftButtonOn = false;
bool g_isRightButtonOn = false;
bool ready = false;
bool initialized = false;
bool clicked = false;
int initCounter = 0;
cv::CascadeClassifier faceDetector;
cv::Ptr<cv::face::Facemark> facemark;
cv::VideoCapture cam;

int main(int argc, char *argv[])
{
    init_GL(argc, argv);

    init();

    set_callback_functions();

    glutMainLoop();

    return 0;
}

void init_GL(int argc, char *argv[])
{
    faceDetector.load("resource/haarcascade_frontalface_alt2.xml");
    facemark = cv::face::FacemarkLBF::create();
    facemark->loadModel("resource/lbfmodel2.yaml");

    cam.open(0);
    cv::namedWindow("win2", cv::WINDOW_AUTOSIZE);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(WINDOW_X, WINDOW_Y);
    glutCreateWindow(WINDOW_NAME);
}

void init()
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
}

void set_callback_functions()
{
    glutDisplayFunc(glut_display);
    glutKeyboardFunc(glut_keyboard);
    glutMouseFunc(glut_mouse);
    glutMotionFunc(glut_motion);
    glutPassiveMotionFunc(glut_motion);
    glutIdleFunc(glut_idle);
}

void glut_keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 'q':
    case 'Q':
    case '\033':
        exit(0);
    case 'r':
    case 'R':
        ready = true;
        break;
    }
    glutPostRedisplay();
}

void glut_mouse(int button, int state, int x, int y)
{
    if (button == GLUT_LEFT_BUTTON)
    {
        if (state == GLUT_UP)
        {
            g_isLeftButtonOn = false;
        }
        else if (state == GLUT_DOWN)
        {
            g_isLeftButtonOn = true;
        }
    }

    if (button == GLUT_RIGHT_BUTTON)
    {
        if (state == GLUT_UP)
        {
            g_isRightButtonOn = false;
        }
        else if (state == GLUT_DOWN)
        {
            g_isRightButtonOn = true;
        }
    }
}

void glut_motion(int x, int y)
{
    static int px = -1, py = -1;
    if (g_isLeftButtonOn == true)
    {
        px = x;
        py = y;
    }
    else if (g_isRightButtonOn == true)
    {
        if (px >= 0 && py >= 0)
        {
            g_distance += (double)(y - py) / 20;
        }
        px = x;
        py = y;
    }
    else
    {
        px = -1;
        py = -1;
    }
    glutPostRedisplay();
}

void glut_idle()
{
    if (cam.read(frame))
    {
        cv::flip(frame, frame, 1);
        std::vector<cv::Rect> faces;
        cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        faceDetector.detectMultiScale(gray, faces, 1.1, 3, CV_HAAR_SCALE_IMAGE, cv::Size(150, 150));
        std::vector<std::vector<cv::Point2f>> landmarks;
        bool success = facemark->fit(frame, faces, landmarks);
        if (success)
        {
            cv::Point2f leftEye, rightEye, nose, topNose, topMouth, bottomMouth, mouth;
            leftEye = (landmarks[0][36] + landmarks[0][39]) / 2;
            rightEye = (landmarks[0][42] + landmarks[0][45]) / 2;
            nose = landmarks[0][30];
            topNose = (leftEye + rightEye) / 2;
            topMouth = (landmarks[0][61] + landmarks[0][62] + landmarks[0][63]) / 3;
            bottomMouth = (landmarks[0][65] + landmarks[0][66] + landmarks[0][67]) / 3;
            mouth = topMouth + bottomMouth / 2;

            if (!initialized)
            {
                cv::circle(frame, cv::Size(WIDTH / 2, HEIGHT / 2), 10, cv::Scalar(0, 0, 255), -1);
                if (ready)
                {
                    if (initCounter++ > INIT_FRAME)
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
                if(sqrt(mouthVec.dot(mouthVec)) > mouthLength * 2)
                    clicked = true;
                else
                    clicked = false;

                cv::Point3f leftVec(leftEye - nose);
                cv::Point3f rightVec(rightEye - nose);
                double leftLen = sqrt(leftVec.dot(leftVec)), rightLen = sqrt(rightVec.dot(rightVec));
                double horizontalAngle_temp = 0.0, verticalAngle_temp = 0.0;
                if (leftLen > rightLen)
                {
                    rightVec.z = sqrt(leftLen * leftLen - rightLen * rightLen);
                    cv::Point3f vertVec = leftVec.cross(rightVec);
                    horizontalAngle_temp = acos(vertVec.dot(cv::Point3f(0, 0, 1.0)) / sqrt(vertVec.dot(vertVec))) * 180 / M_PI;
                }
                else
                {
                    leftVec.z = sqrt(rightLen * rightLen - leftLen * leftLen);
                    cv::Point3f vertVec = leftVec.cross(rightVec);
                    horizontalAngle_temp = -acos(vertVec.dot(cv::Point3f(0, 0, 1.0)) / sqrt(vertVec.dot(vertVec))) * 180 / M_PI;
                }
                horizontalAngle_temp /= 2;

                horizontalAngle = (horizontalAngle * 14 + horizontalAngle_temp) / 15;
                cv::Point3f upVec(topNose - nose);
                cv::Point3f downVec(topMouth - nose);
                double upLen = sqrt(upVec.dot(upVec)), downLen = sqrt(downVec.dot(downVec));

                if (upLen > downLen * verticalRatio)
                {
                    downVec.z = sqrt(pow((upLen / verticalRatio), 2) - pow(downLen, 2));
                    verticalAngle_temp = -asin(downVec.dot(cv::Point3f(0, 0, 1.0)) / sqrt(downVec.dot(downVec))) * 180 / M_PI;
                }
                else
                {
                    upVec.z = sqrt(pow(downLen * verticalRatio, 2) - pow(upLen, 2));
                    verticalAngle_temp = asin(upVec.dot(cv::Point3f(0, 0, 1.0)) / sqrt(upVec.dot(upVec))) * 180 / M_PI;
                }
                verticalAngle_temp /= 2;
                if(clicked)
                    verticalAngle -= 3;
                verticalAngle = (verticalAngle * 14 + verticalAngle_temp) / 15;
                std::ostringstream buf;
                if(clicked)
                    buf << "  CLICKED ";
                else
                    buf << "UNCLICKED ";
                buf << '(' << horizontalAngle << ", " << verticalAngle << ')';
                cv::putText(frame, buf.str(), cv::Point2d(10, 50), 2, 1.0, cv::Scalar(255, 0, 0));
                glutPostRedisplay();
            }
        }
    }
    cv::imshow("win2", frame);
    char c = cv::waitKey(33);
    switch (c)
    {
    case 27: //ESC
    case 'q':
    case 'Q':
        exit(0);
        break;
    case 'r':
    case 'R':
        ready = true;
        break;
    }
}

void glut_display()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(30.0, (double)WINDOW_X / WINDOW_Y, 0.1, 200);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    gluLookAt(0, 0, g_distance,
              -g_distance * sin(horizontalAngle / 180 * M_PI) * cos(verticalAngle / 180 * M_PI), -g_distance * sin(verticalAngle / 180 * M_PI),
              g_distance - g_distance * cos(horizontalAngle / 180 * M_PI) * cos(verticalAngle / 180 * M_PI),
              0, 1.0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_DEPTH_TEST);

    glPushMatrix();
    glColor3f(1.0, 0, 0);
    GLUquadric *sphere;
    sphere = gluNewQuadric();
    gluSphere(sphere, 1.0f, 50, 10);
    gluDeleteQuadric(sphere);
    glPopMatrix();

    glFlush();
    glDisable(GL_DEPTH_TEST);

    glutSwapBuffers();
}

void draw_pyramid()
{
    GLdouble pointO[] = {0.0, 1.0, 0.0};
    GLdouble pointA[] = {1.5, -1.0, 1.5};
    GLdouble pointB[] = {-1.5, -1.0, 1.5};
    GLdouble pointC[] = {-1.5, -1.0, -1.5};
    GLdouble pointD[] = {1.5, -1.0, -1.5};

    glColor3d(1.0, 0.0, 0.0);
    glBegin(GL_TRIANGLES);
    glVertex3dv(pointO);
    glVertex3dv(pointA);
    glVertex3dv(pointB);
    glEnd();

    glColor3d(1.0, 1.0, 0.0);
    glBegin(GL_TRIANGLES);
    glVertex3dv(pointO);
    glVertex3dv(pointB);
    glVertex3dv(pointC);
    glEnd();

    glColor3d(0.0, 1.0, 1.0);
    glBegin(GL_TRIANGLES);
    glVertex3dv(pointO);
    glVertex3dv(pointC);
    glVertex3dv(pointD);
    glEnd();

    glColor3d(1.0, 0.0, 1.0);
    glBegin(GL_TRIANGLES);
    glVertex3dv(pointO);
    glVertex3dv(pointD);
    glVertex3dv(pointA);
    glEnd();

    glColor3d(1.0, 1.0, 1.0);
    glBegin(GL_POLYGON);
    glVertex3dv(pointA);
    glVertex3dv(pointB);
    glVertex3dv(pointC);
    glVertex3dv(pointD);
    glEnd();
}

void draw_cube()
{
    GLdouble pointA[] = {1.0, 1.0, 1.0};
    GLdouble pointB[] = {1.0, -1.0, 1.0};
    GLdouble pointC[] = {-1.0, -1.0, 1.0};
    GLdouble pointD[] = {-1.0, 1.0, 1.0};
    GLdouble pointE[] = {1.0, 1.0, -1.0};
    GLdouble pointF[] = {1.0, -1.0, -1.0};
    GLdouble pointG[] = {-1.0, -1.0, -1.0};
    GLdouble pointH[] = {-1.0, 1.0, -1.0};

    glColor3d(1.0, 0.0, 0.0);
    glBegin(GL_POLYGON);
    glVertex3dv(pointA);
    glVertex3dv(pointB);
    glVertex3dv(pointC);
    glVertex3dv(pointD);
    glEnd();

    glColor3d(0.0, 1.0, 0.0);
    glBegin(GL_POLYGON);
    glVertex3dv(pointA);
    glVertex3dv(pointB);
    glVertex3dv(pointF);
    glVertex3dv(pointE);
    glEnd();

    glColor3d(0.0, 0.0, 1.0);
    glBegin(GL_POLYGON);
    glVertex3dv(pointB);
    glVertex3dv(pointC);
    glVertex3dv(pointG);
    glVertex3dv(pointF);
    glEnd();

    glColor3d(1.0, 1.0, 0.0);
    glBegin(GL_POLYGON);
    glVertex3dv(pointC);
    glVertex3dv(pointD);
    glVertex3dv(pointH);
    glVertex3dv(pointG);
    glEnd();

    glColor3d(1.0, 0.0, 1.0);
    glBegin(GL_POLYGON);
    glVertex3dv(pointA);
    glVertex3dv(pointD);
    glVertex3dv(pointH);
    glVertex3dv(pointE);
    glEnd();

    glColor3d(0.0, 1.0, 1.0);
    glBegin(GL_POLYGON);
    glVertex3dv(pointE);
    glVertex3dv(pointF);
    glVertex3dv(pointG);
    glVertex3dv(pointH);
    glEnd();
}