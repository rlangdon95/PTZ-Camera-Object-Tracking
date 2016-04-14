#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "httpClient.h"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

const static int SENSITIVITY_VALUE = 50;
const static int BLUR_SIZE = 10;
const static int THRESHOLD_FRAMES =  30;

VideoCapture cap;
Mat image;

bool selectObject = false;
int trackObject = -1;
Point origin;
Rect selection;
int vmin = 10, vmax = 256, smin = 30;
bool tracker = false;

string intToString(int number) {

	std::stringstream ss;
	ss << number;
	return ss.str();
}

int main() {

	CvCapture* camera = cvCaptureFromFile("http://root:root@192.168.42.173/axis-cgi/mjpg/video.cgi?resolution=320x240&req_fps=30&.mjpg");
	IplImage* back;

	CvCapture* camera2;
	IplImage* back2;

	Mat bg;
	Mat fg;
	Mat morphed_diff_frame;
	Mat backup_image;
	Mat frame;
	Mat hsv;
	Mat hue;
	Mat mask;
	Mat hist;
	Mat histimg = Mat::zeros(200, 320, CV_8UC3);
	Mat backproj;
	
	Rect trackWindow;

	bool paused = false;
	
    int hsize = 16;
	int temp_counter = 0;
	int cam_pan = -35;
    
	float hranges[] = {0,180};
    const float* phranges = hranges;

	//cap.open(0);
	//cap >> bg;

	back = cvQueryFrame(camera);
	bg = cvarrToMat(back);
	bg.copyTo(morphed_diff_frame);
	cvtColor(bg, bg, CV_BGR2GRAY);
	//imshow("BACKGROUND", bg);

	while(!tracker) {

		//cap >> fg;
		//camera2 = cvCaptureFromFile("http://root:root@192.168.42.173/axis-cgi/mjpg/video.cgi?resolution=320x240&req_fps=30&.mjpg");
		back2 = cvQueryFrame(camera);
		fg = cvarrToMat(back2);
		fg.copyTo(image);
		cvtColor(fg, fg, CV_BGR2GRAY);
		//backgroundSubtract(bg, image, morphed_diff_frame);

		absdiff(bg, fg, morphed_diff_frame);
		threshold(morphed_diff_frame, morphed_diff_frame, SENSITIVITY_VALUE, 255, THRESH_BINARY);

		erode(morphed_diff_frame, morphed_diff_frame, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(morphed_diff_frame, morphed_diff_frame, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(morphed_diff_frame, morphed_diff_frame, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		erode(morphed_diff_frame, morphed_diff_frame, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

		blur(morphed_diff_frame, morphed_diff_frame, Size(BLUR_SIZE, BLUR_SIZE));
		threshold(morphed_diff_frame, morphed_diff_frame, SENSITIVITY_VALUE, 255, THRESH_BINARY);

		erode(morphed_diff_frame, morphed_diff_frame, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(morphed_diff_frame, morphed_diff_frame, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(morphed_diff_frame, morphed_diff_frame, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		erode(morphed_diff_frame, morphed_diff_frame, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

		if(morphed_diff_frame.data == NULL) {

			cout << "NULL" << endl;
			continue;
		}

		imshow("BACKGROUND SUBTRACTION", morphed_diff_frame);

		//BOUNDING BOX
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		Canny(morphed_diff_frame, morphed_diff_frame, 10, 10*3, 3);
		findContours(morphed_diff_frame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		if(contours.size() > 0)
			temp_counter++;

		if(contours.size() == 0)
			temp_counter = 0;

		//cout << "Counter = " << temp_counter << endl;

		if(temp_counter >= THRESHOLD_FRAMES) {
			
			tracker = true;

			int largest_area = 0;
			int largest_contour_index = 0;
			Rect bounding_rect;
			Rect objectBoundingRectangle = Rect(0, 0, 0, 0);

			for(int i = 0; i < (int) (contours.size()); i++) {		                        //iterate through each contour. 

				double a = contourArea(contours[i], false);									//Find the area of contour

				if(a > largest_area) {

					largest_area = (int) a;
					largest_contour_index = i;												//Store the index of largest contour
				}
			}

			objectBoundingRectangle = boundingRect(contours[largest_contour_index]);		//Find the bounding rectangle for biggest contour

			selection = Rect(objectBoundingRectangle.x, objectBoundingRectangle.y, objectBoundingRectangle.width, objectBoundingRectangle.height);
			selection &= Rect(0, 0, image.cols, image.rows);

			image.copyTo(backup_image);
			rectangle(backup_image, objectBoundingRectangle.tl(), objectBoundingRectangle.br(), Scalar(0, 255, 0), 2, 8, 0);
			imshow("ROI", backup_image);
		}

		waitKey(10);
	}

	while(1) {

		//puts("0. HELLO");
		//getchar();
        if(!paused) {

			//puts("1. HELLO");
			//getchar();
            //cap >> frame;
			back = cvQueryFrame(camera);
			frame = cvarrToMat(back);

            if( frame.empty() )
                break;
        }
        
        frame.copyTo(image);
        
        if(!paused) {
            
			cvtColor(image, hsv, CV_BGR2HSV);

			//puts("2. HELLO");
			//getchar();
            
            if(trackObject) {

                int _vmin = vmin, _vmax = vmax;
                
                inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)), Scalar(180, 256, MAX(_vmin, _vmax)), mask);

				/*if(!hue.empty())
					imshow("Hue", hue);

				imshow("Mask", mask);*/

                int ch[] = {0, 0};
                hue.create(hsv.size(), hsv.depth());
                mixChannels(&hsv, 1, &hue, 1, ch, 1);
                
                if( trackObject < 0 ) {

                    Mat roi(hue, selection);
					Mat maskroi(mask, selection);

                    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                    normalize(hist, hist, 0, 255, CV_MINMAX);
                    
                    trackWindow = selection;
                    trackObject = 1;
                    
                    histimg = Scalar::all(0);
                    int binW = histimg.cols / hsize;
                    Mat buf(1, hsize, CV_8UC3);

                    for( int i = 0; i < hsize; i++ )
                        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
                    
					cvtColor(buf, buf, CV_HSV2BGR);
                    
                    for( int i = 0; i < hsize; i++ ) {

                        int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
                        rectangle( histimg, Point(i*binW,histimg.rows), Point((i+1)*binW,histimg.rows - val), Scalar(buf.at<Vec3b>(i)), -1, 8 );
                    }
                }
                
                calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
                backproj &= mask;
                RotatedRect trackBox = CamShift(backproj, trackWindow, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
                
				if(trackWindow.area() <= 1) {

                    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
                    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r, trackWindow.x + r, trackWindow.y + r) & Rect(0, 0, cols, rows);
                }
                
                ellipse(image, trackBox, Scalar(0, 0, 255), 3, CV_AA);

				Point centroid = trackBox.center;
				//cout << centroid.x;
				circle(image, centroid, 10, cv::Scalar(0, 0, 255));
				putText(image, intToString(centroid.x) + " , " + intToString(centroid.y), Point(centroid.x, centroid.y + 20), 1, 1, Scalar(0, 255, 0));

				//puts("3. HELLO");
				cvReleaseCapture(&camera);

				//puts("4. HELLO");
				if(centroid.x > 440) {

					httpClient hc = httpClient("http://root:root@192.168.42.173/axis-cgi/mjpg/video.cgi?resolution=320x240&req_fps=30&.mjpg");
					cout << centroid.x << endl;
					cam_pan += 5;
					hc.setPTZ(cam_pan, -20, 1);
				}

				else if(centroid.x < 200) {

					httpClient hc = httpClient("http://root:root@192.168.42.173/axis-cgi/mjpg/video.cgi?resolution=320x240&req_fps=30&.mjpg");
					cout << centroid.x << endl;
					cam_pan -= 5;
					hc.setPTZ(cam_pan, -20, 1);
				}

				else
					cout << centroid.x << endl;

				//puts("5. HELLO");
				camera = cvCaptureFromFile("http://root:root@192.168.42.173/axis-cgi/mjpg/video.cgi?resolution=320x240&req_fps=30&.mjpg");
				//puts("6. HELLO");
            }
        }

        else if( trackObject < 0 )
            paused = false;
        
        if(selectObject && selection.width > 0 && selection.height > 0) {

            Mat roi(image, selection);
            bitwise_not(roi, roi);
        }
        
        imshow( "CamShift Object Tracker", image );
        
        char c = (char)waitKey(10);

        if( c == 27 )
            break;
        switch(c) {

            case 's':
                trackObject = 0;
                histimg = Scalar::all(0);
                break;

            case 'p':
                paused = !paused;
                break;
        }
    }
    
    return 0;
}