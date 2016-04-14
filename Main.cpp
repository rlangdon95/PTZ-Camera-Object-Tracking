//#include "httpClient.h"
//#include "Objects.h"
//
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>
//using namespace std;
//using namespace cv;
//
//const static int SENSITIVITY_VALUE = 50;
//const static int BLUR_SIZE = 10;
//const static int HISTOGRAM_THRESHOLD = 10;
//const static int THRESHOLD_FRAMES = 30;
//
//const static double LIMIT_VAL_HUE = 0.09;
//const static double LIMIT_VAL_SAT = 0.09;
//const static double LIMIT_VAL_VAL = 0.09;
//
//const int FRAME_WIDTH = 640;
//const int FRAME_HEIGHT = 480;
//const int MAX_NUM_OBJECTS = 50;
//const int MIN_OBJECT_AREA = 20 * 20;
//const int MAX_OBJECT_AREA = (int) (FRAME_HEIGHT*FRAME_WIDTH / 1.3);
//
//int theObject[2] = { 0,0 };
//bool objectDetected = false;
//int hue_max = 0;
//int hue_min = 0;
//int saturation_min = 0;
//int saturation_max = 0;
//int value_min = 0;
//int value_max = 0;
//
//static Mat HSV_Reference_Frame;
//
//static int ROI_x;
//static int ROI_y;
//static int ROI_width;
//static int ROI_height;
//
//int h_peak_y = 10000;
//int s_peak_y = 10000;
//int v_peak_y = 10000;
//int h_peak_x = 10000;
//int s_peak_x = 10000;
//int v_peak_x = 10000;
//
//int h_x = -1;
//int s_x = -1;
//int v_x = -1;
//
//Rect ROI;
//Rect objectBoundingRectangle = Rect(0, 0, 0, 0);
//RNG rng(12345);
//
//Scalar HSVMinimum;
//Scalar HSVMaximum;
//
//bool thread_flag = false;
//int global_flag = 0;
//
//CvCapture *camera;
//IplImage *back;
//Mat bg;
//Mat bg_color;
//
//int avg_x = 0;
//int avg_y = 0;
////bool camera_flag = false;
//
//void setHueRange(int arr[], int max_y, int max_x) {
//
//	int limit = 400 - (int)((double)(400 - max_y) * LIMIT_VAL_HUE);
//	hue_min = 0;
//	hue_max = 512;
//
//	for(int i = max_x; i > 0; i--)
//		if((arr[i] > limit) || (arr[i] == 400)) {
//
//			hue_min = i;
//			break;
//		}
//
//	for(int j = max_x; j < 256; j++)
//		if((arr[j] > limit) || (arr[j] == 400)) {
//
//			hue_max = j;
//			break;
//		}
//}
//
//void setSaturationRange(int arr[], int max_y, int max_x) {
//
//	int limit = 400 - (int)((double)(400 - max_y) * LIMIT_VAL_SAT);
//	saturation_min = 0;
//	saturation_max = 512;
//
//	for(int i = max_x; i > 0; i--)
//		if((arr[i] > limit) || (arr[i] == 400)) {
//
//			saturation_min = i;
//			break;
//		}
//
//	for(int j = max_x; j < 256; j++)
//		if((arr[j] > limit) || (arr[j] == 400)) {
//
//			saturation_max = j;
//			break;
//		}
//}
//
//void setValueRange(int arr[], int max_y, int max_x) {
//
//	int limit = 400 - (int)((double)(400 - max_y) * LIMIT_VAL_VAL);
//	value_min = 0;
//	value_max = 512;
//
//	for(int i = max_x; i > 0; i--)
//		if((arr[i] > limit) || (arr[i] == 400)) {
//
//			value_min = i;
//			break;
//		}
//
//	for(int j = max_x; j < 256; j++)
//		if((arr[j] > limit) || (arr[j] == 400)) {
//
//			value_max = j;
//			break;
//		}
//}
//
//void morphOps(Mat &thresh) {
//
//	//create structuring element that will be used to "dilate" and "erode" image.
//	//the element chosen here is a 3px by 3px rectangle
//	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3,3));
//	
//	//dilate with larger element so make sure object is nicely visible
//	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8,8));
//
//	erode(thresh, thresh, erodeElement);
//	erode(thresh, thresh, erodeElement);
//
//	dilate(thresh, thresh, dilateElement);
//	dilate(thresh, thresh, dilateElement);
//}
//
//void morphOps2(Mat &img_mask) {
//
//	erode(img_mask, img_mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
//	dilate(img_mask, img_mask, cv::getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
//	dilate(img_mask, img_mask, cv::getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
//	erode(img_mask, img_mask, cv::getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
//
//	blur(img_mask, img_mask, Size(BLUR_SIZE, BLUR_SIZE));
//	threshold(img_mask, img_mask, SENSITIVITY_VALUE, 255, cv::THRESH_BINARY);
//
//	erode(img_mask, img_mask, cv::getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
//	dilate(img_mask, img_mask, cv::getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
//	dilate(img_mask, img_mask, cv::getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
//	erode(img_mask, img_mask, cv::getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
//}
//
//void backgroundSubtract(Mat gray_frame1, Mat gray_frame2, Mat morphed_diff_frame) {
//
//	Mat diff_frame, binary_diff_frame;
//
//	absdiff(gray_frame1, gray_frame2, diff_frame);
//	threshold(diff_frame, binary_diff_frame, SENSITIVITY_VALUE, 255, THRESH_BINARY);
//
//	binary_diff_frame.copyTo(morphed_diff_frame);
//
//	erode(binary_diff_frame, morphed_diff_frame, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
//	dilate(morphed_diff_frame, morphed_diff_frame, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
//	dilate(morphed_diff_frame, morphed_diff_frame, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
//	erode(morphed_diff_frame, morphed_diff_frame, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
//
//	blur(morphed_diff_frame, morphed_diff_frame, Size(BLUR_SIZE, BLUR_SIZE));
//	threshold(morphed_diff_frame, morphed_diff_frame, SENSITIVITY_VALUE, 255, THRESH_BINARY);
//
//	erode(morphed_diff_frame, morphed_diff_frame, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
//	dilate(morphed_diff_frame, morphed_diff_frame, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
//	dilate(morphed_diff_frame, morphed_diff_frame, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
//	erode(morphed_diff_frame, morphed_diff_frame, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
//}
//
//string intToString(int number) {
//
//	std::stringstream ss;
//	ss << number;
//	return ss.str();
//}
//
//int gl_cntr = 0;
//
//void drawObject(vector<Objects> theObjects, Mat &frame) {
//
//	try {
//
//		int i;
//		avg_x = 0;
//		avg_y = 0;
//
//		for(i = 0; theObjects.size(); i++) {
//
//			cout << theObjects.at(i).getXPos() << endl;
//			cout << theObjects.at(i).getYPos() << endl;
//
//			circle(frame, Point(theObjects.at(i).getXPos(), theObjects.at(i).getYPos()), 10, cv::Scalar(0, 0, 255));
//			putText(frame, intToString(theObjects.at(i).getXPos()) + " , " + intToString(theObjects.at(i).getYPos()), Point(theObjects.at(i).getXPos(), theObjects.at(i).getYPos() + 20), 1, 1, Scalar(0, 255, 0));
//
//			avg_x += theObjects.at(i).getXPos();
//			avg_y += theObjects.at(i).getYPos();
//		}
//
//		/*if(theObjects.size() > 0) {
//
//			gl_cntr++;
//			camera_flag = true;
//		}*/
//
//		/*if(gl_cntr > 30)
//			gl_cntr = 0;*/
//
//		avg_x /= theObjects.size();
//		avg_y /= theObjects.size();
//
//		//httpClient hc = httpClient("http://root:root@192.168.42.175/axis-cgi/mjpg/video.cgi?resolution=640x480&req_fps=30&.mjpg");
//		//hc.setCenter(theObjects.at(i).getXPos(), theObjects.at(i).getYPos());
//	}
//
//	catch(const std::out_of_range& oor) {
//	
//		cerr << "Out of Range error: " << oor.what() << '\n';
//	}
//}
//
//void trackFilteredObject(Objects theObject, Mat threshold, Mat HSV, Mat &cameraFeed){
//	
//	vector <Objects> obj_vec;
//	Objects obj;
//
//	Mat temp;
//	threshold.copyTo(temp);
//
//	//these two vectors needed for output of findContours
//	vector< vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//	
//	//find contours of filtered image using openCV findContours function
//	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
//	
//	//use moments method to find our filtered object
//	double refArea = 0;
//	bool objectFound = false;
//	if (hierarchy.size() > 0) {
//		
//		int numObjects = hierarchy.size();
//		
//		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
//		if(numObjects<MAX_NUM_OBJECTS) {
//
//			for (int index = 0; index >= 0; index = hierarchy[index][0]) {
//
//				Moments moment = moments((cv::Mat)contours[index]);
//				//double area = moment.m00;
//
//				//if the area is less than 20 px by 20px then it is probably just noise
//				//if the area is the same as the 3/2 of the image size, probably just a bad filter
//				//we only want the object with the largest area so we safe a reference area each
//				//iteration and compare it to the area in the next iteration.
//				if(moment.m00 > MIN_OBJECT_AREA) {
//					
//					obj.setXPos((int) (moment.m10 / moment.m00)); 
//					obj.setYPos((int) (moment.m01 / moment.m00));
//					obj.setType(theObject.getType());
//					obj.setColour(theObject.getColour());
//
//					obj_vec.push_back(obj);
//					objectFound = true;
//				}
//				
//				else objectFound = false;
//			}
//			
//			//let user know you found an object
//			if(objectFound ==true){
//				//draw object location on screen
//				drawObject(obj_vec, cameraFeed);
//			}
//		}
//		
//		else 
//			putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(0, 50), 1, 2, Scalar(0, 0, 255), 2);
//	}
//}
//
//void track() {
//
//	cout << "TRACK FUNCTION" << endl;
//
//	//if we would like to calibrate our filter values, set to true.
//	bool calibrationMode = true;
//
//	Mat cameraFeed;
//	Mat threshold;
//	Mat HSV;
//	
//	bg_color.copyTo(cameraFeed);
//
//	//Matrix to store each frame of the webcam feed
//
//	//video capture object to acquire webcam feed
//	VideoCapture cap;
//
//	//open capture object at location zero (default location for webcam)
//	//set height and width of capture frame
//	cap.set(CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH);
//	cap.set(CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT);
//
//	//CvCapture *camera = cvCaptureFromFile("http://root:root@192.168.42.171/axis-cgi/mjpg/video.cgi?resolution=640x480&req_fps=30&.mjpg");
//	httpClient hc = httpClient("http://root:root@192.168.42.175/axis-cgi/mjpg/video.cgi?resolution=640x480&req_fps=30&.mjpg");
//
//	int local_flag = 0;
//	Mat ref_frame;
//
//	bool rot = true;
//
//	//start an infinite loop where webcam feed is copied to cameraFeed matrix
//	//all of our operations will be performed within this loop
//	while(1) {
//
//		//cap.open(0);
//		//cap >> cameraFeed;
//
//		//convert frame from BGR to HSV colorspace
//		cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
//
//		if(local_flag == 0) {
//		
//			HSV.copyTo(ref_frame);
//			local_flag++;
//		}
//
//		if(true) {
//
//			Objects obj;
//			obj.setHSVMin(HSVMinimum);
//			obj.setHSVMax(HSVMaximum);
//
//			absdiff(ref_frame, HSV, HSV);
//			imshow("HSV New", HSV);
//
//			inRange(HSV, obj.getHSVMin(), obj.getHSVMax(), threshold);
//			morphOps2(threshold);
//			imshow("Threshold NEW", threshold);
//			trackFilteredObject(obj, threshold, HSV, cameraFeed);
//		}
//
//		imshow("Camera Feed", cameraFeed);
//
//		//delay 30ms so that screen can refresh.
//		//image will not appear without this waitKey() command
//		if(waitKey(30) == 27)
//			exit(0);
//
//		vector<vector<Point>> contours;
//		vector<Vec4i> hierarchy;
//		Rect objectBoundingRectangle = Rect(0, 0, 0, 0);
//		RNG rng(12345);
//
//		Canny(threshold, threshold, 10, 10*3, 3);
//		findContours(threshold, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
//
//		int largest_area = 0;
//		int largest_contour_index = 0;
//		Rect bounding_rect;
//
//		for(int i = 0; i < (contours.size()); i++) {//iterate through each contour. 
//
//			double a=contourArea( contours[i],false);  //Find the area of contour
//			if(a>largest_area) {
//				largest_area = (int) a;
//				largest_contour_index = i;                //Store the index of largest contour
//				objectBoundingRectangle = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
//			}
//		}
//
//		int region_x = objectBoundingRectangle.x;
//		int region_width = objectBoundingRectangle.width;
//		int region_y = objectBoundingRectangle.y;
//		int region_height = (int) (objectBoundingRectangle.height / 2);
//
//		int cam_centre_x = region_x + (int) (region_width / 2);
//		int cam_centre_y = region_y + (region_height);
//
//		//if(camera_flag)// && gl_cntr == 30)
//		if(rot) {
//		
//			hc.setAreaZoom((avg_x * 704) / 640, (avg_y * 576) / 480, 100);
//			rot = false;
//		}
//
//		back = cvQueryFrame(camera);
//		cameraFeed = cvarrToMat(back);
//	}
//	
//	return;
//}

/*void track2(Mat image) {

	Mat hsv;
	Mat mask;
	Mat hue;
	Mat hist;
	Mat histimg = Mat::zeros(200, 320, CV_8UC3);

	int hsize = 16;
	int _vmin = vmin;
	int _vmax = vmax;
	int ch[] = {0, 0};

	float hranges[] = {0,180};
    const float* phranges = hranges;

    Rect trackWindow;

	cvtColor(image, hsv, CV_BGR2HSV);

	inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)), Scalar(180, 256, MAX(_vmin, _vmax)), mask);

	hue.create(hsv.size(), hsv.depth());
    mixChannels(&hsv, 1, &hue, 1, ch, 1);

    Mat roi(hue, selection);
    Mat maskroi(mask, selection);
    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
    normalize(hist, hist, 0, 255, CV_MINMAX);

	trackWindow = selection;

	histimg = Scalar::all(0);
	int binW = histimg.cols / hsize;
	Mat buf(1, hsize, CV_8UC3);
	for( int i = 0; i < hsize; i++ )
		buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);

	cvtColor(buf, buf, CV_HSV2BGR);
                    
	for( int i = 0; i < hsize; i++ ) {
	
		int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
		rectangle( histimg, Point(i*binW,histimg.rows),
		Point((i+1)*binW,histimg.rows - val),
		Scalar(buf.at<Vec3b>(i)), -1, 8 );
	}

	calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
    backproj &= mask;
    RotatedRect trackBox = CamShift(backproj, trackWindow, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));

    if( trackWindow.area() <= 1 ) {
                    
        int cols = backproj.cols;
        int rows = backproj.rows;
        int r = (MIN(cols, rows) + 5)/6;

        trackWindow = Rect(trackWindow.x - r, trackWindow.y - r, trackWindow.x + r, trackWindow.y + r) & Rect(0, 0, cols, rows);
    }
                
	ellipse( image, trackBox, Scalar(0, 0, 255), 3, CV_AA );

	if( selectObject && selection.width > 0 && selection.height > 0 ) {
            
        Mat roi(image, selection);
        bitwise_not(roi, roi);
    }

    imshow("CamShift Object Tracker", image);
}*/

//
//int main() {
//
//	camera = cvCaptureFromFile("http://root:root@192.168.42.175/axis-cgi/mjpg/video.cgi?resolution=640x480&req_fps=30&.mjpg");
//	//hc = httpClient("http://root:root@192.168.42.175/axis-cgi/mjpg/video.cgi?resolution=640x480&req_fps=30&.mjpg");
//
//	VideoCapture cap;
//	cap.open(0);
//	Objects obj;
//
//	if (camera==NULL)
//		printf("1. camera is null\n");
//	else
//	    printf("camera is not null\n");
//
//	Mat frame;
//	Mat morphed_diff_frame;
//	Mat frame1;
//
//	back = cvQueryFrame(camera);
//	bg = cvarrToMat(back);
//	bg.copyTo(bg_color);
//	cvtColor(bg, bg, COLOR_BGR2GRAY);
//
//	int ctr = 0;
//	int thresh = 127;
//	int max_thresh = 255;
//
//	int flag = 0;
//	int key = 0;
//	int temp_ctr = 0;
//	int histogram_arr_b[256];
//	int histogram_arr_g[256];
//	int histogram_arr_r[256];
//	int histogram_arr_x[256];
//	int histogram_arr_b_new[256];
//	int histogram_arr_g_new[256];
//	int histogram_arr_r_new[256];
//
//	while(1) {
//	
//		Mat img_bkgmodel;
//
//		back = cvQueryFrame(camera);
//		frame = cvarrToMat(back);
//		frame.copyTo(frame1);
//		cvtColor(frame, frame, COLOR_BGR2GRAY);
//		//cap >> frame;
//
//		if(frame.data == NULL)
//			cout << "HELLO" << endl;
//
//		imshow("LIVE", frame);
//
//		frame.copyTo(morphed_diff_frame);
//		frame1.copyTo(HSV_Reference_Frame);
//
//		backgroundSubtract(bg, frame, morphed_diff_frame);
//
//		if(morphed_diff_frame.data == NULL) {
//
//			cout << "NULL" << endl;
//			continue;
//		}
//
//		imshow("BACKGROUND", morphed_diff_frame);
//
//		if(global_flag == 0)
//			imshow("Background Subtraction Video", morphed_diff_frame);
//
//		//BOUNDING BOX
//		vector<vector<Point>> contours;
//		vector<Vec4i> hierarchy;
//		
//		Canny(morphed_diff_frame, morphed_diff_frame, 10, 10*3, 3);
//		findContours(morphed_diff_frame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
//
//		if(global_flag ==0)
//			imshow("EDGE", morphed_diff_frame);
//
//		if((contours.size() > 0) && (key == 0))
//			temp_ctr++;
//
//		// Draw the histograms for B, G and R
//		/// Establish the number of bins
//  		int histSize = 256;
//		int hist_w = 512;
//		int hist_h = 400;
//  		int bin_w = cvRound((double)hist_w/histSize);
//
//		Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
//
//		if(temp_ctr >= THRESHOLD_FRAMES) {
//
//			cout << "NULL 1" << endl;
//
//			if(frame.empty())
//				cout << "EMPTY" << endl;
//
//			cvtColor(frame1, HSV_Reference_Frame, COLOR_BGR2HSV);
//			cout << "NULL 2" << endl;
//
//			vector<vector<Point>> contours_poly(contours.size());
//			vector<Rect> boundRect(contours.size());
//			vector<Point2f> center(contours.size());
//			vector<float> radius(contours.size());
//			vector<vector<Point>> largestContourVec;
//
//			if(flag == 0) {
//
//				int largest_area = 0;
//				int largest_contour_index = 0;
//				Rect bounding_rect;
//
//				for(int i = 0; i < (contours.size()); i++) {//iterate through each contour. 
//
//					double a=contourArea( contours[i],false);  //Find the area of contour
//					if(a>largest_area) {
//						largest_area = (int) a;
//						largest_contour_index = i;                //Store the index of largest contour
//						objectBoundingRectangle = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
//					}
//				}
//
//				ROI_x = objectBoundingRectangle.x;
//				ROI_width = objectBoundingRectangle.width;
//				ROI_y = objectBoundingRectangle.y;
//				ROI_height = (int) (objectBoundingRectangle.height / 2);
//
//				ROI = Rect(ROI_x, ROI_y, ROI_width, ROI_height);
//
//				origin = Point(ROI_x, ROI_y);
//				selection = Rect(ROI_x, ROI_y, 0, 0);
//				selection.x = MIN(x, origin.x);
//		        selection.y = MIN(y, origin.y);
//   		    selection.width = std::abs(x - origin.x);
//        		selection.height = std::abs(y - origin.y);
//        
//        		selection &= Rect(0, 0, image.cols, image.rows);
//
//				vector<Mat> bgr_planes;
//
//				/// Set the ranges ( for B,G,R) )
//  				float range[] = { 0, 256 } ;
//				const float* histRange = { range };
//
//				int channels[] = {0};
//				bool uniform = true;
//				bool accumulate = false;
//
//				Mat b_hist, g_hist, r_hist;
//				Mat part_of_image = HSV_Reference_Frame(ROI);
//				split(part_of_image, bgr_planes);
//				//imshow("Crop", part_of_image);
//
//				cout << "X 1" << endl;
//
//				/// Compute the histograms:
//  				calcHist(&bgr_planes[0], 1, channels, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
//				calcHist(&bgr_planes[1], 1, channels, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
//				calcHist(&bgr_planes[2], 1, channels, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
//
//  				/// Normalize the result to [ 0, histImage.rows ]
//  				normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//  				normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//  				normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//
//  				/// Draw for each channel
//  				for(int i = 0; i < histSize; i++) {
//
//	  				Point b_pt = Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)));
//	  				Point g_pt = Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)));
//	  				Point r_pt = Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)));
//
//					histogram_arr_x[i] = 2 * b_pt.x;
//					histogram_arr_b[i] = b_pt.y;
//					histogram_arr_g[i] = g_pt.y;
//					histogram_arr_r[i] = r_pt.y;
//  				}
//
//				for(int i = 0; i < (histSize); i++) {
//
//					int left = i - HISTOGRAM_THRESHOLD;
//					int right = i + HISTOGRAM_THRESHOLD;
//					int sum_b = 0;
//					int sum_g = 0;
//					int sum_r = 0;
//
//					if(left < 0)
//						left = 0;
//
//					if(right > (histSize - 1))
//						right = histSize - 1;
//
//					for(int j = left; j <= right; j++) {
//
//						sum_b = sum_b + histogram_arr_b[j];
//						sum_g = sum_g + histogram_arr_g[j];
//						sum_r = sum_r + histogram_arr_r[j];
//					}
//
//					histogram_arr_b_new[i] = sum_b/(right - left + 1);
//					histogram_arr_g_new[i] = sum_g/(right - left + 1);
//					histogram_arr_r_new[i] = sum_r/(right - left + 1);
//
//					if(h_peak_y > histogram_arr_b_new[i]) {
//
//						h_peak_y = histogram_arr_b_new[i];
//						h_peak_x = i * 2;
//					}
//
//					if(s_peak_y > histogram_arr_g_new[i]) {
//
//						s_peak_y = histogram_arr_g_new[i];
//						s_peak_x = i * 2;
//					}
//
//					if(v_peak_y > histogram_arr_r_new[i]) {
//
//						v_peak_y = histogram_arr_r_new[i];
//						v_peak_x = i * 2;
//					}
//				}
//
//				for(int i = 1; i < histSize; i++) {
//
//	  				Point b_pt_1 = Point(bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)));
//	  				Point g_pt_1 = Point(bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)));
//	  				Point r_pt_1 = Point(bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)));
//					
//					Point b_pt_2 = Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)));
//	  				Point g_pt_2 = Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)));
//	  				Point r_pt_2 = Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)));
//
//					b_pt_1.y = histogram_arr_b_new[i - 1];
//					g_pt_1.y = histogram_arr_g_new[i - 1];
//					r_pt_1.y = histogram_arr_r_new[i - 1];
//
//					b_pt_2.y = histogram_arr_b_new[i];
//					g_pt_2.y = histogram_arr_g_new[i];
//					r_pt_2.y = histogram_arr_r_new[i];
//
//      				line(histImage, b_pt_1, b_pt_2, Scalar(255, 0, 0), 2, 8, 0);
//      				line(histImage, g_pt_1, g_pt_2, Scalar(0, 255, 0), 2, 8, 0);
//      				line(histImage, r_pt_1, r_pt_2, Scalar(0, 0, 255), 2, 8, 0);
//  				}
//
//				imshow("Histogram", histImage);
//				
//				setHueRange(histogram_arr_b_new, h_peak_y, h_peak_x / 2);
//				setSaturationRange(histogram_arr_g_new, s_peak_y, s_peak_x / 2);
//				setValueRange(histogram_arr_r_new, v_peak_y, v_peak_x / 2);
//
//				saturation_max = saturation_max / 2;
//				saturation_min = saturation_min / 2;
//
//				cout << hue_max << "   " << hue_min << endl;
//				cout << saturation_max << "   " << saturation_min << endl;
//				cout << value_max << "   " << value_min << endl;
//
//				obj.setHSVMin(Scalar(hue_min, saturation_min, value_min));
//				obj.setHSVMax(Scalar(hue_max, saturation_max, value_max));
//
//				flag = 1;
//			}
//
//			Mat drawing = Mat::zeros(morphed_diff_frame.size(), CV_8UC3);
//			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
//			rectangle(drawing, objectBoundingRectangle.tl(), objectBoundingRectangle.br(), color, 2, 8, 0);
//
//			//Show in a window
//			if(drawing.data == NULL)
//				cout << "NULL" << endl;
//
//			if(global_flag == 0) {
//				
//				namedWindow("Contours", CV_WINDOW_AUTOSIZE);
//				imshow("Contours", drawing);
//			}
//
//			HSVMinimum = Scalar(hue_min, saturation_min, value_min);
//			HSVMaximum = Scalar(hue_max, saturation_max, value_max);
//
//			track();
//		}
//
//		if(waitKey(10) == 33)
//			return 0;
//	}
//
//	cout << "Broken from While" << endl;
//
//	return 0;
//}