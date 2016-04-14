//#include "httpClient.h"
//#include <cstdio>
//using namespace std;
//
//int main() {
//
//	httpClient hc = httpClient("http://root:root@192.168.42.173/axis-cgi/mjpg/video.cgi?resolution=640x480&req_fps=30&.mjpg");
//	//hc.setInitPos(10, 60, 1);
//
//	std::cout << hc.getInitPos() << std::endl;
//
//	while(1) {
//	
//		int p;
//		int t;
//		int z;
//
//		cin >> p;
//		cin >> t;
//		cin >> z;
//
//		hc.setPTZ(p, t, z);
//	}
//
//	return 0;
//}
////
////#include "opencv2/video/tracking.hpp"
////#include "opencv2/imgproc/imgproc.hpp"
////#include "opencv2/highgui/highgui.hpp"
////
////#include <iostream>
////#include <ctype.h>
////
////using namespace cv;
////using namespace std;
////
////Mat image;
////
////bool selectObject = false;
////int trackObject = 0;
////Point origin;
////Rect selection;
////int vmin = 10, vmax = 256, smin = 30;
////
////static void onMouse( int event, int x, int y, int, void* )
////{
////    if( selectObject )
////    {
////        selection.x = MIN(x, origin.x);
////        selection.y = MIN(y, origin.y);
////        selection.width = std::abs(x - origin.x);
////        selection.height = std::abs(y - origin.y);
////        
////        selection &= Rect(0, 0, image.cols, image.rows);
////    }
////    
////    switch( event )
////    {
////        case CV_EVENT_LBUTTONDOWN:
////            origin = Point(x,y);
////            selection = Rect(x,y,0,0);
////            selectObject = true;
////            break;
////        case CV_EVENT_LBUTTONUP:
////            selectObject = false;
////            if( selection.width > 0 && selection.height > 0 )
////                trackObject = -1;
////            break;
////    }
////}
////
////static void help()
////{
////    cout << "\nCAMShift based object tracking\n"
////    "You select a colored object and the algorithm tracks it.\n"
////    "This takes the input from the webcam\n"
////    "Usage: \n"
////    "$ ./main [camera number]\n";
////    
////    cout << "\n\nKeyboard input options: \n"
////    "\tESC - quit the program\n"
////    "\ts - stop the tracking\n"
////    "\tp - pause video\n"
////    "\nTo start tracking an object, select the rectangular region around it with the mouse\n\n";
////}
////
////int main( int argc, const char** argv )
////{
////    
////	httpClient hc = httpClient("http://root:root@192.168.42.175/axis-cgi/mjpg/video.cgi?resolution=640x480&req_fps=30&.mjpg");
////	CvCapture* camera = cvCaptureFromFile("http://root:root@192.168.42.175/axis-cgi/mjpg/video.cgi?resolution=640x480&req_fps=30&.mjpg");
////	IplImage* back;
////
////	help();
////
////    VideoCapture cap;
////    Rect trackWindow;
////    int hsize = 16;
////    float hranges[] = {0,180};
////    const float* phranges = hranges;
////
////	int cam_pan = 0;
////    
////    int camNum = 0;
////    
////    namedWindow( "CamShift Object Tracker", 0 );
////    setMouseCallback( "CamShift Object Tracker", onMouse, 0 );
////    
////    Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
////    bool paused = false;
////    
////    for(;;)
////    {
////        if( !paused )
////        {
////            //cap >> frame;
////			back = cvQueryFrame(camera);
////			frame = cvarrToMat(back);
////
////            if( frame.empty() )
////                break;
////        }
////        
////        frame.copyTo(image);
////        
////        if( !paused )
////        {
////            cvtColor(image, hsv, CV_BGR2HSV);
////            
////            if( trackObject )
////            {
////                int _vmin = vmin, _vmax = vmax;
////                
////                inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
////                        Scalar(180, 256, MAX(_vmin, _vmax)), mask);
////
////				if(!hue.empty())
////					imshow("Hue", hue);
////
////				imshow("Mask", mask);
////
////                int ch[] = {0, 0};
////                hue.create(hsv.size(), hsv.depth());
////                mixChannels(&hsv, 1, &hue, 1, ch, 1);
////                
////                if( trackObject < 0 )
////                {
////                    Mat roi(hue, selection), maskroi(mask, selection);
////                    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
////                    normalize(hist, hist, 0, 255, CV_MINMAX);
////                    
////                    trackWindow = selection;
////                    trackObject = 1;
////                    
////                    histimg = Scalar::all(0);
////                    int binW = histimg.cols / hsize;
////                    Mat buf(1, hsize, CV_8UC3);
////                    for( int i = 0; i < hsize; i++ )
////                        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
////                    cvtColor(buf, buf, CV_HSV2BGR);
////                    
////                    for( int i = 0; i < hsize; i++ )
////                    {
////                        int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
////                        rectangle( histimg, Point(i*binW,histimg.rows),
////                                  Point((i+1)*binW,histimg.rows - val),
////                                  Scalar(buf.at<Vec3b>(i)), -1, 8 );
////                    }
////                }
////                
////                calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
////                backproj &= mask;
////                RotatedRect trackBox = CamShift(backproj, trackWindow,
////                                                TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
////                if( trackWindow.area() <= 1 )
////                {
////                    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
////                    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
////                                       trackWindow.x + r, trackWindow.y + r) &
////                    Rect(0, 0, cols, rows);
////                }
////                
////                ellipse( image, trackBox, Scalar(0, 0, 255), 3, CV_AA );
////
////				Point centroid = trackBox.center;
////
////				if(centroid.x > 500)
////					cam_pan += 5;
////
////				else if(centroid.x > 500)
////					cam_pan -= 5;
////
////				hc.setPTZ(cam_pan, 0, 1);
////            }
////        }
////
////        else if( trackObject < 0 )
////            paused = false;
////        
////        if( selectObject && selection.width > 0 && selection.height > 0 )
////        {
////            Mat roi(image, selection);
////            bitwise_not(roi, roi);
////        }
////        
////        imshow( "CamShift Object Tracker", image );
////        
////        char c = (char)waitKey(10);
////        if( c == 27 )
////            break;
////        switch(c)
////        {
////            case 's':
////                trackObject = 0;
////                histimg = Scalar::all(0);
////                break;
////
////            case 'p':
////                paused = !paused;
////                break;
////                
////            default:
////                ;
////        }
////    }
////    
////    return 0;
////}
//
////#include "httpClient.h"
////
////int main() {
////
////	httpClient hc = httpClient("http://root:root@192.168.42.175/axis-cgi/mjpg/video.cgi?resolution=640x480&req_fps=30&.mjpg");
////	//hc.setInitPos(10, 60, 1);
////
////	std::cout << hc.getInitPos() << std::endl;
////
////	hc.setPTZ(60, 100, 1);
////
////	getchar();
////
////	return 0;
////}
//
//#include "opencv2/video/tracking.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//
//#include <iostream>
//#include <ctype.h>
//
//using namespace cv;
//using namespace std;
//
//Mat image;
//
//bool selectObject = false;
//int trackObject = 0;
//Point origin;
//Rect selection;
//int vmin = 10, vmax = 256, smin = 30;
//
//static void onMouse( int event, int x, int y, int, void* )
//{
//    if( selectObject )
//    {
//        selection.x = MIN(x, origin.x);
//        selection.y = MIN(y, origin.y);
//        selection.width = std::abs(x - origin.x);
//        selection.height = std::abs(y - origin.y);
//        
//        selection &= Rect(0, 0, image.cols, image.rows);
//    }
//    
//    switch( event )
//    {
//        case CV_EVENT_LBUTTONDOWN:
//            origin = Point(x,y);
//            selection = Rect(x,y,0,0);
//            selectObject = true;
//            break;
//        case CV_EVENT_LBUTTONUP:
//            selectObject = false;
//            if( selection.width > 0 && selection.height > 0 )
//                trackObject = -1;
//            break;
//    }
//}
//
//static void help()
//{
//    cout << "\nCAMShift based object tracking\n"
//    "You select a colored object and the algorithm tracks it.\n"
//    "This takes the input from the webcam\n"
//    "Usage: \n"
//    "$ ./main [camera number]\n";
//    
//    cout << "\n\nKeyboard input options: \n"
//    "\tESC - quit the program\n"
//    "\ts - stop the tracking\n"
//    "\tp - pause video\n"
//    "\nTo start tracking an object, select the rectangular region around it with the mouse\n\n";
//}
//
//int main( int argc, const char** argv )
//{    
//	CvCapture* camera = cvCaptureFromFile("http://root:root@192.168.42.175/axis-cgi/mjpg/video.cgi?resolution=640x480&req_fps=30&.mjpg");
//	IplImage* back;
//
//    VideoCapture cap;
//    Rect trackWindow;
//    int hsize = 16;
//    float hranges[] = {0,180};
//    const float* phranges = hranges;
//
//	int cam_pan = 0;
//    
//    int camNum = 0;
//    if(argc == 2)
//        camNum = atoi(argv[1]);
//    
//    namedWindow( "CamShift Object Tracker", 0 );
//    setMouseCallback( "CamShift Object Tracker", onMouse, 0 );
//    
//    Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
//    bool paused = false;
//    
//    for(;;)
//    {
//        if( !paused )
//        {
//            //cap >> frame;
//			back = cvQueryFrame(camera);
//			frame = cvarrToMat(back);
//
//            if( frame.empty() )
//                break;
//        }
//        
//        frame.copyTo(image);
//        
//        if( !paused )
//        {
//            cvtColor(image, hsv, CV_BGR2HSV);
//            
//            if( trackObject )
//            {
//                int _vmin = vmin, _vmax = vmax;
//                
//                inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
//                        Scalar(180, 256, MAX(_vmin, _vmax)), mask);
//
//				/*if(!hue.empty())
//					imshow("Hue", hue);
//
//				imshow("Mask", mask);*/
//
//                int ch[] = {0, 0};
//                hue.create(hsv.size(), hsv.depth());
//                mixChannels(&hsv, 1, &hue, 1, ch, 1);
//                
//                if( trackObject < 0 )
//                {
//                    Mat roi(hue, selection), maskroi(mask, selection);
//                    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
//                    normalize(hist, hist, 0, 255, CV_MINMAX);
//                    
//                    trackWindow = selection;
//                    trackObject = 1;
//                    
//                    histimg = Scalar::all(0);
//                    int binW = histimg.cols / hsize;
//                    Mat buf(1, hsize, CV_8UC3);
//                    for( int i = 0; i < hsize; i++ )
//                        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
//                    cvtColor(buf, buf, CV_HSV2BGR);
//                    
//                    for( int i = 0; i < hsize; i++ )
//                    {
//                        int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
//                        rectangle( histimg, Point(i*binW,histimg.rows),
//                                  Point((i+1)*binW,histimg.rows - val),
//                                  Scalar(buf.at<Vec3b>(i)), -1, 8 );
//                    }
//                }
//                
//                calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
//                backproj &= mask;
//                RotatedRect trackBox = CamShift(backproj, trackWindow,
//                                                TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
//                if( trackWindow.area() <= 1 )
//                {
//                    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
//                    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
//                                       trackWindow.x + r, trackWindow.y + r) &
//                    Rect(0, 0, cols, rows);
//                }
//                
//                ellipse( image, trackBox, Scalar(0, 0, 255), 3, CV_AA );
//				Point centroid = trackBox.center;
//
//				if(centroid.x > 500) {
//
//					httpClient hc = httpClient("http://root:root@192.168.42.175/axis-cgi/mjpg/video.cgi?resolution=640x480&req_fps=30&.mjpg");
//					cout << centroid.x;
//					cam_pan += 5;
//					hc.setPTZ(cam_pan, 0, 1);
//				}
//
//				else if(centroid.x < 180) {
//
//					httpClient hc = httpClient("http://root:root@192.168.42.175/axis-cgi/mjpg/video.cgi?resolution=640x480&req_fps=30&.mjpg");
//					cout << centroid.x;
//					cam_pan -= 5;
//					hc.setPTZ(cam_pan, 0, 1);
//				}
//
//				//hc.setCenter(centroid.x, centroid.y);
//
//				//getchar();
//            }
//        }
//
//        else if( trackObject < 0 )
//            paused = false;
//        
//        if( selectObject && selection.width > 0 && selection.height > 0 )
//        {
//            Mat roi(image, selection);
//            bitwise_not(roi, roi);
//        }
//        
//        imshow( "CamShift Object Tracker", image );
//        
//        char c = (char)waitKey(10);
//        if( c == 27 )
//            break;
//        switch(c)
//        {
//            case 's':
//                trackObject = 0;
//                histimg = Scalar::all(0);
//                break;
//
//            case 'p':
//                paused = !paused;
//                break;
//                
//            default:
//                ;
//        }
//    }
//    
//    return 0;
//}