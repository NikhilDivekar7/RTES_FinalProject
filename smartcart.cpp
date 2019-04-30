#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <unistd.h>
#include <sqlite3.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <semaphore.h>
using namespace std;
using namespace cv;

#define NUM_THREADS 2

VideoCapture capture;
Mat frame, image;
string inputName;
bool tryflip;
CascadeClassifier cascade, nestedCascade, thirdCascade;
double scale;
string cascadeName;
string nestedCascadeName;
string thirdCascadeName;
int monsterFlag = 10;
int total_price = 0;
int monsterPrice = 2;
int countval = 10;
int rc;
sqlite3 *db;

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade, CascadeClassifier& thirdCascade,
                    double scale, bool tryflip );


static void help()
{
    cout << "\nThis program demonstrates the use of cv::CascadeClassifier class to detect objects (Face + eyes). You can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
               "   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
               "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
               "   [--try-flip]\n"
               "   [filename|camera_index]\n\n"
            "see facedetect.cmd for one call:\n"
            "./facedetect --cascade=\"data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"data/haarcascades/haarcascade_eye_tree_eyeglasses.xml\" --scale=1.3\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

void *Service1(void *threadp)
{
    if( inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1) )
    {
        int camera = inputName.empty() ? 0 : inputName[0] - '0';
        if(!capture.open(camera))
        {
            cout << "Capture from camera #" <<  camera << " didn't work" << endl;
            exit(1);
        }
    }
    else if (!inputName.empty())
    {
        image = imread(samples::findFileOrKeep(inputName), IMREAD_COLOR);
        if (image.empty())
        {
            if (!capture.open(samples::findFileOrKeep(inputName)))
            {
                cout << "Could not read " << inputName << endl;
                exit(1);
            }
        }
    }

    if( capture.isOpened() )
    {
        cout << "Video capturing has been started ..." << endl;
	rc = sqlite3_open("prices.db", &db);
    sqlite3_stmt *stmt;
    sqlite3_prepare_v2(db, "SELECT PRICE FROM OBJECT_PRICE WHERE OBJECT_ID LIKE 3", -1, &stmt, NULL);
	int rc = sqlite3_step(stmt);
	monsterPrice = sqlite3_column_int(stmt, 0);
    printf("Starting monster value: %d\n", sqlite3_column_int(stmt, 0));
	//sqlite3_finalize(stmt);


        for(;;)
        {
            capture >> frame;
            if( frame.empty() )
            pthread_exit(NULL);
	
        }
    }
}

void *Service2(void *threadp)
{
	for(;;)
	{	
		    detectAndDraw( frame, cascade, nestedCascade, thirdCascade, scale, tryflip );
		    char c = (char)waitKey(10);
		    if( c == 27 || c == 'q' || c == 'Q' )
		    pthread_exit(NULL);
	}
}


int main (int argc, const char** argv) {
   pthread_t threads[NUM_THREADS];
   int rc;
   int i;

cv::CommandLineParser parser(argc, argv,
        "{help h||}"
        "{cascade|../../opencv/data/haarcascades/cascade_monster.xml|}"
        "{nested-cascade|../../opencv/data/haarcascades/cascade_redbull.xml|}"
        "{scale|1|}{try-flip||}{@filename||}"
    );

if (parser.has("help"))
    {
        help();
        return 0;
    }
    cascadeName = parser.get<string>("cascade");
    nestedCascadeName = parser.get<string>("nested-cascade");
    thirdCascadeName = "../../opencv/data/haarcascades/banana.xml";
    scale = parser.get<double>("scale");
    if (scale < 1)
        scale = 1;
    tryflip = parser.has("try-flip");
    inputName = parser.get<string>("@filename");
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }


    if (!nestedCascade.load(samples::findFileOrKeep(nestedCascadeName)))
        cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
    if (!thirdCascade.load(samples::findFileOrKeep(thirdCascadeName)))
        cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
    if (!cascade.load(samples::findFile(cascadeName)))
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help();
        return -1;
    }
	   	
	struct sched_param rt_param[NUM_THREADS];
    	pthread_attr_t rt_sched_attr[NUM_THREADS];

	int rt_max_prio, rt_min_prio;
	
	rt_max_prio = sched_get_priority_max(SCHED_FIFO);
	rt_min_prio = sched_get_priority_min(SCHED_FIFO);

	//assinging the current main process thread to be of highest priority
	struct sched_param params;
	params.sched_priority = rt_max_prio-2;
	int status = pthread_setschedparam(pthread_self(), SCHED_FIFO, &params);
	if (status != 0) 
	{
		printf("Unsuccessful in setting thread realtime prio\n");
		return 1;     
	}




	for(int i = 0; i < NUM_THREADS; i++)
	{
	
	pthread_attr_init(&rt_sched_attr[i]);
        pthread_attr_setinheritsched(&rt_sched_attr[i], PTHREAD_EXPLICIT_SCHED);
        pthread_attr_setschedpolicy(&rt_sched_attr[i],  SCHED_FIFO);
	pthread_attr_setschedparam(&rt_sched_attr[i], &rt_param[i]);
	}
	
	
	rt_param[0].sched_priority = rt_max_prio;
	rt_param[1].sched_priority = rt_max_prio-1;
	//rt_param[1].sched_priority = rt_max_prio - (2);
	
  
      cout << "main() : creating thread, " << i << endl;
      //rc = pthread_create(&threads[0], &rt_sched_attr[0], Service1, (void *)(0));
	rc = pthread_create(&threads[0], &rt_sched_attr[0], Service1, (void *)(0));
       if (rc) {
         cout << "Error:unable to create thread," << rc << endl;
         exit(-1);
	}

	rc = pthread_create(&threads[1], &rt_sched_attr[1], Service2, (void *)(0));
       if (rc) {
         cout << "Error:unable to create thread," << rc << endl;
         exit(-1);
      }
   
   pthread_join(threads[0], NULL);
   pthread_join(threads[1], NULL);
   pthread_exit(NULL);
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade, CascadeClassifier& thirdCascade,
                    double scale, bool tryflip )
{
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =
    {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };
    Mat gray, smallImg;

    cvtColor( img, gray, COLOR_BGR2GRAY );
    double fx = 1 / (scale);
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT );
    equalizeHist( smallImg, smallImg );

    t = (double)getTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE,
        Size(30, 30) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CASCADE_FIND_BIGGEST_OBJECT
                                 //|CASCADE_DO_ROUGH_SEARCH
                                 |CASCADE_SCALE_IMAGE,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)getTickCount() - t;
    printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    if(faces.size() <= 0)
    {
	printf("No monster\n");
	countval--;
	if(countval <= 0 && total_price != 0)
	{
		total_price -= monsterPrice;
		countval = 0;
	}
    }
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
	    printf("Radius: %d\n", radius);
            if(radius > 145)
	    {
		
	    }		
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
	{
            rectangle( img, Point(cvRound(r.x*scale), cvRound(r.y*scale)),
                       Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);
	    printf("Monster detected\n");
		total_price += monsterPrice;
		countval = 10;
	    putText(img, "Monster", Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale - 5)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

	}
        /*if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg( r );
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
            1.1, 2, 0
            //|CASCADE_FIND_BIGGEST_OBJECT
            //|CASCADE_DO_ROUGH_SEARCH
            //|CASCADE_DO_CANNY_PRUNING
            |CASCADE_SCALE_IMAGE,
            Size(30, 30) );
        for ( size_t j = 0; j < nestedObjects.size(); j++ )
        {
            Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }*/
    }

    cvtColor( img, gray, COLOR_BGR2GRAY );
    fx = 1 / (scale);
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT );
    equalizeHist( smallImg, smallImg );

    t = (double)getTickCount();
    nestedCascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE,
        Size(30, 30) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        nestedCascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CASCADE_FIND_BIGGEST_OBJECT
                                 //|CASCADE_DO_ROUGH_SEARCH
                                 |CASCADE_SCALE_IMAGE,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)getTickCount() - t;
    printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
	    printf("Radius: %d\n", radius);
            if(radius > 145)
		printf("Face detected\n");
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( img, Point(cvRound(r.x*scale), cvRound(r.y*scale)),
                       Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);
	putText(img, "Redbull", Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale - 5)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        /*if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg( r );
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
            1.1, 2, 0
            //|CASCADE_FIND_BIGGEST_OBJECT
            //|CASCADE_DO_ROUGH_SEARCH
            //|CASCADE_DO_CANNY_PRUNING
            |CASCADE_SCALE_IMAGE,
            Size(30, 30) );
        for ( size_t j = 0; j < nestedObjects.size(); j++ )
        {
            Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }*/
    }

	cvtColor( img, gray, COLOR_BGR2GRAY );
    fx = 1 / (scale);
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT );
    equalizeHist( smallImg, smallImg );

    t = (double)getTickCount();
    thirdCascade.detectMultiScale( smallImg, faces,
        1.1, 4, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE,
        Size(30, 30) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        thirdCascade.detectMultiScale( smallImg, faces2,
                                 1.1, 4, 0
                                 //|CASCADE_FIND_BIGGEST_OBJECT
                                 //|CASCADE_DO_ROUGH_SEARCH
                                 |CASCADE_SCALE_IMAGE,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)getTickCount() - t;
    printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
	    printf("Radius: %d\n", radius);
            if(radius > 145)
		printf("Face detected\n");
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( img, Point(cvRound(r.x*scale), cvRound(r.y*scale)),
                       Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);
	putText(img, "Banana", Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale - 5)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        /*if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg( r );
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
            1.1, 2, 0
            //|CASCADE_FIND_BIGGEST_OBJECT
            //|CASCADE_DO_ROUGH_SEARCH
            //|CASCADE_DO_CANNY_PRUNING
            |CASCADE_SCALE_IMAGE,
            Size(30, 30) );
        for ( size_t j = 0; j < nestedObjects.size(); j++ )
        {
            Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }*/
    }
	printf("Total price: %d\n", total_price);
    imshow( "result", img );
	usleep(10000);
	total_price = 0;
}
