#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <unistd.h>
#include <sqlite3.h>
#include <sched.h>
#include <time.h>
#include <semaphore.h>
#include <time.h>
#include <syslog.h>
#include <sys/time.h>
#include <errno.h>

using namespace std;
using namespace cv;

#define USEC_PER_MSEC (1000)
#define NANOSEC_PER_SEC (1000000000)
#define NUM_THREADS 3
#define TRUE (1)
#define FALSE (0)
#define sequencePeriods 900
#define USEC_PER_MSEC (1000)
#define NANOSEC_PER_SEC (1000000000)
#define NUM_CPU_CORES (2)

VideoCapture capture;
Mat frame, image;
string inputName;
bool tryflip;
CascadeClassifier cascade, nestedCascade, thirdCascade;
double scale;
string cascadeName;
string nestedCascadeName;
string thirdCascadeName;
int total_price = 0;
int monsterPrice = 2;
int redbullPrice = 2;
int countval = 10;
int countval1 = 10;
int rc;
sqlite3 *db;
sem_t semS1, semS2,semS3;
int WCET1=0;
int WCET2=0;
int abortTest=FALSE;
int abortS1=FALSE, abortS2=FALSE, abortS3=FALSE, abortS4=FALSE, abortS5=FALSE, abortS6=FALSE, abortS7=FALSE;

struct timeval start_time_val;

typedef struct
{
    int threadIdx;
    //unsigned long long sequencePeriods;
} threadParams_t;


void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade, CascadeClassifier& thirdCascade,
                    double scale, bool tryflip );


void initCamera()
{
	/*if( inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1) )
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
    	}*/

    	if( capture.isOpened() )
    	{
        	cout << "Video capturing has been started ..." << endl;
		
		rc = sqlite3_open("prices.db", &db);
    		sqlite3_stmt *stmt;
    		sqlite3_prepare_v2(db, "SELECT PRICE FROM OBJECT_PRICE WHERE OBJECT_ID LIKE 3", -1, &stmt, NULL);
		int rc = sqlite3_step(stmt);
		monsterPrice = sqlite3_column_int(stmt, 0);
    		printf("Starting monster value: %d\n", sqlite3_column_int(stmt, 0));
   		
		sqlite3_stmt *stmt1;
   		sqlite3_prepare_v2(db, "SELECT PRICE FROM OBJECT_PRICE WHERE OBJECT_ID LIKE 2", -1, &stmt1, NULL);
		rc = sqlite3_step(stmt1);
		redbullPrice = sqlite3_column_int(stmt1, 0);
    		printf("Starting redbull value: %d\n", sqlite3_column_int(stmt1, 0));
    	}
    	
	capture >> frame;
    	if(frame.empty())
    		pthread_exit(NULL);
}

static void help()
{
    cout << " For debug \n" << endl;
}

void *Sequencer(void *threadp)
{
	printf("Here seq\n");
    	struct timeval current_time_val;
    	struct timespec delay_time = {0,300000000}; // delay for 100 msec, 10 Hz
    	struct timespec remaining_time;

    	double current_time;
    	double residual;
    	int rc, delay_cnt=0;
    	unsigned long long seqCnt=0;

    	threadParams_t *threadParams = (threadParams_t *)threadp;

    	// threadParams[0].sequencePeriods=900;
    
	// printf("Here\n");
  	// gettimeofday(&current_time_val, (struct timezone *)0);
   	// syslog(LOG_CRIT, "Sequencer thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
    	//printf("Sequencer thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

    	do
    	{
        	delay_cnt=0; residual=0.0;

        	//gettimeofday(&current_time_val, (struct timezone *)0);
        	//syslog(LOG_CRIT, "Sequencer thread prior to delay @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
        	do
       		{
            		rc=nanosleep(&delay_time, &remaining_time);

            		if(rc == EINTR)
            		{ 
                		residual = remaining_time.tv_sec + ((double)remaining_time.tv_nsec / (double)NANOSEC_PER_SEC);

                		if(residual > 0.0) printf("residual=%lf, sec=%d, nsec=%d\n", residual, (int)remaining_time.tv_sec, (int)remaining_time.tv_nsec);
 
                		delay_cnt++;
            		}
		    	else if(rc < 0)
		    	{
				perror("Sequencer nanosleep");
				exit(-1);
		    	}  
        	}while((residual > 0.0) && (delay_cnt < 100));

        	// seqCnt++;
        	// gettimeofday(&current_time_val, (struct timezone *)0);
        	// syslog(LOG_CRIT, "Sequencer cycle %llu @ sec=%d, msec=%d\n", seqCnt, (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

  	     	// if(delay_cnt > 1) printf("Sequencer looping delay %d\n", delay_cnt);
		struct timeval previous, current;

        	// Release each service at a sub-rate of the generic sequencer rate

        	// Servcie_1 = RT_MAX-1	@ 3 Hz
        	if((seqCnt % 15) == 0) 
		{
			sem_post(&semS1);
			gettimeofday(&current, (struct timezone *)0);
			int microseconds = (current.tv_sec - previous.tv_sec) * 1000000 + ((int)current.tv_usec - (int)previous.tv_usec);
	    		int TimeforSeq = microseconds/1000;	
			printf("Sequencer time difference: %d\n", TimeforSeq);
			previous = current;
		}

        	// Service_2 = RT_MAX-2	@ 1 Hz
        	if((seqCnt % 15) == 0) sem_post(&semS2);

        	// gettimeofday(&current_time_val, (struct timezone *)0);
        	// syslog(LOG_CRIT, "Sequencer release all sub-services @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

    	}while(!abortTest);

    	sem_post(&semS1); sem_post(&semS2);
    	abortS1=TRUE; abortS2=TRUE;
    	pthread_exit((void *)0);
}

void *Service1(void *threadp)
{
	struct timeval start_time, end_time;

        for(;;)
        {
		// sem_wait(&semS3);
	    	sem_wait(&semS1);
	    	// cout << "In service 1 "<< endl;
	    	gettimeofday(&start_time, (struct timezone *)0);
	    	// printf("Service 1 starts @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
	   	// printf("In service 1\n");
            	capture >> frame;
		 
            	if(frame.empty())
            		pthread_exit(NULL);
	    	char c = (char)waitKey(10);
  	    	
		if( c == 27 || c == 'q' || c == 'Q' )
			break;
	    	//pthread_exit(NULL);
  	   	gettimeofday(&end_time, (struct timezone *)0);
	    	// printf("Service 1 stop @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
	    	int microseconds = (end_time.tv_sec - start_time.tv_sec) * 1000000 + ((int)end_time.tv_usec - (int)start_time.tv_usec);
	    	int TimeforService = microseconds/1000;	
	    	// printf("WCET of service 1 is %d \n", milliseconds);
            	if (TimeforService > WCET1)
		{
			WCET1 = TimeforService;
		}
	    	//sem_post(&semS3);
        }
}

void *Service2(void *threadp)
{
	struct timeval start_time, end_time;
	sem_post(&semS3);
	for(;;)
	{	
		sem_wait(&semS2);
		// cout << "In service 2 "<< endl;
		gettimeofday(&start_time, (struct timezone *)0);
		// usleep(10000);
		detectAndDraw( frame, cascade, nestedCascade, thirdCascade, scale, tryflip );
		    
		char c = (char)waitKey(10);
		if( c == 27 || c == 'q' || c == 'Q' )
			break;
		gettimeofday(&end_time, (struct timezone *)0);
	    	// printf("Service 1 stop @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
	    	int microseconds = (end_time.tv_sec - start_time.tv_sec) * 1000000 + ((int)end_time.tv_usec - (int)start_time.tv_usec);
	    	int TimeforService = microseconds/1000;	
	    	// printf("WCET of service 1 is %d \n", milliseconds);
            	if (TimeforService > WCET2)
		{
			WCET2 = TimeforService;
		}
		// pthread_exit(NULL);
		// sem_post(&semS3);
	}
}


int main (int argc, const char** argv) {
   gettimeofday(&start_time_val, (struct timezone *)0);
   pthread_t threads[NUM_THREADS];
   int rc,scope;
   cpu_set_t threadcpu;
   int i;
   threadParams_t threadParams[NUM_THREADS];
   struct sched_param main_param;
   pthread_attr_t main_attr;
   cpu_set_t allcpuset;

   sem_init (&semS1, 0, 0);
   sem_init (&semS2, 0, 0);
   sem_init (&semS3, 0, 0);
   pid_t mainpid;
   //sem_post(&semS1);
   
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
	  
initCamera();
 	
	struct sched_param rt_param[NUM_THREADS];
    	pthread_attr_t rt_sched_attr[NUM_THREADS];

	int rt_max_prio, rt_min_prio;
	
	rt_max_prio = sched_get_priority_max(SCHED_FIFO);
	rt_min_prio = sched_get_priority_min(SCHED_FIFO);

	//assinging the current main process thread to be of highest priority
	/*struct sched_param params;
	params.sched_priority = rt_max_prio;
	int status = pthread_setschedparam(pthread_self(), SCHED_FIFO, &params);
	if (status != 0) 
	{
		printf("Unsuccessful in setting thread realtime prio\n");
		return 1;     
	}
	 cpu_set_t allcpuset;
 	CPU_ZERO(&allcpuset);

   	for(i=0; i < 1; i++)
      	CPU_SET(i, &allcpuset);

	for(int i = 0; i < NUM_THREADS; i++)
	{
	pthread_attr_init(&rt_sched_attr[i]);
        pthread_attr_setinheritsched(&rt_sched_attr[i], PTHREAD_EXPLICIT_SCHED);
        pthread_attr_setschedpolicy(&rt_sched_attr[i],  SCHED_FIFO);
	pthread_attr_setschedparam(&rt_sched_attr[i], &rt_param[i]);
	}*/
	////
	rc=sched_getparam(mainpid, &main_param);
    main_param.sched_priority=rt_max_prio;
    rc=sched_setscheduler(getpid(), SCHED_FIFO, &main_param);
    if(rc < 0) perror("main_param");
    //print_scheduler();


    pthread_attr_getscope(&main_attr, &scope);

    if(scope == PTHREAD_SCOPE_SYSTEM)
      printf("PTHREAD SCOPE SYSTEM\n");
    else if (scope == PTHREAD_SCOPE_PROCESS)
      printf("PTHREAD SCOPE PROCESS\n");
    else
      printf("PTHREAD SCOPE UNKNOWN\n");

    printf("rt_max_prio=%d\n", rt_max_prio);
    printf("rt_min_prio=%d\n", rt_min_prio);

	CPU_ZERO(&allcpuset);
	for(i=0; i < NUM_CPU_CORES; i++)
       CPU_SET(i, &allcpuset);

    for(i=0; i < NUM_THREADS; i++)
    {

      //CPU_ZERO(&threadcpu);
      //CPU_SET(3, &threadcpu);

      rc=pthread_attr_init(&rt_sched_attr[i]);
      rc=pthread_attr_setinheritsched(&rt_sched_attr[i], PTHREAD_EXPLICIT_SCHED);
      rc=pthread_attr_setschedpolicy(&rt_sched_attr[i], SCHED_FIFO);
      //rc=pthread_attr_setaffinity_np(&rt_sched_attr[i], sizeof(cpu_set_t), &threadcpu);

      rt_param[i].sched_priority=rt_max_prio-i;
      pthread_attr_setschedparam(&rt_sched_attr[i], &rt_param[i]);

      threadParams[i].threadIdx=i;
    }
	////
	rt_param[0].sched_priority = rt_max_prio;
	rt_param[1].sched_priority = rt_max_prio-2;
	rt_param[2].sched_priority = rt_max_prio-5;
		//rt_param[1].sched_priority = rt_max_prio - (2);
	
  
      cout << "main() : creating thread, " << i << endl;
      //rc = pthread_create(&threads[0], &rt_sched_attr[0], Service1, (void *)(0));
	rc = pthread_create(&threads[0], &rt_sched_attr[0], Sequencer, (void *)(0));
       if (rc) {
         cout << "Error:unable to create thread," << rc << endl;
         exit(-1);
	}

	rc = pthread_create(&threads[1], &rt_sched_attr[1], Service1, (void *)&(threadParams[1]));
       if (rc) {
         cout << "Error:unable to create thread," << rc << endl;
         exit(-1);
      }

	rc = pthread_create(&threads[2], &rt_sched_attr[2], Service2, (void *)&(threadParams[2]));
       if (rc) {
         cout << "Error:unable to create thread," << rc << endl;
         exit(-1);
      }
	
   
   pthread_join(threads[0], NULL);
   pthread_join(threads[1], NULL);
   pthread_join(threads[2], NULL);
   pthread_exit(NULL);
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade, CascadeClassifier& thirdCascade,
                    double scale, bool tryflip )
{
    static int iterationCnt = 0;
    double t = 0;
    vector<Rect> faces, faces2,faces3, faces4;
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
    //printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    if(faces.size() <= 0)
    {
	/*printf("No monster\n");
	countval--;
	if(countval <= 0 && total_price != 0)
	{
		total_price -= monsterPrice;
		countval = 0;
	}*/
    }
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        //double aspect_ratio = (double)r.width/r.height;
        /*if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
	    printf("Radius: %d\n", radius);
            if(radius > 145)
	    {
		
	    }		
            circle( img, center, radius, color, 3, 8, 0 );
        }*/
        //else
	{
            rectangle( img, Point(cvRound(r.x*scale), cvRound(r.y*scale)),
                       Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);
	    //printf("Monster detected\n");
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
    nestedCascade.detectMultiScale( smallImg, faces3,
        1.1, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE,
        Size(30, 30) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        nestedCascade.detectMultiScale( smallImg, faces4,
                                 1.1, 2, 0
                                 //|CASCADE_FIND_BIGGEST_OBJECT
                                 //|CASCADE_DO_ROUGH_SEARCH
                                 |CASCADE_SCALE_IMAGE,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces4.begin(); r != faces4.end(); ++r )
        {
            faces3.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)getTickCount() - t;
    //printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    if(faces3.size() <= 0)
    {
	/*printf("No redbull\n");
	countval1--;
	if(countval1 <= 0 && total_price != 0)
	{
		total_price -= redbullPrice;
		countval1 = 0;
	}*/
    }
    for ( size_t i = 0; i < faces3.size(); i++ )
    {
        Rect r = faces3[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        /*double aspect_ratio = (double)r.width/r.height;
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
        else*/
	{
            rectangle( img, Point(cvRound(r.x*scale), cvRound(r.y*scale)),
                       Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);
	    //printf("Redbull detected\n");
		total_price += redbullPrice;
		countval1 = 10;
	    putText(img, "Redbull", Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale - 5)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }
	//putText(img, "Redbull", Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale - 5)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
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

	/*cvtColor( img, gray, COLOR_BGR2GRAY );
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
    //}
	if(iterationCnt % 10 == 0)
	{
	printf("Total price: %d\n", total_price);
	printf("WCET1= %d ms\n",WCET1);
        printf("WCET2= %d ms\n",WCET2);
	iterationCnt = 0;
	}
    imshow( "result", img );
	//usleep(10000);
	total_price = 0;
}
