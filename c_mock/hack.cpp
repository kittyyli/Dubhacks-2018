#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include <iostream>
#include <cstdio>

using namespace cv;
//For compatibility with opencv2
namespace cv
{
    using std::vector;
}

//./cv_practice /mnt/c/Users/Sasha/Downloads/tennisball2.jpg

Mat color_corrected(Mat img)
{
    Mat lab_img;
    cvtColor(img, lab_img, CV_BGR2Lab);
    vector<Mat> lab_planes(3);
    split(lab_img, lab_planes);

    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4);
    Mat dst;
    clahe->apply(lab_planes[0], dst);
    dst.copyTo(lab_planes[0]);
    merge(lab_planes, lab_img);

    Mat corrected;
    cvtColor(lab_img, corrected, CV_Lab2BGR);
    return(corrected);
}

Mat threshold_image(Mat img)
{
    Mat hsv(img.rows, img.cols, CV_8UC3);
    cvtColor(img, hsv, CV_BGR2HSV);
    Mat thresh(img.rows, img.cols, CV_8UC1);
    inRange(hsv, Scalar(0.11 * 256, 0.60 * 256, 0.20 * 256), Scalar(0.14 * 256, 1.0 * 255.0, 1.0 * 256), thresh);
    return(thresh);
}

Mat morphed_img(Mat mask)
{
    Mat se21 = getStructuringElement(MORPH_RECT, Size(21, 21));
    Mat se11 = getStructuringElement(MORPH_RECT, Size(11, 11));

    morphologyEx(mask, mask, MORPH_CLOSE, se21);
    morphologyEx(mask, mask, MORPH_OPEN, se11);

    GaussianBlur(mask, mask, Size(15, 15), 0, 0);
    return(mask);   
}

Mat overall_filter(Mat img)
{
//    Mat corrected = color_corrected(img);
    Mat mask = threshold_image(img);
    Mat filtered = morphed_img(mask);
    return(filtered);
}

void hough_circles_identifier(Mat src)
{
    Mat hough_in = overall_filter(src);
    vector<Vec3f> circles;
    /// Apply the Hough Transform to find the circles
    HoughCircles(hough_in, circles, CV_HOUGH_GRADIENT, 1.1, hough_in.rows/10, 100, 40, 0, 0);

    printf("circles: %lu\n", circles.size());
    /// Draw the circles detected
    for(size_t i = 0; i < circles.size(); i++)
    {
	Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
	int radius = cvRound(circles[i][2]);
	// circle center
	circle(src, center, 3, Scalar(0,255,0), -1, 8, 0);
	// circle outline
	circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);
    }

    /// Show your results
    namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
    imshow("Hough Circle Transform Demo", hough_in);
}

/*
  take your post-processed images and get the connected components, draw a bounding square around them,
  take the inscribed circle of the bounding square, then calculate a coverage overlap
  that will give you a "percent like a circle" metric
  and you can tune that threshold to whatever is best for your application
 */
void connected_components_identifier(Mat src)
{
    Mat filtered = overall_filter(src);
    Mat labels;
    int components = connectedComponents(filtered, labels);
    printf("%d connected components\n", components);
    vector<Rect> rectangles;
    for(int i = 1; i < components; i++)
    {
	Mat component_i;
	inRange(labels, Scalar(i), Scalar(i), component_i);
	Rect r = boundingRect(component_i);
	
	Point center = (r.tl() + r.br()) / 2;
	int radius = abs(r.tl().y - center.y);
	Mat circ(component_i.size(), component_i.type());
	circ = Scalar(0, 0, 0);
	circle(circ, center, radius, i, -1);

	Mat intersection;
	bitwise_and(circ, component_i, intersection);
	Mat union_;
	bitwise_or(circ, component_i, union_);

	printf("intersection: %d\n", countNonZero(intersection));
	printf("union: %d\n", countNonZero(union_));

	float iou = (float)countNonZero(intersection) / (float)countNonZero(union_);

	printf("%.3f iou\n", iou);

#define CIRCLE_THRESH 0.8f

	if(iou >= CIRCLE_THRESH)
	    rectangles.push_back(r);

#undef CIRCLE_THRESH
    }

    for(Rect r : rectangles)
    {
	rectangle(src, r.tl(), r.br(), Scalar(255, 0, 0), 1);
    }
    
    normalize(labels, labels, 0, 255, NORM_MINMAX, CV_8U);
    namedWindow("Connected Components Transform", CV_WINDOW_AUTOSIZE);
    imshow("Connected Components Transform", src);
}

Mat art(Mat src)
{
    Mat res;
    Canny(src, res, 150, 200, 3);
/*    Mat se1 = getStructuringElement(MORPH_RECT, Size(1, 1));
    Mat se3 = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat se9 = getStructuringElement(MORPH_RECT, Size(9, 9));
    morphologyEx(res, res, MORPH_DILATE, se3);
    morphologyEx(res, res, MORPH_ERODE, se3);*/
    return res;
}

int main(int argc, char** argv)
{
    VideoCapture cap(0);
    if(!cap.isOpened())
    {
        printf("failed to open webcam\n");
        return -1;
    }
    namedWindow("bork", CV_WINDOW_AUTOSIZE);
    for(;;)
    {
        Mat src;
        cap >> src;
        printf("got frame\n");
        Mat bork = art(src);
        imshow("bork", bork);
        if(waitKey(30) >= 0) break;
    }
    return 0;
}
