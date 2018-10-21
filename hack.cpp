#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include <iostream>
#include <cstdio>
#include <queue>
#include <algorithm>
#include <chrono>
#include <bitset>

using namespace cv;
using namespace std;
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

#define SE(x) Mat se##x = getStructuringElement(MORPH_RECT, Size(x, x));

Mat art(Mat src, bool beat = false)
{
#if 0
    Mat res;
    blur(src, res, Size(3, 3));
    Canny(src, res, 100, 200, 3);
    SE(1);
    SE(3);
    SE(5);
    Mat se_circle = getStructuringElement(MORPH_ELLIPSE, Size(7,  7));
    morphologyEx(res, res, MORPH_DILATE, se_circle);
    morphologyEx(res, res, MORPH_ERODE, se_circle);
    morphologyEx(res, res, MORPH_OPEN, se1);
    Mat labels;
    int components = connectedComponents(res, labels);
    assert(components > 0);
    constexpr int frames_per_component = 16;
    static int selected;
    static Scalar color;
    static int t;
    static Scalar sub;
    if(t % frames_per_component == 0 || selected >= components)
    {
        selected = rand() % components;
        color = Scalar(rand() % 125 + 128, rand() % 125 + 128, rand() % 125 + 128);
        sub = color / (float)frames_per_component;
        t = 0;
    }

    Mat selected_comp;    
    inRange(labels, Scalar(selected), Scalar(selected), selected_comp);
    Rect r = boundingRect(selected_comp);
    Point pt(r.x + rand() % r.width, r.y + rand() % r.height);
    
    Mat mask;
    copyMakeBorder(res, mask, 1, 1, 1, 1, BORDER_REPLICATE);
    Mat res2;
    cvtColor(res, res2, CV_GRAY2BGR);
    
    floodFill(res2, mask, pt, color, 0, Scalar(), Scalar(), (128 << 8) | 4);
    t++;
    color -= sub;
/*    morphologyEx(res, res, MORPH_DILATE, se3);
    morphologyEx(res, res, MORPH_ERODE, se5);
    morphologyEx(res, res, MORPH_DILATE, se5);
    morphologyEx(res, res, MORPH_DILATE, se5);
    bitwise_not(res, res);
    SE(11);
    morphologyEx(res, res, MORPH_ERODE, se3);
    morphologyEx(res, res, MORPH_DILATE, se3);
    morphologyEx(res, res, MORPH_DILATE, se5);*/
    return res2;
#elif 0
    Mat res;
    blur(src, res, Size(3, 3));
    Canny(src, res, 100, 200, 3);
    SE(1);
    SE(3);
    SE(5);
    SE(11);
    Mat se_circle = getStructuringElement(MORPH_ELLIPSE, Size(7,  7));
    morphologyEx(res, res, MORPH_DILATE, se_circle);
    morphologyEx(res, res, MORPH_ERODE, se_circle);
    morphologyEx(res, res, MORPH_OPEN, se1);
    Mat labels;
    int components = connectedComponents(res, labels);
    assert(components > 0);
    vector<Rect> rects;
    for(int i = 0; i < components; i++)
    {
        Mat selected_comp;
        inRange(labels, Scalar(i), Scalar(i), selected_comp);
        Rect r = boundingRect(selected_comp);
        rects.push_back(r);
    }

    vector<vector<int>> graph(components);
    for(int i = 0; i < components - 1; i++)
    {
        for(int j = i + 1; j < components; j++)
        {
            Rect r1 = rects[i];
            Rect r2 = rects[j];
            if(
                (
                    (r1.x + r1.width + 1 >= r2.x || r2.x + r2.width + 1 >= r1.x)
//                    && ((r1.y <= r2.y && r2.y <= r1.y + r1.height) || (r2.y <= r1.y && r1.y <= r2.y + r2.height))
                    ) ||
                (
                    (r1.y + r1.height + 1 == r2.y || r2.y + r2.height + 1 == r1.y)
//                    && ((r1.x <= r2.x && r2.x <= r1.x + r1.width) || (r2.x <= r1.x && r1.x <= r2.x + r2.width))
                    )
                )
            {
                graph[i].push_back(j);
                graph[j].push_back(i);
            }
        }
    }

    Mat mask;
    copyMakeBorder(res, mask, 1, 1, 1, 1, BORDER_REPLICATE);
    Mat res2;
    cvtColor(res, res2, CV_GRAY2BGR);
    
    vector<bool> available(components);
    vector<int> colors(components);
    fill(available.begin(), available.end(), false);
    fill(colors.begin(), colors.end(), -1);
    colors[0] = 0;
    for(int i = 1; i < components; i++)
    {
        for(int j : graph[i])
        {
            if(colors[j] != -1)
                available[colors[j]] = true;
        }
        int color;
        for(color = 0; color < components; color++)
            if(!available[color])
                break;
        colors[i] = color;
        for(int j : graph[i])
            if(colors[j] != -1)
                available[colors[j]] = false;
    }

    vector<Scalar> color_vals;
    for(int i = 0; i < components; i++)
    {
        auto &rect = rects[i];
        Point pt = Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
        int color = colors[i];
        if(i >= color_vals.size())
            color_vals.push_back(Scalar(rand() % 255, rand() % 255, rand() % 255));
        floodFill(res2, mask, pt, color_vals[i], 0, Scalar(), Scalar(), (128 << 8) | 4);
    }
    return res2;    
#elif 1
    static float target_hue;
    if(beat)
    {
        target_hue += 120 + rand() % 120;
        target_hue = fmod(target_hue, 360);
    }
    Mat blurred;
    blur(src, blurred, Size(3, 3));
    Mat canny_output;
    Canny(blurred, canny_output, 100, 200, 3);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
    Vec3f hsv;
    hsv[0] = target_hue;
    hsv[1] = 200 + rand() % 55;
    hsv[2] = 200 + rand() % 55;
    for(int i = 0; i < contours.size(); i++)
    {
	Mat hsv2(1, 1, CV_8UC3, Scalar(fmod(hsv[0] + rand() % 40 - 20 + 360, 360), hsv[1], hsv[2]));
	Mat rgb;
	cvtColor(hsv2, rgb, CV_HSV2BGR);
 	Scalar color = Scalar((int)rgb.at<cv::Vec3b>(0, 0)[0],(int)rgb.at<cv::Vec3b>(0, 0)[1],(int)rgb.at<cv::Vec3b>(0, 0)[2]);
        drawContours(drawing, contours, i, color, 1, CV_AA, hierarchy, 0, Point());
    }
    constexpr int kernel_size = 5;
    Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / ((float)(kernel_size * kernel_size));
//    filter2D(drawing, drawing, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);

/*    constexpr float contrast = 4.0f;
    constexpr float brightness = 0.0f;
    for(int y = 0; y < drawing.rows; y++)
        for(int x = 0; x < drawing.cols; x++)
            for(int c = 0; c < 3; c++)
                drawing.at<Vec3b>(y, x)[c] = saturate_cast<unsigned char>(contrast * drawing.at<Vec3b>(y, x)[c] + brightness);
*/
    return drawing;
#elif 0
    Mat blurred;
    blur(src, blurred, Size(3, 3));
    Mat canny_output;
    Canny(blurred, canny_output, 100, 200, 3);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
    for(int i = 0; i < contours.size(); i++)
    {
        Rect rect = boundingRect(contours[i]);
        if(rand() % 8 == 0)
        {
            for(int y = rect.y; y < rect.y + rect.height; y++)
                for(int x = rect.x; x < rect.x + rect.width; x++)
                {
                    double dist = pointPolygonTest(contours[i], Point2f(x, y), true);
                    int col = 0;
                    if(dist < 0)
                        col = 2;
                    drawing.at<Vec3b>(y, x)[col] = round(128 * dist);
                }
        }
        else
        {
            Scalar color(rand() % 255, rand() % 255, rand() % 255);
            drawContours(drawing, contours, i, color, 1, CV_AA, hierarchy, 0, Point());
        }
    }
    return drawing;
#endif
}

struct mp3_header
{
    int blank : 6;
    uint8_t channel_mode : 2;
    bool priv : 1;
    bool padding : 1;
    uint8_t freq : 2;
    uint8_t bitrate_idx : 4;
    bool protection : 1;
    uint8_t layer : 2;
    uint8_t audio_version : 2;
    uint16_t frame_sync : 11;
};

struct wav_header 
{
    uint32_t chunk_id;
    uint32_t chunk_size;
    uint32_t format;
    uint32_t subchunk_id;
    uint32_t subchunk1_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    uint32_t subchunk2_id;
    uint32_t subchunk2_size;
};

int bitrates[][5] =
{
    {0, 0, 0, 0, 0},
    {32, 32, 32, 32, 8},
    {64, 48, 40, 48, 16},
    {96, 56, 48, 56, 24},
    {128, 64, 56, 64, 32},
    {160, 80, 64, 80, 40},
    {192, 96, 80, 96, 48},
    {224, 112, 96, 112, 56},
    {256, 128, 112, 128, 64},
    {288, 160, 128, 144, 80},
    {320, 192, 160, 160, 96},
    {352, 224, 192, 176, 112},
    {384, 256, 224, 192, 128},
    {416, 320, 256, 224, 144},
    {448, 384, 320, 256, 160},
    {-1, -1, -1, -1, -1},
};

int sample_rates[][4] = 
{
    {11025, 0, 22050, 44100},
    {12000, 0, 24000, 48000},
    {8000, 0, 16000, 32000},
};

int bitrate_idx(uint8_t mpeg_version, uint8_t layer)
{
    switch(mpeg_version)
    {
    case 0:
    case 2:
	if(layer == 1)
	    return 3;
	return 4;
    case 3:
	return 3 - layer;
    default:
	return -1;
    }
}

template <typename T, int channels>
double detect_beat(wav_header header, T *samples)
{
    constexpr double C = 2.3;
    constexpr int num_seconds = 10;
    samples += header.sample_rate * channels * num_seconds * 2;

    double e = 0;
    int num_of_beats = 0;
    for(int i = 0; i < header.sample_rate * channels * num_seconds; i++)
	e += samples[i] * samples[i];
    e /= (header.sample_rate * channels * num_seconds);
    int curr_idx = 0;
    while(curr_idx < header.sample_rate * channels * num_seconds)
    {
	double local_e = 0;
	for(int j = 0; j < 1024 * channels; j++)
	{
	    local_e += samples[curr_idx] * samples[curr_idx];
	    curr_idx++;
	}
	local_e /= (1024 * channels);
	if(C * e <= local_e) 
	    num_of_beats++;
    }
    return (double)num_of_beats / num_seconds;
}

struct player
{
    player(char *argv)
    {
	string command = "exec afplay \"";
	command += argv;
	command += "\" &";
	system(command.c_str());
    }

    ~player()
    {
	system("killall afplay");
    }
};

int main(int argc, char** argv)
{
    player p(argv[1]);

    FILE *f = fopen(argv[1], "r");
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    rewind(f);
    wav_header header;
    fread(&header, sizeof(header), 1, f);
    uint8_t *samples = (uint8_t *)malloc(size - sizeof(header));
    fread(samples, sizeof(uint8_t), size - sizeof(header), f);
    string aIn = argv[2];
    string parsed = aIn.substr(0, aIn.find(" "));
    double beat_timing = atof(parsed.c_str()) / 60.0;
    int64_t first_beat = round(atof(argv[3]) / 1000.0f);
    int64_t ms_per_beat = round(1000 / beat_timing);
    
    constexpr int fps = 10;
    constexpr int total_duration = 1000 / fps;
    srand(time(NULL));
    VideoCapture cap(0);
    if(!cap.isOpened())
    {
        printf("failed to open webcam\n");
        return -1;
    }
    namedWindow("bork", CV_WINDOW_AUTOSIZE);
    int64_t ms_since_beat = ms_per_beat - first_beat;
    for(;;)
    {
        bool beat = false;
        if(ms_since_beat >= ms_per_beat)
        {
            cout << "beat, " << ms_since_beat << "ms " << endl;
            beat = true;
            ms_since_beat = 0;
        }
        auto start = std::chrono::system_clock::now();
        Mat src;
        cap >> src;
        Mat bork = art(src, beat);
        imshow("bork", bork);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<float> duration = end - start;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        ms_since_beat += ms.count();
        if(waitKey(16) >= 0) break;
    }
    return 0;
/*
//    uint8_t *mp3 = (uint8_t *)malloc(size);
//    fread(mp3, sizeof(uint8_t), size, f);
    

    for(size_t i = 0; i < size; i++)
    {
	mp3_header h = *reinterpret_cast<mp3_header *>(mp3 + i);
	uint32_t h2 = *reinterpret_cast<uint32_t *>(&h);
//	bitset<32> bits(h2);
//	cout << bits << endl;
//	printf("%u \n", h.audio_version);
	//sorry we can only deal with MPEG-2
//	assert(h.audio_version == 0x2);
	// 108 bytes per frame, 4 bytes for header
	for (size_t j = i++; j < 26; j++) {
//	    amp.push_back(
	
	if (h.frame_sync != 0b11111111111)
	    continue;
	
	int bit_rate = bitrates[h.bitrate_idx][bitrate_idx(h.audio_version, h.layer)] * 1000;
	int sample_rate = sample_rates[h.freq][h.layer];
	int frame_length = (12 * bit_rate / sample_rate + 32 * h.padding) * 4;
	if(h.layer < 3)
	    frame_length = 144 * bit_rate / sample_rate + 8 * h.padding;
    }
*/
    return 0;
}
