/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_runtime.h>

#include <libsgm.h>

#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

struct device_buffer
{
	device_buffer() : data(nullptr) {}
	device_buffer(size_t count) { allocate(count); }
	void allocate(size_t count) { cudaMalloc(&data, count); }
	~device_buffer() { cudaFree(data); }
	void* data;
};

// Camera Parameters
struct CameraParameters
{
	float fu;                 //!< focal length x (pixel)
	float fv;                 //!< focal length y (pixel)
	float u0;                 //!< principal point x (pixel)
	float v0;                 //!< principal point y (pixel)
	float baseline;           //!< baseline (meter)
	float height;             //!< height position (meter), ignored when ROAD_ESTIMATION_AUTO
	float tilt;               //!< tilt angle (radian), ignored when ROAD_ESTIMATION_AUTO
	float p0;
	float p1;
};

// Transformation between pixel coordinate and world coordinate
struct CoordinateTransform
{
	CoordinateTransform(const CameraParameters& camera) : camera(camera)
	{
		bf = camera.baseline * camera.fu;
		invfu = 1.f / camera.fu;
		invfv = 1.f / camera.fv;
	}

	inline cv::Point3f imageToWorld(const cv::Point2f& pt, float d) const
	{
		const float u = pt.x;
		const float v = pt.y;

		const float Zc = bf / d;
		const float Xc = invfu * ((u - camera.u0) * Zc - camera.p0);
		const float Yc = invfv * ((v - camera.v0) * Zc - camera.p1);

		return cv::Point3f(Xc, Yc, Zc);
	}

	CameraParameters camera;
	float sinTilt, cosTilt, bf, invfu, invfv;
};

template <class... Args>
static std::string format_string(const char* fmt, Args... args)
{
	const int BUF_SIZE = 1024;
	char buf[BUF_SIZE];
	std::snprintf(buf, BUF_SIZE, fmt, args...);
	return std::string(buf);
}

static cv::Scalar computeColor(float val)
{
	const float hscale = 6.f;
	float h = 0.6f * (1.f - val), s = 1.f, v = 1.f;
	float r, g, b;

	static const int sector_data[][3] =
	{ { 1,3,0 },{ 1,0,2 },{ 3,0,1 },{ 0,2,1 },{ 0,1,3 },{ 2,1,0 } };
	float tab[4];
	int sector;
	h *= hscale;
	if (h < 0)
		do h += 6; while (h < 0);
	else if (h >= 6)
		do h -= 6; while (h >= 6);
	sector = cvFloor(h);
	h -= sector;
	if ((unsigned)sector >= 6u)
	{
		sector = 0;
		h = 0.f;
	}

	tab[0] = v;
	tab[1] = v*(1.f - s);
	tab[2] = v*(1.f - s*h);
	tab[3] = v*(1.f - s*(1.f - h));

	b = tab[sector_data[sector][0]];
	g = tab[sector_data[sector][1]];
	r = tab[sector_data[sector][2]];
	return 255 * cv::Scalar(b, g, r);
}

void reprojectPointsTo3D(const cv::Mat& disparity, const CameraParameters& camera, std::vector<cv::Point3f>& points, bool subpixeled)
{
	CV_Assert(disparity.type() == CV_32F);

	CoordinateTransform tf(camera);

	points.clear();
	points.reserve(disparity.rows * disparity.cols);

	for (int y = 0; y < disparity.rows; y++)
	{
		for (int x = 0; x < disparity.cols; x++)
		{
			const float d = disparity.at<float>(y, x);
			if (d > 0)
				points.push_back(tf.imageToWorld(cv::Point(x, y), d));
		}
	}
}

void drawPoints3D(const std::vector<cv::Point3f>& points, cv::Mat& draw)
{
	const int SIZE_X = 1024;
	const int SIZE_Z = 1024;
	const int maxz = 80; // [meter]
	const double pixelsPerMeter = 1. * SIZE_Z / maxz;

	draw = cv::Mat::zeros(SIZE_Z, SIZE_X, CV_8UC3);

	for (const cv::Point3f& pt : points)
	{
		const float X = pt.x;
		const float Z = pt.z;

		const int u = cvRound(pixelsPerMeter * X) + SIZE_X / 2;
		const int v = SIZE_Z - cvRound(pixelsPerMeter * Z);

		const cv::Scalar color = computeColor(std::min(Z, 1.f * maxz) / maxz);
		cv::circle(draw, cv::Point(u, v), 1, color);
	}
}

void saveVectorToFile(std::vector<cv::Point3f> points, std::string file) 
{
	std::ofstream outFile(file);
	for (const cv::Point3f e : points) {
		outFile << e.x << ";";
		outFile << e.y << ";";
		outFile << e.z << "\n";
	}	
}

int main(int argc, char* argv[])
{
	// point clouds are saved
	if (argc < 4) {
		std::cout << "usage: " << argv[0] << " left-image-format right-image-format camera.xml [disp_size] [subpixel_enable(0: false, 1:true)]" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	const int first_frame = 1;
	cv::FileStorage fs(format_string(argv[3], first_frame), cv::FileStorage::READ);
	const int disp_size = argc >= 5 ? std::stoi(argv[4]) : 128;
	const bool subpixel = argc >= 6 ? std::stoi(argv[5]) != 0 : true;
	const int input_depth = 8;
	const int output_depth = 16;

	std::stringstream fileLocation;
	for (int frame_no = first_frame;; frame_no++) {

		cv::Mat I1 = cv::imread(format_string(argv[1], frame_no), -1);
		cv::Mat I2 = cv::imread(format_string(argv[2], frame_no), -1);
		if (I1.empty() || I2.empty()) {
			break;
		}

		cv::Mat I1_Gray, I2_Gray;
		cv::cvtColor(I1, I1_Gray, cv::COLOR_BGR2GRAY);
		cv::cvtColor(I2, I2_Gray, cv::COLOR_BGR2GRAY);
		
	        const int width = I1.cols;
	        const int height = I1.rows;
	        const int input_bytes = width * height * sizeof(uint8_t);
	        const int output_bytes = width * height * sizeof(int16_t);

	        const sgm::StereoSGM::Parameters params{6, 96, 0.95f, subpixel};
	        sgm::StereoSGM sgm(width, height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA, params);

	        cv::Mat disparity(height, width, CV_16SC1);
	        cv::Mat disparity_8u, disparity_32f, disparity_color, draw;
        	std::vector<cv::Point3f> points;

	        device_buffer d_I1(input_bytes), d_I2(input_bytes), d_disparity(output_bytes);
		
		// update camera parameters
		cv::FileStorage fs(format_string(argv[3], frame_no), cv::FileStorage::READ);
		ASSERT_MSG(fs.isOpened(), "camera.xml read failed.");
		CameraParameters camera;
		camera.fu = fs["FocalLengthX"];
		camera.fv = fs["FocalLengthY"];
		camera.u0 = fs["CenterX"];
		camera.v0 = fs["CenterY"];
		camera.baseline = fs["BaseLine"];
		camera.tilt = fs["Tilt"];
		camera.p0 = fs["P0"];
		camera.p1 = fs["P1"];
		// TODO: Read from xml file.
                //camera.p0 = 4.575831000000e+01;
		//camera.p1 = -3.454157000000e-01;

		cudaMemcpy(d_I1.data, I1_Gray.data, input_bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_I2.data, I2_Gray.data, input_bytes, cudaMemcpyHostToDevice);

		const auto t1 = std::chrono::system_clock::now();

		sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
		cudaDeviceSynchronize();

		const auto t2 = std::chrono::system_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		const double fps = 1e6 / duration;

		cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);

		disparity.convertTo(disparity_32f, CV_32FC1, subpixel ? 1. / sgm::StereoSGM::SUBPIXEL_SCALE : 1);
		reprojectPointsTo3D(disparity_32f, camera, points, subpixel);
		fileLocation.str("");
		fileLocation << "./csv/" << std::setfill('0') << std::setw(6) << frame_no << ".csv";
		saveVectorToFile(points, fileLocation.str());
		
		/*
		if (I1.type() != CV_8U) {
                        cv::normalize(I1, I1, 0, 255, cv::NORM_MINMAX);
                        I1.convertTo(I1, CV_8U);
                }
                if (I2.type() != CV_8U) {
                        cv::normalize(I2, I2, 0, 255, cv::NORM_MINMAX);
                        I2.convertTo(I2, CV_8U);
                }
		   
		drawPoints3D(points, draw);

		disparity_32f.convertTo(disparity_8u, CV_8U, 255. / disp_size);
		cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
		disparity_color.setTo(cv::Scalar(0, 0, 0), disparity_32f < 0); // invalid disparity will be negative
		cv::putText(disparity_color, format_string("sgm execution time: %4.1f[msec] %4.1f[FPS]", 1e-3 * duration, fps),
			cv::Point(50, 50), 2, 0.75, cv::Scalar(255, 255, 255));
		cv::imwrite("left_image.png", I1);
		cv::imwrite("right_image.png", I2);
		cv::imwrite("disparity.png", disparity_color);
		cv::imwrite("points.png", draw);
		return 0;*/
		std::cerr << "Processed frame no " << frame_no << std::endl;
	}

	return 0;
}
