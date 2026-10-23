#include "pipeline.hpp"

#define OFFSET 5
#define OFFSET_Y 5

using namespace std;
using namespace cv;

// Глобальные переменные и вспомогательная функция для обработки прерывания
std::shared_ptr<AutopilotInterface> autopilot_interface_quit_;
std::shared_ptr<UartInterface> port_quit_;
std::shared_ptr<CameraInterface> cam_quit_;
void quit_handler( int sig );


// Cоздаем обработчик m_frameProcessor для детекции, сравнения особых точек
Pipeline::Pipeline(int threshold, int octaves)
	: frameProcessor_("BRISK", threshold, octaves)
{
}

// Основной пайплайн
int Pipeline::process_video()
{
	Mat first, second;

	// FeatureInfo - структура для хранения ключевых точек и их дескрипторов 
	FeatureInfo firstInfo, secondInfo;

	cv::Rect cropRect;
	cv::Point2f shift;
	OpticalFlowLkt opticalflow;

	std::shared_ptr<UartInterface> port = std::make_shared<UartInterface>("/dev/ttyACM0", 115200);
	std::shared_ptr<AutopilotInterface> autopilot = std::make_shared<AutopilotInterface>(port);

	port_quit_ = port;
	autopilot_interface_quit_ = autopilot;
	signal(SIGINT,quit_handler);

	std::shared_ptr<CameraInterface> cam = std::make_shared<CameraInterface>();
	cam_quit_ = cam;   // иначе quit_handler обращается к пустому указателю при Ctrl+C
	cam->open();

	// Если захват видео не удался - вывод сообщения и завершение программы
	if (!cam->is_opened_){ 
		std::cout << "Video source is not opened" << std::endl;
		cam->close();
		return -1;
	}

	port->start();
	autopilot->start();

	// Если захватили кадр - начинаем обработку
	if (cam->camera_connected())
	{
		// Загружаем изображение. Загружаем в second, чтобы далее сравнивать соседние кадры
		// Т.е. меняем second и first местами каждый раз, загружаем последующее изображение в 
		// second
		second = cam->get_frame();
		previous_img_capture_time_ = std::chrono::high_resolution_clock::now();

		// Cоздаем шаблон, с разрешением на 10 пикселей меньше по высоте и ширине исходного
		cropRect = Rect(OFFSET_Y, OFFSET, second.cols-2*OFFSET_Y, second.rows-2*OFFSET);

		// std::ostringstream saving_path;
		// output_folder_ = create_output_folder();
    	// saving_path << output_folder_ << "/frame_" << std::setfill('0') << std::setw(6) << save_counter_++ << ".jpg";
    	// cv::imwrite(saving_path.str(), second);

		// Обрезаем исходное изображение по шаблону (по 5 пикселей с каждой стороны).
		// Т.к. наибольшие искажения наблюдаются в близи к краям изображения, то просто обрезаем их 
		second = Mat(second, cropRect);

		// Переводим в градацию серого
		cv::cvtColor(second,second,cv::COLOR_BGR2GRAY);

		// // Определяем ключевые точки изображения и соответствующие им дескрипторы
		// secondInfo = frameProcessor_.GetKeypointData(second);

		shift = opticalflow.get_optical_flow(second);
	}

	float camera_vfov = calculate_vertical_fov(camera_hfov_, second.cols + 2*OFFSET, second.rows + 2*OFFSET_Y);

	float pixels_per_radian_h = (second.cols + 2*OFFSET) / (camera_hfov_*M_PI / 180);
    float pixels_per_radian_v = (second.rows + 2*OFFSET_Y) / (camera_vfov*M_PI / 180);

	// Пока можем захватывать кадры - обработка
	while (cam->camera_connected())
	{
		first = second.clone();
		swap(firstInfo, secondInfo);
		
		second = cam->get_frame();

		auto frame_grabbed_time = std::chrono::high_resolution_clock::now();
		auto time_diff_btwn_capturing_imgs = std::chrono::duration_cast<std::chrono::microseconds>(frame_grabbed_time - previous_img_capture_time_);
		float diff_btwn_capturing_imgs_sec = time_diff_btwn_capturing_imgs.count()/1000000.0;

		// Если кадр оказался пустым, пропускаем итерацию
		if (second.rows == 0 || second.cols == 0){
			continue;
		}
		
		// std::ostringstream saving_path;
    	// saving_path << output_folder_ << "/frame_" << std::setfill('0') << std::setw(6) << save_counter_++ << ".jpg";
    	// cv::imwrite(saving_path.str(), second);

		second = Mat(second, cropRect);
		cv::cvtColor(second,second,cv::COLOR_BGR2GRAY);
		
		if (show_video_) {
			cv::imshow("video", second);
			cv::waitKey(5);
		}

		// // Сравниваем соседние кадры
		// auto result = frameProcessor_.MatchImages(first, firstInfo, second, secondInfo);

		// // Получаем на основе сравнения матрицу афинных преобразований
		// cv::Mat move = moves_estimator_.EstimateMovements(result);

		shift = opticalflow.get_optical_flow(second);

		// ---- Качество ----
		int quality;
		int good_points = opticalflow.last_good_points();
		if (good_points > 0) {
			dropout_frames_ = 0;
			quality = 255 * good_points / std::max(1, opticalflow.max_good_points());
			quality = std::max(quality_measured_min_, std::min(255, quality));
		} else if (opticalflow.last_was_forecast()) {
			dropout_frames_ = 0;
			quality = quality_forecast_;
		} else {
			dropout_frames_++;
			quality = (dropout_frames_ <= max_dropout_frames_) ? quality_dropout_ : 0;
		}

		// ---- Угловая скорость потока, рад/с (по точному дробному смещению) ----
		float rate_x = 0.0f;
		float rate_y = 0.0f;
		if (quality > 0 && diff_btwn_capturing_imgs_sec > 0.0f) {
			rate_x = shift.x / (pixels_per_radian_h * diff_btwn_capturing_imgs_sec);
			rate_y = shift.y / (pixels_per_radian_v * diff_btwn_capturing_imgs_sec);
		} else {
			flow_x_residual_ = 0.0f;
			flow_y_residual_ = 0.0f;
		}

		// ---- Интервал между отправками ----
		// ArduPilot 4.4-4.6 умножает flow_x на интервал между приходом сообщений,
		// поэтому делим на интервал между отправками, а не между кадрами
		auto send_time = std::chrono::high_resolution_clock::now();
		float dt_send = diff_btwn_capturing_imgs_sec;
		if (have_last_send_time_) {
			dt_send = std::chrono::duration<float>(send_time - last_send_time_).count();
		}
		if (dt_send <= 0.001f || dt_send > 0.5f) {
			dt_send = (diff_btwn_capturing_imgs_sec > 0.001f) ? diff_btwn_capturing_imgs_sec : 0.04f;
		}
		last_send_time_ = send_time;
		have_last_send_time_ = true;

		// ---- Целые flow_x/y для 4.4-4.6 с переносом остатка округления ----
		float fx = flow_gain_k_ * rate_x / dt_send + flow_x_residual_;
		float fy = flow_gain_k_ * rate_y / dt_send + flow_y_residual_;
		float int_flow_x = std::max(-32767.0f, std::min(32767.0f, std::round(fx)));
		float int_flow_y = std::max(-32767.0f, std::min(32767.0f, std::round(fy)));
		flow_x_residual_ = fx - int_flow_x;
		flow_y_residual_ = fy - int_flow_y;

		// flow_rate для 4.7+ масштабируется тем же K, чтобы FLOW_FXSCALER был одинаковым на всех прошивках
		autopilot->write_optical_flow(int_flow_x, int_flow_y,
									  flow_gain_k_ * rate_x, flow_gain_k_ * rate_y, quality);
		std::clog << "shift: " << shift.x << " " << shift.y
				  << "  rate: " << rate_x << " " << rate_y
				  << "  quality: " << quality << std::endl;

		previous_img_capture_time_ = frame_grabbed_time;
	}

	// Закрываем файлы и источник видео
	port->stop();
	autopilot->stop();
	cam->close();

	return 0;
}


float Pipeline::calculate_vertical_fov(float hfov_deg, int width, int height) {
	/**
	* Рассчитывает вертикальный FOV на основе горизонтального FOV и разрешения изображения
	* tan(vfov/2) = tan(hfov/2) * (height/weight)
	*/

	// Проверка входных данных для предотвращения математических ошибок
    if (width <= 0) 
        return 0;
    if (height <= 0)
        return 0;
    if (hfov_deg <= 0 || hfov_deg >= 180)
        return 0;

    try {
        float hfov_rad = hfov_deg * M_PI / 180;
        float aspect_ratio = (float)height / width;

        float vfov_rad = 2 * std::atan(std::tan(hfov_rad / 2) * aspect_ratio);
        float vfov_deg = vfov_rad * 180 / M_PI;

        if (vfov_deg <= 0 || vfov_deg >= 180) {
            return 0;
        }

        return vfov_deg;
    } 
	catch (const std::exception& e) {
        fprintf(stderr,"Mathematical error calculating VFOV\n");
        return 0;
    }
}


void quit_handler(int sig)
{

	printf("\n");
	printf("TERMINATING AT USER REQUEST\n");
	printf("\n");

	try {
		autopilot_interface_quit_->handle_quit(sig);
	}
	catch (int error){}

	try {
		port_quit_->stop();
	}
	catch (int error){}

		try {
		cam_quit_->close();
	}
	catch (int error){}

	// Завершение программы
	exit(0);

}

std::string Pipeline::create_output_folder() 
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);

    std::ostringstream oss;
    oss << "/home/teleskret/work/frame_shifts/build/" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    std::string folder_name = oss.str();

    mkdir(folder_name.c_str(), 0777);
    return folder_name;
}