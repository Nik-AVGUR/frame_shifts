#ifndef PIPELINE_H_
#define PIPELINE_H_

#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <chrono>
#include <algorithm>
#include <cmath>

// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui.hpp>
// #include <ctime>
// #include <cmath>

#include "image_data.hpp"
#include "frame_processor.hpp"
#include "moves_estimator.hpp"
#include "optical_flow_lkt.hpp"
#include "autopilot_interface.hpp"
#include "uart_interface.hpp"
#include "camera_interface.hpp"

class Pipeline
{
public:

	explicit Pipeline(int threshold = 30, int octaves = 3);

	int process_video();

protected:

	FrameProcessor frameProcessor_;
	MovesEstimator moves_estimator_;

	std::vector<ImageData> matched_data_;
	std::string outFile_;

	std::string output_folder_;
	int save_counter_ = 0;

	// clock_t previous_img_capture_time_ = 0;
	std::chrono::_V2::system_clock::time_point previous_img_capture_time_;
	
	//Для 9 mm - 48.3
	//Для 13 mm - 33.9
	float camera_hfov_ = 48.3f;   // было int: 48.3 -> 48, 33.9 -> 33 (ошибка масштаба до 2.7%)

	// Показ видео в окне (только для отладки на столе: на борту без дисплея не нужен, waitKey добавляет >=5 мс)
	bool show_video_ = false;

	// ---- Совместимость с любой прошивкой ArduPilot ----
	// ArduPilot 4.7+ берет поток из flow_rate_x/y (float, рад/с).
	// ArduPilot 4.4-4.6 берет только целые flow_x/y и считает flowRate = flow_x * dt * (1 + 0.001*FLOW_FXSCALER),
	// где dt - интервал между приходом сообщений на полетник.
	// Оба поля заполняются так, чтобы при ОДНОМ И ТОМ ЖЕ FLOW_FXSCALER полетник получал одинаковую скорость:
	//   flow_rate = K * rate,  flow_x = round(K * rate / dt_send),  FLOW_FXSCALER = FLOW_FYSCALER = 1000/K - 1000.
	// K увеличивает разрешение целого flow_x в K раз (шаг = dt/K рад/с).
	// K = 4 -> FLOW_FXSCALER = FLOW_FYSCALER = -750 (затем уточнить калибровкой, покачав борт руками).
	float flow_gain_k_ = 4.0f;

	// Остаток округления flow_x/y переносится на следующее сообщение (среднее значение сохраняется)
	float flow_x_residual_ = 0.0f;
	float flow_y_residual_ = 0.0f;

	std::chrono::high_resolution_clock::time_point last_send_time_;
	bool have_last_send_time_ = false;

	// ---- Качество (OF.Qual в логе полетника) ----
	// Измерено по точкам: 255 * (число точек / максимум), но не меньше quality_measured_min_
	// Прогноз на время калибровки тепловизора: quality_forecast_
	// Точки не найдены: первые max_dropout_frames_ кадров - нулевой поток с quality_dropout_
	// (FlowHold сохраняет интегральную составляющую), дальше - quality 0 (FlowHold работает как AltHold,
	// EKF перестает использовать поток). Все ненулевые значения должны быть выше FHLD_QUAL_MIN.
	int quality_measured_min_ = 100;
	int quality_forecast_ = 60;
	int quality_dropout_ = 30;
	int max_dropout_frames_ = 12;   // ~0.5 с при 25 кадр/с
	int dropout_frames_ = 0;

	float calculate_vertical_fov(float hfov_deg, int width, int height);
	std::string create_output_folder();
};

#endif // PIPELINE_H_
