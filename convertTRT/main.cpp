#include "utils.h"
#include "semantic.h"
#include "detection.h"

int main()
{
    //cv::Mat image = cv::imread("C:\\Users\\Renyupeng\\Desktop\\10000x10000.jpg");
    //cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    //std::vector<cv::Mat> channels;
    //cv::split(image, channels);

    //// std::vector<Barrier*> output;
    //split_param param;
    //param.crop_h = 512;
    //param.crop_w = 512;
    //param.overlap_h = 0.0;
    //param.overlap_w = 0.0;
    //param.merge_h = image.rows;
    //param.merge_w = image.cols;

    semantic* s2t = new semantic("model_s.wts", "model_s.engine");
    s2t->convert2engine();

    detection* d2t = new detection("model_d.wts", "model_d.engine");
    d2t->convert2engine();

    cv::Mat output;
    std::vector<Barrier*> output_det;

    /*

    detection* det = new detection("yolov5m.engine", param);
    semantic* seg = new semantic("unet.engine", param);
    for (int i = 0; i < 100; i++)
    {
        
        //auto start = std::chrono::system_clock::now();
        //seg->process(channels, output);
        //auto end = std::chrono::system_clock::now();
        //std::cout << "seg process time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


        //cv::imshow("unet", output);
        //cv::waitKey(0);
        
        auto start = std::chrono::system_clock::now();
        det->process(channels, output_det);
        auto end = std::chrono::system_clock::now();
        std::cout << "det process time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        for (size_t j = 0; j < output_det.size(); j++) {
            cv::Rect r(cv::Point(output_det[j]->x, output_det[j]->y), cv::Point(output_det[j]->x + output_det[j]->width, output_det[j]->y + output_det[j]->height));

            cv::rectangle(image, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(image, std::to_string((int)output_det[j]->classid) + " " + std::to_string(output_det[j]->confidence), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
        cv::imshow("det", image);
        cv::imwrite("det.jpg", image);
        cv::waitKey(0);
    }

    */
}