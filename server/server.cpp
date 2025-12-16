#include <grpcpp/grpcpp.h>
#include "ocr_service.grpc.pb.h"
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <chrono>
#include <algorithm>
#include <fstream>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using ocr::OCRService;
using ocr::ImageRequest;
using ocr::OCRResponse;

// Structure to hold both text and processed image
struct OCRResult {
    std::string text;
    double time_ms;
    std::vector<uint8_t> processed_image;
};

// Thread pool task structure
struct OCRTask {
    ImageRequest request;
    OCRResponse* response;
    std::condition_variable* cv;
    std::mutex* mtx;
    bool* completed;
};

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<OCRTask> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    
    OCRResult process_image(const std::vector<uint8_t>& image_data) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Write image data to temporary file
        std::string temp_filename = "ocr_temp_" + 
            std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) + ".png";
        std::ofstream temp_file(temp_filename, std::ios::binary);
        temp_file.write(reinterpret_cast<const char*>(image_data.data()), image_data.size());
        temp_file.close();
        
        tesseract::TessBaseAPI api;
        const std::string lang = "eng";
        const char* tessdata_path = "./tessdata";
        
        if (api.Init(tessdata_path, lang.c_str(), tesseract::OEM_LSTM_ONLY)) {
            std::remove(temp_filename.c_str());
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            return {"[ERROR: Tesseract initialization failed]", elapsed, {}};
        }
        
        api.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
        
        Pix* image = pixRead(temp_filename.c_str());
        std::remove(temp_filename.c_str());
        
        if (!image) {
            api.End();
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            return {"[ERROR: Unable to open image]", elapsed, {}};
        }
        
        // Preprocessing
        Pix* gray = pixConvertTo8(image, false);
        pixDestroy(&image);
        
        int w = pixGetWidth(gray);
        int h = pixGetHeight(gray);
        
        Pix* scaled = gray;
        if (w < 500 || h < 250) {
            float scale = std::max(500.0f / w, 250.0f / h);
            scaled = pixScale(gray, scale, scale);
            pixDestroy(&gray);
        }
        
        Pix* sharpened = pixUnsharpMaskingGray(scaled, 5, 2.5);
        if (sharpened) {
            pixDestroy(&scaled);
            scaled = sharpened;
        }
        
        Pix* contrast = pixContrastNorm(NULL, scaled, 50, 50, 130, 2, 2);
        if (contrast) {
            pixDestroy(&scaled);
            scaled = contrast;
        }
        
        Pix* binary = pixOtsuThreshOnBackgroundNorm(scaled, NULL, 10, 10, 100, 50, 10, 10, 10, 0.1, NULL);
        Pix* final_image = binary ? binary : scaled;
        
        // Save the processed image to memory
        std::string processed_filename = "processed_" + 
            std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) + ".png";
        pixWrite(processed_filename.c_str(), final_image, IFF_PNG);
        
        // Read processed image into vector
        std::ifstream proc_file(processed_filename, std::ios::binary);
        std::vector<uint8_t> processed_data(
            (std::istreambuf_iterator<char>(proc_file)),
            std::istreambuf_iterator<char>());
        proc_file.close();
        std::remove(processed_filename.c_str());
        
        // Perform OCR
        api.SetImage(final_image);
        char* raw_text = api.GetUTF8Text();
        std::string text = raw_text ? raw_text : "";
        
        text.erase(std::remove_if(text.begin(), text.end(), 
            [](unsigned char c) { return c > 127 || (c < 32 && c != ' ' && c != '\n'); }), text.end());
        
        if (text.empty() || text.length() < 2) {
            api.SetPageSegMode(tesseract::PSM_SINGLE_WORD);
            api.SetImage(final_image);
            delete[] raw_text;
            raw_text = api.GetUTF8Text();
            text = raw_text ? raw_text : "";
            
            text.erase(std::remove_if(text.begin(), text.end(), 
                [](unsigned char c) { return c > 127 || (c < 32 && c != ' ' && c != '\n'); }), text.end());
        }
        
        delete[] raw_text;
        if (binary) pixDestroy(&binary);
        pixDestroy(&scaled);
        api.End();
        
        // Clean up text
        text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
        text.erase(std::remove(text.begin(), text.end(), '\t'), text.end());
        
        size_t start_pos = text.find_first_not_of(" \n");
        size_t end_pos = text.find_last_not_of(" \n");
        if (start_pos != std::string::npos && end_pos != std::string::npos) {
            text = text.substr(start_pos, end_pos - start_pos + 1);
        } else {
            text = "";
        }
        
        if (text.empty()) {
            text = "[UNREADABLE]";
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        
        return {text, elapsed, processed_data};
    }
    
    void worker() {
        while (true) {
            OCRTask task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                condition.wait(lock, [this] { return stop || !tasks.empty(); });
                
                if (stop && tasks.empty()) return;
                
                task = std::move(tasks.front());
                tasks.pop();
            }
            
            // Process the image
            std::cout << "[Worker " << std::this_thread::get_id() << "] Processing: " 
                      << task.request.filename() << std::endl;
            
            std::vector<uint8_t> image_data(task.request.image_data().begin(), 
                                           task.request.image_data().end());
            auto result = process_image(image_data);
            
            // Fill response
            task.response->set_image_id(task.request.image_id());
            task.response->set_filename(task.request.filename());
            task.response->set_extracted_text(result.text);
            task.response->set_processing_time_ms(result.time_ms);
            task.response->set_success(true);
            task.response->set_processed_image(result.processed_image.data(), result.processed_image.size());
            
            std::cout << "[Worker " << std::this_thread::get_id() << "] Completed: " 
                      << task.request.filename() << " - \"" << result.text << "\"" << std::endl;
            
            // Notify completion
            {
                std::lock_guard<std::mutex> lock(*task.mtx);
                *task.completed = true;
            }
            task.cv->notify_one();
        }
    }
    
public:
    ThreadPool(size_t num_threads) : stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] { worker(); });
        }
        std::cout << "[ThreadPool] Started with " << num_threads << " worker threads" << std::endl;
    }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
    }
    
    void enqueue(OCRTask task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.push(std::move(task));
        }
        condition.notify_one();
    }
};

class OCRServiceImpl final : public OCRService::Service {
private:
    ThreadPool thread_pool;
    
public:
    OCRServiceImpl(size_t num_threads) : thread_pool(num_threads) {}
    
    Status ProcessImage(ServerContext* context, const ImageRequest* request,
                       grpc::ServerWriter<OCRResponse>* writer) override {
        std::cout << "\n[Server] Received image: " << request->filename() 
                  << " (Batch: " << request->batch_id() << ", ID: " << request->image_id() << ")" << std::endl;
        
        OCRResponse response;
        std::mutex mtx;
        std::condition_variable cv;
        bool completed = false;
        
        OCRTask task{*request, &response, &cv, &mtx, &completed};
        thread_pool.enqueue(task);
        
        // Wait for completion
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&completed] { return completed; });
        }
        
        // Send response back to client
        writer->Write(response);
        
        std::cout << "[Server] Sent response for: " << request->filename() << std::endl;
        
        return Status::OK;
    }
};

void RunServer(const std::string& server_address, size_t num_threads) {
    OCRServiceImpl service(num_threads);
    
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "\n=== OCR Server Running ===" << std::endl;
    std::cout << "Listening on: " << server_address << std::endl;
    std::cout << "Worker threads: " << num_threads << std::endl;
    std::cout << "Press Ctrl+C to stop...\n" << std::endl;
    
    server->Wait();
}

int main(int argc, char** argv) {
    std::string server_address = "0.0.0.0:50051";
    size_t num_threads = 4;
    
    if (argc > 1) {
        server_address = argv[1];
    }
    if (argc > 2) {
        num_threads = std::stoi(argv[2]);
    }
    
    RunServer(server_address, num_threads);
    
    return 0;
}