#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QPushButton>
#include <QProgressBar>
#include <QFileDialog>
#include <QLabel>
#include <QMessageBox>
#include <QThread>
#include <QPixmap>
#include <QScrollArea>
#include <QScrollBar>
#include <QMap>
#include <grpcpp/grpcpp.h>
#include "ocr_service.grpc.pb.h"
#include <fstream>
#include <memory>

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using ocr::OCRService;
using ocr::ImageRequest;
using ocr::OCRResponse;

class OCRClient {
private:
    std::unique_ptr<OCRService::Stub> stub_;
    
public:
    OCRClient(std::shared_ptr<Channel> channel)
        : stub_(OCRService::NewStub(channel)) {}
    
    bool ProcessImage(const std::string& filename, const std::vector<uint8_t>& image_data,
                     int batch_id, int image_id, std::string& result, double& time_ms,
                     std::vector<uint8_t>& processed_image) {
        ImageRequest request;
        request.set_filename(filename);
        request.set_image_data(image_data.data(), image_data.size());
        request.set_batch_id(batch_id);
        request.set_image_id(image_id);
        
        ClientContext context;
        std::unique_ptr<grpc::ClientReader<OCRResponse>> reader(
            stub_->ProcessImage(&context, request));
        
        OCRResponse response;
        if (reader->Read(&response)) {
            result = response.extracted_text();
            time_ms = response.processing_time_ms();
            
            const std::string& img_data = response.processed_image();
            processed_image.assign(img_data.begin(), img_data.end());
            
            Status status = reader->Finish();
            return status.ok();
        }
        
        return false;
    }
};

class ProcessThread : public QThread {
    Q_OBJECT
    
private:
    OCRClient* client;
    QString filename;
    std::vector<uint8_t> image_data;
    int batch_id;
    int image_id;
    
public:
    ProcessThread(OCRClient* client, const QString& filename, 
                 const std::vector<uint8_t>& data, int batch_id, int image_id)
        : client(client), filename(filename), image_data(data), 
          batch_id(batch_id), image_id(image_id) {}
    
signals:
    void resultReady(int id, QString filename, QString text, double time_ms, QByteArray processedImage);
    void processingError(int id, QString filename, QString error);
    
protected:
    void run() override {
        std::string result;
        double time_ms;
        std::vector<uint8_t> processed_image;
        
        try {
            bool success = client->ProcessImage(
                filename.toStdString(), image_data, batch_id, image_id, result, time_ms, processed_image);
            
            if (success) {
                QByteArray imgData(reinterpret_cast<const char*>(processed_image.data()), 
                                  processed_image.size());
                emit resultReady(image_id, filename, QString::fromStdString(result), time_ms, imgData);
            } else {
                emit processingError(image_id, filename, "gRPC call failed");
            }
        } catch (const std::exception& e) {
            emit processingError(image_id, filename, QString::fromStdString(e.what()));
        }
    }
};

class ResultWidget : public QWidget {
    Q_OBJECT
    
private:
    QLabel* imageLabel;
    QLabel* textLabel;
    bool isProcessing;
    
public:
    ResultWidget(QWidget* parent = nullptr) 
        : QWidget(parent), isProcessing(true) {
        
        setFixedSize(130, 130);
        
        QVBoxLayout* layout = new QVBoxLayout(this);
        layout->setSpacing(5);
        layout->setContentsMargins(8, 8, 8, 8);
        
        // Image placeholder
        imageLabel = new QLabel();
        imageLabel->setStyleSheet("background-color: white; border: 1px solid #999;");
        imageLabel->setAlignment(Qt::AlignCenter);
        imageLabel->setFixedSize(114, 80);
        layout->addWidget(imageLabel, 0, Qt::AlignCenter);
        
        // Text label
        textLabel = new QLabel("In progress");
        textLabel->setWordWrap(true);
        textLabel->setAlignment(Qt::AlignCenter);
        textLabel->setStyleSheet("font-size: 10px; color: #ccc;");
        textLabel->setFixedHeight(30);
        layout->addWidget(textLabel);
        
        setStyleSheet("ResultWidget { background-color: #3a3a3a; border-radius: 4px; }");
    }
    
    void setResult(const QString& text, const QByteArray& imageData) {
        isProcessing = false;
        
        QPixmap pixmap;
        if (pixmap.loadFromData(imageData)) {
            QPixmap scaled = pixmap.scaled(114, 80, Qt::KeepAspectRatio, Qt::SmoothTransformation);
            imageLabel->setPixmap(scaled);
        } else {
            imageLabel->setText("Error");
            imageLabel->setStyleSheet("background-color: #ffe0e0; border: 1px solid #999; color: red;");
        }
        
        QString displayText = text;
        if (displayText.length() > 30) {
            displayText = displayText.left(27) + "...";
        }
        textLabel->setText(displayText);
        textLabel->setStyleSheet("font-size: 10px; color: white;");
    }
    
    void setError(const QString& error) {
        isProcessing = false;
        imageLabel->setText("ERROR");
        imageLabel->setStyleSheet("background-color: #ffe0e0; border: 1px solid #999; color: red; font-size: 10px;");
        
        textLabel->setText(error.left(30));
        textLabel->setStyleSheet("font-size: 9px; color: #ff6666;");
    }
};

class OCRWindow : public QMainWindow {
    Q_OBJECT
    
private:
    OCRClient* client;
    QPushButton* uploadButton;
    QProgressBar* progressBar;
    QWidget* resultsContainer;
    QGridLayout* resultsLayout;
    QScrollArea* scrollArea;
    
    int current_batch_id;
    int total_images;
    int completed_images;
    std::vector<ProcessThread*> active_threads;
    QMap<int, ResultWidget*> resultWidgets;
    
    static const int COLUMNS = 4;
    
public:
    OCRWindow(OCRClient* client, QWidget* parent = nullptr) 
        : QMainWindow(parent), client(client), current_batch_id(1), 
          total_images(0), completed_images(0) {
        
        setWindowTitle("Distributed OCR System");
        setMinimumSize(620, 580);
        
        setStyleSheet(
            "QMainWindow { background-color: #2b2b2b; }"
            "QLabel { color: white; }"
            "QPushButton { "
            "   background-color: #4a4a4a; "
            "   color: white; "
            "   border: none; "
            "   border-radius: 4px; "
            "   padding: 8px; "
            "   font-size: 12px; "
            "}"
            "QPushButton:hover { background-color: #5a5a5a; }"
            "QProgressBar { "
            "   border: 1px solid #555; "
            "   border-radius: 3px; "
            "   text-align: center; "
            "   background-color: #3a3a3a; "
            "   color: white; "
            "   height: 20px; "
            "}"
            "QProgressBar::chunk { background-color: #4a90e2; }"
        );
        
        QWidget* centralWidget = new QWidget(this);
        centralWidget->setStyleSheet("background-color: #2b2b2b;");
        QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);
        mainLayout->setSpacing(10);
        mainLayout->setContentsMargins(15, 15, 15, 15);
        
        // Upload button
        uploadButton = new QPushButton("Upload Images");
        uploadButton->setMinimumHeight(35);
        connect(uploadButton, &QPushButton::clicked, this, &OCRWindow::onUploadClicked);
        mainLayout->addWidget(uploadButton);
        
        // Progress bar
        progressBar = new QProgressBar();
        progressBar->setMinimum(0);
        progressBar->setMaximum(100);
        progressBar->setValue(0);
        progressBar->setTextVisible(true);
        progressBar->setFormat("%p%");
        mainLayout->addWidget(progressBar);
        
        // Results scroll area
        scrollArea = new QScrollArea();
        scrollArea->setWidgetResizable(true);
        scrollArea->setStyleSheet(
            "QScrollArea { "
            "   background-color: #2b2b2b; "
            "   border: none; "
            "}"
            "QScrollBar:vertical { "
            "   background: #2b2b2b; "
            "   width: 12px; "
            "   margin: 0px; "
            "}"
            "QScrollBar::handle:vertical { "
            "   background: #555; "
            "   border-radius: 6px; "
            "   min-height: 20px; "
            "}"
            "QScrollBar::handle:vertical:hover { background: #666; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        );
        
        resultsContainer = new QWidget();
        resultsContainer->setStyleSheet("background-color: #2b2b2b;");
        resultsLayout = new QGridLayout(resultsContainer);
        resultsLayout->setSpacing(10);
        resultsLayout->setContentsMargins(5, 5, 5, 5);
        resultsLayout->setAlignment(Qt::AlignTop | Qt::AlignLeft);
        
        scrollArea->setWidget(resultsContainer);
        mainLayout->addWidget(scrollArea);
        
        setCentralWidget(centralWidget);
    }
    
    ~OCRWindow() {
        for (auto thread : active_threads) {
            thread->wait();
            delete thread;
        }
    }
    
private slots:
    void onUploadClicked() {
        QStringList filenames = QFileDialog::getOpenFileNames(
            this, "Select Images", "", 
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)");
        
        if (filenames.isEmpty()) return;
        
        // Clear previous results if starting new batch
        if (total_images > 0 && completed_images == total_images) {
            current_batch_id++;
            clearResults();
        }
        
        for (const QString& filename : filenames) {
            std::ifstream file(filename.toStdString(), std::ios::binary);
            if (!file.is_open()) {
                QMessageBox::warning(this, "Error", 
                    "Could not open file: " + filename);
                continue;
            }
            
            std::vector<uint8_t> image_data(
                (std::istreambuf_iterator<char>(file)),
                std::istreambuf_iterator<char>());
            file.close();
            
            QString basename = QFileInfo(filename).fileName();
            
            int image_id = total_images;
            
            // Create placeholder widget
            ResultWidget* widget = new ResultWidget();
            int row = image_id / COLUMNS;
            int col = image_id % COLUMNS;
            resultsLayout->addWidget(widget, row, col);
            resultWidgets[image_id] = widget;
            
            // Create processing thread
            ProcessThread* thread = new ProcessThread(
                client, basename, image_data, current_batch_id, image_id);
            
            connect(thread, &ProcessThread::resultReady, 
                    this, &OCRWindow::onResultReady);
            connect(thread, &ProcessThread::processingError, 
                    this, &OCRWindow::onProcessingError);
            connect(thread, &ProcessThread::finished, 
                    thread, &ProcessThread::deleteLater);
            
            active_threads.push_back(thread);
            total_images++;
            
            thread->start();
        }
        
        updateProgress();
    }
    
    void onResultReady(int id, QString filename, QString text, double time_ms, QByteArray processedImage) {
        completed_images++;
        
        if (resultWidgets.contains(id)) {
            resultWidgets[id]->setResult(text, processedImage);
        }
        
        updateProgress();
        
        if (completed_images == total_images) {
            QMessageBox::information(this, "Complete", 
                QString("Successfully processed all %1 images!").arg(total_images));
        }
    }
    
    void onProcessingError(int id, QString filename, QString error) {
        completed_images++;
        
        if (resultWidgets.contains(id)) {
            resultWidgets[id]->setError(error);
        }
        
        updateProgress();
    }
    
    void updateProgress() {
        if (total_images == 0) {
            progressBar->setValue(0);
        } else {
            int progress = (completed_images * 100) / total_images;
            progressBar->setValue(progress);
        }
    }
    
    void clearResults() {
        total_images = 0;
        completed_images = 0;
        
        QLayoutItem* item;
        while ((item = resultsLayout->takeAt(0)) != nullptr) {
            delete item->widget();
            delete item;
        }
        
        resultWidgets.clear();
        progressBar->setValue(0);
    }
};

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    
    std::string server_address = "localhost:50051";
    if (argc > 1) {
        server_address = argv[1];
    }
    
    auto channel = grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
    OCRClient client(channel);
    
    OCRWindow window(&client);
    window.show();
    
    return app.exec();
}

#include "client.moc"