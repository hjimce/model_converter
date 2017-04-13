#include <iostream>
#include <tiny_dnn/tiny_dnn.h>

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
//构建网络模型
void construct_net(network<sequential> &nn)
{
    nn << convolutional_layer<relu>(32, 32, 5, 1,
                                     6, padding::same)
    << max_pooling_layer<identity>(32, 32, 6, 2)
    << convolutional_layer<relu>(16, 16, 5, 6,
                                  16,  padding::same)
    << max_pooling_layer<identity>(16, 16, 16, 2)
    << convolutional_layer<relu>(8, 8, 5, 16, 32,
                                  padding::same)
    << max_pooling_layer<identity>(8, 8, 32, 2)
    << convolutional_layer<relu>(4, 4, 1, 32, 32,
                                    padding::same)
    << max_pooling_layer<identity>(4, 4, 32, 2)
    << fully_connected_layer<softmax>(120, 10);
}
//加载标签文件
void load_label_flie(std::string label_flie_path,std::vector<std::pair<std::string,int>>&image_paths){
    image_paths.clear();
    std::ifstream file(label_flie_path);
    std::string line;
    while(std::getline(file,line)){
        std::stringstream linestream(line);
        std::string image_path;
        int label;
        linestream>>image_path>>label;
        std::cout<<image_path<<label<<std::endl;
        image_paths.push_back(std::pair<std::string,int>(image_path,label));
    }

}
void load_batch(std::vector<std::pair<std::string,int>>&image_paths,int batch_size,
                std::vector<vec_t> &train_images,std::vector<label_t>& train_labels){
//std::shuffle(image_paths.begin(),image_paths.end(),std::default_random_engine ());
    for(int i=0;i<batch_size;i++){

    }



}

void train_model(const std::string &data_dir_path,
                        double learning_rate,
                        const int n_train_epochs,
                        const int n_minibatch) {
    // specify loss-function and learning strategy
    network<sequential> nn;
    construct_net(nn);
    adagrad optimizer;


    std::cout << "load models..." << std::endl;

    // load MNIST dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;


    std::cout << "start training" << std::endl;

    progress_display disp(train_images.size());
    timer t;

    optimizer.alpha *=
            std::min(tiny_dnn::float_t(4),
                     static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

    int epoch = 1;
    // create callback
    auto on_enumerate_epoch = [&]() {
        std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
        << t.elapsed() << "s elapsed." << std::endl;
        ++epoch;
        tiny_dnn::result res = nn.test(test_images, test_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

    // training
    nn.train<>(optimizer, train_images, train_labels, n_minibatch,
                  n_train_epochs, on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);
    // save network model & trained weights
    nn.save("LeNet-model");
}


int main(int argc, char **argv) {

    std::vector<std::pair<std::string,int>>image_paths;
    load_label_flie("./data/train.txt",image_paths);
    int nepoch=10;
    for(int i=0;i<nepoch;i++){

    }


    return 0;
}